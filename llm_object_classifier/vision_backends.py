"""Ollama and OpenAI transports; record unverifiable GPU/seed explicitly."""
from __future__ import annotations
import base64
import json
import os
import time
import urllib.error
import urllib.request
from .experiment import strict_json
from .vision_pipeline import GPURequiredError


def request(url, body=None, timeout=600, headers=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_text = response.read().decode("utf-8")
            try:
                return strict_json(raw_text)
            except Exception as exc:
                exc.result = {"response_text": raw_text}
                raise
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors='replace')
        failure = RuntimeError(f"HTTP {exc.code}: {body_text[:2000]}")
        failure.result = {"http_status": exc.code, "response_text": body_text}
        raise failure from None


class Backend:
    def __init__(self, spec, timeout=600):
        self.spec, self.timeout = spec, timeout
        self.url = spec["base_url"].rstrip("/")
        self.headers = {}
        if spec.get("api_key_env"):
            token = os.environ.get(spec["api_key_env"])
            if not token:
                raise ValueError(f"Missing credential environment variable {spec['api_key_env']}.")
            self.headers = {"Authorization": "Bearer " + token}

    def metadata(self, vision=False):
        if self.spec["backend"] == "openai":
            return {"backend": "openai", "requested_model": self.spec["model"],
                    "declared_revision": self.spec.get("revision"), "gpu_verified": False,
                    "note": "Check GPU, precision, revision and applied settings on the server."}
        info = request(self.url + "/api/show", {"model": self.spec["model"]}, self.timeout, self.headers)
        if vision and "vision" not in info.get("capabilities", []):
            raise ValueError(f"{self.spec['model']} does not report vision support.")
        precision = info.get("details", {}).get("quantization_level", "").upper()
        if self.spec.get("precision") and precision != self.spec["precision"]:
            raise ValueError(f"Expected {self.spec['precision']}, server reports {precision!r}.")
        return {"backend": "ollama", "show": info,
                "server_version": request(self.url + "/api/version", timeout=self.timeout, headers=self.headers)}

    def generate(self, system, prompt, opts, image=None, mime="image/jpeg"):
        if self.spec["backend"] == "ollama":
            user = {"role": "user", "content": prompt}
            if image is not None:
                user["images"] = [base64.b64encode(image).decode()]
            body = {"model": self.spec["model"], "messages": [{"role": "system", "content": system}, user],
                    "stream": False, "think": False, "format": "json", "keep_alive": "5m", "options": opts}
            endpoint, unsupported = "/api/chat", []
        else:
            text = system + "\n\n" + prompt if self.spec.get("system_as_user") else prompt
            content = text
            if image is not None:
                content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(image).decode()}}]
            messages = [{"role": "user", "content": content}]
            if not self.spec.get("system_as_user"):
                messages.insert(0, {"role": "system", "content": system})
            body = {"model": self.spec["model"], "messages": messages, "stream": False,
                    "temperature": opts["temperature"], "top_p": opts["top_p"], "max_tokens": opts["num_predict"]}
            unsupported = ["num_ctx", "top_k", "presence_penalty", "num_gpu"]
            if self.spec.get("send_seed"):
                body["seed"] = opts["seed"]
            else:
                unsupported.append("seed")
            if self.spec.get("json_mode"):
                body["response_format"] = {"type": "json_object"}
            endpoint = "/chat/completions"
        # Save the actual request but not a second copy of base64 image bytes.
        logged_body = json.loads(json.dumps(body))
        for message in logged_body["messages"]:
            if "images" in message:
                message["images"] = ["<image bytes omitted; see image_sha256>"]
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        item["image_url"] = {"url": "<image bytes omitted; see image_sha256>"}
        started = time.perf_counter()
        try:
            raw = request(self.url + endpoint, body, self.timeout, self.headers)
        except Exception as exc:
            exc.result = {**getattr(exc, "result", {}), "request": logged_body,
                          "requested_options": opts, "unsupported_options": unsupported,
                          "elapsed_seconds": time.perf_counter() - started, "gpu_verified": False}
            raise
        elapsed = time.perf_counter() - started
        if self.spec["backend"] == "ollama":
            content, finish = raw.get("message", {}).get("content", ""), raw.get("done_reason")
        else:
            choice = raw.get("choices", [{}])[0]
            content, finish = choice.get("message", {}).get("content", ""), choice.get("finish_reason")
        result = {"content": content, "elapsed_seconds": elapsed, "finish_reason": finish,
                  "requested_options": opts, "unsupported_options": unsupported,
                  "request": logged_body, "response": raw, "gpu_verified": False,
                  "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if self.spec["backend"] == "ollama" and self.spec.get("require_gpu"):
            try:
                state = request(self.url + "/api/ps", timeout=self.timeout, headers=self.headers)
                entry = next((m for m in state.get("models", [])
                              if self.spec["model"] in (m.get("name"), m.get("model"))), None)
                if entry is None or entry.get("size_vram", 0) <= 0:
                    raise ValueError("Requested model has no reported VRAM allocation.")
                if entry.get("size", 0) > 0 and entry["size_vram"] < entry["size"]:
                    raise ValueError("Partial CPU offload detected in model allocation.")
                result.update({"gpu": entry, "gpu_verified": True})
            except Exception as exc:
                failure = GPURequiredError(f"GPU verification failed: {exc}. Recreate the project's Ollama container with the GPU override.")
                failure.result = result
                raise failure from exc
        return result

    def unload(self):
        """Release the current Ollama model between comparison models."""
        if self.spec["backend"] == "ollama":
            request(self.url + "/api/generate", {"model": self.spec["model"], "keep_alive": 0}, self.timeout, self.headers)
