from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .schema import Frame


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: float = 120.0
    extra_body: dict[str, Any] | None = None

    @classmethod
    def from_file(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            base_url=raw["base_url"].rstrip("/"),
            model=raw["model"],
            temperature=float(raw.get("temperature", 0.0)),
            max_tokens=int(raw.get("max_tokens", 512)),
            timeout_seconds=float(raw.get("timeout_seconds", 120.0)),
            extra_body=raw.get("extra_body"),
        )


@dataclass(frozen=True)
class LlmResult:
    content: str
    elapsed_seconds: float


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def classify(self, frame: Frame) -> LlmResult:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(frame)},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.config.extra_body:
            body.update(self.config.extra_body)

        request = urllib.request.Request(
            url=f"{self.config.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Request failed: {exc.reason}") from exc
        elapsed = time.perf_counter() - start

        response_data = json.loads(raw_response)
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenAI-compatible response: {raw_response}") from exc
        if not isinstance(content, str):
            raise RuntimeError("OpenAI-compatible response content is not a string.")
        return LlmResult(content=content, elapsed_seconds=elapsed)

