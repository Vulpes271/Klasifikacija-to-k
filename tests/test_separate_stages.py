"""Real HTTP regressions for isolated image description and text reasoning stages."""
import argparse
import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from llm_object_classifier import stage1_describe, stage2_reason
from llm_object_classifier.description_prompts import validate_description
from llm_object_classifier.experiment import load_config, strict_json
from llm_object_classifier.image_dataset import load_images, prepare
from llm_object_classifier.vision_backends import Backend


def description(label="chair", oid=5):
    return {"scene_description": "A room with a chair.", "objects": [{"id": oid, "label": label,
            "observation": "A wooden chair on four visible legs.",
            "attachment_evidence": "appears_freestanding", "identifiable": True}]}


class SeparateStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.bad = {}
        self.gpu = True
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/ps":
                    self.reply({"models": [{"name": "model-a", "size_vram": 123 if owner.gpu else 0},
                                           {"name": "model-b", "size_vram": 123 if owner.gpu else 0}]})
                else:
                    self.reply({"version": "test-server"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.calls.append((self.path, body))
                if self.path == "/api/show":
                    self.reply({"capabilities": ["vision"], "details": {"quantization_level": "BF16"}})
                    return
                if self.path == "/api/generate":
                    self.reply({"done": True})
                    return
                image_request = ("images" in body["messages"][-1] or isinstance(body["messages"][-1]["content"], list))
                payload = description() if image_request else {"objects": [{"id": 5, "classification": "movable", "reason": "Can be repositioned."}]}
                content = json.dumps(payload)
                if body["model"] in owner.bad:
                    content = owner.bad[body["model"]]
                if self.path == "/chat/completions":
                    self.reply({"model": body["model"], "choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"completion_tokens": 20}})
                else:
                    self.reply({"message": {"content": content}, "done_reason": "stop", "eval_count": 20})

            def reply(self, result):
                data = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.images = self.root / "rgb"
        self.images.mkdir()
        for name in ("one", "two", "three"):
            Image.new("RGB", (8, 6), "blue").save(self.images / f"{name}.jpg")
        self.input = self.root / "frames.jsonl"
        prepare(self.images, self.input, ["*.jpg"], 2, 1)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def config(self, stage=1, backend="ollama", both=True):
        raw = {"models": [{"name": n, "model": m, "backend": backend, "base_url": self.url,
                           "require_gpu": backend == "ollama", "system_as_user": backend == "openai"}
                          for n, m in [("a", "model-a"), ("b", "model-b")][:2 if both else 1]],
               "temperatures": [0.0], "seeds": [42]}
        path = self.root / f"config-{stage}-{backend}.json"
        path.write_text(json.dumps(raw))
        return load_config(path, stage)

    def describe(self, config=None, name="descriptions", input=None):
        args = argparse.Namespace(input=input or self.input, output_dir=self.root / name, limit=2, frame_id=None)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = stage1_describe.run(args, config or self.config())
        return rc, args.output_dir

    def test_two_models_same_images_prompts_and_no_classification(self):
        rc, output = self.describe()
        self.assertEqual(rc, 0)
        calls = [b for path, b in self.calls if path == "/api/chat"]
        self.assertEqual(len(calls), 4)
        for first, second in zip(calls[:2], calls[2:]):
            self.assertEqual(first["messages"], second["messages"])
            self.assertEqual(first["options"], second["options"])
        result = json.loads((output / "a/t0-s42/frame-000001.json").read_text())
        self.assertTrue(result["gpu_verified"])
        self.assertEqual(result["annotation_mode"], "model_discovery")
        self.assertNotIn("classification", result["parsed"]["objects"][0])
        self.assertIn("<image bytes omitted", result["request"]["messages"][1]["images"][0])
        self.assertIn("data:image/jpeg;base64,", (output / "comparison.html").read_text())
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["models"]["a"]["ok"], 2)
        self.assertEqual(summary["models"]["b"]["objects_returned"], 2)

    def test_duplicate_json_rejected_raw_preserved_other_model_continues(self):
        self.bad["model-a"] = '{"scene_description":"room","scene_description":"other","objects":[]}'
        rc, output = self.describe()
        self.assertEqual(rc, 2)
        bad = json.loads((output / "a/t0-s42/frame-000001.json").read_text())
        self.assertEqual(bad["content"], self.bad["model-a"])
        self.assertIn("Duplicate JSON key", bad["error"])
        self.assertEqual(json.loads((output / "summary.json").read_text())["models"]["b"]["ok"], 2)

    def test_gpu_fallback_fails_run_and_retains_response(self):
        self.gpu = False
        rc, output = self.describe()
        self.assertEqual(rc, 1)
        result = json.loads((output / "a/t0-s42/frame-000001.json").read_text())
        self.assertEqual(result["status"], "error")
        self.assertIn("GPU verification", result["error"])
        self.assertIn('"objects"', result["content"])
        self.assertEqual(len([b for p, b in self.calls if p == "/api/chat"]), 1)

    def test_stage2_uses_frozen_captions_after_photos_are_deleted(self):
        _, first = self.describe()
        for image in self.images.glob("*.jpg"):
            image.unlink()
        self.calls.clear()
        args = argparse.Namespace(captions=first / "a/t0-s42", output_dir=self.root / "reason", limit=None)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = stage2_reason.run(args, self.config(stage=2))
        self.assertEqual(rc, 0)
        calls = [b for path, b in self.calls if path == "/api/chat"]
        self.assertEqual(len(calls), 4)
        self.assertTrue(all("images" not in message for call in calls for message in call["messages"]))
        self.assertEqual(calls[0]["messages"], calls[2]["messages"])
        result = json.loads((args.output_dir / "a/t0-s42/frame-000001.json").read_text())
        self.assertEqual(result["parsed"]["objects"][0]["classification"], "movable")
        self.assertEqual(len((args.output_dir / "classifications.csv").read_text().splitlines()), 5)

    def test_stage2_skips_failed_captions_and_rejects_duplicate_reason_keys(self):
        self.bad["model-a"] = "not json"
        _, first = self.describe()
        self.calls.clear()
        self.bad.clear()
        args = argparse.Namespace(captions=first / "a/t0-s42", output_dir=self.root / "skipped", limit=None)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage2_reason.run(args, self.config(stage=2, both=False)), 2)
        self.assertFalse(any(p == "/api/chat" for p, _ in self.calls))
        self.bad["model-a"] = '{"objects":[{"id":5,"classification":"fixed","classification":"movable","reason":"chair"}]}'
        args.captions, args.output_dir = first / "b/t0-s42", self.root / "bad-reason"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage2_reason.run(args, self.config(stage=2, both=False)), 2)
        result = json.loads((args.output_dir / "a/t0-s42/frame-000001.json").read_text())
        self.assertIn("Duplicate JSON key", result["error"])

    def test_openai_single_user_mode_logs_unsupported_controls(self):
        rc, output = self.describe(self.config(backend="openai", both=False))
        self.assertEqual(rc, 0)
        body = next(b for path, b in self.calls if path == "/chat/completions")
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertTrue(body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("seed", body)
        self.assertNotIn("response_format", body)
        result = json.loads((output / "a/t0-s42/frame-000001.json").read_text())
        self.assertFalse(result["gpu_verified"])
        self.assertIn("seed", result["unsupported_options"])
        self.assertIn("top_k", result["unsupported_options"])
        self.assertIn("Describe an RGB", body["messages"][0]["content"][0]["text"])

    def test_labeled_instances_keep_ids_labels_and_boxes(self):
        record = {"frame_id": "instance-test", "image_path": str(self.images / "one.jpg"),
                  "annotation_kind": "object_instance", "objects": [{"id": 5, "label": "chair", "bbox_xyxy": [1, 1, 4, 5]}]}
        self.input.write_text(json.dumps(record))
        rc, output = self.describe()
        self.assertEqual(rc, 0)
        body = next(b for path, b in self.calls if path == "/api/chat")
        text = body["messages"][1]["content"]
        self.assertIn('"bbox_xyxy": [1, 1, 4, 5]', text)
        self.assertIn('"annotation_kind": "object_instance"', text)
        parsed = description(oid="5")
        with self.assertRaises(ValueError):
            validate_description(parsed, record["objects"])
        with self.assertRaises(ValueError):
            validate_description(description(label="stool"), record["objects"])

    def test_selection_is_deterministic_and_checks_modified_images(self):
        second = self.root / "second.jsonl"
        prepare(self.images, second, ["*.jpg"], 2, 1)
        self.assertEqual(self.input.read_bytes(), second.read_bytes())
        with self.assertRaises(ValueError):
            prepare(self.images, self.input, ["*.jpg"], 2, 1)
        record = json.loads(self.input.read_text().splitlines()[0])
        Image.new("RGB", (10, 10), "red").save(record["image_path"])
        with self.assertRaisesRegex(ValueError, "image changed"):
            load_images(self.input)

    def test_stage1_must_not_classify_and_refuses_old_results_overwrite(self):
        parsed = description()
        parsed["objects"][0]["classification"] = "fixed"
        with self.assertRaisesRegex(ValueError, "Stage 1"):
            validate_description(parsed)
        self.describe()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.describe()

    def test_wrong_precision_is_reported_without_inference(self):
        config = self.config(both=False)
        config["models"][0]["precision"] = "F16"
        rc, output = self.describe(config)
        self.assertEqual(rc, 2)
        self.assertFalse(any(p == "/api/chat" for p, _ in self.calls))
        self.assertIn("Expected F16", (output / "a/t0-s42/frame-000001.json").read_text())

    def test_http_transport_does_not_log_credentials(self):
        config = self.config(backend="openai", both=False)
        config["models"][0]["api_key_env"] = "EXPERIMENT_TEST_TOKEN"
        with patch.dict("os.environ", {"EXPERIMENT_TEST_TOKEN": "do-not-write-me"}):
            _, output = self.describe(config)
        for path in output.rglob("*.json"):
            self.assertNotIn("do-not-write-me", path.read_text())

    def test_configuration_and_nonfinite_or_duplicate_json_are_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                strict_json(text)
        path = self.root / "bad.json"
        for extra in ({"max_tokens": 0}, {"detail": "wrong"}, {"seeds": [True]}, {"unexpected": 1}):
            path.write_text(json.dumps({"models": [{"name": "x", "model": "x"}], **extra}))
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                load_config(path, 1)
        path.write_text(json.dumps({"models": [{"name": "x", "model": "x", "backend": "openai", "require_gpu": True}]}))
        with self.assertRaisesRegex(ValueError, "cannot verify GPU"):
            load_config(path, 1)


if __name__ == "__main__":
    unittest.main()
