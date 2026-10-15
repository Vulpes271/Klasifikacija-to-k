"""Regression tests for editable two-stage experiments, without model downloads."""
import contextlib
import copy
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from llm_object_classifier.vision_config import parse_args
from llm_object_classifier.vision_pipeline import main, run
from llm_object_classifier.vision_prompts import CAPTION_SYSTEM, REASON_SYSTEM, caption_system
from tests import test_vision_pipeline as fixtures


class VisionSettingsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.VisionPipelineTests()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixture.setup_input(self.root)

    def config(self, raw):
        path = self.root / "settings.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_defaults_keep_working_prompts_and_sampling(self):
        args = parse_args(["--output-dir", "results/test"])
        self.assertEqual(args.vision_model, "qwen3.5:4b")
        self.assertEqual(args.reason_model, "qwen3.5:4b")
        self.assertEqual(args.temperatures, [0, 0.3, 0.7])
        self.assertEqual(args.caption_temperature, 0)
        self.assertEqual(caption_system(args.caption_detail), CAPTION_SYSTEM)
        self.assertEqual(args.num_predict, 2048)
        self.assertIsNone(args.caption_num_predict)

    def test_config_paths_are_relative_to_config_and_cli_wins(self):
        path = self.config({"vision_model": "vision:selected", "reason_model": "reason:config",
                            "reason_prompt_file": "prompt with spaces.txt", "input": "objects.jsonl",
                            "output_dir": "run", "caption_detail": "standard",
                            "caption_temperature": 0.2, "temperatures": [0.1, 0.5]})
        args = parse_args(["--config", str(path), "--reason-model", "reason:cli",
                           "--temperatures", "0.8", "--output-dir", "results/override"])
        self.assertEqual(args.vision_model, "vision:selected")
        self.assertEqual(args.reason_model, "reason:cli")
        self.assertEqual(args.temperatures, [0.8])
        self.assertEqual(args.caption_temperature, 0.2)
        self.assertEqual(args.input, self.root / "objects.jsonl")
        self.assertEqual(args.reason_prompt_file, self.root / "prompt with spaces.txt")
        self.assertEqual(args.output_dir, Path("results/override"))
        self.assertEqual(parse_args(["--config", str(path)]).output_dir, self.root / "run")

    def test_bad_configuration_fails_before_network_calls(self):
        cases = [[], {"vision_modle": "typo"}, {"caption_detail": "huge"},
                 {"temperatures": 0.5}, {"temperatures": []}, {"temperatures": [True]},
                 {"temperatures": [float("nan")]}, {"temperatures": [-1]},
                 {"temperatures": [0.1, 0.1]}, {"caption_temperature": float("inf")},
                 {"require_gpu": "false"}, {"num_predict": True}, {"num_predict": 1.5},
                 {"caption_num_predict": 0}, {"timeout": -1}, {"vision_model": ""},
                 {"reason_model": "model\nsecond"}, {"reason_prompt_file": 123}]
        for raw in cases:
            with self.subTest(raw=raw), contextlib.redirect_stderr(io.StringIO()), \
                    patch("llm_object_classifier.vision_pipeline.request_json") as request:
                with self.assertRaises(SystemExit) as error:
                    main(["--config", str(self.config(raw)), "--output-dir", str(self.root / "bad")])
                self.assertEqual(error.exception.code, 2)
                request.assert_not_called()

    def test_prompt_file_errors_fail_before_model_calls(self):
        args = self.fixture.args(self.root, "bad")
        args.reason_prompt_file = self.root / "prompt.txt"
        for content in (None, "  \n"):
            if content is not None:
                args.reason_prompt_file.write_text(content, encoding="utf-8")
            with self.subTest(content=content), patch("llm_object_classifier.vision_pipeline.request_json") as request:
                with self.assertRaises((OSError, ValueError)):
                    run(args)
                request.assert_not_called()

    def test_detail_changes_word_guidance_without_conflicting_brief_limits(self):
        for detail, scene, objects in (("brief", 35, 25), ("standard", 80, 60), ("detailed", 160, 120)):
            system = caption_system(detail)
            self.assertIn(f"at most {scene} words", system)
            self.assertIn(f"at most {objects} words", system)
            self.assertIn("Do NOT assign fixed/movable classes", system)
            if detail != "brief":
                self.assertNotIn("at most 25 words", system)
                self.assertNotIn("at most 35 words", system)

    def mocked_run(self, args, needs_caption):
        answers = ([self.fixture.result(self.fixture.caption())] if needs_caption else [])
        answers += [self.fixture.result(self.fixture.prediction())] * len(args.temperatures)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("llm_object_classifier.vision_pipeline.request_json", return_value={"capabilities": ["vision"]}), \
                patch("llm_object_classifier.vision_pipeline.chat", side_effect=answers) as chat:
            self.assertEqual(run(args), 0)
            self.assertEqual(chat.call_count, len(answers))
            return chat.call_args_list

    def test_caption_settings_invalidate_cache_but_reasoning_settings_reuse_it(self):
        base = self.fixture.args(self.root, "original")
        base.temperatures = [0.0]
        self.mocked_run(base, True)
        for index, (key, value, needs_caption) in enumerate([
            ("caption_detail", "detailed", True), ("caption_temperature", 0.4, True),
            ("vision_model", "different:vision", True), ("caption_num_predict", 4096, True),
            ("reason_model", "different:reason", False), ("temperatures", [0.9], False),
        ]):
            args = copy.copy(base)
            args.output_dir = self.root / f"changed-{index}"
            setattr(args, key, value)
            with self.subTest(key=key):
                self.mocked_run(args, needs_caption)
                summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["caption_cached"], 0 if needs_caption else 1)

    def test_editing_prompt_files_reuses_caption_and_records_exact_text(self):
        args = self.fixture.args(self.root, "original")
        args.temperatures = [0.0]
        args.reason_prompt_file = self.root / "task.txt"
        args.reason_system_file = self.root / "system.txt"
        for index, instructions in enumerate(("Bodi previden pri omarah.", "Presodi vidne pritrditve.")):
            args.output_dir = self.root / f"prompt-{index}"
            args.reason_prompt_file.write_text(instructions, encoding="utf-8-sig")
            system = REASON_SYSTEM + f"\nExperiment {index}."
            args.reason_system_file.write_text(system, encoding="utf-8")
            calls = self.mocked_run(args, index == 0)
            reasoning = calls[-1]
            self.assertEqual(reasoning.args[2], system)
            prompt = reasoning.args[3]
            self.assertTrue(prompt.startswith(instructions + "\n"))
            payload = json.loads(prompt.split("\n", 1)[1])
            self.assertEqual(payload["objects"], self.fixture.objects())
            self.assertEqual(payload["visual_observations"], self.fixture.caption())
            manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["reason_instructions"], instructions)
            self.assertEqual(manifest["reason_system"], system)

    def test_caption_token_budget_is_independent_of_reasoning_budget(self):
        args = self.fixture.args(self.root, "tokens")
        args.temperatures = [0.0]
        args.num_predict = 512
        args.caption_num_predict = 4096
        calls = self.mocked_run(args, True)
        self.assertEqual(calls[0].kwargs["options"]["num_predict"], 4096)
        self.assertEqual(calls[1].kwargs["options"]["num_predict"], 512)

    def test_print_models_resolves_configuration_without_contacting_server(self):
        path = self.config({"vision_model": "vision:custom", "reason_model": "reason:custom"})
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                patch("llm_object_classifier.vision_pipeline.request_json") as request:
            self.assertEqual(main(["--config", str(path), "--output-dir", "unused", "--print-models"]), 0)
            request.assert_not_called()
        self.assertEqual(output.getvalue().splitlines(), ["vision:custom", "reason:custom"])

    def test_configured_pipeline_sends_models_prompts_and_temperatures_over_http(self):
        # Exercise the real urllib transport against a local Ollama-shaped server.
        fixture = self.fixture
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.reply({"version": "test"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    self.reply({"capabilities": ["vision"]})
                else:
                    requests.append(body)
                    result = fixture.caption() if "images" in body["messages"][1] else fixture.prediction()
                    self.reply({"message": {"content": json.dumps(result)}})

            def reply(self, body):
                data = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            (self.root / "reason.txt").write_text("Use visible evidence carefully.", encoding="utf-8")
            path = self.config({"input": "objects.jsonl", "cache_dir": "cache", "output_dir": "http-run",
                                "base_url": f"http://127.0.0.1:{server.server_port}",
                                "vision_model": "image:selected", "reason_model": "text:selected",
                                "caption_detail": "detailed", "caption_temperature": 0.2,
                                "temperatures": [0.4, 0.8], "caption_num_predict": 4096,
                                "num_predict": 512, "reason_prompt_file": "reason.txt"})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(path)]), 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual([r["model"] for r in requests], ["image:selected", "text:selected", "text:selected"])
        self.assertEqual([r["options"]["temperature"] for r in requests], [0.2, 0.4, 0.8])
        self.assertEqual([r["options"]["num_predict"] for r in requests], [4096, 512, 512])
        self.assertIn("at most 120 words", requests[0]["messages"][0]["content"])
        self.assertIn("images", requests[0]["messages"][1])
        for request in requests[1:]:
            self.assertNotIn("images", request["messages"][1])
            self.assertTrue(request["messages"][1]["content"].startswith("Use visible evidence carefully."))


if __name__ == "__main__":
    unittest.main()
