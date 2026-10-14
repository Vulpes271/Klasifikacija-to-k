import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tests import test_vision_pipeline as fixtures
from llm_object_classifier.vision_pipeline import (
    GPURequiredError, chat, loaded_gpu, run, variants,
)


class GPUSweepTests(unittest.TestCase):
    def test_request_image_only_goes_to_caption_and_options_are_sent(self):
        responses = [{"message": {"content": "{}"}},
                     {"models": [{"name": "model:bf16", "size_vram": 1024}]}]
        with patch("llm_object_classifier.vision_pipeline.request_json", side_effect=responses) as request:
            result = chat("http://fake", "model:bf16", "observe", "look", 0.7, 43,
                          image=b"image", options={"top_k": 40}, require_gpu=True)
        body = request.call_args_list[0].args[2]
        self.assertEqual(body["options"]["num_gpu"], 999)
        self.assertEqual(body["options"]["top_k"], 40)
        self.assertEqual(body["options"]["seed"], 43)
        self.assertIn("images", body["messages"][1])
        self.assertGreater(result["gpu"]["size_vram"], 0)
        with patch("llm_object_classifier.vision_pipeline.request_json", return_value={"message": {"content": "{}"}}) as request:
            chat("http://fake", "model:bf16", "classify", "description", 0, 42)
        self.assertNotIn("images", request.call_args.args[2]["messages"][1])

    def test_gpu_requires_requested_model_not_other_gpu_job(self):
        for models in ([], [{"name": "wanted", "size_vram": 0}],
                       [{"name": "other", "size_vram": 1234}]):
            with self.subTest(models=models), patch("llm_object_classifier.vision_pipeline.request_json", return_value={"models": models}):
                with self.assertRaises(GPURequiredError):
                    loaded_gpu("http://fake", "wanted")

    def test_sweep_preserves_all_files_and_reuses_caption_across_reason_models(self):
        fixture = fixtures.VisionPipelineTests()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            fixture.setup_input(root)
            profiles = root / "profiles.json"
            profiles.write_text(json.dumps([{"name": "base", "options": {}},
                                            {"name": "k40", "options": {"top_k": 40}}]))
            args = fixture.args(root, "sweep")
            args.profiles, args.seeds = profiles, [42, 43]
            self.assertEqual(len(variants(args)), 12)
            info = {"capabilities": ["vision"], "details": {"quantization_level": "BF16"}}
            with patch("llm_object_classifier.vision_pipeline.request_json", return_value=info), patch("llm_object_classifier.vision_pipeline.chat", side_effect=[fixture.result(fixture.caption())] + [fixture.result(fixture.prediction())] * 12) as call:
                self.assertEqual(run(args), 0)
                self.assertEqual(call.call_count, 13)
                self.assertTrue(all(c.kwargs["options"]["top_k"] == 40 for c in call.call_args_list[-6:]))
            self.assertEqual(len(list((root / "sweep/frame-001").glob("reasoning-*.json"))), 12)
            with (root / "sweep/results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 24)
            self.assertEqual(len({r["experiment"] for r in rows}), 12)
            args.output_dir, args.reason_model = root / "quant", "quantized"
            with patch("llm_object_classifier.vision_pipeline.request_json", return_value=info), patch("llm_object_classifier.vision_pipeline.chat", side_effect=[fixture.result(fixture.prediction())] * 12) as call:
                self.assertEqual(run(args), 0)
                self.assertEqual(call.call_count, 12)
            self.assertEqual(json.loads((root / "quant/summary.json").read_text())["caption_cached"], 1)
            args.output_dir, args.num_ctx = root / "new-context", 4096
            with patch("llm_object_classifier.vision_pipeline.request_json", return_value=info), patch("llm_object_classifier.vision_pipeline.chat", side_effect=[fixture.result(fixture.caption())] + [fixture.result(fixture.prediction())] * 12) as call:
                self.assertEqual(run(args), 0)
                self.assertEqual(call.call_count, 13)

    def test_unquantized_flag_rejects_quantized_weights(self):
        fixture = fixtures.VisionPipelineTests()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture.setup_input(root)
            args = fixture.args(root, "bad")
            args.require_unquantized = True
            with patch("llm_object_classifier.vision_pipeline.request_json", return_value={"capabilities": ["vision"], "details": {"quantization_level": "Q4_K_M"}}), patch("llm_object_classifier.vision_pipeline.chat") as call:
                with self.assertRaisesRegex(ValueError, "expected unquantized"):
                    run(args)
                call.assert_not_called()

    def test_gpu_failure_aborts_instead_of_marking_success(self):
        fixture = fixtures.VisionPipelineTests()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            fixture.setup_input(root)
            args = fixture.args(root, "cpu")
            args.require_gpu = True
            with patch("llm_object_classifier.vision_pipeline.request_json", return_value={"capabilities": ["vision"]}), patch("llm_object_classifier.vision_pipeline.chat", side_effect=GPURequiredError("CPU fallback")):
                with self.assertRaises(GPURequiredError):
                    run(args)

    def test_invalid_profile_and_duplicate_seeds_rejected(self):
        fixture = fixtures.VisionPipelineTests()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture.args(root, "run")
            args.seeds = [42, 42]
            with self.assertRaises(ValueError): variants(args)
            args.seeds = [42]
            args.profiles = root / "profiles.json"
            for options in ({"temperature": 1}, {"top_p": 5}, {"top_k": 0.5}, {"num_ctx": -1}):
                args.profiles.write_text(json.dumps([{"name": "bad", "options": options}]))
                with self.subTest(options=options), self.assertRaises(ValueError): variants(args)


if __name__ == "__main__":
    unittest.main()
