import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from llm_object_classifier.vision_pipeline import exact_items, region_objects, run, validate_caption

class VisionPipelineTests(unittest.TestCase):
    def objects(self):
        return [{"id": 1, "label": "wall"}, {"id": 5, "label": "chair"}]
    def caption(self):
        return {"scene_description": "An indoor room.", "objects": [
            {"id": 1, "observation": "Room wall", "attachment_evidence": "visible_installation", "identifiable": True},
            {"id": 5, "observation": "Chair on legs", "attachment_evidence": "appears_freestanding", "identifiable": True}]}
    def result(self, parsed):
        return {"content": json.dumps(parsed), "elapsed_seconds": 1}
    def prediction(self):
        return {"objects": [{"id": 1, "classification": "fixed", "reason": "Structural wall."},
                            {"id": 5, "classification": "movable", "reason": "A person can move it."}]}
    def test_exact_id_types_missing_and_duplicates(self):
        for items in ([{"id": "1"}, {"id": 5}], [{"id": 1}], [{"id": 1}, {"id": 1}], [{"id": True}, {"id": 5}]):
            with self.assertRaises(ValueError): exact_items(self.objects(), items)
    def test_caption_rejects_invented_evidence_and_missing_objects(self):
        data = self.caption()
        data["objects"][0]["attachment_evidence"] = "probably_screwed_to_wall"
        with self.assertRaises(ValueError): validate_caption(self.objects(), data)
        data = self.caption()
        data["objects"].pop()
        with self.assertRaises(ValueError): validate_caption(self.objects(), data)
    def test_semantic_union_box_not_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = Image.new("P", (4, 2))
            mask.putpalette([v for i in range(256) for v in (i, i, i)])
            mask.putdata([5, 0, 0, 5, 1, 1, 0, 0])
            mask.save(root / "mask.png")
            objects = region_objects({"objects": self.objects(), "mask_path": "mask.png"}, root)
            self.assertEqual(objects[1]["bbox_xyxy"], [0, 0, 4, 1])
            self.assertEqual(objects[1]["region_kind"], "semantic_class_region")
            self.assertEqual(len(objects), 2)
    def args(self, root, name):
        return argparse.Namespace(input=root / "objects.jsonl", output_dir=root / name,
            cache_dir=root / "cache", base_url="http://fake", vision_model="qwen3.5:4b",
            reason_model="qwen3.5:4b", temperatures=[0.0, 0.3, 0.7], caption_temperature=0.0,
            limit=20, seed=42, timeout=600)
    def setup_input(self, root):
        Image.new("RGB", (4, 2)).save(root / "rgb.jpg")
        (root / "objects.jsonl").write_text(json.dumps({"frame_id": "f1", "objects": self.objects(), "image_path": "rgb.jpg"}) + "\n")
    def test_two_stages_cache_and_temperature_error_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_input(root)
            bad = self.prediction()
            bad["objects"][0]["id"] = "1"
            with patch("llm_object_classifier.vision_pipeline.request_json", return_value={"capabilities": ["vision"]}), patch("llm_object_classifier.vision_pipeline.chat", side_effect=[self.result(self.caption()), self.result(self.prediction()), self.result(bad), self.result(self.prediction())]) as chat:
                self.assertEqual(run(self.args(root, "run1")), 0)
                self.assertIsInstance(chat.call_args_list[0].args[6], bytes)
                self.assertEqual(len(chat.call_args_list[1].args), 6)
                self.assertIn("visual_observations", chat.call_args_list[1].args[3])
            summary = json.loads((root / "run1/summary.json").read_text())
            self.assertEqual(summary["reasoning"]["0.3"]["error"], 1)
            self.assertEqual(summary["reasoning"]["0.7"]["ok"], 1)
            with patch("llm_object_classifier.vision_pipeline.request_json", return_value={"capabilities": ["vision"]}), patch("llm_object_classifier.vision_pipeline.chat", side_effect=[self.result(self.prediction())] * 3) as chat:
                run(self.args(root, "run2"))
                self.assertEqual(chat.call_count, 3)
            self.assertEqual(json.loads((root / "run2/summary.json").read_text())["caption_cached"], 1)
    def test_bad_caption_skips_reasoning_and_keeps_raw_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_input(root)
            with patch("llm_object_classifier.vision_pipeline.request_json", return_value={"capabilities": ["vision"]}), patch("llm_object_classifier.vision_pipeline.chat", return_value={"content": "bad json", "elapsed_seconds": 1}) as chat:
                self.assertEqual(run(self.args(root, "run1")), 2)
                self.assertEqual(chat.call_count, 1)
            summary = json.loads((root / "run1/summary.json").read_text())
            self.assertEqual(summary["caption_error"], 1)
            self.assertEqual(summary["reasoning"]["0.0"]["skipped"], 1)
            error = json.loads((root / "run1/frame-001/caption_error.json").read_text())
            self.assertEqual(error["response"]["content"], "bad json")

if __name__ == "__main__": unittest.main()
