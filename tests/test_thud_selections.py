"""Ensure the committed 2 x 50 THUD selections remain runnable after cloning."""
import hashlib
import json
from pathlib import Path
import unittest
from llm_object_classifier.image_dataset import load_images
from llm_object_classifier.description_prompts import user_prompt


class ThudSelectionTests(unittest.TestCase):
    def test_all_selected_photos_labels_and_boxes_reach_stage1(self):
        root = Path(__file__).resolve().parents[1]
        for name, count in (("thud_dense_test50", 678), ("thud_agtea_test50", 1021)):
            with self.subTest(dataset=name):
                folder = root / "data" / name
                if not folder.exists():
                    self.skipTest("Committed data is not mounted in this environment.")
                frames = load_images(folder / "objects.jsonl", limit=50)
                self.assertEqual(len(frames), 50)
                self.assertEqual(len({f["record"]["frame_id"] for f in frames}), 50)
                self.assertEqual(len({f["image_sha256"] for f in frames}), 50)
                self.assertEqual(sum(len(f["regions"]) for f in frames), count)
                for frame in frames:
                    record = frame["record"]
                    self.assertEqual(frame["image_size"], [960, 540])
                    annotation = json.loads((folder / record["annotation_path"]).read_text())
                    self.assertEqual(len(annotation["shapes"]), len(frame["regions"]))
                    payload = json.loads(user_prompt(frame).split("\n", 1)[1])
                    self.assertEqual(payload["image_size"], [960, 540])
                    for original, supplied in zip(record["objects"], payload["supplied_objects"]):
                        self.assertEqual(supplied["id"], original["id"])
                        self.assertEqual(supplied["label"], original["label"])
                        self.assertEqual(supplied["bbox_xyxy"], original["bbox_xyxy"])
                        self.assertEqual(supplied["region_kind"], "object_instance")
                for line in (folder / "SHA256SUMS.txt").read_text().splitlines():
                    expected, filename = line.split(maxsplit=1)
                    self.assertEqual(hashlib.sha256((folder / filename).read_bytes()).hexdigest(), expected)
