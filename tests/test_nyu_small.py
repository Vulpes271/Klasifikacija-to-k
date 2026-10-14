import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from scripts.download_nyu_sample import objects_from_mask, download_sample
from llm_object_classifier.schema import validate_frame


class NyuSmallTests(unittest.TestCase):
    def test_raw_ids_are_one_based_and_zero_is_unlabelled(self):
        mask = Image.new("P", (4, 1))
        mask.putdata([0, 1, 5, 5])
        objects = objects_from_mask(mask, {"0": "wall", "4": "chair"})
        self.assertEqual([(o["id"], o["label"], o["pixel_count"]) for o in objects],
                         [(1, "wall", 1), (5, "chair", 2)])
        validate_frame({"frame_id": "sample", "objects": objects})

    def test_multiple_stools_are_not_claimed_to_be_instances(self):
        mask = Image.new("L", (3, 1))
        mask.putdata([5, 0, 5])
        objects = objects_from_mask(mask, {"4": "chair"})
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["annotation_kind"], "semantic_class_region")

    def test_reject_rgb_mask_and_unknown_ids(self):
        with self.assertRaises(ValueError):
            objects_from_mask(Image.new("RGB", (1, 1)), {})
        with self.assertRaises(ValueError):
            objects_from_mask(Image.new("L", (1, 1), 41), {})

    def test_does_not_overwrite_existing_input(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "objects.jsonl").write_text("existing\n")
            with patch("scripts.download_nyu_sample.fetch") as fetch:
                with self.assertRaises(ValueError):
                    download_sample(path)
                fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
