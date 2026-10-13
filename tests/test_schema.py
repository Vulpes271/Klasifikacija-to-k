import json
import unittest

from llm_object_classifier.schema import validate_classifications, validate_frame


class SchemaTests(unittest.TestCase):
    def test_requires_exact_ids(self):
        frame = validate_frame(
            {
                "frame_id": "frame-1",
                "objects": [{"id": 1, "label": "chair"}, {"id": 2, "label": "wall"}],
            }
        )
        with self.assertRaisesRegex(ValueError, "Missing classifications"):
            validate_classifications(
                frame,
                {"objects": [{"id": 1, "classification": "movable", "reason": "Chair."}]},
            )

    def test_rejects_bad_class(self):
        frame = validate_frame({"frame_id": "frame-1", "objects": [{"id": 1, "label": "chair"}]})
        with self.assertRaisesRegex(ValueError, "invalid classification"):
            validate_classifications(
                frame,
                {"objects": [{"id": 1, "classification": "dynamic", "reason": "No."}]},
            )


if __name__ == "__main__":
    unittest.main()

