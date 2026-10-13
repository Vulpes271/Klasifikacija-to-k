import json
import tempfile
import unittest
from pathlib import Path

from llm_object_classifier.cli import main
from llm_object_classifier.evaluate import evaluate_results


class EvaluateAndThudTests(unittest.TestCase):
    def test_evaluate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results.jsonl"
            results.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "frame_id": "f1",
                        "objects": [
                            {"id": 1, "classification": "movable"},
                            {"id": 2, "classification": "fixed"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            reference = root / "reference.csv"
            reference.write_text(
                "frame_id,object_id,classification\nf1,1,movable\nf1,2,movable\n",
                encoding="utf-8",
            )
            metrics = evaluate_results(results, reference)
            self.assertEqual(metrics["evaluated_objects"], 2)
            self.assertEqual(metrics["accuracy"], 0.5)
            self.assertEqual(metrics["movable_predicted_fixed_rate"], 0.5)

    def test_prepare_thud_synthetic_captures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "captures_000.json"
            source.write_text(
                json.dumps(
                    {
                        "captures": [
                            {
                                "step": 0,
                                "filename": "RGB/rgb_0.png",
                                "annotations": [
                                    {
                                        "values": [
                                            {
                                                "label_name": "Rowing Machine",
                                                "instance_id": 304,
                                                "label_id": 4,
                                            }
                                        ]
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "out.jsonl"
            code = main(["prepare-thud", "--input", str(source), "--output", str(output)])
            self.assertEqual(code, 0)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["objects"][0]["id"], 304)
            self.assertEqual(record["objects"][0]["label"], "Rowing Machine")


if __name__ == "__main__":
    unittest.main()

