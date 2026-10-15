from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schema import ALLOWED_CLASSES

CLASSES = ["fixed", "movable", "uncertain"]


def load_reference(path: Path) -> dict[tuple[str, str], str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_id", "object_id", "classification"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Reference CSV must contain frame_id,object_id,classification columns.")
        reference: dict[tuple[str, str], str] = {}
        for row in reader:
            classification = (row.get("classification") or "").strip()
            if classification not in ALLOWED_CLASSES:
                raise ValueError(f"Invalid reference class: {classification!r}")
            key = ((row.get("frame_id") or "").strip(), (row.get("object_id") or "").strip())
            if not key[0] or not key[1]:
                raise ValueError("Reference rows need non-empty frame_id and object_id.")
            reference[key] = classification
        return reference


def iter_predictions(results_jsonl: Path):
    with results_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok":
                continue
            for item in record.get("objects", []):
                yield {
                    "frame_id": record["frame_id"],
                    "object_id": str(item["id"]),
                    "classification": item["classification"],
                }


def evaluate_results(results_jsonl: Path, reference_csv: Path) -> dict[str, Any]:
    reference = load_reference(reference_csv)
    confusion: dict[str, Counter[str]] = {name: Counter() for name in CLASSES}
    total = 0
    correct = 0
    movable_total = 0
    movable_as_fixed = 0

    for prediction in iter_predictions(results_jsonl):
        key = (prediction["frame_id"], prediction["object_id"])
        truth = reference.get(key)
        if truth is None:
            continue
        pred = prediction["classification"]
        confusion[truth][pred] += 1
        total += 1
        correct += int(truth == pred)
        if truth == "movable":
            movable_total += 1
            movable_as_fixed += int(pred == "fixed")

    metrics: dict[str, Any] = {
        "evaluated_objects": total,
        "accuracy": correct / total if total else None,
        "confusion_matrix": {truth: {pred: confusion[truth][pred] for pred in CLASSES} for truth in CLASSES},
        "per_class": {},
        "movable_predicted_fixed_rate": movable_as_fixed / movable_total if movable_total else None,
    }
    for cls in CLASSES:
        tp = confusion[cls][cls]
        fp = sum(confusion[truth][cls] for truth in CLASSES if truth != cls)
        fn = sum(confusion[cls][pred] for pred in CLASSES if pred != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics["per_class"][cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[cls].values()),
        }
    return metrics


