from __future__ import annotations

import csv
from pathlib import Path

from .schema import ALLOWED_CLASSES

DEFAULT_LABEL_MAP: dict[str, str] = {
    "wall": "fixed",
    "floor": "fixed",
    "ceiling": "fixed",
    "door": "fixed",
    "window": "fixed",
    "stairs": "fixed",
    "elevator": "fixed",
    "pillar": "fixed",
    "column": "fixed",
    "chair": "movable",
    "stool": "movable",
    "person": "movable",
    "human": "movable",
    "pedestrian": "movable",
    "box": "movable",
    "bag": "movable",
    "trash can": "movable",
    "freestanding table": "movable",
    "table": "movable",
    "desk": "movable",
    "cabinet": "uncertain",
    "shelf": "uncertain",
    "counter": "uncertain",
    "machine": "uncertain",
}


def load_label_map(path: Path | None) -> dict[str, str]:
    mapping = dict(DEFAULT_LABEL_MAP)
    if path is None:
        return mapping
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "classification"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Label map CSV must contain label,classification columns.")
        for row in reader:
            label = (row.get("label") or "").strip().lower()
            classification = (row.get("classification") or "").strip()
            if not label:
                continue
            if classification not in ALLOWED_CLASSES:
                raise ValueError(f"Invalid class for label {label!r}: {classification!r}")
            mapping[label] = classification
    return mapping


def classify_label(label: str, mapping: dict[str, str]) -> tuple[str, str]:
    key = label.strip().lower()
    classification = mapping.get(key, "uncertain")
    if classification == "uncertain":
        return classification, "No reliable fixed/movable rule for this label."
    return classification, f"Reference label map classifies {label!r} as {classification}."

