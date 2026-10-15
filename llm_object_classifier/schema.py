from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ALLOWED_CLASSES = {"fixed", "movable", "uncertain"}


@dataclass(frozen=True)
class ObjectItem:
    id: int | str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    frame_id: str
    scene: str | None
    objects: list[ObjectItem]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Classification:
    id: int | str
    classification: str
    reason: str


def validate_frame(raw: dict[str, Any]) -> Frame:
    if not isinstance(raw, dict):
        raise ValueError("Frame record must be a JSON object.")
    frame_id = raw.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("frame_id must be a non-empty string.")
    scene = raw.get("scene")
    if scene is not None and not isinstance(scene, str):
        raise ValueError("scene must be a string when present.")
    raw_objects = raw.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("objects must be a list.")

    seen: set[int | str] = set()
    objects: list[ObjectItem] = []
    for item in raw_objects:
        if not isinstance(item, dict):
            raise ValueError(f"Frame {frame_id}: each object must be an object.")
        object_id = item.get("id")
        if not isinstance(object_id, (int, str)) or object_id == "":
            raise ValueError(f"Frame {frame_id}: object id must be an int or non-empty string.")
        if object_id in seen:
            raise ValueError(f"Frame {frame_id}: duplicate object id {object_id!r}.")
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Frame {frame_id}: object {object_id!r} has invalid label.")
        seen.add(object_id)
        metadata = {k: v for k, v in item.items() if k not in {"id", "label"}}
        objects.append(ObjectItem(id=object_id, label=label.strip(), metadata=metadata))

    metadata = {k: v for k, v in raw.items() if k not in {"frame_id", "scene", "objects"}}
    return Frame(frame_id=frame_id, scene=scene, objects=objects, metadata=metadata)


def validate_classifications(frame: Frame, raw: dict[str, Any]) -> list[Classification]:
    if not isinstance(raw, dict):
        raise ValueError("LLM response must be a JSON object.")
    raw_objects = raw.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("LLM response must contain an objects list.")

    expected = {item.id for item in frame.objects}
    seen: set[int | str] = set()
    classifications: list[Classification] = []
    for item in raw_objects:
        if not isinstance(item, dict):
            raise ValueError("Each response object must be a JSON object.")
        object_id = item.get("id")
        if object_id not in expected:
            raise ValueError(f"Unexpected object id in response: {object_id!r}.")
        if object_id in seen:
            raise ValueError(f"Duplicate object id in response: {object_id!r}.")
        classification = item.get("classification")
        if classification not in ALLOWED_CLASSES:
            raise ValueError(
                f"Object {object_id!r} has invalid classification {classification!r}."
            )
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Object {object_id!r} has missing reason.")
        seen.add(object_id)
        classifications.append(
            Classification(
                id=object_id,
                classification=classification,
                reason=reason.strip(),
            )
        )

    missing = expected - seen
    if missing:
        raise ValueError(f"Missing classifications for object ids: {sorted(missing, key=str)!r}.")
    return classifications


def frame_to_json(frame: Frame) -> dict[str, Any]:
    data: dict[str, Any] = {"frame_id": frame.frame_id, "objects": []}
    if frame.scene is not None:
        data["scene"] = frame.scene
    data.update(frame.metadata)
    objects = []
    for item in frame.objects:
        object_data = {"id": item.id, "label": item.label}
        object_data.update(item.metadata)
        objects.append(object_data)
    data["objects"] = objects
    return data


