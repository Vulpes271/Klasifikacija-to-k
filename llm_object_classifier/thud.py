from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


class UnsupportedThudFormatError(ValueError):
    pass


def iter_json_members(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".json"):
                    continue
                with archive.open(name) as handle:
                    yield name, json.loads(handle.read().decode("utf-8"))
    elif path.is_dir():
        for json_path in sorted(path.rglob("*.json")):
            with json_path.open("r", encoding="utf-8") as handle:
                yield str(json_path.relative_to(path)), json.load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield path.name, json.load(handle)


def detect_scene(member_name: str) -> str | None:
    parts = PurePosixPath(member_name.replace("\\", "/")).parts
    for root_name in ("Real_Scenes", "Synthetic_Scenes"):
        if root_name in parts:
            index = parts.index(root_name)
            if index + 1 < len(parts):
                return parts[index + 1]
    return None


def frame_id_from_member(member_name: str) -> str:
    return PurePosixPath(member_name.replace("\\", "/")).stem


def extract_unity_captures(member_name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    captures = data.get("captures")
    if not isinstance(captures, list):
        raise UnsupportedThudFormatError("Expected documented synthetic THUD captures list.")
    frames: list[dict[str, Any]] = []
    for capture in captures:
        if not isinstance(capture, dict):
            continue
        objects = []
        for annotation in capture.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            values = annotation.get("values")
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict):
                    continue
                label = value.get("label_name")
                object_id = value.get("instance_id", value.get("label_id"))
                if isinstance(label, str) and isinstance(object_id, (int, str)):
                    metadata = {
                        key: value[key]
                        for key in ("label_id", "translation", "size", "rotation", "velocity", "acceleration")
                        if key in value
                    }
                    item = {"id": object_id, "label": label}
                    item.update(metadata)
                    objects.append(item)
        frame_id = str(capture.get("step", capture.get("id", frame_id_from_member(member_name))))
        filename = capture.get("filename")
        frame = {
            "frame_id": frame_id,
            "scene": detect_scene(member_name),
            "objects": objects,
            "source_member": member_name,
        }
        if isinstance(filename, str):
            frame["image_path"] = filename
        frames.append(frame)
    return frames


def extract_real_2d(member_name: str, data: dict[str, Any]) -> dict[str, Any]:
    raise UnsupportedThudFormatError(
        "The public THUD/THUD++ overview documents the Label/2D_Object_Detection path, "
        "but not the real-scene JSON schema. Provide one real frame-*.json sample to finish this extractor."
    )


def convert_thud_records(path: Path, source_format: str = "auto") -> Iterator[dict[str, Any]]:
    for member_name, data in iter_json_members(path):
        normalized = member_name.replace("\\", "/")
        if source_format in {"auto", "synthetic-captures"} and "captures" in data:
            yield from extract_unity_captures(member_name, data)
            continue
        if source_format in {"auto", "real-2d"} and "/Label/2D_Object_Detection/" in f"/{normalized}":
            yield extract_real_2d(member_name, data)
            continue
        if source_format != "auto":
            raise UnsupportedThudFormatError(f"{member_name}: not compatible with {source_format}.")


