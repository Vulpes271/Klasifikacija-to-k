from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schema import Frame, validate_frame


def iter_jsonl_frames(path: Path, limit: int | None = None, every_n: int = 1) -> Iterator[Frame]:
    if every_n < 1:
        raise ValueError("every_n must be at least 1.")
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if (index - 1) % every_n != 0:
                continue
            raw = json.loads(line)
            yield validate_frame(raw)
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def load_completed_frame_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok" and isinstance(record.get("frame_id"), str):
                completed.add(record["frame_id"])
    return completed


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

