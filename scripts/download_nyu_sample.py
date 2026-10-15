"""Download a few real NYUv2 RGB/semantic pairs, never the full dataset."""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image

DATASET = "wyrx/NYUDepthv2_seg"
REVISION = "a5d5e3b0ee744b2f03628c7ef282a2d55fd2b833"
MAP_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/id2label.json"


def fetch(url: str, max_bytes: int = 20_000_000) -> bytes:
    """Bound downloads and retry transient failures without printing signed URLs."""
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nyuv2-small-test/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("Unexpectedly large response; download stopped.")
            return data
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError("Download failed. Check internet access and retry.") from None
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def objects_from_mask(mask: Image.Image, id2label: dict[str, str]) -> list[dict]:
    # This NYUv2 mirror stores raw NYU40 IDs 1..40, with 0 unlabelled.
    # Its id2label.json uses keys 0..39, hence subtract ONE from each pixel ID.
    # Preserve palette indices; converting a P-mode mask to L would corrupt IDs.
    if mask.mode not in {"P", "L", "I", "I;16", "I;16B", "I;16L"}:
        raise ValueError(f"Expected indexed/integer semantic mask, got {mask.mode}.")
    pixels = mask.get_flattened_data() if hasattr(mask, "get_flattened_data") else mask.getdata()
    counts = Counter(pixels)
    objects = []
    for raw_id, pixels in sorted(counts.items()):
        if raw_id == 0:
            continue
        if not isinstance(raw_id, int) or not 1 <= raw_id <= 40:
            raise ValueError(f"Unexpected NYU40 pixel ID: {raw_id!r}")
        label = id2label.get(str(raw_id - 1))
        if not isinstance(label, str) or not label:
            raise ValueError(f"Missing class name for pixel ID {raw_id}")
        objects.append({"id": raw_id, "label": label, "pixel_count": pixels,
                        "annotation_kind": "semantic_class_region"})
    return objects


def download_sample(output_dir: Path, count: int = 10, offset: int = 0,
                    include_rgb: bool = True) -> Path:
    if not 1 <= count <= 100 or offset < 0 or offset + count > 654:
        raise ValueError("Use 1..100 images and offset+count <= 654 (test split).")
    output = output_dir / "objects.jsonl"
    if output.exists():
        raise ValueError(f"{output} already exists. Use a new output directory.")
    id2label = json.loads(fetch(MAP_URL))
    if len(id2label) != 40 or id2label.get("0") != "wall":
        raise ValueError("The upstream class mapping changed; inspect before using it.")
    query = urllib.parse.urlencode({"dataset": DATASET, "config": "default",
                                   "split": "test", "offset": offset, "length": count})
    page = json.loads(fetch(f"https://datasets-server.huggingface.co/rows?{query}"))
    rows = page.get("rows", [])
    if len(rows) != count or any(row.get("truncated_cells") for row in rows):
        raise ValueError("Incomplete Dataset Viewer response; no dataset exported.")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    downloaded_bytes = 0
    for item in rows:
        index, row = item["row_idx"], item["row"]
        frame_id = f"nyuv2-test-{index:06d}"
        mask_bytes = fetch(row["label"]["src"])
        downloaded_bytes += len(mask_bytes)
        with Image.open(io.BytesIO(mask_bytes)) as mask:
            mask.load()
            objects = objects_from_mask(mask, id2label)
            size = list(mask.size)
        mask_name = f"{frame_id}.labels.png"
        (output_dir / mask_name).write_bytes(mask_bytes)
        record = {"frame_id": frame_id, "scene": "indoor (room type not supplied)",
                  "objects": objects, "mask_path": mask_name, "image_size": size,
                  "source_dataset": DATASET, "source_split": "test",
                  "source_row": index, "annotation_kind": "semantic_class_regions"}
        if include_rgb:
            rgb_bytes = fetch(row["image"]["src"])
            downloaded_bytes += len(rgb_bytes)
            with Image.open(io.BytesIO(rgb_bytes)) as image:
                image.load()
                if list(image.size) != size:
                    raise ValueError("RGB and semantic mask dimensions differ.")
            image_name = f"{frame_id}.jpg"
            (output_dir / image_name).write_bytes(rgb_bytes)
            record["image_path"] = image_name
        records.append(record)
        print(f"Downloaded {len(records)}/{count}: {frame_id}, {len(objects)} classes", flush=True)
    # Publish the JSONL only after every requested image has succeeded.
    temp = output.with_suffix(".jsonl.tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    temp.replace(output)
    provenance = {"dataset": DATASET, "mapping_revision": REVISION,
                  "source": f"https://huggingface.co/datasets/{DATASET}",
                  "original": "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html",
                  "downloaded_image_bytes": downloaded_bytes, "frames": count,
                  "note": "Semantic classes only, NOT individual object instances. "
                          "No camera poses or fixed/movable ground truth. "
                          "Sample intended for pipeline testing, not a final benchmark."}
    (output_dir / "source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Ready: {output}; downloaded image files {downloaded_bytes / 1_000_000:.2f} MB")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/nyu-small"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    try:
        download_sample(args.output_dir, args.count, args.offset, not args.labels_only)
    except (ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
