"""Prepare deterministic RGB selections; annotations remain optional and explicit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image
from .experiment import sha256, strict_json
from .schema import validate_frame
from .vision_pipeline import typed_id


def load_images(manifest, limit=20, frame_ids=None):
    selected, seen = [], set()
    with Path(manifest).open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            record = strict_json(line)
            fid = record.get("frame_id")
            if not isinstance(fid, str) or not fid:
                raise ValueError("Each frame needs a nonempty frame_id.")
            if frame_ids and fid not in frame_ids:
                continue
            if fid in seen:
                raise ValueError(f"Duplicate frame_id: {fid}")
            seen.add(fid)
            name = record.get("image_path")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{fid}: missing image_path.")
            path = Path(name)
            if not path.is_absolute():
                path = Path(manifest).resolve().parent / path
            data = path.read_bytes()
            digest = sha256(data)
            if record.get("image_sha256") and digest != record["image_sha256"]:
                raise ValueError(f"{fid}: image changed since selection was prepared.")
            with Image.open(path) as image:
                image.verify()
                mime = Image.MIME.get(image.format, "application/octet-stream")
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError(f"{fid}: use JPEG, PNG or WEBP.")
            regions = None
            if "objects" in record:
                validate_frame(record)
                for obj in record["objects"]:
                    typed_id(obj["id"])
                # Keep instance boxes and annotation kind. Never call them semantic unions.
                regions = [{k: o[k] for k in ("id", "label", "bbox_xyxy", "region_kind", "annotation_kind") if k in o}
                           for o in record["objects"]]
                if record.get("mask_path") and not any("bbox_xyxy" in o for o in regions):
                    if record.get("annotation_kind") != "semantic_class_regions":
                        raise ValueError("mask_path currently requires annotation_kind=semantic_class_regions; supply explicit instance boxes otherwise.")
                    from .vision_pipeline import region_objects
                    regions = region_objects(record, Path(manifest).resolve().parent)
            selected.append({"record": record, "path": str(path.resolve()), "image": data,
                             "image_sha256": digest, "mime": mime, "regions": regions})
            if len(selected) >= limit:
                break
    if not selected:
        raise ValueError("No selected images found.")
    if frame_ids and set(frame_ids) - seen:
        raise ValueError(f"Frame IDs not found within selection: {set(frame_ids) - seen}")
    return selected


def prepare(images_dir, output, patterns, limit, every_n):
    images_dir, output = Path(images_dir).resolve(), Path(output)
    if not images_dir.is_dir():
        raise ValueError(f"Missing image directory: {images_dir}")
    if output.exists():
        raise ValueError(f"Selection exists: {output}. Choose a new filename.")
    paths = sorted({p for pattern in patterns for p in images_dir.rglob(pattern) if p.is_file()})
    paths = paths[::every_n][:limit]
    if not paths:
        raise ValueError("No images match the patterns. Point to RGB images, not masks/depth.")
    records = []
    for path in paths:
        relative = path.relative_to(images_dir).as_posix()
        with Image.open(path) as image:
            image.verify()
        records.append({"frame_id": "image-" + sha256(relative.encode())[:16],
                        "image_path": str(path), "source_relative_path": relative,
                        "image_sha256": sha256(path.read_bytes()),
                        "selection_mode": "rgb_only_no_ground_truth"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patterns", nargs="+", default=["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--every-n", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1 or args.every_n < 1:
        parser.error("limit and every-n must be positive.")
    try:
        records = prepare(args.images_dir, args.output, args.patterns, args.limit, args.every_n)
        print(f"Selected {len(records)} RGB images: {args.output}")
        print("No object annotations were inferred from dataset filenames.")
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
