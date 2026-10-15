"""Select a reproducible test sample from one real THUD capture.

Uses existing LabelMe rectangle annotations, Pillow, and the standard library.
The default dense mode prioritizes frames filled with foreground objects.
Does not infer mobility ground truth or change the source capture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageStat

BACKGROUND_LABELS = {"window", "door", "wall", "floor", "ceiling", "stairs", "lift"}


def foreground_coverage(objects, width, height):
    """Exact rectangle union; an occupancy proxy, not object segmentation."""
    boxes = [o["bbox_xyxy"] for o in objects if o["label"].casefold() not in BACKGROUND_LABELS]
    xs = sorted({x for box in boxes for x in (box[0], box[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        end, covered = 0.0, 0.0
        for start, stop in intervals:
            covered += max(0, stop - max(start, end))
            end = max(end, stop)
        area += (right - left) * covered
    return area / (width * height)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def describe_image(path):
    with Image.open(path) as original:
        rgb = original.convert("RGB")
        gray = rgb.convert("L")
        small = gray.resize((17, 8))
        pixels = list(small.getdata())
        dhash = 0
        for y in range(8):
            for x in range(16):
                dhash = (dhash << 1) | (pixels[y * 17 + x] > pixels[y * 17 + x + 1])
        edge_image = gray.resize((640, 360)).filter(ImageFilter.FIND_EDGES).crop((1, 1, 639, 359))
        sharpness = ImageStat.Stat(edge_image).var[0]
        colors = tuple(channel / 255 for pixel in rgb.resize((16, 9)).getdata() for channel in pixel)
        histogram = gray.histogram()
        return {
            "width": rgb.width, "height": rgb.height,
            "edge_variance": sharpness,
            "dark_fraction": sum(histogram[:6]) / (rgb.width * rgb.height),
            "bright_fraction": sum(histogram[250:]) / (rgb.width * rgb.height),
            "dhash": dhash, "colors": colors,
        }


def distance(a, b):
    hamming = bin(a["dhash"] ^ b["dhash"]).count("1") / 128
    color = sum(abs(x - y) for x, y in zip(a["colors"], b["colors"])) / len(a["colors"])
    return 0.5 * hamming + 0.5 * min(1, color * 3)


def load_capture(source):
    rows = []
    for annotation_path in sorted((source / "Label/2D_Object_Detection").glob("*.json")):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = source / "RGB" / (annotation_path.stem + ".png")
        if not image_path.is_file():
            raise ValueError(f"Missing RGB image: {image_path}")
        metrics = describe_image(image_path)
        w, h = metrics["width"], metrics["height"]
        if (annotation["imageWidth"], annotation["imageHeight"]) != (w, h):
            raise ValueError(f"Annotation dimensions disagree: {annotation_path}")
        objects = []
        for index, shape in enumerate(annotation["shapes"], 1):
            if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) != 2:
                raise ValueError(f"Expected rectangle: {annotation_path}, shape {index}")
            (xa, ya), (xb, yb) = shape["points"]
            x1, x2 = max(0, min(xa, xb)), min(w, max(xa, xb))
            y1, y2 = max(0, min(ya, yb)), min(h, max(ya, yb))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid rectangle: {annotation_path}, shape {index}")
            objects.append({
                "id": index, "label": shape["label"], "bbox_xyxy": [x1, y1, x2, y2],
                "region_kind": "object_instance", "source_shape_index": index - 1,
            })
        useful = [o for o in objects if
                  (o["bbox_xyxy"][2] - o["bbox_xyxy"][0]) *
                  (o["bbox_xyxy"][3] - o["bbox_xyxy"][1]) >= w * h * 0.001
                  and min(o["bbox_xyxy"][2] - o["bbox_xyxy"][0],
                          o["bbox_xyxy"][3] - o["bbox_xyxy"][1]) >= 10]
        rows.append({
            "frame_id": annotation_path.stem.removesuffix(".color"),
            "frame_number": int(annotation_path.stem.split("-")[1].split(".")[0]),
            "image": image_path, "annotation_path": annotation_path,
            "objects": objects, "object_count": len(objects), "usable_objects": len(useful),
            "class_count": len({o["label"] for o in useful}), **metrics,
            "foreground_box_coverage": foreground_coverage(objects, w, h),
        })
    if not rows:
        raise ValueError("No real-scene 2D annotations found.")
    return rows


def select(rows, count, mode="dense"):
    # Reject the least sharp 10% within this capture. This is a relative proxy,
    # not an objective camera-independent blur measurement.
    cutoff = sorted(r["edge_variance"] for r in rows)[len(rows) // 10]
    dense = mode == "dense"
    rules = {"mode": mode, "min_usable_objects": 8 if dense else 14,
             "min_classes": 4 if dense else 6,
             "min_foreground_box_coverage": 0.75 if dense else 0,
             "min_frame_number_gap": 3 if dense else 4,
             "min_bbox_area_fraction": 0.001, "min_bbox_side_px": 10,
             "sharpness_bottom_fraction_excluded": 0.1,
             "edge_variance_cutoff": cutoff, "min_perceptual_distance": 0.07}
    pool = [r for r in rows if r["usable_objects"] >= rules["min_usable_objects"]
            and r["class_count"] >= rules["min_classes"]
            and r["foreground_box_coverage"] >= rules["min_foreground_box_coverage"]
            and r["edge_variance"] >= cutoff]
    sharpness_values = sorted(r["edge_variance"] for r in rows)
    for row in pool:
        row["quality"] = (0.40 * min(row["usable_objects"], 30) / 30
                          + 0.35 * min(row["class_count"], 12) / 12
                          + 0.25 * sharpness_values.index(row["edge_variance"]) / (len(rows) - 1))
        row["novelty"] = 1.0
        if dense:
            row["quality"] = (0.70 * row["foreground_box_coverage"]
                              + 0.15 * min(row["usable_objects"], 25) / 25
                              + 0.15 * sharpness_values.index(row["edge_variance"]) / (len(rows) - 1))
    chosen = []
    available = list(pool)
    while available and len(chosen) < count:
        ranked = sorted(available, key=lambda r: ((0.85 if dense else 0.65) * r["quality"]
                                                  + (0.15 if dense else 0.35) * r["novelty"],
                                                  -r["frame_number"]), reverse=True)
        best = None
        for candidate in ranked:
            remaining = [r for r in available
                         if abs(r["frame_number"] - candidate["frame_number"]) >= rules["min_frame_number_gap"]
                         and distance(r, candidate) >= rules["min_perceptual_distance"]]
            # Preserve enough temporal slots to reach the requested sample size.
            last, capacity = -100000, 0
            for row in sorted(remaining, key=lambda r: r["frame_number"]):
                if row["frame_number"] - last >= rules["min_frame_number_gap"]:
                    capacity += 1
                    last = row["frame_number"]
            if capacity >= count - len(chosen) - 1:
                best, next_available = candidate, remaining
                break
        if best is None:
            break
        chosen.append(best)
        for row in next_available:
            row["novelty"] = min(row["novelty"], distance(row, best))
        available = next_available
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} frames meet the constraints; requested {count}.")
    return sorted(chosen, key=lambda r: r["frame_number"]), len(pool), rules


def export(source, output, rows, chosen, pool_size, rules):
    parts = source.parts
    scene = parts[parts.index("Real_Scenes") + 1] if "Real_Scenes" in parts else source.parent.parent.name
    output.mkdir(parents=True, exist_ok=False)
    for name in ("images", "annotations", "preview"):
        (output / name).mkdir()
    records, checksums = [], []
    gallery = []
    for row in chosen:
        image_name = row["image"].name
        shutil.copy2(row["image"], output / "images" / image_name)
        raw = json.loads(row["annotation_path"].read_text(encoding="utf-8"))
        raw["imageData"] = None  # Preserve shapes without duplicating base64 RGB.
        raw["imagePath"] = "../images/" + image_name
        annotation_name = row["annotation_path"].name
        write_json(output / "annotations" / annotation_name, raw)
        digest = hashlib.sha256((output / "images" / image_name).read_bytes()).hexdigest()
        checksums.append(digest + "  images/" + image_name)
        records.append({
            "frame_id": row["frame_id"], "scene": scene, "objects": row["objects"],
            "image_path": "images/" + image_name, "image_size": [row["width"], row["height"]],
            "annotation_path": "annotations/" + annotation_name,
            "source_capture": source.as_posix(), "source_image": row["image"].as_posix(),
            "object_count": row["object_count"], "class_count": row["class_count"],
            "foreground_box_coverage": row["foreground_box_coverage"],
        })
        with Image.open(row["image"]) as original:
            thumbnail = original.convert("RGB")
            thumbnail.thumbnail((480, 270))
            thumbnail.save(output / "preview" / (row["frame_id"] + ".jpg"), quality=85)
        labels = ", ".join(sorted({o["label"] for o in row["objects"]}))
        gallery.append(f'<figure><a href="images/{image_name}"><img loading="lazy" '
                       f'src="preview/{row["frame_id"]}.jpg"></a><figcaption>'
                       f'{row["frame_id"]}: {row["object_count"]} objektov, '
                       f'{row["class_count"]} vrst; pokritost okvirjev {row["foreground_box_coverage"]:.0%}'
                       f'<br>{html.escape(labels)}</figcaption></figure>')
    if len(set(checksums)) != len(checksums) or len({line.split()[0] for line in checksums}) != len(chosen):
        raise ValueError("Exact duplicate image in selection.")
    (output / "objects.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    fields = ["frame_id", "frame_number", "object_count", "usable_objects", "class_count",
              "edge_variance", "dark_fraction", "bright_fraction", "foreground_box_coverage", "selected"]
    selected_ids = {r["frame_id"] for r in chosen}
    with (output / "selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in fields[:-1]}, "selected": row["frame_id"] in selected_ids})
    label_instances = Counter(o["label"] for row in chosen for o in row["objects"])
    label_frames = Counter(label for row in chosen for label in {o["label"] for o in row["objects"]})
    counts = [r["object_count"] for r in chosen]
    classes = [r["class_count"] for r in chosen]
    pairs = [distance(a, b) for i, a in enumerate(chosen) for b in chosen[i + 1:]]
    summary = {
        "source_capture": source.as_posix(), "audited_frames": len(rows),
        "eligible_frames": pool_size, "selected_frames": len(chosen),
        "objects_total": sum(counts), "objects_min": min(counts), "objects_max": max(counts),
        "objects_mean": statistics.mean(counts), "classes_min": min(classes),
        "classes_max": max(classes), "classes_mean": statistics.mean(classes),
        "distinct_classes": len(label_instances), "instances_per_class": dict(sorted(label_instances.items())),
        "frames_per_class": dict(sorted(label_frames.items())),
        "selection_rules": rules,
        "foreground_box_coverage_min": min(r["foreground_box_coverage"] for r in chosen),
        "foreground_box_coverage_mean": statistics.mean(r["foreground_box_coverage"] for r in chosen),
        "coverage_excluded_labels": sorted(BACKGROUND_LABELS),
        "coverage_interpretation": "Union of foreground annotation rectangles; not segmentation or an exact empty-space fraction",
        "actual_min_frame_number_gap": min(b["frame_number"] - a["frame_number"] for a, b in zip(chosen, chosen[1:])),
        "actual_min_perceptual_distance": min(pairs), "mobility_ground_truth": False,
        "instance_ids": "1-based annotation shape index within each image; not tracking IDs",
    }
    write_json(output / "summary.json", summary)
    for page in range(math.ceil(len(chosen) / 25)):
        canvas = Image.new("RGB", (1600, 1050), "white")
        draw = ImageDraw.Draw(canvas)
        for position, row in enumerate(chosen[page * 25:(page + 1) * 25]):
            x, y = (position % 5) * 320, (position // 5) * 210
            with Image.open(row["image"]) as original:
                thumb = original.convert("RGB")
                thumb.thumbnail((320, 180))
                canvas.paste(thumb, (x, y))
            draw.text((x + 4, y + 182), f'{row["frame_id"]} | objects={row["object_count"]} | boxes={row["foreground_box_coverage"]:.0%}', fill="black")
        canvas.save(output / f"contact_sheet_{page + 1:02d}.jpg", quality=92)
    density_note = ('<p>Glavno merilo izbora je zapolnjenost kadra s predmeti. '
                    'Pokritost okvirjev je približek iz pravokotnih oznak; '
                    'ne pomeni natančnega deleža pik predmetov.</p>') if rules["mode"] == "dense" else ''
    (output / "gallery.html").write_text(
        f'<!doctype html><html lang="sl"><meta charset="utf-8"><title>THUD {html.escape(scene)}: testne slike</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#f4f4f4}main{display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}figure{margin:0;'
        'background:white;padding:10px}img{width:100%;height:auto}figcaption{line-height:1.5}</style>'
        f'<h1>THUD {html.escape(scene)}: {len(chosen)} testnih slik</h1><p>{min(counts)}–{max(counts)} označenih '
        f'objektov na sliko; povprečno {statistics.mean(counts):.1f}. Klik odpre izvirno sliko.</p>'
        + density_note + '<main>' + "\n".join(gallery) + '</main></html>', encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/THUD_Robot/Real_Scenes/store/dynamic/Capture_1"))
    parser.add_argument("--output", type=Path, default=Path("data/thud_dense_test50"))
    parser.add_argument("--mode", choices=("dense", "diverse"), default="dense")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    if args.count < 2:
        parser.error("--count must be at least 2")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory.")
    rows = load_capture(args.source)
    chosen, pool_size, rules = select(rows, args.count, args.mode)
    summary = export(args.source, args.output, rows, chosen, pool_size, rules)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
