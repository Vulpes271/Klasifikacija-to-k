from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .baseline import classify_label, load_label_map
from .client import ModelConfig, OpenAICompatibleClient
from .evaluate import evaluate_results
from .io_utils import append_jsonl, iter_jsonl_frames, load_completed_frame_ids, write_csv_rows
from .schema import frame_to_json, validate_classifications
from .thud import UnsupportedThudFormatError, convert_thud_records

CSV_FIELDS = [
    "run_id",
    "model",
    "temperature",
    "max_tokens",
    "status",
    "frame_id",
    "scene",
    "object_id",
    "label",
    "classification",
    "reason",
    "elapsed_seconds",
    "error",
]


def make_run_dir(base: Path, name: str | None = None) -> Path:
    run_name = name or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = base / run_name
    suffix = 1
    while path.exists():
        suffix += 1
        path = base / f"{run_name}-{suffix}"
    path.mkdir(parents=True)
    return path


def command_run(args: argparse.Namespace) -> int:
    config = ModelConfig.from_file(str(args.config))
    run_dir = args.output_dir if args.resume else make_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    csv_path = run_dir / "results.csv"
    completed = load_completed_frame_ids(results_path) if args.resume else set()
    client = OpenAICompatibleClient(config)

    summary = {"ok": 0, "error": 0, "skipped": 0}
    for frame in iter_jsonl_frames(args.input, limit=args.limit, every_n=args.every_n):
        if frame.frame_id in completed:
            summary["skipped"] += 1
            continue
        raw_response = ""
        elapsed = 0.0
        try:
            llm_result = client.classify(frame)
            raw_response = llm_result.content
            elapsed = llm_result.elapsed_seconds
            parsed = json.loads(raw_response)
            classifications = validate_classifications(frame, parsed)
            record = {
                "status": "ok",
                "frame_id": frame.frame_id,
                "scene": frame.scene,
                "model": config.model,
                "settings": {
                    "base_url": config.base_url,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                },
                "elapsed_seconds": elapsed,
                "raw_response": raw_response,
                "objects": [
                    {
                        "id": item.id,
                        "label": next(obj.label for obj in frame.objects if obj.id == item.id),
                        "classification": item.classification,
                        "reason": item.reason,
                    }
                    for item in classifications
                ],
            }
            append_jsonl(results_path, record)
            rows = [
                {
                    "run_id": run_dir.name,
                    "model": config.model,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "status": "ok",
                    "frame_id": frame.frame_id,
                    "scene": frame.scene,
                    "object_id": item["id"],
                    "label": item["label"],
                    "classification": item["classification"],
                    "reason": item["reason"],
                    "elapsed_seconds": elapsed,
                    "error": "",
                }
                for item in record["objects"]
            ]
            write_csv_rows(csv_path, rows, CSV_FIELDS)
            summary["ok"] += 1
        except Exception as exc:
            error = str(exc)
            record = {
                "status": "error",
                "frame_id": frame.frame_id,
                "scene": frame.scene,
                "model": config.model,
                "elapsed_seconds": elapsed,
                "raw_response": raw_response,
                "error": error,
                "input": frame_to_json(frame),
            }
            append_jsonl(results_path, record)
            write_csv_rows(
                csv_path,
                [
                    {
                        "run_id": run_dir.name,
                        "model": config.model,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "status": "error",
                        "frame_id": frame.frame_id,
                        "scene": frame.scene,
                        "elapsed_seconds": elapsed,
                        "error": error,
                    }
                ],
                CSV_FIELDS,
            )
            summary["error"] += 1

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return 0


def command_baseline(args: argparse.Namespace) -> int:
    mapping = load_label_map(args.label_map)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    results_path = run_dir / "results.jsonl"
    csv_path = run_dir / "results.csv"
    start = time.perf_counter()
    frames = 0
    objects = 0
    for frame in iter_jsonl_frames(args.input, limit=args.limit, every_n=args.every_n):
        frames += 1
        elapsed = time.perf_counter() - start
        classified = []
        rows = []
        for item in frame.objects:
            classification, reason = classify_label(item.label, mapping)
            classified.append(
                {
                    "id": item.id,
                    "label": item.label,
                    "classification": classification,
                    "reason": reason,
                }
            )
            rows.append(
                {
                    "run_id": run_dir.name,
                    "model": "label-map-baseline",
                    "temperature": "",
                    "max_tokens": "",
                    "status": "ok",
                    "frame_id": frame.frame_id,
                    "scene": frame.scene,
                    "object_id": item.id,
                    "label": item.label,
                    "classification": classification,
                    "reason": reason,
                    "elapsed_seconds": elapsed,
                    "error": "",
                }
            )
            objects += 1
        append_jsonl(
            results_path,
            {
                "status": "ok",
                "frame_id": frame.frame_id,
                "scene": frame.scene,
                "model": "label-map-baseline",
                "elapsed_seconds": elapsed,
                "objects": classified,
            },
        )
        write_csv_rows(csv_path, rows, CSV_FIELDS)
    summary = {"frames": frames, "objects": objects, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def command_prepare_thud(args: argparse.Namespace) -> int:
    written = 0
    errors = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        try:
            for index, record in enumerate(convert_thud_records(args.input, args.source_format), start=1):
                if args.every_n > 1 and (index - 1) % args.every_n != 0:
                    continue
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                if args.limit is not None and written >= args.limit:
                    break
        except UnsupportedThudFormatError as exc:
            errors.append(str(exc))
    report = {"output": str(args.output), "frames_written": written, "errors": errors}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors and written == 0 else 0


def command_evaluate(args: argparse.Namespace) -> int:
    metrics = evaluate_results(args.results, args.reference)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local LLMs for fixed/movable object classification.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run LLM classification on standardized JSONL input.")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    run_parser.add_argument("--run-name")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--every-n", type=int, default=1)
    run_parser.add_argument("--resume", action="store_true", help="Use --output-dir as an existing run dir.")
    run_parser.set_defaults(func=command_run)

    baseline_parser = subparsers.add_parser("baseline", help="Run label-map baseline without an LLM.")
    baseline_parser.add_argument("--input", type=Path, required=True)
    baseline_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    baseline_parser.add_argument("--run-name")
    baseline_parser.add_argument("--label-map", type=Path)
    baseline_parser.add_argument("--limit", type=int)
    baseline_parser.add_argument("--every-n", type=int, default=1)
    baseline_parser.set_defaults(func=command_baseline)

    prepare_parser = subparsers.add_parser("prepare-thud", help="Convert supported THUD JSON files to JSONL.")
    prepare_parser.add_argument("--input", type=Path, required=True, help="THUD folder, zip, or JSON file.")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--source-format", choices=["auto", "synthetic-captures", "real-2d"], default="auto")
    prepare_parser.add_argument("--limit", type=int)
    prepare_parser.add_argument("--every-n", type=int, default=1)
    prepare_parser.set_defaults(func=command_prepare_thud)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a results JSONL against manual reference CSV.")
    eval_parser.add_argument("--results", type=Path, required=True)
    eval_parser.add_argument("--reference", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path)
    eval_parser.set_defaults(func=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


