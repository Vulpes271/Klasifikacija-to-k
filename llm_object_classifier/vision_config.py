"""CLI and optional JSON settings for the two-stage pipeline.

Priority: command line > configuration file > original working defaults.
Paths inside JSON are relative to that file, so Linux runs do not depend on
the directory from which the configuration was created.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .vision_prompts import CAPTION_DETAIL


def build_parser():
    parser = argparse.ArgumentParser(description="Image observations followed by mobility reasoning.")
    parser.add_argument("--config", type=Path, help="Optional JSON settings; CLI flags override its values.")
    parser.add_argument("--input", type=Path, default=Path("data/nyu20/objects.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/vision-caption-cache"))
    parser.add_argument("--base-url", default="http://ollama:11434")
    parser.add_argument("--vision-model", default="qwen3.5:4b")
    parser.add_argument("--reason-model", default="qwen3.5:4b")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.3, 0.7],
                        help="Stage 2 temperatures; each value creates a separate experiment.")
    parser.add_argument("--caption-temperature", type=float, default=0.0, help="Stage 1 temperature.")
    parser.add_argument("--caption-detail", choices=CAPTION_DETAIL, default="brief",
                        help="Stage 1 scene/object word budgets: brief 35/25, standard 80/60, detailed 160/120.")
    parser.add_argument("--reason-prompt-file", type=Path,
                        help="UTF-8 stage 2 task instructions; labels and observations are appended automatically.")
    parser.add_argument("--reason-system-file", type=Path,
                        help="Optional replacement for the complete stage 2 system prompt, including JSON rules.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", help="Reasoning seeds; caption always uses --seed.")
    parser.add_argument("--profiles", type=Path, help="JSON sampling profiles; crossed with temperatures and seeds.")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--caption-num-predict", type=int,
                        help="Stage 1 token limit; defaults to --num-predict for compatibility.")
    parser.add_argument("--require-gpu", action="store_true", help="Abort on CPU fallback; records /api/ps per request.")
    parser.add_argument("--require-unquantized", action="store_true", help="Check both models' reported weight precision.")
    parser.add_argument("--print-models", action="store_true",
                        help="Print the two selected model names and exit without contacting Ollama.")
    return parser


def config_defaults(parser, path):
    """Reject misspelled keys and wrong JSON types instead of silently ignoring them."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object.")
    actions = {a.dest: a for a in parser._actions if a.dest not in {"help", "config", "print_models"}}
    unknown = set(raw) - set(actions)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    result = {}
    for key, value in raw.items():
        action = actions[key]
        values = value if action.nargs == "+" else [value]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list.")
        converted = []
        for item in values:
            if action.type is Path:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"{key} must be a nonempty path string.")
                item = Path(item)
                if not item.is_absolute():
                    item = path.resolve().parent / item
            elif isinstance(action, argparse._StoreTrueAction):
                if type(item) is not bool:
                    raise ValueError(f"{key} must be a JSON boolean.")
            elif action.type in (int, float):
                valid_types = (int,) if action.type is int else (int, float)
                if type(item) not in valid_types:
                    raise ValueError(f"{key} has an invalid numeric type.")
                item = action.type(item)
            elif not isinstance(item, str) or not item.strip():
                raise ValueError(f"{key} must be a nonempty string.")
            if action.choices is not None and item not in action.choices:
                raise ValueError(f"Invalid {key}: {item!r}; choose from {list(action.choices)}")
            converted.append(item)
        result[key] = converted if action.nargs == "+" else converted[0]
    return result


def parse_args(argv=None):
    # Read just --config first, then let argparse apply CLI overrides normally.
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    selected, _ = preliminary.parse_known_args(argv)
    parser = build_parser()
    if selected.config:
        try:
            defaults = config_defaults(parser, selected.config)
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot load configuration: {exc}")
        parser.set_defaults(**defaults)
        if "output_dir" in defaults:
            next(a for a in parser._actions if a.dest == "output_dir").required = False
    args = parser.parse_args(argv)
    if args.limit < 1 or len(set(args.temperatures)) != len(args.temperatures):
        parser.error("limit must be positive and temperatures must be unique.")
    if any(not math.isfinite(t) or t < 0 for t in [*args.temperatures, args.caption_temperature]):
        parser.error("Temperatures must be finite and nonnegative.")
    for key in ("timeout", "num_ctx", "num_predict", "caption_num_predict"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"{key} must be positive.")
    for key in ("base_url", "vision_model", "reason_model"):
        value = getattr(args, key)
        if not value.strip() or "\n" in value or "\r" in value:
            parser.error(f"{key} must be a nonempty single line.")
    return args
