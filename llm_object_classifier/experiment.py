"""Shared validation and output helpers for separate experiment stages."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def slug(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Use a name with letters, digits, _, . or -: {value!r}")
    if value in {"manifest.json", "summary.json", "comparison.html", "descriptions.csv", "classifications.csv"}:
        raise ValueError("Name conflicts with a reserved output filename.")
    return value


def new_run(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Output already exists: {path}. Choose a new run name.")
    path.mkdir(parents=True)
    return path


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value, name, lower=0, upper=None, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'}.")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside the allowed range.")
    return value


def load_config(path, stage):
    config = read_json(path)
    allowed = {"models", "temperatures", "seeds", "max_tokens", "num_ctx", "top_p", "top_k",
               "presence_penalty", "timeout", "detail", "system_file", "instruction_file"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Configuration must contain documented keys only.")
    defaults = {"temperatures": [0.0] if stage == 1 else [0.0, 0.3, 0.7], "seeds": [42],
                "max_tokens": 4096 if stage == 1 else 2048, "num_ctx": 8192,
                "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
                "timeout": 600, "detail": "detailed"}
    config = {**defaults, **config}
    for key in ("temperatures", "seeds"):
        values = config[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list.")
        for v in values:
            finite_number(v, key, lower=-2147483648 if key == "seeds" else 0, integer=key == "seeds")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {key}.")
    for key in ("max_tokens", "num_ctx", "timeout", "top_k"):
        finite_number(config[key], key, lower=0 if key == "top_k" else 1, integer=True)
    finite_number(config["top_p"], "top_p", lower=0, upper=1)
    if config["top_p"] == 0:
        raise ValueError("top_p must be > 0.")
    finite_number(config["presence_penalty"], "presence_penalty", upper=2)
    if config["detail"] not in {"brief", "standard", "detailed"}:
        raise ValueError("detail must be brief, standard or detailed.")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list.")
    names = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Each model must be an object.")
        known = {"name", "backend", "model", "base_url", "require_gpu", "precision", "system_as_user",
                 "send_seed", "json_mode", "revision", "api_key_env"}
        if set(model) - known:
            raise ValueError(f"Unknown model options: {set(model) - known}")
        name = slug(model.get("name"))
        if name in names:
            raise ValueError("Model names must be unique.")
        names.add(name)
        backend = model.setdefault("backend", "ollama")
        if backend not in {"ollama", "openai"}:
            raise ValueError("backend must be ollama or openai.")
        model.setdefault("base_url", "http://ollama:11434" if backend == "ollama" else "http://host.docker.internal:8000")
        for key in ("model", "base_url"):
            value = model.get(key)
            if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
                raise ValueError(f"Invalid model {key}.")
        if not model["base_url"].startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://.")
        for key, default in (("require_gpu", backend == "ollama"), ("system_as_user", False),
                             ("send_seed", False), ("json_mode", False)):
            if type(model.setdefault(key, default)) is not bool:
                raise ValueError(f"{key} must be a JSON boolean.")
        if backend == "openai" and (model["require_gpu"] or model.get("precision")):
            raise ValueError("OpenAI API cannot verify GPU or precision; check the server separately.")
        if model.get("precision") not in {None, "BF16", "F16", "F32", "Q4_K_M", "Q8_0"}:
            raise ValueError("Unsupported precision check.")
    for key in ("system_file", "instruction_file"):
        if key in config:
            p = config[key]
            if not isinstance(p, str) or not p:
                raise ValueError(f"{key} must be a path string.")
            p = Path(p)
            if not p.is_absolute():
                p = Path(path).resolve().parent / p
            text = p.read_text(encoding="utf-8-sig").strip()
            if not text:
                raise ValueError(f"Empty {key}: {p}")
            config[key] = str(p)
            config[key + "_text"] = text
    return config


def combinations(config):
    for temperature in config["temperatures"]:
        temperature_text = str(float(temperature)).removesuffix(".0")
        for seed in config["seeds"]:
            yield {"id": f"t{temperature_text}-s{seed}", "temperature": temperature, "seed": seed}


def options(config, variant):
    return {"temperature": variant["temperature"], "seed": variant["seed"],
            "num_predict": config["max_tokens"], "num_ctx": config["num_ctx"],
            "top_p": config["top_p"], "top_k": config["top_k"],
            "presence_penalty": config["presence_penalty"], "num_gpu": 999}
