"""Skupno branje/pisanje, kontrolne vsote in nespremenjene nastavitve generiranja."""
import hashlib
import json
import math
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


def finite_number(value, name, lower=0, upper=None, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'}.")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside the allowed range.")
    return value


def options(config, variant):
    return {"temperature": variant["temperature"], "seed": variant["seed"],
            "num_predict": config["max_tokens"], "num_ctx": config["num_ctx"],
            "top_p": config["top_p"], "top_k": config["top_k"],
            "presence_penalty": config["presence_penalty"], "num_gpu": 999}
