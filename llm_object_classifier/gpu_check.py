"""Fail early if the requested Ollama model does not execute on a GPU."""
import argparse
import json
from .vision_pipeline import chat, request_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://ollama:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--require-unquantized", action="store_true")
    args = parser.parse_args()
    try:
        info = request_json(args.base_url, "/api/show", {"model": args.model})
        precision = info.get("details", {}).get("quantization_level", "").upper()
        if args.require_unquantized and precision not in {"BF16", "F16", "F32"}:
            raise ValueError(f"Expected BF16/F16/F32, got {precision!r}.")
        response = chat(args.base_url, args.model, "Return JSON only.",
                        'Return {"ok":true}', 0, 42,
                        options={"num_predict": 16}, require_gpu=True)
        print(json.dumps({"model": args.model, "precision": precision, "gpu": response["gpu"]}, indent=2))
    except Exception as exc:
        parser.exit(1, f"GPU preflight failed: {exc}\n")


if __name__ == "__main__":
    main()
