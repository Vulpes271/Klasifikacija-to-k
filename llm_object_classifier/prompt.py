from __future__ import annotations

import json

from .schema import Frame

SYSTEM_PROMPT = """You classify already detected indoor objects for long-term robotic mapping.

Classes:
- fixed: permanently attached structural or room element suitable for a long-term map.
- movable: an object that can be moved even if it is currently stationary, including people.
- uncertain: the label and available context are insufficient for a reliable decision.

Important rules:
- Stationary does not mean permanently fixed.
- Do not guess that a cabinet, shelf, desk, counter, or table is built-in unless that information is available.
- Preserve every input object id exactly once.
- Return strict JSON only, without markdown or explanatory text outside JSON.
"""


def build_user_prompt(frame: Frame) -> str:
    payload = {
        "frame_id": frame.frame_id,
        "scene": frame.scene,
        "objects": [{"id": item.id, "label": item.label} for item in frame.objects],
        "output_schema": {
            "objects": [
                {
                    "id": "same id as input",
                    "classification": "fixed | movable | uncertain",
                    "reason": "short English reason",
                }
            ]
        },
    }
    return (
        "Classify each object using only the label and scene context below.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

