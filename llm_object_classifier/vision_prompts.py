"""Dobesedno sestavi uporabniški LLM vhod iz zamrznjenega opisa in regij."""
import json

REASON_INSTRUCTION = "Classify every ID using the labels and visual observations below:"


def reason_prompt(frame_id, objects, caption, instructions=None):
    """Ohrani celoten opis, vse regije in točne tipe ID-jev brez dodajanja podatkov."""
    return (REASON_INSTRUCTION if instructions is None else instructions) + "\n" + json.dumps(
        {"frame_id": frame_id, "objects": [{"id": o["id"], "label": o["label"]} for o in objects],
         "visual_observations": caption}, ensure_ascii=False)
