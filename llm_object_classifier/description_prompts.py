"""Stage 1 describes visible evidence, independently of mobility classification."""
import json
from .vision_pipeline import ATTACHMENTS, exact_items, typed_id

VERSION = "separate-description-v1"
SYSTEM = '''Describe an RGB photograph for later robotic scene analysis.
Report visible evidence only. Do NOT classify objects as fixed or movable.
Describe each object's appearance, location, supports/clearance, contact with other
objects, wheels/legs, handles, and clearly visible mounting or integration.
Do not infer hidden screws, weight, fastening, function, or permanent attachment.
Hanging from a hook, proximity to a wall, and standing still do not prove permanent
installation. Say what is visible and what cannot be established. Do not hallucinate.
The input tells you whether supplied IDs are semantic class regions, object
instances, or unspecified annotations. Semantic class regions may contain several
objects; describe differences and use mixed evidence when appropriate.
Some supplied boxes are group annotations (e.g. Products): do not assume that
one annotated box means one physical object. Coordinates refer to original image
pixels; image_size is [width, height].
When supplied_objects is null, identify visible objects yourself and assign unique
integer IDs within this photograph. These are tentative detections, not dataset
annotations. Describe separate visible instances separately when possible.
When supplied_objects is a list, copy ALL its IDs and labels exactly once; do not
add objects. If a supplied object cannot be identified, use identifiable=false.
Return only JSON with scene_description (up to SCENE_WORDS words) and objects.
Each object must contain id, label, observation (up to OBJECT_WORDS words),
attachment_evidence (visible_installation, appears_freestanding, not_visible, mixed),
and identifiable (boolean). visible_installation requires visible evidence of
mounting/integration; uncertainty must be stated in the observation.
No markdown. Text fields must be in English.'''


def system_prompt(detail):
    scene, obj = {"brief": (35, 25), "standard": (80, 60), "detailed": (160, 120)}[detail]
    return SYSTEM.replace("SCENE_WORDS", str(scene)).replace("OBJECT_WORDS", str(obj))


def user_prompt(frame):
    return "Describe the photograph using these inputs:\n" + json.dumps({
        "frame_id": frame["record"]["frame_id"],
        "image_size": frame["image_size"],
        "annotation_kind": frame["record"].get("annotation_kind", "unspecified"),
        "supplied_objects": frame["regions"]}, ensure_ascii=False)


def validate_description(parsed, supplied=None, expected_schema=None):
    if not isinstance(parsed, dict) or not isinstance(parsed.get("scene_description"), str) or not parsed["scene_description"].strip():
        raise ValueError("Missing scene_description.")
    objects = parsed.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Missing objects list.")
    if supplied is not None:
        exact_items(supplied, objects)
        labels = {typed_id(o["id"]): o["label"] for o in supplied}
    seen = set()
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("Each object must be an object.")
        key = typed_id(obj.get("id"))
        if key in seen:
            raise ValueError("Duplicate object ID.")
        seen.add(key)
        if any(k in obj for k in ("classification", "reason")):
            raise ValueError("Stage 1 must not emit mobility classifications/reasons.")
        for field in ("label", "observation"):
            if not isinstance(obj.get(field), str) or not obj[field].strip():
                raise ValueError(f"Missing {field}.")
        if supplied is not None and obj["label"] != labels[key]:
            raise ValueError("A supplied label was changed.")
        if obj.get("attachment_evidence") not in ATTACHMENTS or type(obj.get("identifiable")) is not bool:
            raise ValueError("Invalid evidence or identifiable field.")
    rich = parsed.get("schema_version") == "rich-description-v2"
    if expected_schema == "rich-v2" and not rich:
        raise ValueError("Missing rich-description-v2 schema_version.")
    if "schema_version" in parsed or "relations" in parsed:
        if not rich:
            raise ValueError("Unknown or missing description schema_version.")
        from .rich_description import validate_rich
        validate_rich(parsed)
    return parsed
