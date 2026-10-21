"""Validate explicit visual evidence and a graph of local object IDs.

No coordinates or stability labels are estimated here. Strings describe image
coordinates in words; relations are assertions by the VLM, not verified facts.
Legacy fields remain available to old CSV consumers and the text-only reasoner.
"""
from .vision_pipeline import typed_id

VERSION = "rich-description-v2"
VISUAL_FIELDS = ("appearance", "image_location", "support_contacts", "installation_evidence")
PREDICATES = {"on", "under", "beside", "in_front_of", "behind", "attached_to", "inside", "supports", "overlapping"}
OCCLUSION = {"none", "partial", "heavy", "unknown"}


def validate_rich(parsed):
    if set(parsed) != {"schema_version", "scene_description", "objects", "relations"}:
        raise ValueError("Rich description must contain only schema_version, scene_description, objects, relations.")
    if parsed["schema_version"] != VERSION:
        raise ValueError("Unknown rich description version.")
    ids = {typed_id(o["id"]) for o in parsed["objects"]}
    for obj in parsed["objects"]:
        required = {"id", "label", "observation", "attachment_evidence", "identifiable", "occlusion", *VISUAL_FIELDS}
        if set(obj) != required:
            raise ValueError("Rich object fields missing or unexpected.")
        if type(obj["id"]) is not int or obj["id"] < 1:
            raise ValueError("Rich local IDs must be positive integers.")
        for field in VISUAL_FIELDS:
            if not isinstance(obj[field], str) or not obj[field].strip():
                raise ValueError(f"Missing rich evidence field: {field}.")
        if obj["occlusion"] not in OCCLUSION:
            raise ValueError("Invalid occlusion value.")
    if not isinstance(parsed["relations"], list):
        raise ValueError("Missing relations list.")
    seen = set()
    for relation in parsed["relations"]:
        if not isinstance(relation, dict) or set(relation) != {"subject_id", "predicate", "object_id", "evidence"}:
            raise ValueError("Invalid relation fields.")
        subject, target = typed_id(relation["subject_id"]), typed_id(relation["object_id"])
        if subject not in ids or target not in ids or subject == target:
            raise ValueError("Relation must reference two different existing typed IDs.")
        if relation["predicate"] not in PREDICATES:
            raise ValueError("Unknown relation predicate.")
        key = (subject, relation["predicate"], target)
        if key in seen:
            raise ValueError("Duplicate relation.")
        seen.add(key)
        if not isinstance(relation["evidence"], str) or not relation["evidence"].strip():
            raise ValueError("Relation requires visible evidence text.")
    return parsed
