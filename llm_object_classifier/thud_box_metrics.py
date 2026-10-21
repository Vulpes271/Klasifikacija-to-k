"""Auditable one-to-one region matching. No guessed captions or semantic scores."""


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def normalize(label):
    return ' '.join(label.lower().split())


def match_regions(predictions, references, policy):
    aliases = policy['aliases']
    vocabulary = set(aliases.values())
    refs = [(i,r) for i,r in enumerate(references) if r['label'] in vocabulary]
    candidates = {}
    for p, pred in enumerate(predictions):
        canonical = aliases.get(normalize(pred['label']))
        possible = [(i, iou(pred['bbox_xyxy'], ref['bbox_xyxy'])) for i,ref in refs if canonical == ref['label']]
        candidates[p] = [(i,overlap) for i,overlap in sorted(possible, key=lambda x:(-x[1], x[0]))
                         if overlap >= policy['iou_threshold']]
    assigned = {}

    def augment(p, visited):
        for r, overlap in candidates[p]:
            if r in visited:
                continue
            visited.add(r)
            if r not in assigned or augment(assigned[r], visited):
                assigned[r] = p
                return True
        return False

    for p in range(len(predictions)):
        augment(p, set())
    pairs = [{'prediction_index': p, 'reference_index': r,
              'prediction_label': predictions[p]['label'], 'reference_label': references[r]['label'],
              'reference_id': references[r]['id'],
              'prediction_bbox_xyxy': predictions[p]['bbox_xyxy'],
              'reference_bbox_xyxy': references[r]['bbox_xyxy'],
              'iou': iou(predictions[p]['bbox_xyxy'], references[r]['bbox_xyxy'])}
             for r,p in sorted(assigned.items())]
    return {'pairs': pairs, 'matched': len(pairs), 'eligible_reference_regions': len(refs),
            'excluded_reference_regions': len(references)-len(refs), 'prediction_regions': len(predictions),
            'mapped_prediction_regions': sum(normalize(p['label']) in aliases for p in predictions),
            'unmatched_prediction_indices': sorted(set(range(len(predictions)))-set(assigned.values())),
            'unmatched_eligible_reference_indices': sorted(set(i for i,_ in refs)-set(assigned))}
