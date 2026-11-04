"""Skupne meritve CUDA in preverjanje izvornih okvirjev; brez vezave na nabor."""
from collections import Counter
import time

def environment(torch, model):
    """Zapiše dejanske različice in zavrne parametre zunaj GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback forbidden')
    devices = Counter(str(p.device) for p in model.parameters())
    if any(not d.startswith('cuda') for d in devices):
        raise RuntimeError('Parameters outside GPU: ' + str(devices))
    return {'python': __import__('sys').version, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'transformers': __import__('transformers').__version__,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_capability': list(torch.cuda.get_device_capability(0)),
            'compiled_arches': torch.cuda.get_arch_list(),
            'parameter_devices': dict(devices),
            'parameter_dtypes': dict(Counter(str(p.dtype) for p in model.parameters())),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'gpu_verified': True}

def begin_measure(torch):
    """Sinhronizira CUDA pred merjenjem in ponastavi vrh porabe VRAM."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()

def end_measure(torch, start):
    """Po koncu CUDA dela vrne dejanski čas in vrh porabe pomnilnika."""
    torch.cuda.synchronize()
    return {'elapsed_seconds': time.perf_counter() - start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'gpu_verified': True}

def validate_boxes(parsed, task, size):
    """Zavrne neveljavne okvirje; koordinat ne popravlja ali omejuje na sliko."""
    value = parsed.get(task)
    if task == '<MORE_DETAILED_CAPTION>':
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Empty detailed caption')
        return
    if not isinstance(value, dict) or set(value) != {'bboxes', 'labels'}:
        raise ValueError('Missing labels/bboxes')
    if len(value['bboxes']) != len(value['labels']):
        raise ValueError('Unpaired labels/bboxes')
    import math
    for label, box in zip(value['labels'], value['bboxes']):
        if not isinstance(label, str) or not label.strip():
            raise ValueError('Empty detection label')
        if len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
            raise ValueError('Malformed bbox')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            raise ValueError('Out-of-image or inverted bbox: ' + str(box))
