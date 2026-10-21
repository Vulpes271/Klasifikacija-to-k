"""Shared files/measurement helpers for the two isolated native GPU runners.

Only the reference-free RGB manifests enter inference. Dataset annotations are
read later by the offline evaluator. Existing outputs (including errors) resume
without another inference request; changed settings or images are rejected.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import time
from zoneinfo import ZoneInfo

from .experiment import read_json, sha256, write_json
from .image_dataset import load_images
from .thud_experiment import immutable_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/thud-extended'
REPORT = ROOT / 'reports/THUD_VMESNO'
DATASETS = ('thud_dense_test50', 'thud_agtea_test50')


def progress(message):
    timestamp = datetime.now(ZoneInfo('Europe/Ljubljana')).isoformat(timespec='seconds')
    print(timestamp, message, flush=True)
    with (REPORT / 'NAPREDEK.md').open('a') as stream:
        stream.write(f'\n- {timestamp}: {message}\n')


def frames():
    selected = []
    audit = read_json(REPORT / 'data-audit.json')
    for dataset in DATASETS:
        source = ROOT / 'results/thud-part2/inputs' / f'{dataset}-discovery-all.jsonl'
        images = load_images(source, 50)
        assert len(images) == 50
        assert [x['image_sha256'] for x in images] == [x['image_sha256'] for x in audit[dataset]['frames']]
        for index, image in enumerate(images):
            if set(image['record']) != {'frame_id', 'image_path', 'image_sha256', 'image_size'} or image['regions'] is not None:
                raise ValueError('Annotation leak in discovery manifest')
            selected.append({'dataset': dataset, 'index': index,
                             'frame_id': image['record']['frame_id'],
                             'image_path': image['path'], 'image_sha256': image['image_sha256'],
                             'image_size': image['image_size'],
                             'subset20': index in audit[dataset]['subset_indices_zero_based']})
    return selected


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = frames()
    immutable_json(OUT / 'inputs.json', selected)
    index = OUT / 'previous-results-index.json'
    if not index.exists():
        paths = sorted(p for folder in ('thud-vmesno', 'thud-part2')
                       for p in (ROOT / 'results' / folder).glob('*/*/t*/frame-*.json'))
        immutable_json(index, {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in paths})
    return selected


def environment(torch, model):
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
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def end_measure(torch, start):
    torch.cuda.synchronize()
    return {'elapsed_seconds': time.perf_counter() - start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'gpu_verified': True}


def snapshot(destination, config, selected):
    # Snapshot the inference implementation, not just an eventual commit SHA.
    immutable_json(destination / 'manifest.json', {'config': config, 'frames': selected})
    stamp = str(time.time_ns())
    files = [p for folder in ('llm_object_classifier', 'scripts', 'configs', 'docker')
             for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.sh', '.txt', '.json', '.Dockerfile'}]
    commit = os.environ.get('THUD_SOURCE_COMMIT')
    if not commit:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write_json(destination / 'source-snapshots' / (stamp + '.json'), {
        'git_commit': commit,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in files},
        'timestamp': datetime.now(ZoneInfo('Europe/Ljubljana')).isoformat()})


def saved_path(destination, frame, task):
    return destination / frame['dataset'] / task / f"frame-{frame['index'] + 1:06d}.json"


def check_saved(path, frame):
    if not path.exists():
        return False
    row = read_json(path)
    if row['image_sha256'] != frame['image_sha256']:
        raise ValueError('Changed saved input: ' + str(path))
    return True


def validate_boxes(parsed, task, size):
    """Reject malformed/out-of-image boxes; never silently clamp predictions."""
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


def draw_boxes(image_path, parsed, task, output):
    from PIL import Image, ImageDraw
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for i, (box, label) in enumerate(zip(parsed[task]['bboxes'], parsed[task]['labels']), 1):
        color = ('#ff3232', '#00c3ff', '#ffe600', '#48ff80', '#ff70ff')[i % 5]
        draw.rectangle(box, outline=color, width=3)
        caption = f'{i}: {label}'
        x, y = box[0], max(0, box[1] - 13)
        draw.text((x, y), caption, fill=color, stroke_width=1, stroke_fill='black')
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
