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
from .gpu_runtime import environment, begin_measure, end_measure, validate_boxes
from .image_dataset import load_images
from .thud_experiment import immutable_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/thud-extended'
REPORT = ROOT / 'reports/THUD_VMESNO'
DATASETS = ('thud_dense_test50', 'thud_agtea_test50')


def progress(message):
    timestamp = datetime.now(ZoneInfo('Europe/Ljubljana')).isoformat(timespec='seconds')
    print(timestamp, message, flush=True)
    target = Path(os.environ.get('THUD_PROGRESS_FILE', str(REPORT / 'NAPREDEK.md')))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a') as stream:
        stream.write(f'\n- {timestamp}: {message}\n')


def native_selection(path=None):
    """Optional reference-free selection; old experiments keep their defaults."""
    if path is None:
        return prepare()
    selected = read_json(path)
    allowed = {'dataset','index','frame_id','image_path','image_sha256',
               'image_size','subset20','source_frame_id'}
    if not isinstance(selected,list) or not selected:
        raise ValueError('Empty native selection')
    for row in selected:
        if set(row)-allowed or not allowed-{'source_frame_id'} <= set(row):
            raise ValueError('Unexpected fields / annotation leak in native selection')
        if sha256(Path(row['image_path']).read_bytes()) != row['image_sha256']:
            raise ValueError('Selected RGB changed')
    if len({r['frame_id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate selection frame ID')
    return selected


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
