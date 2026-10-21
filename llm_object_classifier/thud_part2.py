"""Part two: reuse part-one inputs, enrich observations, freeze and classify text.

Uses the existing two stages, with separate local outputs and immutable configs.
No part-one inference is repeated. Schema failures are retained as observations.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import os
import subprocess
import time
from types import SimpleNamespace
from . import stage1_describe, stage2_reason
from .experiment import load_config, read_json, sha256, write_json
from .image_dataset import load_images
from .thud_experiment import ROOT, REPORT, DATASETS, MODELS, OUT as PART1, immutable_json
from .thud_metrics import aggregate
from .vision_backends import Backend, request

OUT = ROOT / 'results/thud-part2'


def progress(message):
    print(message, flush=True)
    with (REPORT / 'NAPREDEK.md').open('a') as stream:
        stream.write('\n- ' + datetime.now(ZoneInfo('Europe/Ljubljana')).isoformat(timespec='seconds') + ': ' + message + '\n')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = read_json(REPORT / 'data-audit.json')
    for ds in DATASETS:
        for subset in ('all', 'pilot', 'subset'):
            source = PART1 / 'inputs' / f'{ds}-discovery-{subset}.jsonl'
            dest = OUT / 'inputs' / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.read_bytes() != source.read_bytes():
                raise ValueError('Changed discovery input: ' + str(dest))
            if not dest.exists():
                dest.write_bytes(source.read_bytes())
        frames = load_images(OUT / 'inputs' / f'{ds}-discovery-all.jsonl', 50)
        if len(frames) != 50 or [f['image_sha256'] for f in frames] != [f['image_sha256'] for f in audit[ds]['frames']]:
            raise ValueError('Part-one selection changed')
        if any(set(f['record']) != {'frame_id', 'image_path', 'image_sha256', 'image_size'} for f in frames):
            raise ValueError('Discovery input contains annotation metadata')
    immutable_json(OUT / 'plan.json', {
        'part1_commit': '7509607', 'stage1_measured_calls': 300, 'pilot_measured_calls': 12,
        'stage2_new_positions': 100, 'stage2_reused_positions': 20,
        'stage1_temperature': 0, 'stage1_seed': 42, 'stage2_temperature': .3, 'stage2_seed': 42,
        'stage2_model': MODELS[0]['model'], 'stage2_prompt': 'thud-short-no-examples.txt',
        'subset': {ds: audit[ds]['subset_indices_zero_based'] for ds in DATASETS},
        'nvila': 'Not running: retain part-one diagnostic; no second setup attempt or inference claimed.',
        'comparison': 'Old detailed T0 for all models; old short T0 BF16 on same20 separately. No causal schema-only improvement claim.',
        'boxes': 'No predicted boxes; image-relative location text is not precise localization.',
    })
    index = OUT / 'part1-results-index.json'
    if not index.exists():
        immutable_json(index, {str(p.relative_to(PART1)): sha256(p.read_bytes())
                              for p in sorted(PART1.glob('*/*/t*/frame-*.json'))})


def reuse_pilot(dest, config):
    """Reuse identical pilot calls, including errors; do not measure them twice."""
    pilot = dest.with_name(dest.name.replace('rich-', 'pilot-', 1))
    if not (pilot/'complete.json').exists():
        return
    manifest = read_json(pilot/'manifest.json')
    if manifest['config'] != config:
        raise ValueError('Cannot reuse pilot with changed settings/prompt')
    copied = {}
    for source in sorted(pilot.glob('*/t*/frame-*.json')) + sorted(pilot.glob('*/warmup.json')):
        relative = source.relative_to(pilot)
        target = dest/relative
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Conflicting existing pilot reuse')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        copied[str(relative)] = sha256(source.read_bytes())
    immutable_json(dest/'reused-pilot.json', {'source':str(pilot), 'sha256':copied,
        'note':'These are identical existing calls, including rejected outputs and warmup; not newly measured.'})


def configuration(spec, url, stage=1):
    raw = read_json(ROOT / 'configs/stage1/thud-part2.json')
    raw['models'] = [{**spec, 'base_url': url, 'require_gpu': True}]
    if stage == 1:
        raw['description_schema'] = 'rich-v2'
        raw['system_file'] = str(ROOT / 'configs/prompts/thud-description-rich.txt')
    else:
        raw.pop('description_schema', None)
        raw['temperatures'] = [.3]
        raw['system_file'] = str(ROOT / 'configs/prompts/thud-short-no-examples.txt')
        raw['instruction_file'] = str(ROOT / 'configs/prompts/reasoning.txt')
    return raw


def job_records(path):
    return [read_json(p) for p in sorted(path.glob('*/t*/frame-*.json'))]


def execute(name, raw, stage, source, count):
    config_path = OUT / 'configs' / (name + '.json')
    immutable_json(config_path, raw)
    config = load_config(config_path, stage)
    dest = OUT / name
    done = dest / 'complete.json'
    if (dest / 'manifest.json').exists():
        old = read_json(dest / 'manifest.json')
        if old['config'] != config:
            raise ValueError('Completed/resumed job configuration changed: ' + name)
        if stage == 1 and old['input_sha256'] != sha256(source.read_bytes()):
            raise ValueError('Description input changed: ' + name)
        if stage == 2:
            current = [{'path': str(p), 'sha256': sha256(p.read_bytes())} for p in sorted(source.glob('frame-*.json'))]
            if current != old['source_captions']:
                raise ValueError('Frozen captions changed: ' + name)
    if done.exists():
        if len(job_records(dest)) != count:
            raise ValueError('Completed job has unexpected record count: ' + name)
        return dest
    if stage == 1 and name.startswith('rich-'):
        reuse_pilot(dest, config)
    progress('2. del: zagon ' + name)
    start = time.perf_counter()
    args = SimpleNamespace(output_dir=dest, resume=True, warmup=True, limit=count, frame_id=None, input=source, captions=source)
    code = (stage1_describe if stage == 1 else stage2_reason).run(args, config)
    Backend(config['models'][0], config['timeout']).unload()
    if code == 1:
        raise RuntimeError('GPU/preparation failure; partial outputs preserved, experiment stopped')
    if len(job_records(dest)) != count:
        raise RuntimeError('Incomplete job, no completion marker written')
    write_json(done, {'exit_code': code, 'end_to_end_seconds': time.perf_counter()-start,
                      'completed_at': datetime.now(ZoneInfo('Europe/Ljubljana')).isoformat(timespec='seconds')})
    progress(f'2. del: končano {name}, izhod {code}, {time.perf_counter()-start:.1f} s')
    return dest


def freeze(ds, model, kind):
    audit = read_json(REPORT / 'data-audit.json')[ds]
    source = (OUT / f'rich-{model}-{ds}' if kind == 'rich' else PART1 / f'baseline-{model}-discovery-{ds}') / model / 't0-s42'
    dest = OUT / 'frozen' / kind / model / ds
    for i in audit['subset_indices_zero_based']:
        p = source / f'frame-{i+1:06d}.json'
        # Rejected source rows stay rejected; stage2 records an explicit skipped slot.
        immutable_json(dest / p.name, read_json(p))
    return dest


def snapshot(url, phase):
    files = [p for folder in ('llm_object_classifier', 'configs', 'scripts')
             for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.txt', '.json', '.sh'}]
    write_json(OUT/'source-snapshots'/f'{phase}-{time.time_ns()}.json', {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], text=True),
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in sorted(files)},
        'ollama_version': request(url + '/api/version'), 'tags': request(url + '/api/tags'),
        'phase': phase, 'timestamp': datetime.now(ZoneInfo('Europe/Ljubljana')).isoformat(timespec='seconds'),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'pilot', 'describe', 'reason', 'all'], default='all')
    parser.add_argument('--base-url', default=os.environ.get('THUD_OLLAMA_URL', 'http://172.18.0.2:11434'))
    args = parser.parse_args()
    prepare()
    if args.phase == 'prepare':
        return
    snapshot(args.base_url, args.phase)
    if args.phase in {'pilot', 'all'}:
        for spec in MODELS:
            for ds in DATASETS:
                execute(f'pilot-{spec["name"]}-{ds}', configuration(spec, args.base_url), 1,
                        OUT/'inputs'/f'{ds}-discovery-pilot.jsonl', 2)
        rows = [r for path in sorted(OUT.glob('pilot-*')) if path.is_dir() for r in job_records(path)]
        a = aggregate(rows)
        write_json(OUT / 'pilot-estimate.json', {'pilot': a, 'expected_full_stage1_calls': 300,
            'estimated_stage1_seconds': a['mean_s']*300 if a['mean_s'] is not None else None,
            'note': 'Linear pre-run estimate; output lengths and shared GPU may change duration.'})
        progress('2. del: pilot shranjen, ocena osnovnih opisov ' + str(round((a['mean_s'] or 0)*300/60, 1)) + ' min')
    if args.phase in {'describe', 'all'}:
        for spec in MODELS:
            for ds in DATASETS:
                execute(f'rich-{spec["name"]}-{ds}', configuration(spec, args.base_url), 1,
                        OUT/'inputs'/f'{ds}-discovery-all.jsonl', 50)
    if args.phase in {'reason', 'all'}:
        for spec in MODELS:
            for ds in DATASETS:
                kinds = ['rich'] + (['legacy'] if spec['name'] != MODELS[0]['name'] else [])
                for kind in kinds:
                    frozen = freeze(ds, spec['name'], kind)
                    execute(f'reason-{kind}-{spec["name"]}-{ds}', configuration(MODELS[0], args.base_url, 2), 2, frozen, 10)
        reused = {}
        for ds in DATASETS:
            path = PART1/f'reason-short-no-examples-t0.3-s42-{ds}'
            reused[ds] = {'path': str(path), 'records': 10,
                          'sha256': {str(p.relative_to(path)): sha256(p.read_bytes()) for p in sorted(path.glob('*/t*/frame-*.json'))}}
        immutable_json(OUT/'reused-stage2.json', reused)
    progress('2. del: zahtevana faza ' + args.phase + ' zaključena; surovi odgovori so v results/thud-part2/')


if __name__ == '__main__':
    main()
