"""Portable stage-one comparison only. Never runs inference or reads annotations.

The export copies original response bytes, records their provenance, and derives
literal display views and repeat differences using the existing report helpers.
Malformed model answers remain malformed; format validity is not object accuracy.
"""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import math
import os
import re
from pathlib import Path
import shutil
import statistics

from .experiment import read_json, write_json, sha256
from .thud_extended_common import ROOT, native_selection, saved_path, draw_boxes
from .thud_extended_report import repetition_view, repetition_consistency
from .thud_florence import TASKS
from .thud_qwen3_vl import OUT, MODEL, REVISION, SEEDS

TARGET = ROOT / 'analiza rezultatov vmesna predstavitev'
OLD = ROOT / 'results/ponovitve/stage1'
MODELS = {'florence': 'microsoft/Florence-2-large',
          'nvila': 'Efficient-Large-Model/NVILA-8B', 'qwen3-vl': MODEL}
REVISIONS = {'florence': '21a599d414c4d928c9032694c424fb94458e3594',
             'nvila': '049743cc51bdd8872cdf696a1a58b03fef2c367e', 'qwen3-vl': REVISION}


def leaf_name(frame):
    return frame['dataset'] + '--' + frame['source_frame_id']


def csv_file(path, rows, columns=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def portable_frame(frame):
    return {**frame, 'image_path': str(Path(frame['image_path']).relative_to(ROOT)),
            'published_original': leaf_name(frame) + '/original.png'}


def run_root(model, repeat, roots=None):
    """Explicit alternate roots; legacy THUD commands retain their old paths."""
    return ((roots[model] if roots else (OUT / 'qwen3-vl' if model == 'qwen3-vl' else OLD / model)) / repeat)


def audit_protocol(selected, roots=None):
    """Fail closed if a reused run changed seeds, prompts, revisions or RGB bytes."""
    configs = {}
    for model in MODELS:
        configs[model] = {}
        for repeat, seed in SEEDS.items():
            root = run_root(model, repeat, roots)
            manifest = read_json(root / 'manifest.json')
            if manifest['frames'] != selected:
                raise ValueError(f'{model}/{repeat}: changed image selection')
            config = manifest['config']
            assert config['model'] == MODELS[model] and config['revision'] == REVISIONS[model]
            assert config['seed'] == seed and config['repeat'] == repeat
            if model == 'florence':
                assert config['tasks'] == TASKS and not config['do_sample']
                assert config['num_beams'] == 3 and config['max_new_tokens'] == 1024
            else:
                assert config['do_sample'] and config['temperature'] == .3
                assert config['top_p'] == .8 and config['top_k'] == 20
                assert config['repetition_penalty'] == 1.05 and config['num_beams'] == 1
                assert config['max_new_tokens'] == 4096
            configs[model][repeat] = config
        normalized = [{k: v for k, v in c.items() if k not in ('repeat', 'seed')}
                      for c in configs[model].values()]
        assert normalized[0] == normalized[1] == normalized[2], f'{model}: settings changed'
    # The added Qwen connection uses exactly the existing NVILA instruction.
    assert configs['nvila']['r1']['system_prompt'] == configs['qwen3-vl']['r1']['instruction']
    return configs


def audit_nvila_initialization():
    """Keep the corrected same-weight runs; never import the old seed-43 load."""
    expected = {'input': '72a9c9d364f93bf4dfd74f43be8ea47ee48b1beda30c27b6ce1a2ab79205c20a',
                'output': 'dc01da616896fdf214e9732855aff5bb803c175103a122bca503748d209e919a'}
    for repeat in ('r2', 'r3'):
        paths = list((OLD / 'nvila' / repeat / 'environments').glob('*.json'))
        assert paths, 'Missing initialization audit'
        for path in paths:
            env = read_json(path)
            assert env['initialization_seed'] == 42 and env['embedding_tail_sha256'] == expected
    return {'r2_r3_initialization_seed': 42, 'r2_r3_embedding_tail_sha256': expected,
            'r1': 'Original r1 loader used args.seed=42 before loading. No embedding fingerprint was captured in r1.',
            'r1_evidence': 'provenance/nvila/r1/source-snapshots and prior published pinned source; no claim of a retrospectively measured fingerprint.',
            'excluded_attempts': 'results/ponovitve/diagnostics/nvila-init-seed-*; not imported into this package.'}


def native_prompts(row, model):
    if model == 'nvila':
        user = '<image>' + row['system_prompt'] + '\n\n' + row['user_prompt']
        chat = '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n' + user + '<|im_end|>\n<|im_start|>assistant\n'
        assert row.get('native_chat_prompt_text', chat) == chat
        return {'native_system_prompt': 'You are a helpful assistant', 'native_user_prompt': user,
                'native_chat_prompt_text': chat,
                'provenance': 'runtime capture' if 'native_chat_prompt_text' in row else
                'reconstructed from pinned VILA AUTO tokenizer template and extract_media; original components remain in raw.json'}
    if model == 'florence':
        # Read the checkpoint's own expansion, rather than invent a text prompt.
        import ast
        processor_path = ROOT / 'results/thud-extended/models/florence/processing_florence2.py'
        tree = ast.parse(processor_path.read_text())
        expansion = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                         if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and
                            t.attr == 'task_prompts_without_inputs' for t in n.targets))
        return {'native_system_prompt': row.get('system_prompt'), 'native_user_prompt': row['prompt'],
                'provenance': 'runtime task token plus expansion reconstructed from the pinned checkpoint processor',
                'processor_prompt': expansion[row['prompt']],
                'processor_source_sha256': sha256(processor_path.read_bytes())}
    return {k: row[k] for k in ('system_prompt', 'user_prompt', 'native_chat_prompt_text',
                               'instruction', 'generic_input_prompt')}


def copy_provenance(configs, selected):
    provenance = TARGET / 'provenance'
    for model in MODELS:
        for repeat in SEEDS:
            source = (OUT / 'qwen3-vl' if model == 'qwen3-vl' else OLD / model) / repeat
            destination = provenance / model / repeat
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / 'manifest.json', {'config': configs[model][repeat],
                       'frames': [portable_frame(f) for f in selected],
                       'original_manifest_sha256': sha256((source / 'manifest.json').read_bytes()),
                       'original_manifest_path': str((source / 'manifest.json').relative_to(ROOT)),
                       'reused_existing_generation': model != 'qwen3-vl'})
            for folder in ('environments', 'source-snapshots'):
                if (source / folder).exists():
                    shutil.copytree(source / folder, destination / folder, dirs_exist_ok=True)
            # Warmups are explicitly outside the 180 logical comparison positions.
            warmups = list((source / 'warmups').glob('*.json'))
            if warmups:
                write_json(destination / 'warmup-index.json', [
                    {'path': str(p.relative_to(ROOT)), 'sha256': sha256(p.read_bytes()),
                     'call_id': read_json(p).get('call_id'), 'excluded_from_quality': True,
                     'elapsed_seconds': read_json(p).get('elapsed_seconds')} for p in warmups])
    for filename in ('environment.json', 'setup/qwen-api.json'):
        path = OUT / filename
        if path.exists():
            dest = provenance / 'qwen3-vl' / Path(filename).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    for name in ('setup/nvila-r3-segfault.json', 'setup/nvila-initialization-correction.json'):
        path = ROOT / 'results/ponovitve' / name
        if path.exists(): shutil.copyfile(path, provenance / Path(name).name)
    # Preserve the failed API warmup separately from the actual 60 new answers.
    for path in (OUT / 'warmups').glob('*.json'):
        destination = provenance / 'qwen3-vl' / 'warmups' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for path in (OUT / 'setup').glob('*'):
        if path.name in {'docker-images.txt', 'nvidia-smi.txt', 'weights-provenance.json', 'nvila-fallback-log-audit.json'}:
            if path.name == 'nvidia-smi.txt':
                write_json(provenance / 'nvidia-smi.json', {'source_sha256': sha256(path.read_bytes()),
                                                          'stdout': path.read_text()})
                (provenance / path.name).unlink(missing_ok=True)
            else:
                shutil.copyfile(path, provenance / path.name)
    processor = ROOT / 'results/thud-extended/models/florence/processing_florence2.py'
    # Preserve the unmodified upstream source as JSON text, including its
    # whitespace/license, without introducing source-format warnings in Git.
    write_json(provenance / 'florence/processing_florence2.source.json',
               {'source_sha256': sha256(processor.read_bytes()), 'source_utf8': processor.read_text()})
    (provenance / 'florence/processing_florence2.py').unlink(missing_ok=True)
    # Tiny checkpoint configs document preprocessing without publishing weights.
    for name in ('config.json', 'generation_config.json', 'preprocessor_config.json',
                 'video_preprocessor_config.json', 'chat_template.json'):
        path = OUT / 'models/qwen3-vl' / name
        if path.exists(): shutil.copyfile(path, provenance / 'qwen3-vl' / name)
    for name, destination in [('progress.md', 'qwen3-vl/generation-progress.md'),
                              ('gpu-samples.jsonl', 'gpu-samples-qwen.jsonl'),
                              ('logs/tests.log', 'tests.txt')]:
        path = OUT / name
        if path.exists(): shutil.copyfile(path, provenance / destination)


def collect(selected, configs, *, target=None, roots=None, reused=None, copy_original=True):
    target = TARGET if target is None else Path(target)
    reused = {'florence', 'nvila'} if reused is None else set(reused)
    records, sources, combined = [], [], []
    for frame in selected:
        image_root = target / leaf_name(frame)
        image_root.mkdir(parents=True, exist_ok=True)
        if copy_original:
            shutil.copyfile(frame['image_path'], image_root / 'original.png')
            assert sha256((image_root / 'original.png').read_bytes()) == frame['image_sha256']
        for model in MODELS:
            for repeat, seed in SEEDS.items():
                root = run_root(model, repeat, roots)
                destination = image_root / model / repeat
                destination.mkdir(parents=True, exist_ok=True)
                tasks = TASKS if model == 'florence' else {'description': None}
                for task in tasks:
                    path = saved_path(root, frame, task if model != 'nvila' else 'rich')
                    row = read_json(path)
                    assert row['frame_id'] == frame['frame_id'] and row['image_sha256'] == frame['image_sha256']
                    assert row['model'] == MODELS[model] and row['seed'] == seed and row['repeat'] == repeat
                    assert row['gpu_verified'] and row['call_id'] and row['finished_at_utc']
                    if model != 'florence':
                        expected = configs['nvila'][repeat]['system_prompt']
                        assert row.get('instruction', row.get('system_prompt')) == expected
                        gen = row['generation_config']
                        for key in ('do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'num_beams', 'max_new_tokens'):
                            assert gen[key] == configs[model][repeat][key], (model, repeat, key)
                    else:
                        assert row['prompt'] == TASKS[task]
                        for key in ('do_sample', 'num_beams', 'max_new_tokens', 'seed'):
                            assert row['generation'][key] == configs[model][repeat][key]
                    relative = destination.relative_to(target)
                    # Exact source bytes, including invalid JSON text and complete tracebacks.
                    raw_path = destination / (task + '.raw.json')
                    shutil.copyfile(path, raw_path)
                    raw_sha = sha256(path.read_bytes())
                    assert sha256(raw_path.read_bytes()) == raw_sha
                    source = {'frame_id': frame['frame_id'], 'model': model, 'repeat': repeat, 'task': task,
                              'source_path': str(path.relative_to(ROOT)), 'source_sha256': raw_sha,
                              'published_path': str(raw_path.relative_to(target)), 'call_id': row['call_id'],
                              'finished_at_utc': row['finished_at_utc'], 'reused_existing_generation': model in reused}
                    sources.append(source)
                    row.update(model_name=model, task=task, original_record_sha256=raw_sha)
                    row['view'] = repetition_view(row)
                    metadata = {
                        **source, 'model_id': MODELS[model], 'model_revision': REVISIONS[model],
                        'settings': configs[model][repeat],
                        **{k: row[k] for k in ('status', 'syntax_valid', 'schema_valid', 'error', 'elapsed_seconds',
                            'output_tokens', 'peak_allocated_bytes', 'peak_reserved_bytes', 'gpu_verified') if k in row},
                        'quality_accuracy': None, 'quality_note': 'No human reference assessment; format validity is not detection correctness.'}
                    write_json(destination / (task + '.result.json'), {
                        'view': row['view'], 'prompts': native_prompts(row, model), 'metadata': metadata,
                        'raw_response': task + '.raw.json',
                        'note': 'Literal extraction only; unavailable parse does not mean an empty scene. Full raw answer is in raw_response, including invalid JSON.'})
                    if model == 'florence' and task in ('od', 'dense') and row['status'] == 'ok':
                        draw_boxes(frame['image_path'], row['parsed'], TASKS[task], destination / (task + '.predictions.jpg'))
                    row['published_directory'] = str(relative)
                    records.append(row)
                if model == 'florence':
                    path = saved_path(root, frame, 'combined')
                    row = read_json(path)
                    assert row['image_sha256'] == frame['image_sha256'] and row['complete_three_task_measurement']
                    shutil.copyfile(path, destination / 'combined.raw.json')
                    combined.append({**row, 'model_name': model, 'task': 'combined', 'repeat': repeat})
    return records, sources, combined


def summarize(records, combined, reused=None):
    reused = {'florence', 'nvila'} if reused is None else set(reused)
    groups = defaultdict(list)
    for row in records + combined:
        groups[(row['model_name'], row['task'])].append(row)
    summary = []
    for (model, task), rows in groups.items():
        times = sorted(r['elapsed_seconds'] for r in rows)
        summary.append({'model': model, 'task': task, 'n': len(rows),
            'answers_with_content': sum(bool(r.get('content')) for r in rows) if task != 'combined' else None,
            'schema_or_native_parse_ok': sum(r.get('status') == 'ok' for r in rows) if task != 'combined' else None,
            'errors': sum(r.get('status') == 'error' for r in rows),
            'object_list_available': sum(r.get('view', {}).get('object_list_available', False) for r in rows) if task != 'combined' else None,
            'mean_seconds': statistics.mean(times), 'median_seconds': statistics.median(times),
            'p90_seconds_nearest_rank': times[math.ceil(.9 * len(times))-1],
            'sum_seconds': sum(times),
            'max_allocated_GiB': max(r['peak_allocated_bytes'] for r in rows) / 2**30,
            'max_reserved_GiB': max(r['peak_reserved_bytes'] for r in rows) / 2**30,
            'precision': 'BF16' if model == 'qwen3-vl' else 'FP16',
            'reused': model in reused, 'detection_accuracy': None})
    return summary


def escaped(value):
    return html.escape(str(value))


def pre(value):
    content=escaped(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)).replace('\r','&#13;')
    # HTML entities preserve literal model whitespace without source-format noise.
    content=re.sub(r'(?m)[ \t]+$',lambda m:''.join('&#'+str(ord(c))+';' for c in m[0]),content)
    return '<pre>' + content + '</pre>'


STYLE = '''body{font:16px system-ui;margin:24px;background:#f4f5f8;color:#17202a}a{color:#1459a1}img{max-width:100%;height:auto}.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.model,article{background:white;padding:14px;border:1px solid #ccd3dd;border-radius:7px;margin-bottom:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}summary{cursor:pointer;font-weight:600;padding:6px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd3dd;padding:7px;text-align:left}nav a{display:inline-block;margin:5px}.note{padding:12px;background:#fff0cf}h1{font-size:1.6em}h2{font-size:1.15em}@media(max-width:1100px){.columns{grid-template-columns:1fr}}'''


def document(title, body):
    return '<!doctype html><html lang="sl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + escaped(title) + '</title><style>' + STYLE + '</style><body>' + body + '</body></html>'


def render(selected, records, summary, *, target=None, reused=None, introduction=None):
    target = TARGET if target is None else Path(target)
    reused = {'florence', 'nvila'} if reused is None else set(reused)
    groups = defaultdict(list)
    for row in records:
        groups[(row['frame_id'], row['model_name'], row['repeat'])].append(row)
    intro = '<h1>Prvi korak: opis slike, objekti in lokacije</h1><p>20 istih slik · 3 VLM · 3 ponovitve. Brez referenčnih oznak na vhodu. Brez drugega koraka in PDF.</p>'
    intro += '<p class="note">Enakost odgovorov in veljaven zapis nista dokaz zaznavne pravilnosti. Florence vrne lastne okvirje; NVILA in Qwen pri tem protokolu nimata okvirjev. Referenčnih okvirjev ta paket ne izrisuje. Stari časi Florence/NVILA so ponovno uporabljene meritve z drugega dne, ne nove sočasne meritve.</p>'
    if introduction is not None: intro = introduction
    intro += '<nav><a href="README.md">Protokol in omejitve</a><a href="summary.csv">Čas in VRAM</a><a href="response-metrics.csv">Posamezne meritve</a><a href="repeat-differences.csv">Razlike ponovitev</a><a href="manual-review-empty.csv">Prazen ročni pregled</a><a href="verification.json">Popolnost</a><a href="provenance.json">Izvor vseh odgovorov</a></nav>'
    intro += '<table><tr><th>Model / naloga</th><th>N</th><th>Mediana s</th><th>VRAM allocated GiB</th><th>Napake zapisa</th></tr>'
    for row in summary:
        intro += f'<tr><td>{escaped(row["model"])} / {escaped(row["task"])}</td><td>{row["n"]}</td><td>{row["median_seconds"]:.3f}</td><td>{row["max_allocated_GiB"]:.2f}</td><td>{row["errors"]}</td></tr>'
    intro += '</table><h2>Izberite sliko</h2><nav>'
    for frame in selected:
        name = leaf_name(frame)
        intro += f'<a href="{name}/index.html">{escaped(frame["frame_id"])}</a>'
        body = '<p><a href="../index.html">← Vse slike</a></p><h1>' + escaped(frame['frame_id']) + '</h1>'
        original = os.path.relpath(target / frame['published_original'], target / name) if 'published_original' in frame else 'original.png'
        body += f'<a href="{original}"><img src="{original}" alt="Originalna slika brez oznak"></a><p>Original, brez referenčnih okvirjev. SHA256: ' + frame['image_sha256'] + '</p><div class="columns">'
        for model, model_id in MODELS.items():
            body += '<section class="model"><h2>' + escaped(model_id) + '</h2><p>' + ('Ponovna uporaba izvornih GPU klicev 26. 10. 2026.' if model in reused else 'Nove GPU generacije za ta paket.') + '</p>'
            if model != 'florence': body += '<p>Model ni vrnil okvirjev; lokacije in odnosi so v besedilu spodaj.</p>'
            for repeat in SEEDS:
                body += '<article><h2>' + repeat + ' · seed ' + str(SEEDS[repeat]) + '</h2>'
                for row in groups[(frame['frame_id'], model, repeat)]:
                    task = row['task']; link = model + '/' + repeat + '/' + task
                    body += '<h3>' + escaped(TASKS.get(task, task)) + '</h3><p>' + f'{row["elapsed_seconds"]:.3f} s · allocated {row["peak_allocated_bytes"]/2**30:.2f} GiB · reserved {row["peak_reserved_bytes"]/2**30:.2f} GiB · status {escaped(row["status"])}' + '</p>'
                    if row.get('error'): body += '<p class="note">Napaka zapisa ali izvajanja: ' + escaped(row['error']) + '</p>'
                    if model == 'florence' and task in ('od', 'dense') and row['status'] == 'ok':
                        body += f'<a href="{link}.predictions.jpg"><img src="{link}.predictions.jpg" alt="Florence lastne napovedi {task} {repeat}"></a>'
                    body += '<p>Izluščenje: ' + escaped(row['view']['extraction']) + '</p>'
                    if row['view'].get('scene_description'): body += '<details open><summary>Celoten opis</summary>' + pre(row['view']['scene_description']) + '</details>'
                    body += '<details open><summary>Celoten seznam objektov</summary>' + pre({'available': row['view'].get('object_list_available', False), 'objects': row['view']['objects']}) + '</details>'
                    body += '<details><summary>Celoten surovi odgovor</summary>' + pre(row.get('content', '')) + '</details>'
                    body += '<details><summary>Dejanski prompti</summary>' + pre(native_prompts(row, model)) + '</details>'
                    body += f'<p><a href="{link}.raw.json">Celoten izvorni odgovor in tokeni</a> · <a href="{link}.result.json">Celoten opis, objekti, prompti in meritve</a></p>'
                if model == 'florence': body += f'<p><a href="{model}/{repeat}/combined.raw.json">Ločena meritev skupnega postopka</a></p>'
                body += '</article>'
            body += '</section>'
        body += '</div>'
        (target / name / 'index.html').write_text(document(frame['frame_id'], body), encoding='utf-8')
    (target / 'index.html').write_text(document('Primerjava prvega koraka', intro + '</nav>'), encoding='utf-8')


def verify(selected, records, sources, differences):
    expected = {(f['frame_id'], m, r, t) for f in selected for m in MODELS for r in SEEDS
                for t in (TASKS if m == 'florence' else ['description'])}
    actual = {(r['frame_id'], r['model_name'], r['repeat'], r['task']) for r in records}
    assert actual == expected and len(records) == len(expected) == 300
    assert len({r['call_id'] for r in records}) == 300, 'Copied generation ID'
    assert len({(r['frame_id'], r['model_name'], r['repeat']) for r in records}) == 180
    assert Counter(f['dataset'] for f in selected) == {'thud_dense_test50': 10, 'thud_agtea_test50': 10}
    assert len(differences) == 300
    assert len(list(TARGET.glob('*--*/florence/*/*.predictions.jpg'))) == 120
    for model in ('nvila', 'qwen3-vl'):
        assert not list(TARGET.glob('*--*/' + model + '/*/*.predictions.jpg'))
    for source, row in zip(sources, records):
        original = ROOT / source['source_path']
        published = TARGET / source['published_path']
        assert original.read_bytes() == published.read_bytes()
        assert read_json(published).get('content', '') == row.get('content', '')
        result = read_json(published.parent / (row['task'] + '.result.json'))
        assert result['view'] == row['view'] and result['prompts'] == native_prompts(row, row['model_name'])
    # Check every local HTML link, and all empty review cells, without browser dependencies.
    from html.parser import HTMLParser
    links = []
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            links.extend(v for k, v in attrs if k in ('href', 'src'))
    checked = 0
    for path in TARGET.rglob('*.html'):
        links.clear(); Links().feed(path.read_text())
        for link in links:
            assert not link.startswith(('http:', 'https:', '/')), 'Non-portable HTML asset'
            assert (path.parent / link).is_file(), (path, link)
            checked += 1
    with (TARGET / 'manual-review-empty.csv').open() as stream:
        review = list(csv.DictReader(stream))
    assert len(review) == 180
    assert all(not value for row in review for key, value in row.items() if key.startswith('review_'))
    result = {'complete': True, 'images': 20, 'datasets': {'thud_dense_test50': 10, 'thud_agtea_test50': 10},
              'models': 3, 'repeats': 3, 'logical_positions': 180, 'native_task_answers': 300,
              'florence_answers': 180, 'nvila_answers': 60, 'qwen3_vl_answers': 60,
              'distinct_original_call_ids': 300, 'reused_answers': 240, 'new_answers': 60,
              'byte_identical_raw_exports': 300, 'full_text_exports_verified': 300,
              'florence_prediction_images': 120, 'borrowed_prediction_images': 0,
              'pairwise_repeat_rows': len(differences), 'local_html_links_verified': checked,
              'manual_review_rows': 180, 'manual_review_fields_all_empty': True,
              'reference_annotations_sent_to_models': False, 'reference_overlays': False,
              'stage2_executed': False, 'pdf_generated': False,
              'detection_accuracy': None, 'accuracy_note': 'No manual/reference scoring claimed.',
              'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(TARGET / 'verification.json', result)
    return result


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    # This file is itself linked by the index; never leave stale success behind
    # while regenerating or checking an incomplete publication.
    write_json(TARGET / 'verification.json', {'complete': False, 'verification_in_progress': True})
    selected = native_selection(ROOT / 'results/ponovitve/frames.json')
    configs = audit_protocol(selected)
    write_json(TARGET / 'nvila-initialization-audit.json', audit_nvila_initialization())
    copy_provenance(configs, selected)
    records, sources, combined = collect(selected, configs)
    differences = repetition_consistency(records)
    # Missing parses must not become an apparently measured empty list/Jaccard.
    for row in differences:
        for side, available in zip(('left', 'right'), row['object_list_available']):
            if not available: row[side + '_objects'] = None
        if not all(row['object_list_available']):
            row['name_jaccard'] = None
            row['name_intersection'] = None
            row['name_union'] = None
            row['names_added'] = None
            row['names_removed'] = None
            row['name_group_details_changed'] = None
            row['possible_alias_changes'] = None
    write_json(TARGET / 'repeat-differences.json', differences)
    csv_file(TARGET / 'repeat-differences.csv', differences,
             list(dict.fromkeys(k for row in differences for k in row)))
    summary = summarize(records, combined)
    write_json(TARGET / 'summary.json', summary)
    csv_file(TARGET / 'summary.csv', summary)
    metrics = [{'frame_id': r['frame_id'], 'model': r['model_name'], 'repeat': r['repeat'],
                'task': r['task'], 'seed': r['seed'], 'call_id': r['call_id'],
                'status': r['status'], 'generation_has_content': bool(r.get('content')),
                'object_list_available': r['view'].get('object_list_available', False),
                'object_entries': len(r['view']['objects']) if r['view'].get('object_list_available', False) else None,
                'scene_characters': len(r['view'].get('scene_description', '')),
                'output_tokens': r.get('output_tokens'), 'elapsed_seconds': r['elapsed_seconds'],
                'tokens_per_end_to_end_second': r['output_tokens'] / r['elapsed_seconds'] if r.get('output_tokens') else None,
                'peak_allocated_bytes': r['peak_allocated_bytes'], 'peak_reserved_bytes': r['peak_reserved_bytes'],
                'error': r.get('error'), 'reused_existing_generation': r['model_name'] != 'qwen3-vl'} for r in records]
    csv_file(TARGET / 'response-metrics.csv', metrics)
    write_json(TARGET / 'provenance.json', sources)
    write_json(TARGET / 'manifest.json', {'frames': [portable_frame(f) for f in selected],
               'models': MODELS, 'revisions': REVISIONS, 'seeds': SEEDS, 'configurations': configs,
               'reused_generation_date': '2026-10-26', 'new_generation_date': 'see per-call UTC timestamps',
               'reference_data_used_for_inference': False})
    review = [{'frame_id': f['frame_id'], 'model': m, 'repeat': r,
               'result_page': leaf_name(f) + '/index.html',
               **{k: '' for k in ('reviewer', 'review_date', 'review_scene_correctness',
                  'review_false_positives', 'review_missing_objects', 'review_locations_relations',
                  'review_duplicate_or_group_objects', 'review_box_quality', 'review_notes')}}
              for f in selected for m in MODELS for r in SEEDS]
    csv_file(TARGET / 'manual-review-empty.csv', review)
    render(selected, records, summary)
    result = verify(selected, records, sources, differences)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
