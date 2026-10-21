"""Offline audit, comparative tables and report appendix for rich descriptions.

Counts of fields/edges measure output structure, not visual correctness. Pairing
old/new decisions uses unique exact normalized names, never cross-run local IDs.
All original responses, rejected candidates and review work remain preserved.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
from .description_prompts import validate_description
from .experiment import read_json, sha256, strict_json, write_csv, write_json
from .image_dataset import load_images
from .schema import validate_frame, validate_classifications
from .thud_experiment import ROOT, REPORT, DATASETS, MODELS, OUT as PART1
from .thud_part2 import OUT, job_records
from .thud_metrics import aggregate, name_overlap
from .vision_pipeline import exact_items


def expected_jobs():
    jobs = []
    for spec in MODELS:
        model = spec['name']
        for ds in DATASETS:
            for kind, count in [('pilot', 2), ('rich', 50), ('reason-rich', 10), ('reason-legacy', 10)]:
                reused = kind == 'reason-legacy' and model == MODELS[0]['name']
                path = (PART1/f'reason-short-no-examples-t0.3-s42-{ds}' if reused else OUT/f'{kind}-{model}-{ds}')
                jobs.append({'name': f'{kind}-{model}-{ds}', 'dataset': ds, 'caption_model': model,
                             'kind': kind, 'expected': count, 'path': path, 'reused': reused})
    return jobs


def collect():
    jobs = []
    for job in expected_jobs():
        path = job['path']
        jobs.append({**job, 'records': job_records(path) if path.exists() else [],
                     'complete': (path/'complete.json').exists(),
                     'warmups': [read_json(p) for p in path.glob('*/warmup.json')] if path.exists() else [],
                     'end_to_end_seconds': read_json(path/'complete.json')['end_to_end_seconds'] if (path/'complete.json').exists() else None})
    return jobs


def copied_warmup(job):
    path = job['path']/'reused-pilot.json'
    return path.exists() and any(p.endswith('/warmup.json') for p in read_json(path)['sha256'])


def verify(jobs, require_complete=False):
    errors, missing = [], []
    audit = read_json(REPORT/'data-audit.json')
    for job in jobs:
        name, path, ds = job['name'], job['path'], job['dataset']
        if not job['complete']:
            missing.append(name)
            continue
        rows = job['records']
        if len(rows) != job['expected']:
            errors.append(name + ': unexpected record count')
        manifest = read_json(path/'manifest.json')
        reuse_file = path/'reused-pilot.json'
        if reuse_file.exists():
            reuse = read_json(reuse_file)
            for relative, digest in reuse['sha256'].items():
                if sha256((path/relative).read_bytes()) != digest or sha256((Path(reuse['source'])/relative).read_bytes()) != digest:
                    errors.append(name + ': reused pilot differs from original')
        is_stage1 = job['kind'] in {'pilot', 'rich'}
        expected_indices = (list(range(2)) if job['kind'] == 'pilot' else list(range(50))) if is_stage1 else audit[ds]['subset_indices_zero_based']
        expected_fids = [audit[ds]['frames'][i]['discovery_id'] for i in expected_indices]
        if [r['frame_id'] for r in rows] != expected_fids:
            errors.append(name + ': wrong ordered image selection')
        if is_stage1:
            if manifest['config'].get('description_schema') != 'rich-v2':
                errors.append(name + ': wrong description schema')
            input_path = OUT/'inputs'/f'{ds}-discovery-{"pilot" if job["kind"]=="pilot" else "all"}.jsonl'
            if manifest['input_sha256'] != sha256(input_path.read_bytes()):
                errors.append(name + ': changed source manifest')
            frames = load_images(input_path, job['expected'])
            if [r['image_sha256'] for r in rows] != [f['image_sha256'] for f in frames]:
                errors.append(name + ': changed image bytes')
        for row_index, row in enumerate(rows):
            try:
                if is_stage1:
                    if set(row['source_frame']) != {'frame_id', 'image_path', 'image_sha256', 'image_size'}:
                        raise ValueError('annotation metadata in discovery source')
                    if row['variant'] != {'id': 't0-s42', 'temperature': 0.0, 'seed': 42}:
                        raise ValueError('wrong stage1 T/seed')
                else:
                    entry = manifest['source_captions'][row_index]
                    source_path = Path(entry['path'])
                    if sha256(source_path.read_bytes()) != entry['sha256'] or entry['sha256'] != row['caption_sha256']:
                        raise ValueError('changed frozen caption bytes')
                    source = read_json(source_path)
                    if source['frame_id'] != row['frame_id']:
                        raise ValueError('caption frame mismatch')
                    original = ((OUT/f'rich-{job["caption_model"]}-{ds}' if job['kind']=='reason-rich' else PART1/f'baseline-{job["caption_model"]}-discovery-{ds}') / job['caption_model']/'t0-s42'/source_path.name)
                    if source_path.read_bytes() != original.read_bytes():
                        raise ValueError('frozen caption differs from original saved response')
                    if row['status'] == 'skipped':
                        if source['status'] == 'ok' or 'request' in row:
                            raise ValueError('invalid skip')
                        continue
                    if row['variant'] != {'id': 't0.3-s42', 'temperature': .3, 'seed': 42}:
                        raise ValueError('wrong stage2 T/seed')
                if 'response' in row:
                    if not row.get('gpu_verified') or row['gpu']['size_vram'] <= 0 or row['gpu']['size_vram'] < row['gpu'].get('size', 0):
                        raise ValueError('CPU/partial/unverified GPU response')
                if 'request' in row:
                    messages = row['request']['messages']
                    payload = strict_json(messages[-1]['content'].split('\n', 1)[1])
                    if is_stage1:
                        if set(payload) != {'frame_id', 'image_size', 'annotation_kind', 'supplied_objects'} or payload['supplied_objects'] is not None or payload['annotation_kind'] != 'unspecified':
                            raise ValueError('annotation leakage')
                        if payload['frame_id'] != row['frame_id'] or payload['image_size'] != frames[row_index]['image_size']:
                            raise ValueError('request/image mismatch')
                    else:
                        if any('images' in m or isinstance(m.get('content'), list) for m in messages):
                            raise ValueError('image in stage2 request')
                        if payload['visual_observations'] != source['parsed']:
                            raise ValueError('structured evidence/relations changed in stage2')
                        expected_objects = [{'id': o['id'], 'label': o['label']} for o in source['parsed']['objects']]
                        if payload['objects'] != expected_objects:
                            raise ValueError('reasoner IDs/labels differ from caption')
                if row['status'] == 'ok':
                    parsed = strict_json(row['content'])
                    if parsed != row['parsed']:
                        raise ValueError('parsed response differs from raw JSON')
                    if is_stage1:
                        validate_description(parsed, expected_schema='rich-v2')
                    else:
                        objects = [{'id': o['id'], 'label': o['label']} for o in source['parsed']['objects']]
                        exact_items(objects, parsed.get('objects'))
                        validate_classifications(validate_frame({'frame_id': row['frame_id'], 'objects': objects}), parsed)
            except Exception as exc:
                errors.append(name + '/' + row['frame_id'] + ': ' + str(exc))
        for warmup in job['warmups']:
            if 'response' in warmup and not warmup.get('gpu_verified'):
                errors.append(name + ': unverified warmup GPU')
    index_path = OUT/'part1-results-index.json'
    preserved = 0
    if index_path.exists():
        for relative, digest in read_json(index_path).items():
            if sha256((PART1/relative).read_bytes()) != digest:
                errors.append('Part-one result changed: ' + relative)
            else:
                preserved += 1
    else:
        errors.append('Missing part-one preservation index')
    result = {'complete_jobs': len(jobs)-len(missing), 'expected_jobs': len(jobs),
              'missing_jobs': missing, 'invariant_errors': errors, 'preserved_part1_responses': preserved,
              'complete': not missing, 'passed': not errors and (not require_complete or not missing),
              'note': 'Schema errors/skipped invalid sources are measured outcomes, not invariant errors.'}
    write_json(REPORT/'part2-verification.json', result)
    return result


def structure(rows):
    valid = [r['parsed'] for r in rows if r['status']=='ok']
    objects = [o for p in valid for o in p['objects']]
    relations = [r for p in valid for r in p.get('relations', [])]
    return {'valid_descriptions': len(valid), 'objects': len(objects), 'relations': len(relations),
            'with_relations': sum(bool(p.get('relations')) for p in valid),
            'unknown_installation': sum(o.get('installation_evidence', '').strip().lower()=='unknown' for o in objects),
            'unknown_support': sum(o.get('support_contacts', '').strip().lower()=='unknown' for o in objects),
            'occlusion': dict(Counter(o.get('occlusion', 'unstructured') for o in objects)),
            'predicates': dict(Counter(r['predicate'] for r in relations)),
            'note': 'Output structure and literal unknown counts, not visual correctness or relation recall.'}


def reason_labels(row):
    if not row or row['status'] != 'ok':
        return {}
    caption = read_json(row['caption_source'])['parsed']
    labels = {(type(o['id']), o['id']): o['label'] for o in caption['objects']}
    result = {}
    for obj in row['parsed']['objects']:
        label = ' '.join(labels[(type(obj['id']), obj['id'])].lower().split())
        result.setdefault(label, []).append(obj)
    return {label: objects[0] for label, objects in result.items() if len(objects)==1}


def decision_pairs(jobs):
    pairs = []
    indexed = {(j['kind'], j['caption_model'], j['dataset']): j for j in jobs}
    audit = read_json(REPORT/'data-audit.json')
    for spec in MODELS:
        for ds in DATASETS:
            before = {r['frame_id']: r for r in indexed['reason-legacy', spec['name'], ds]['records']}
            after = indexed['reason-rich', spec['name'], ds]['records']
            for row in after:
                old = before.get(row['frame_id'])
                old_labels, new_labels = reason_labels(old), reason_labels(row)
                for label in sorted(set(old_labels) & set(new_labels)):
                    a, b = old_labels[label], new_labels[label]
                    source = next(f for f in audit[ds]['frames'] if f['discovery_id']==row['frame_id'])
                    pairs.append({'dataset': ds, 'caption_model': spec['name'], 'frame_id': row['frame_id'],
                                  'image_sha256': source['image_sha256'], 'normalized_label': label,
                                  'old_id': a['id'], 'rich_id': b['id'], 'old_class': a['classification'],
                                  'rich_class': b['classification'], 'changed': a['classification']!=b['classification'],
                                  'old_reason': a['reason'], 'rich_reason': b['reason'],
                                  'old_caption_sha256': old['caption_sha256'], 'rich_caption_sha256': row['caption_sha256'],
                                  'matching': 'Unique exact lowercase whitespace-normalized label in each same-image output; not verified instance identity.'})
    write_json(REPORT/'part2-decision-pairs.json', pairs)
    write_csv(REPORT/'part2-decision-pairs.csv', pairs, list(pairs[0]) if pairs else ['matching'])
    return pairs


def rejected_candidates(jobs):
    candidates, index = [], []
    for job in jobs:
        if job['reused']:
            continue
        for row in job['records']:
            if row['status'] != 'error':
                continue
            item = {'job':job['name'], 'frame_id':row['frame_id'], 'error':row.get('error'),
                    'accepted':False, 'content_sha256':sha256(row.get('content','').encode())}
            try:
                parsed = strict_json(row.get('content', ''))
            except (ValueError, TypeError):
                item['syntax_valid'] = False
            else:
                item['syntax_valid'] = True
                candidates.append({**item, 'rejected_candidate':parsed})
            index.append(item)
    (OUT/'rejected-candidates.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in candidates))
    write_json(REPORT/'part2-rejected-index.json', index)
    return dict(Counter(item['error'] for item in index))


def review(jobs, frames):
    from .thud_report import review_svg
    audit = read_json(REPORT/'data-audit.json')
    notes = read_json(REPORT/'part2-manual-notes.json') if (REPORT/'part2-manual-notes.json').exists() else {}
    doc = ['<!doctype html><meta charset="utf-8"><title>THUD 2. del – primerjava</title><style>body{font:15px sans-serif;max-width:1300px;margin:auto}section{margin:30px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #aaa;padding:12px;margin:6px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:12px}</style><h1>2. del: isti podnabor 20 slik</h1><p>Referenčni okvirji so namenjeni pregledu, ne vhodu modela. Lokalni ID različnih opisov ne pomeni istega objekta. Strukturna veljavnost ne potrjuje vidnih dejstev. Sistematična ročna polja ostajajo pending.</p>']
    manual_rows, comparison_rows = [], []
    for ds in DATASETS:
        for i in audit[ds]['subset_indices_zero_based']:
            f = frames[ds][i]
            doc.append('<section><h2>' + html.escape(ds+'/'+f['record']['frame_id']) + '</h2>' + review_svg(f))
            note = notes.get(ds+'/'+f['record']['frame_id'])
            if note:
                doc.append('<p><b>Delni vizualni pregled:</b> '+html.escape(note)+'</p>')
            doc.append('<div class="grid">')
            for spec in MODELS:
                model = spec['name']
                old = read_json(PART1/f'baseline-{model}-discovery-{ds}'/model/'t0-s42'/f'frame-{i+1:06d}.json')
                rich_job = next(j for j in jobs if j['kind']=='rich' and j['caption_model']==model and j['dataset']==ds)
                rich = next((r for r in rich_job['records'] if r['image_sha256']==f['image_sha256']), None)
                for kind, row in [('legacy', old), ('rich', rich)]:
                    if row is None:
                        continue
                    doc.append('<details open><summary>'+html.escape(model+' / '+kind+' / '+row['status'])+'</summary><pre>'+html.escape(json.dumps(row.get('parsed', {'error':row.get('error'), 'content':row.get('content')}), ensure_ascii=False, indent=2))+'</pre></details>')
                if rich:
                    before = [o['label'] for o in old.get('parsed', {}).get('objects', [])]
                    after = [o['label'] for o in rich.get('parsed', {}).get('objects', [])]
                    comparison_rows.append({'dataset':ds, 'frame_id':f['record']['frame_id'], 'model':model,
                        'image_sha256':f['image_sha256'], 'old_status':old['status'], 'rich_status':rich['status'],
                        'old_objects':len(before), 'rich_objects':len(after), **name_overlap(after, before)})
                    manual_rows.append({'dataset':ds, 'frame_id':f['record']['frame_id'], 'model':model,
                        'image_sha256':f['image_sha256'], 'review_status':'pending', 'missed':'',
                        'wrong_or_hallucinated':'', 'locations':'', 'occlusion':'', 'supports':'', 'installation':'', 'relations':'', 'notes':''})
            for job in jobs:
                if not job['kind'].startswith('reason') or job['dataset']!=ds:
                    continue
                row = next((r for r in job['records'] if r['frame_id']==audit[ds]['frames'][i]['discovery_id']), None)
                if row:
                    doc.append('<details><summary>'+html.escape(job['name']+' / '+row['status'])+'</summary><pre>'+html.escape(json.dumps(row.get('parsed', {'error':row.get('error'),'content':row.get('content')}), ensure_ascii=False, indent=2))+'</pre></details>')
            doc.append('</div></section>')
    (OUT/'manual-review.html').write_text('\n'.join(doc))
    path = REPORT/'part2-manual-review.csv'
    key = lambda r: (r['image_sha256'], r['model'])
    if path.exists():
        with path.open() as stream:
            previous = {key(r): r for r in csv.DictReader(stream)}
        for row in manual_rows:
            if key(row) in previous:
                for field in ('review_status', 'missed', 'wrong_or_hallucinated', 'locations', 'occlusion', 'supports', 'installation', 'relations', 'notes'):
                    row[field] = previous[key(row)].get(field, row[field])
    write_csv(path, manual_rows, list(manual_rows[0]) if manual_rows else ['review_status'])
    write_json(REPORT/'part2-image-comparisons.json', comparison_rows)
    write_csv(REPORT/'part2-image-comparisons.csv', comparison_rows, list(comparison_rows[0]) if comparison_rows else ['model'])


def latex(jobs, stats, summary, pairs):
    from .thud_report import tex, table, number
    pages = []
    comparisons, structures = [], []
    for spec in MODELS:
        model = spec['name']
        new = [r for j in jobs if j['kind']=='rich' and j['caption_model']==model for r in j['records']]
        old = [r for ds in DATASETS for r in job_records(PART1/f'baseline-{model}-discovery-{ds}')]
        for name, rows in [('1. del, podrobni', old), ('2. del, bogati', new)]:
            a = aggregate(rows)
            comparisons.append([tex(model), name, f'{a["ok"]}/{a["attempted"]}', number(a['median_s']), number(a['p90_s']), number(a['mean_s']), number(a['mean_output_tokens'], 0), number(a['generation_tokens_s']), number(a['max_vram_bytes']/2**30, 2)])
        s = structure(new)
        structures.append([tex(model), s['objects'], s['relations'], f'{s["with_relations"]}/{s["valid_descriptions"]}', s['unknown_installation'], s['unknown_support']])
    pages.append(r'''\section*{2. del: bogatejši samostojni opis (21. 10. 2026)}
Nadaljevanje uporablja obstoječe izhode 1. dela (objavljen commit \texttt{7509607}); uspešni stari klici se ne ponovijo. Vsi trije delujoči modeli prejmejo iste slikovne bajte vseh 100 slik, brez referenčnih imen, ID, okvirjev ali poti anotacij. NVILA še nima preverjenega slikovnega API; njegova že shranjena diagnostika se uporabi brez novega prenosa ali trditve o inferenci.
\subsection*{Izhodna shema in dokazi}
\texttt{rich-description-v2} ohrani \texttt{scene\_description} in stare združljive podatke objektov, doda \texttt{appearance}, \texttt{image\_location}, \texttt{occlusion}, \texttt{support\_contacts}, \texttt{installation\_evidence} ter seznam \texttt{relations}. Odnos navaja začetni/ciljni lokalni ID, predikat in vidni dokaz. Validator zavrne manjkajoče podatke, podvojene ID/odnose, sklice na neobstoječe ID, samopovezave in razrede stabilnosti v prvem koraku. Neznano je izrecno \texttt{unknown}. Nobeno besedilno polje samo ne dokazuje pravilnosti trditve.
Položaji opisujejo levo/desno v sliki, ne svetovnih koordinat. Skritih vijakov, teže ali trajnosti ne zahtevamo; bližina ali negibnost ni namestitev. ID so lokalni, brez sledenja skozi čas. Napovedanih okvirjev ni; besedni položaj ni meritev natančne lokalizacije. Zato ni IoU ali mAP in na teh rezultatih še ni podlage za uvedbo posebnega detektorja.
\subsection*{Primerjava časa in oblike izhoda}
'''+table(['Opisovalnik','Različica','OK/vsi','Med.','p90','Povp.','Tok.','tok/s','GiB'], comparisons, 'llrrrrrrr')+r'''
Časi so sekunde na sliko, tudi za zavrnjene odgovore z izmerjenim časom. GiB je največja API dodelitev VRAM. Ogrevanje in nalaganje sta ločena v \texttt{part2-aggregates.json}. GPU je deljen; obremenitev in dolžina odgovora omejujeta vzročno primerjavo hitrosti. Stari prompt je podroben, novi je drugačen prompt s strožjo shemo; razlike ne pripisujemo samo strukturiranju.
Prekopirano pilotno ogrevanje ne pomeni ogretega modela po novem nalaganju. Prvi novi klic (indeks 3) posamezne skupine lahko vključuje nalaganje; celotni čas ostane v tabeli, API \texttt{load\_duration} je ohranjen ločeno. To je dodatna omejitev neposredne primerjave hitrosti, ne prikrita izločitev počasnih klicev.
'''+table(['Model','Objekti','Odnosi','Opisi z odnosi','Namest. unknown','Podpora unknown'], structures, 'lrrrrr')+r'''
Število odnosov/objektov in prisotnost polj merita strukturo, ne pravilnosti, priklica ali vidnih dokazov. Neznano šteje le dobesedni \texttt{unknown}, ne vseh izrazov negotovosti. Vsi sprejeti odnosi imajo veljavne sklice in neprazno besedilo dokaza; ta pogoj ne potrdi vsebine fotografije.
''')
    reasoning = []
    for spec in MODELS:
        for kind, label in [('reason-legacy', 'stari opis'), ('reason-rich', 'bogati opis')]:
            rows = [r for j in jobs if j['kind']==kind and j['caption_model']==spec['name'] for r in j['records']]
            a = aggregate(rows)
            c = Counter(o['classification'] for r in rows if r['status']=='ok' for o in r['parsed']['objects'])
            reasoning.append([tex(spec['name']), label, f'{a["ok"]}/{a["attempted"]}', a['skipped'], c['fixed'], c['movable'], c['uncertain'], number(a['median_s'])])
    paired = [[tex(s['name']), sum(p['caption_model']==s['name'] for p in pairs), sum(p['changed'] for p in pairs if p['caption_model']==s['name'])] for s in MODELS]
    v = summary['verification']
    t = summary.get('tests', {})
    diagnostic = summary.get('initial_pilot_diagnostic', {})
    initial = diagnostic.get('first_pilot', {})
    control = []
    for label, rows in [('1. del kratek', [r for ds in DATASETS for r in job_records(PART1/f'extra-short-t0-{ds}')]),
                        ('2. del bogati', [r for j in jobs if j['kind']=='rich' and j['caption_model']==MODELS[0]['name'] for r in j['records'] if int(r['frame_id'].split('-')[-1])-1 in read_json(REPORT/'data-audit.json')[j['dataset']]['subset_indices_zero_based']])]:
        a = aggregate(rows)
        control.append([label, f'{a["ok"]}/{a["attempted"]}', number(a['median_s']), a['objects']])
    pages.append(r'''\section*{2. del: vpliv opisov na besedilno presojo}
Isti izbor 10+10 slik ostane zamrznjen. Razvrščevalnik Qwen BF16 dobi celoten shranjeni opis, strukturirane objekte in odnose, brez odpiranja fotografije. Uporabljen je isti kratki prompt brez primerov iz 1. dela pri T=.3, seed 42; izbran je zaradi uspešne izpolnitve sheme, ne potrjene vsebinske prednosti. Stare BF16 odločitve se uporabijo ponovno, manjkajoče kontrole starih Q4/Gemma opisov se izvedejo le enkrat.
'''+table(['Opisovalnik','Vhod','OK/vsi','Preskoki','F','M','U','Med. [s]'], reasoning, 'llrrrrrr')+r'''
F/M/U so števila odločitev v sprejetih odgovorih. Preskok pomeni zavrnjen vhodni opis; ostane v imenovalcu vseh 20 položajev. Drugačno število objektov med starimi in novimi opisi pomeni, da skupnih ID ne moremo obravnavati kot iste instance.
'''+table(['Opisovalnik','Pari imen','Spremembe razreda'], paired, 'lrr')+r'''
Pari so le enolična, natančno enaka imena po pretvorbi v male črke in normalizaciji presledkov v obeh veljavnih izhodih iste slike. Podvojena imena se izključijo; sinonimov in okvirjev ne ujemamo. To ni potrjeno ujemanje instanc; skupinska imena so posebej problematična. Sprememba ali strinjanje ni pravilnost. Konkretni pari, ID, SHA256 obeh opisov in oba modelova razloga so v \texttt{part2-decision-pairs.csv}.
\subsection*{Ožja kontrola prompta BF16 na istih 20 slikah}
'''+table(['Različica opisa BF16','OK/vsi','Med. [s]','Objekti'], control, 'lrrr')+r'''
Oba prompta sta brez vsebinskih primerov in pri T=0, seed 42. Kratki stari opis se ponovno uporabi iz 1. dela. Nova shema in besedilo se še vedno spreminjata skupaj, zato to ni izolirana ablacija sheme. Za Q4/Gemma primerljivega starega kratkega opisa ni; dodatnih starih klicev za to ne izvajamo.
\subsection*{Obseg, revizija in ponovljivost}
'''+tex(f'Položajev glavne primerjave: {summary["new_main"]["attempted"]}; odgovorov {summary["new_main"]["json_text_available_calls"]}; preskokov {summary["new_main"]["skipped"]}. Pilot: {summary["pilot"]["attempted"]} odgovorov, od tega {summary["pilot_reused_in_main_slots"]} prekopiranih v glavno primerjavo, brez ponovnega klica. Enkratnih novih odgovorov končnega prompta: {summary["unique_final_prompt_responses"]}. Ločena nova ogrevanja: {summary["warmup_calls"]}; {summary["warmup_reused_slots"]} njihovih kopij se ne šteje ponovno. Revizija {v["complete_jobs"]}/{v["expected_jobs"]} skupin, kršitev {len(v["invariant_errors"])}; nespremenjenih starih odgovorov {v["preserved_part1_responses"]}. Testi: {t.get("tests", "--")} ({t.get("result", "še ni izvedeno")}).')+r'''
'''+tex(f'Predhodna diagnostika prvega prompta: {initial.get("ok",0)}/{initial.get("attempted",0)} po shemi, {diagnostic.get("warmup_calls",0)} ločenih ogrevanj. Zaradi izpuščenih tipov je dodana izrecna JSON tipovna predloga brez vsebinskih primerov; validator in generacijske nastavitve niso razrahljani. Diagnostični odgovori so ohranjeni ločeno in niso prišteti končni primerjavi; ne gre za prikrito popravljanje izhodov.')+r'''
'''+tex(f'End-to-end čas novih faz brez predhodne diagnostike: {number(summary["new_end_to_end_seconds"]/60)} min; pilotne kopije in 20 starih presoj niso šteti ponovno. Nalaganje in ogrevanje sta podrobno ločena v agregatih; warmup_reused_from_pilot označi kopijo ogrevanja, new_warmup_calls pa dejanske nove klice. Za prihodnje sveže zagone je kopiranje ogrevanja odpravljeno; izvirne meritve in njihov nalagalni strošek ostanejo nespremenjeni.')+r'''
\begin{verbatim}
bash scripts/thud_part2.sh --phase all
python3 -m llm_object_classifier.thud_part2_report --require-complete
python3 -m llm_object_classifier.thud_report
bash scripts/compile_thud_report.sh
\end{verbatim}
Surovi izhodi: \texttt{results/thud-part2/}; izvorni 1. del ostane v \texttt{results/thud-vmesno/}. Dopolnjeni pregled s fotografijami, referenčnimi okvirji, vsemi strukturami in odločitvami: \texttt{results/thud-part2/manual-review.html}. Konfiguracije, SHA256 kode, stanje Gita in modeli so v lokalnih manifestih/snapshotih. Ponovni zagon ohrani tudi zavrnitve in preskoke; spremenjeni vhodi se zavrnejo.
''')
    notes = read_json(REPORT/'part2-manual-notes.json') if (REPORT/'part2-manual-notes.json').exists() else {}
    cases = read_json(REPORT/'cases.json')
    for offset in (0, 2):
        page = r'\section*{2. del: dejanski primeri bogatejšega opisa}'
        for case in cases[offset:offset+2]:
            ds, i = case['dataset'], case['index']
            page += r'\subsection*{'+tex(ds+'/'+case['frame_id'])+r'}\begin{center}\includegraphics[height=3.2cm,width=.8\linewidth,keepaspectratio]{'+case['image']+r'}\end{center}'
            entries = []
            for spec in MODELS:
                job = next(j for j in jobs if j['kind']=='rich' and j['dataset']==ds and j['caption_model']==spec['name'])
                row = next((r for r in job['records'] if r['frame_id']==f'image-{i+1:06d}'), None)
                if row is None:
                    continue
                parsed = row.get('parsed', {})
                labels = {o['id']: o['label'] for o in parsed.get('objects', [])}
                relation = next(iter(parsed.get('relations', [])), None)
                text = (labels[relation['subject_id']]+' '+relation['predicate']+' '+labels[relation['object_id']]+': '+relation['evidence']) if relation else row.get('error', 'Brez odnosov.')
                count_text = str(len(parsed.get('objects', [])))+'/'+str(len(parsed.get('relations', []))) if row['status']=='ok' else 'zavrnjeno'
                entries.append([tex(spec['name']), tex(count_text), tex(text[:210])])
            page += table(['Model','Obj./odnosi','Dejanski prvi modelov odnos ali napaka'], entries, r'p{2.2cm}p{2.1cm}p{10.7cm}')
            page += r'\textbf{Lastna presoja:} '+tex(notes.get(ds+'/'+case['frame_id'], 'Delni vizualni pregled še ni zapisan.'))+'\n'
        if offset == 2:
            examples = []
            for spec in MODELS:
                choices = [p for p in pairs if p['caption_model']==spec['name']]
                chosen = next((p for p in choices if p['changed']), choices[0] if choices else None)
                if chosen:
                    examples.append([tex(spec['name']+' / '+chosen['frame_id']+' / '+chosen['normalized_label']),
                                     tex(chosen['old_class']+' -> '+chosen['rich_class']), tex(chosen['rich_reason'])])
            if examples:
                page += r'\subsection*{Dejanski besedni razlogi pri parih imen}' + table(
                    ['Opisovalnik / slika / ime','Staro -> bogato','Razlog modela na bogatem opisu'], examples,
                    r'p{5.6cm}p{2.8cm}p{6.5cm}')
                page += 'To so modelove utemeljitve, ne potrjena referenca; ujemanje imena ne potrdi iste instance.\n'
            conclusion = read_json(REPORT/'part2-conclusion.json') if (REPORT/'part2-conclusion.json').exists() else {}
            page += r'\subsection*{Sklep in naslednji poskus}'+tex(conclusion.get('text', 'Strukturiran izhod omogoča pregled podpor, zakritosti in izrecnih odnosov. Vsebinska prednost še ni potrjena; sistematični ročni pregled ostane pending. Brez referenc stabilnosti ne poročamo o accuracy. Pred končno izbiro prompta je potreben ročni pregled istega podnabora.'))
        pages.append(page)
    (REPORT/'part2.tex').write_text('\n\\newpage\n'.join(pages)+'\n')


def generate(require_complete=False):
    jobs = collect()
    verification = verify(jobs, require_complete)
    stats = []
    for job in jobs:
        a = aggregate(job['records'])
        record = {**{k:job[k] for k in ('name','kind','dataset','caption_model','complete','reused','end_to_end_seconds')},
                      'warmup_calls': len(job['warmups']), 'warmup_seconds': sum(r.get('elapsed_seconds',0) for r in job['warmups']),
                      'warmup_load_seconds': sum(r.get('response',{}).get('load_duration',0)/1e9 for r in job['warmups']),
                      **a}
        record['pilot_reused'] = (job['path']/'reused-pilot.json').exists()
        record['warmup_reused_from_pilot'] = copied_warmup(job)
        record['new_warmup_calls'] = 0 if record['warmup_reused_from_pilot'] or job['reused'] else len(job['warmups'])
        if job['kind'] in {'pilot','rich'}:
            record.update({'structure_'+k:v for k,v in structure(job['records']).items()})
        stats.append(record)
    write_json(REPORT/'part2-aggregates.json', stats)
    fields = list(dict.fromkeys(k for s in stats for k in s))
    write_csv(REPORT/'part2-aggregates.csv', stats, fields)
    main_rows = [r for j in jobs if not j['reused'] and j['kind']!='pilot' for r in j['records']]
    pilot = [r for j in jobs if j['kind']=='pilot' for r in j['records']]
    warmups = [r for j in jobs if not j['reused'] for r in j['warmups']]
    reused_pilot_rows = [r for j in jobs if j['kind']=='rich' and (j['path']/'reused-pilot.json').exists() for r in j['records'][:2]]
    unique_warmups = [r for j in jobs if not j['reused'] and not copied_warmup(j) for r in j['warmups']]
    summary = {'new_main':aggregate(main_rows), 'pilot':aggregate(pilot),
               'pilot_reused_in_main_slots':len(reused_pilot_rows),
               'unique_final_prompt_responses':sum('response' in r for r in main_rows+pilot)-sum('response' in r for r in reused_pilot_rows),
               'reused_stage2':aggregate([r for j in jobs if j['reused'] for r in j['records']]),
               'warmup_calls':len(unique_warmups), 'warmup_gpu_verified':sum(r.get('gpu_verified',False) for r in unique_warmups),
               'warmup_reused_slots':len(warmups)-len(unique_warmups),
               'verification': verification, 'raw_outputs':str(OUT),
               'tests':read_json(REPORT/'part2-tests.json') if (REPORT/'part2-tests.json').exists() else {},
               'initial_pilot_diagnostic':read_json(REPORT/'part2-pilot-revision.json') if (REPORT/'part2-pilot-revision.json').exists() else {},
               'plan':read_json(OUT/'plan.json'),
               'pilot_estimate':read_json(OUT/'pilot-estimate.json') if (OUT/'pilot-estimate.json').exists() else {},
               'new_end_to_end_seconds':sum(j['end_to_end_seconds'] or 0 for j in jobs if not j['reused'])}
    provenance = OUT/'part1-results-index.json'
    summary['part1_preservation_index'] = {'path':str(provenance), 'sha256':sha256(provenance.read_bytes())} if provenance.exists() else None
    summary['pilot_reuse_policy'] = read_json(OUT/'pilot-reuse-policy.json') if (OUT/'pilot-reuse-policy.json').exists() else None
    snapshots = sorted((OUT/'source-snapshots').glob('all-*.json'))
    snapshot = read_json(snapshots[0]) if snapshots else {}
    write_json(REPORT/'part2-provenance.json', {'inference_source_snapshot':snapshot,
        'report_generator_sha256':sha256(Path(__file__).read_bytes()),
        'initial_part1_commit':'7509607', 'raw_outputs':str(OUT)})
    write_json(REPORT/'part2-model-metadata.json', {
        spec['name']: {j['dataset']:read_json(j['path']/'manifest.json')['model_metadata'][spec['name']] for j in jobs
                       if j['kind']=='rich' and j['caption_model']==spec['name'] and (j['path']/'manifest.json').exists()}
        for spec in MODELS})
    write_json(REPORT/'part2-summary.json', summary)
    pairs = decision_pairs(jobs)
    write_json(REPORT/'part2-errors.json', rejected_candidates(jobs))
    frames = {ds:load_images(ROOT/'data'/ds/'objects.jsonl',50) for ds in DATASETS}
    review(jobs, frames)
    latex(jobs, stats, summary, pairs)
    print(json.dumps(verification, indent=2))
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    verification = generate(args.require_complete)
    if not verification['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
