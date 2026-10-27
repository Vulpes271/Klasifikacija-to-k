"""Scripted comparisons and offline browser for frozen stage-two decisions.

Exact-ID comparisons apply ONLY to the same frozen input. Across VLM sources
and VLM repeats, unique exact normalized names are tentative text matches, never
confirmed physical identities. Repeated names are left ambiguous and unmatched.
"""
from collections import Counter, defaultdict
from itertools import combinations, product
import csv
import html
import json
import re
import statistics
import shutil

from .experiment import read_json, write_json, sha256
from .vision_prompts import reason_prompt
from .thud_stage2_presentation import ROOT, SOURCE, TARGET, OUT, VLMS, SEEDS, PROMPTS, MODELS
from .thud_stage1_presentation import csv_file, document, pre as source_pre, escaped


def pre(value):
    # Preserve literal model whitespace in the DOM without trailing spaces in
    # generated source files. JSON responses themselves are never modified.
    return re.sub(r'(?m)[ \t]+$', lambda m: ''.join('&#'+str(ord(c))+';' for c in m[0]), source_pre(value))


def input_key(row):
    return row['frame_id'], row['vlm'], row['vlm_repeat']


def label_key(label):
    return ' '.join(label.lower().split())


def typed(identifier):
    return type(identifier).__name__, identifier


def restore_public_record(published):
    """Reconstruct the exact logical request using checked shared text files.

    This also lets a fresh Git checkout analyse the published results without
    the ignored local runner folders or installed models.
    """
    r=dict(published)
    if r['status']=='skipped':return r
    user=(ROOT/r['user_prompt_path']).read_text()
    system_file=(ROOT/r['system_prompt_path']).read_bytes()
    system=system_file.decode().strip()
    assert sha256(user.encode())==r['user_prompt_sha256']
    assert sha256(system_file)==r['system_prompt_file_sha256']
    assert sha256(system.encode())==r['system_prompt_sha256']
    r.update(user_prompt=user,system_prompt=system)
    if 'response' in r:r['content']=r['response'].get('message',{}).get('content','')
    if 'request_without_messages' in r:
        r['request']={**r['request_without_messages'],'messages':[
            {'role':'system','content':system},{'role':'user','content':user}]}
    r['parallel_slots']=r.get('parallel_slots_requested',1)
    return r


def load():
    index = read_json(TARGET / 'frozen/index.json')
    frozen = {}
    for row in index:
        path = ROOT / row['path']
        assert sha256(path.read_bytes()) == row['sha256']
        frozen[row['input_index']] = {**row, 'document': read_json(path)}
    records = []
    if not (OUT/'runs').exists():
        for path in sorted((TARGET/'responses').rglob('responses.json')):
            bundle=read_json(path)
            for prompt,runs in bundle.items():
                for repeat,published in runs.items():
                    r=restore_public_record(published)
                    assert r['prompt_name']==prompt and r['llm_repeat']==repeat
                    assert r['caption_sha256']==frozen[r['input_index']]['sha256']
                    records.append(r)
        return index,frozen,records
    for model in MODELS:
        for prompt in PROMPTS:
            for repeat, seed in SEEDS.items():
                directory = OUT / 'runs' / model['name'] / prompt / repeat / model['name'] / f't0.3-s{seed}'
                for number, source in frozen.items():
                    path = directory / f'frame-{number:06d}.json'
                    if not path.exists(): continue
                    r = read_json(path)
                    assert r['caption_sha256'] == source['sha256']
                    assert r['variant']['seed'] == seed and r['variant']['temperature'] == .3
                    assert r['model'] == model['model'] and r['model_name'] == model['name']
                    r.update(input_index=number, vlm=source['vlm'], vlm_repeat=source['vlm_repeat'],
                             llm=model['name'], prompt_name=prompt, llm_repeat=repeat,
                             original_record=str(path.relative_to(ROOT)), original_record_sha256=sha256(path.read_bytes()))
                    records.append(r)
    return index, frozen, records


def destination(source, llm, prompt):
    return (TARGET / 'responses' / source['frame_id'].replace('/', '--') /
            source['vlm'] / ('vlm-' + source['vlm_repeat']) / llm / prompt)


def runtime_slots(record):
    requested=record.get('parallel_slots',record.get('parallel_slots_requested',1))
    if requested==1:return requested,1
    path=OUT/'setup/runtime-parallel.json'
    if not path.exists():path=TARGET/'provenance/runtime-parallel.json'
    runtime=read_json(path) if path.exists() else {}
    model=record.get('llm',record.get('model_name'))
    context=str(record.get('requested_options',{}).get('num_ctx'))
    actual=runtime.get('actual_model_slots_by_context',{}).get(model,{}).get(context)
    if actual is None:actual=runtime.get('actual_model_slots',{}).get(model)
    return requested,actual


def public_record(record, source, user_prompt_path, system_path):
    """Lossless de-duplication: only fields with a checked shared copy become refs.

    The entire Ollama response, including rejected text and timing, stays intact.
    Original local runner files remain unchanged. Published references reconstruct
    the exact full request; no image payload exists in this experiment.
    """
    r = {k: v for k, v in record.items() if k not in ('content', 'user_prompt', 'system_prompt', 'request', 'parsed', 'candidate_parsed')}
    r.update(frozen_input_path=source['path'], frozen_input_sha256=source['sha256'],
             user_prompt_path=str(user_prompt_path.relative_to(ROOT)),
             system_prompt_path=str(system_path.relative_to(ROOT)),
             user_prompt_sha256=sha256(user_prompt_path.read_bytes()),
             system_prompt_file_sha256=sha256(system_path.read_bytes()),
             system_prompt_sha256=sha256(record['system_prompt'].encode()),
             full_response_text_path='response.message.content')
    # The runner's field records OLLAMA_NUM_PARALLEL, which the qwen35
    # architecture currently overrides to one. Preserve that distinction.
    requested,actual=runtime_slots(record)
    r.pop('parallel_slots',None)
    r['parallel_slots_requested']=requested
    r['parallel_slots_actual']=actual
    r['parallel_runtime_note']='Requested setting versus observed n_seq_max; see provenance/runtime-parallel.json and runtime evidence.'
    # Parser outputs are retained as well; accepted decisions are never inferred
    # from an invalid response during statistical comparisons.
    for key in ('parsed', 'candidate_parsed'):
        if key in record: r[key] = record[key]
    if 'response' in record:
        assert record.get('content','') == record['response'].get('message',{}).get('content','')
    elif 'content' in record:
        r['content'] = record['content']
    if 'request' in record:
        request = record['request']; messages = request['messages']
        assert len(messages) == 2 and all('images' not in m for m in messages)
        assert messages[0] == {'role': 'system', 'content': record['system_prompt']}
        assert messages[1] == {'role': 'user', 'content': record['user_prompt']}
        r['request_without_messages'] = {k: v for k, v in request.items() if k != 'messages'}
        r['request_message_references'] = [
            {'role': 'system', 'path': r['system_prompt_path'], 'text_transform': 'strip'},
            {'role': 'user', 'path': r['user_prompt_path']}]
    return r


def decisions(record):
    if record is None or record['status'] != 'ok': return None
    return {typed(o['id']): o for o in record['parsed']['objects']}


def direct_pair(left, right, kind):
    """Compare the same frozen input: a local ID has a documented shared scope."""
    base = {'kind': kind, 'input_index': left['input_index'], 'frame_id': left['frame_id'],
            'vlm': left['vlm'], 'vlm_repeat': left['vlm_repeat'],
            'left': [left['llm'], left['prompt_name'], left['llm_repeat']],
            'right': [right['llm'], right['prompt_name'], right['llm_repeat']],
            'left_status': left['status'], 'right_status': right['status'],
            'same_frozen_input_sha256': left['caption_sha256'] == right['caption_sha256']}
    a, b = decisions(left), decisions(right)
    comparable = base['same_frozen_input_sha256'] and a is not None and b is not None and a.keys() == b.keys()
    base.update(comparable=comparable, compared_objects=0, class_changes=0, reason_changes=0,
                class_agreement=None, raw_equal=None)
    if comparable:
        changed = [{'id': a[k]['id'], 'left': a[k], 'right': b[k]} for k in a if a[k] != b[k]]
        class_changes = sum(a[k]['classification'] != b[k]['classification'] for k in a)
        base.update(compared_objects=len(a), class_changes=class_changes,
                    reason_changes=sum(a[k]['reason'] != b[k]['reason'] for k in a),
                    class_agreement=(len(a)-class_changes)/len(a) if a else None,
                    raw_equal=left.get('content') == right.get('content'), changes=changed)
    return base


def name_pair(left, right, frozen, kind):
    """Conservative name candidates across documents; never match their IDs."""
    base = {'kind': kind, 'frame_id': left['frame_id'], 'llm': left['llm'],
            'prompt': left['prompt_name'], 'llm_repeat': left['llm_repeat'],
            'left_input': left['input_index'], 'right_input': right['input_index'],
            'left_vlm': left['vlm'], 'left_vlm_repeat': left['vlm_repeat'],
            'right_vlm': right['vlm'], 'right_vlm_repeat': right['vlm_repeat'],
            'left_status': left['status'], 'right_status': right['status'],
            'matching': 'UNCONFIRMED unique exact normalized names only; no physical identity established',
            'comparable_outputs': False, 'candidate_name_pairs': 0,
            'candidate_class_changes': None, 'candidate_agreement': None,
            'left_unmatched_names': None, 'right_unmatched_names': None, 'ambiguous_names': None}
    a, b = decisions(left), decisions(right)
    if a is None or b is None: return base
    by_name = []
    for record, predicted in ((left, a), (right, b)):
        groups = defaultdict(list)
        for obj in frozen[record['input_index']]['document']['parsed']['objects']:
            groups[label_key(obj['label'])].append((obj, predicted[typed(obj['id'])]))
        by_name.append(groups)
    x, y = by_name
    common = sorted(set(x) & set(y)); unique = [n for n in common if len(x[n]) == len(y[n]) == 1]
    pairs = [{'name': n, 'left_object_id': x[n][0][0]['id'], 'right_object_id': y[n][0][0]['id'],
              'left_label': x[n][0][0]['label'], 'right_label': y[n][0][0]['label'],
              'left_observation':x[n][0][0].get('observation'),'right_observation':y[n][0][0].get('observation'),
              'left': x[n][0][1], 'right': y[n][0][1]} for n in unique]
    changes = sum(p['left']['classification'] != p['right']['classification'] for p in pairs)
    base.update(comparable_outputs=True, candidate_name_pairs=len(pairs), candidate_class_changes=changes,
        candidate_agreement=(len(pairs)-changes)/len(pairs) if pairs else None,
        left_unmatched_names=sorted(set(x)-set(unique)), right_unmatched_names=sorted(set(y)-set(unique)),
        ambiguous_names=[n for n in common if n not in unique], candidate_pairs=pairs)
    return base


def comparisons(records, frozen):
    lookup = {(r['input_index'], r['llm'], r['prompt_name'], r['llm_repeat']): r for r in records}
    source_lookup = {input_key(s): n for n, s in frozen.items()}
    results = defaultdict(list)
    llms = [m['name'] for m in MODELS]
    def compare(a, b, kind, names=False):
        if a in lookup and b in lookup:
            l, r = lookup[a], lookup[b]
            results[kind].append(name_pair(l, r, frozen, kind) if names else direct_pair(l, r, kind))
    for number in frozen:
        for llm, prompt in product(llms, PROMPTS):
            for a, b in combinations(SEEDS, 2): compare((number,llm,prompt,a),(number,llm,prompt,b),'llm-repeats')
        for llm, repeat in product(llms, SEEDS): compare((number,llm,'short',repeat),(number,llm,'long',repeat),'prompts')
        for prompt, repeat in product(PROMPTS, SEEDS):
            for a, b in combinations(llms,2): compare((number,a,prompt,repeat),(number,b,prompt,repeat),'llms')
    frames = sorted({s['frame_id'] for s in frozen.values()})
    for frame in frames:
        for llm, prompt, repeat in product(llms,PROMPTS,SEEDS):
            for vlm in VLMS:
                for a,b in combinations(SEEDS,2):
                    i,j=source_lookup[(frame,vlm,a)],source_lookup[(frame,vlm,b)]
                    compare((i,llm,prompt,repeat),(j,llm,prompt,repeat),'vlm-repeats',True)
            for vlm_repeat in SEEDS:
                for a,b in combinations(VLMS,2):
                    i,j=source_lookup[(frame,a,vlm_repeat)],source_lookup[(frame,b,vlm_repeat)]
                    compare((i,llm,prompt,repeat),(j,llm,prompt,repeat),'vlms',True)
    return results


def aggregate(records, frozen):
    groups = defaultdict(list)
    for r in records: groups[(r['llm'],r['prompt_name'],r['vlm'],r['vlm_repeat'],r['llm_repeat'])].append(r)
    rows = []
    for key, values in sorted(groups.items()):
        times = [r['elapsed_seconds'] for r in values if 'request' in r and 'elapsed_seconds' in r]
        counts = Counter(o['classification'] for r in values if r['status']=='ok' for o in r['parsed']['objects'])
        reason_words=[len(o['reason'].split()) for r in values if r['status']=='ok' for o in r['parsed']['objects']]
        rows.append(dict(zip(('llm','prompt','vlm','vlm_repeat','llm_repeat'),key)) | {
            'positions':len(values), **{s:sum(r['status']==s for r in values) for s in ('ok','error','skipped')},
            'attempted_calls':sum('request' in r for r in values),
            'completed_response_records':sum('response' in r for r in values),
            'attempts_without_full_response':sum('request' in r and 'response' not in r for r in values),
            'gpu_verified':sum(bool(r.get('gpu_verified')) for r in values),
            'mean_seconds':statistics.mean(times) if times else None,
            'median_seconds':statistics.median(times) if times else None,
            'contexts_used':dict(Counter(str(r.get('requested_options',{}).get('num_ctx')) for r in values if 'request' in r)),
            'max_reported_model_vram_bytes':max((r.get('gpu',{}).get('size_vram',0) for r in values),default=0),
            'mean_decode_seconds':statistics.mean(r['response']['eval_duration']/1e9 for r in values if 'eval_duration' in r.get('response',{})) if any('eval_duration' in r.get('response',{}) for r in values) else None,
            'fixed':counts['fixed'],'movable':counts['movable'],'uncertain':counts['uncertain'],
            'mean_reason_words':statistics.mean(reason_words) if reason_words else None,
            'max_reason_words':max(reason_words,default=0),'reasons_over_12_words':sum(n>12 for n in reason_words),
            'accuracy':None, 'note':'Class counts and agreement are not correctness; native regions can overlap.'})
    return rows


def flat_tables(records,frozen):
    """One call row per position/attempt; object decisions remain scoped by SHA."""
    calls=[];predictions=[]
    def call_row(r,role):
        source=frozen[r['input_index']];response=r.get('response',{});slots=runtime_slots(r)
        return {'attempt_role':role,'input_index':r['input_index'],'frame_id':r['frame_id'],
            'vlm':r['vlm'],'vlm_repeat':r['vlm_repeat'],'llm':r['llm'],'prompt':r['prompt_name'],
            'llm_repeat':r['llm_repeat'],'frozen_sha256':source['sha256'],'status':r['status'],
            'call_attempted':'request' in r,'response_received':'response' in r,
            'num_ctx':r.get('requested_options',{}).get('num_ctx'),'seed':r['variant']['seed'],
            'parallel_slots_requested':slots[0] if 'request' in r else None,
            'parallel_slots_actual':slots[1] if 'request' in r else None,
            'elapsed_http_seconds':r.get('elapsed_seconds'),'prompt_tokens':response.get('prompt_eval_count'),
            'output_tokens':response.get('eval_count'),
            **{name+'_seconds':response[field]/1e9 if field in response else None for name,field in
               [('load','load_duration'),('prefill','prompt_eval_duration'),('decode','eval_duration'),('server_total','total_duration')]},
            'model_vram_bytes':r.get('gpu',{}).get('size_vram'),'model_digest':r.get('gpu',{}).get('digest'),
            'gpu_verified':r.get('gpu_verified') if 'request' in r else None,'error':r.get('error'),
            'response_created_at':response.get('created_at')}
    for r in records:
        call=call_row(r,'main');calls.append(call)
        common={k:call[k] for k in ('input_index','frame_id','vlm','vlm_repeat','llm','prompt','llm_repeat','frozen_sha256','status','error')}
        if r['status']!='ok':
            predictions.append({**common,**{k:None for k in ('object_id_json','object_id_type','label','source_task','source_region_id','classification','reason')}})
            continue
        source={typed(o['id']):o for o in frozen[r['input_index']]['document']['parsed']['objects']}
        for prediction in r['parsed']['objects']:
            obj=source[typed(prediction['id'])]
            predictions.append({**common,'object_id_json':json.dumps(obj['id'],ensure_ascii=False),
                'object_id_type':type(obj['id']).__name__,'label':obj['label'],
                'source_task':obj.get('source_task'),'source_region_id':obj.get('source_region_id'),
                'classification':prediction['classification'],'reason':prediction['reason']})
    diagnostic_root=OUT/'diagnostics/context-16384'
    if not diagnostic_root.exists():diagnostic_root=TARGET/'diagnostics/context-16384'
    for path in diagnostic_root.glob('*/*/*/*/*/frame-*.json'):
        r=read_json(path);parts=path.relative_to(diagnostic_root).parts
        number=int(path.stem.split('-')[-1]);s=frozen[number]
        r.update(input_index=number,vlm=s['vlm'],vlm_repeat=s['vlm_repeat'],llm=parts[0],prompt_name=parts[1],llm_repeat=parts[2])
        calls.append(call_row(r,'context-diagnostic'))
    csv_file(TARGET/'calls.csv',calls)
    csv_file(TARGET/'classifications.csv',predictions)


def render(index, frozen, records):
    by_input = defaultdict(dict)
    for r in records: by_input[r['input_index']][(r['llm'],r['prompt_name'],r['llm_repeat'])]=r
    frames = list(dict.fromkeys(r['frame_id'] for r in index))
    manifest = read_json(SOURCE/'manifest.json'); frame_info={f['frame_id']:f for f in manifest['frames']}
    states=Counter(r['status'] for r in records)
    top='<h1>Drugi korak: razvrščanje zamrznjenih opisov</h1><p>VLM ponovitev in LLM ponovitev sta ločeni. LLM ne prejme slike. Strinjanje ni pravilnost.</p>'
    top+=f'<p>Shranjenih položajev: {len(records)}/3240. Veljavni po shemi: {states["ok"]}, napake: {states["error"]}, preskoki: {states["skipped"]}. Glavnih inferenčnih klicev: {sum("request" in r for r in records)}; arhiviranih neuspešnih kontekstnih poskusov: 9.</p><nav>'
    for name in ('README.md','plan.json','summary.csv','calls.csv','classifications.csv','comparison-summary.csv','manual-review-empty.csv','verification.json','frozen/index.json','execution-context-policy.json','diagnostics/index.json'):
        top+=f'<a href="{name}">{name}</a> '
    top+='</nav><p>Podrobne tabele razlik: '
    for name in ('llm-repeats','prompts','llms','vlm-repeats','vlms'):
        top+=f'<a href="comparisons/{name}.csv">{name}</a> '
    top+='</p><h2>Slike</h2><nav>'
    for frame in frames:
        slug=frame.replace('/','--');top+=f'<a href="images/{slug}.html">{escaped(frame)}</a>'
        info=frame_info[frame];image=info['published_original']
        body=f'<p><a href="../index.html">← Vse slike</a> · <a href="../../index.html">Prvi korak</a></p><h1>{escaped(frame)}</h1>'
        body+=f'<a href="../../{image}"><img src="../../{image}" alt="Original za ročni pregled; LLM te slike ne dobi"></a>'
        body+='<p>Slika samo za vaš pregled; LLM prejme izključno spodnje zamrznjeno besedilo.</p>'
        for vlm in VLMS:
            body+='<h2>'+escaped(vlm)+'</h2>'
            for vlm_repeat in SEEDS:
                source=next(s for s in frozen.values() if input_key(s)==(frame,vlm,vlm_repeat));number=source['input_index']
                filename=f'{slug}--{vlm}--{vlm_repeat}.html'
                body+=f'<a href="../inputs/{filename}">{escaped(vlm)} VLM-{vlm_repeat} · {source["object_entries"]} regij/objektnih zapisov · {escaped(source["status"])}</a><br>'
                detail=f'<p><a href="../images/{slug}.html">← Slika in izvori</a></p><h1>{escaped(frame)} / {vlm} / VLM-{vlm_repeat}</h1>'
                detail+='<details><summary>Izvorna slika za človekov pregled; ni vhod LLM</summary>'
                detail+=f'<a href="../../{image}"><img src="../../{image}" alt="Original samo za ročni pregled"></a>'
                if vlm=='florence':
                    directory=image.rsplit('/',1)[0]
                    for task in ('od','dense'):
                        predicted=f'../../{directory}/florence/{vlm_repeat}/{task}.predictions.jpg'
                        detail+=f'<p>Florence {task}: native ID regije znotraj te naloge; transportni ID je v zamrznjenem dokumentu.</p><a href="{predicted}"><img src="{predicted}" alt="Florence napovedi za ročni pregled"></a>'
                detail+='</details>'
                detail+=f'<p>SHA256 zamrznjenega vhoda: {source["sha256"]}</p><p><a href="../frozen/frame-{number:06d}.json">Celoten zamrznjeni vhod</a></p>'
                if source['status']!='ok':detail+='<p class="note">Preskok vseh 18 kombinacij: '+escaped(source['error'])+'</p>'
                elif source['partial_input']:detail+='<p class="note">Delni nevalidirani NVILA vhod: celoten raw tekst ohranjen, seznam dobesedno izluščen; trditve niso preverjene.</p>'
                detail+='<details><summary>Celoten vhodni opis in seznam</summary>'+pre(source['document'])+'</details><div class="columns">'
                for spec in MODELS:
                    llm=spec['name'];detail+='<section class="model"><h2>'+escaped(spec['model'])+'</h2>'
                    for prompt in PROMPTS:
                        detail+='<h3>'+('Kratki' if prompt=='short' else 'Dolgi')+' prompt brez primerov</h3>'
                        if source['status']=='ok':
                            detail+='<table><tr><th>Lokalni ID / oznaka</th><th>LLM-r1</th><th>LLM-r2</th><th>LLM-r3</th></tr>'
                            for obj in source['document']['parsed']['objects']:
                                detail+='<tr><td>'+escaped(obj['id'])+' / '+escaped(obj['label'])+'</td>'
                                for repeat in SEEDS:
                                    value=by_input[number].get((llm,prompt,repeat));predicted=decisions(value)
                                    decision=predicted.get(typed(obj['id'])) if predicted is not None else None
                                    detail+='<td>'+(escaped(decision['classification'])+'<br>'+escaped(decision['reason']) if decision else escaped(value['status'] if value else 'pending'))+'</td>'
                                detail+='</tr>'
                            detail+='</table>'
                        for llm_repeat in SEEDS:
                            r=by_input[number].get((llm,prompt,llm_repeat));detail+='<article><h3>LLM-'+llm_repeat+' · seed '+str(SEEDS[llm_repeat])+'</h3>'
                            if r is None: detail+='<p>Manjkajoč položaj.</p></article>';continue
                            leaf=destination(source,llm,prompt).relative_to(TARGET)
                            detail+='<p>Status '+escaped(r['status'])+(' · '+f'{r["elapsed_seconds"]:.3f} s' if 'elapsed_seconds' in r else '')+'</p>'
                            if r.get('error'):detail+='<p class="note">'+escaped(r['error'])+'</p>'
                            if r['status']=='ok':
                                labels={typed(o['id']):o['label'] for o in source['document']['parsed']['objects']}
                                detail+='<table><tr><th>ID / oznaka</th><th>Odločitev</th><th>Celoten razlog</th></tr>'
                                for obj in r['parsed']['objects']:
                                    detail+='<tr><td>'+escaped(obj['id'])+' / '+escaped(labels[typed(obj['id'])])+'</td><td>'+escaped(obj['classification'])+'</td><td>'+escaped(obj['reason'])+'</td></tr>'
                                detail+='</table>'
                            detail+='<details><summary>Celoten surovi odgovor</summary>'+pre(r.get('content',''))+'</details>'
                            detail+=f'<p><a href="../{leaf.parent}/responses.json">Vseh šest odzivov tega LLM, GPU, nastavitve, izvor</a>'
                            if r['status']!='skipped':
                                detail+=f' · <a href="../user-prompts/input-{number:06d}-{prompt}.txt">Dejanski uporabniški prompt</a> · <a href="../prompts/{prompt}-no-examples.txt">Sistemski prompt</a>'
                            detail+='</p></article>'
                    detail+='</section>'
                detail+='</div>'
                (TARGET/'inputs'/filename).write_text(document(frame+' '+vlm+' '+vlm_repeat,detail))
        (TARGET/'images'/(slug+'.html')).write_text(document(frame,body))
    top+='</nav>'
    (TARGET/'index.html').write_text(document('Drugi korak: primerjava',top))


def verify():
    index,frozen,records=load();plan=read_json(TARGET/'plan.json')
    assert len(index)==180 and len({s['frame_id'] for s in frozen.values()})==20
    for frame in {s['frame_id'] for s in frozen.values()}:
        assert {(s['vlm'],s['vlm_repeat']) for s in frozen.values() if s['frame_id']==frame}==set(product(VLMS,SEEDS))
    for source in frozen.values():
        objects=source['document'].get('parsed',{}).get('objects',[])
        assert len({typed(o['id']) for o in objects})==len(objects)
    expected={(n,m['name'],p,r) for n in frozen for m in MODELS for p in PROMPTS for r in SEEDS}
    actual={(r['input_index'],r['llm'],r['prompt_name'],r['llm_repeat']) for r in records}
    assert actual==expected and len(records)==plan['logical_positions']==3240
    counts=Counter(r['status'] for r in records);attempted=sum('request' in r for r in records)
    assert counts['skipped']==plan['planned_skips']==144 and attempted==plan['planned_measured_calls']==3096
    assert all(r.get('gpu_verified') for r in records if r['status']=='ok')
    digests={spec['name']:sorted({r.get('gpu',{}).get('digest') for r in records
        if r['llm']==spec['name'] and r.get('gpu',{}).get('digest')}) for spec in MODELS}
    assert all(len(values)==1 for values in digests.values()), 'Model revision changed during comparison'
    identities=[(r['model'],r['response']['created_at']) for r in records if 'response' in r]
    assert len(set(identities))==len(identities), 'Repeated physical response identity'
    diagnostics=[]
    diagnostic_root=OUT/'diagnostics/context-16384'
    if not diagnostic_root.exists():diagnostic_root=TARGET/'diagnostics/context-16384'
    for path in diagnostic_root.glob('*/*/*/*/*/frame-*.json'):
        old=read_json(path);assert old['status']=='error', 'A successful call was repeated'
        diagnostics.append(old)
    assert len(diagnostics)==9
    all_ids=identities+[(r['model'],r['response']['created_at']) for r in diagnostics if 'response' in r]
    assert len(set(all_ids))==len(all_ids)
    policy=read_json(TARGET/'execution-context-policy.json')['overrides_by_frozen_sha256']
    response_index=read_json(TARGET/'responses-index.json')
    assert len(response_index)==540 and sum(r['positions'] for r in response_index)==3240
    assert all(r['complete_six_positions'] and sha256((TARGET/r['path']).read_bytes())==r['sha256'] for r in response_index)
    for r in records:
        source=frozen[r['input_index']];leaf=destination(source,r['llm'],r['prompt_name'])
        published=read_json(leaf.parent/'responses.json')[r['prompt_name']][r['llm_repeat']]
        if 'response' in r:assert published['response']==r['response']
        if r['status']!='skipped':
            system=(TARGET/'prompts'/(r['prompt_name']+'-no-examples.txt')).read_text().strip()
            user=(TARGET/'user-prompts'/f'input-{r["input_index"]:06d}-{r["prompt_name"]}.txt').read_text()
            assert user==r['user_prompt'] and system==r['system_prompt']
            payload=json.loads(user.split('\n',1)[1])
            assert payload['visual_observations']==source['document']['parsed']
            assert payload['frame_id']==source['frame_id']
            assert payload['objects']==[{'id':o['id'],'label':o['label']} for o in source['document']['parsed']['objects']]
            expected_ctx=policy.get(source['sha256'],{}).get('num_ctx',16384)
            assert r['requested_options']['num_ctx']==expected_ctx
            expected_options={'temperature':.3,'seed':SEEDS[r['llm_repeat']],'num_predict':4096,
                'num_ctx':expected_ctx,'top_p':.8,'top_k':20,'presence_penalty':1.5,'num_gpu':999}
            assert r['request']['options']==r['requested_options']==expected_options
            assert r['request']['format']=='json' and r['request']['think'] is False and r['request']['stream'] is False
        assert published['frozen_input_sha256']==source['sha256']
    from html.parser import HTMLParser
    from urllib.parse import unquote
    links=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):links.extend(v for k,v in attrs if k in ('href','src'))
    link_count=0
    for path in TARGET.rglob('*.html'):
        links.clear();Links().feed(path.read_text())
        for link in links:
            assert (path.parent/unquote(link)).is_file(),(path,link)
            link_count+=1
    with (TARGET/'manual-review-empty.csv').open() as stream:
        reviews=list(csv.DictReader(stream))
    assert len(reviews)==3240
    assert all(not value for row in reviews for key,value in row.items() if key.startswith('review'))
    with (TARGET/'calls.csv').open() as stream:call_rows=list(csv.DictReader(stream))
    assert len(call_rows)==3249 and sum(r['call_attempted']=='True' for r in call_rows)==3105
    assert sum(r['attempt_role']=='main' for r in call_rows)==3240
    assert all(r['parallel_slots_actual'] for r in call_rows if r['call_attempted']=='True')
    summary=read_json(TARGET/'summary.json')
    assert len(summary)==162 and all(r['positions']==20 for r in summary)
    comparison_counts={'llm-repeats':3240,'prompts':1620,'llms':3240,'vlm-repeats':3240,'vlms':3240}
    for kind,expected_count in comparison_counts.items():
        with (TARGET/'comparisons'/(kind+'.jsonl')).open() as stream:
            assert sum(1 for _ in stream)==expected_count
    protected_path=OUT/'setup/stage1-preservation.json'
    if not protected_path.exists():protected_path=TARGET/'provenance/stage1-preservation.json'
    protected=read_json(protected_path)
    assert all(sha256((ROOT/path).read_bytes())==digest for path,digest in protected.items())
    result={'complete':True,'logical_positions':len(records),'actual_measured_calls':attempted+len(diagnostics),
        'main_calls':attempted,'archived_failed_context_attempts':len(diagnostics),'successful_calls_repeated':0,
        'statuses':dict(counts),'inputs':len(index),'usable_inputs':plan['usable_inputs'],
        'skipped_inputs':plan['unusable_inputs'],'skipped_positions':counts['skipped'],
        'pilots_included_in_measured_calls':3,'physical_warmups':0,
        'unique_physical_response_ids':len(set(identities)), 'html_links_verified':link_count,
        'model_digests':{name:values[0] for name,values in digests.items()},
        'unique_response_ids_including_diagnostics':len(set(all_ids)),
        'main_completed_response_records':len(identities),
        'main_attempts_without_full_response':attempted-len(identities),
        'completed_response_records_including_diagnostics':len(all_ids),
        'api_call_attempts_including_diagnostics':attempted+len(diagnostics),
        'manual_review_rows':len(reviews),'manual_review_fields_all_empty':True,
        'comparison_pairs':comparison_counts,
        'stage1_files_unchanged':len(protected),
        'stage1_calls':0,'images_sent_to_llms':False,'reference_stability_labels_used':False,
        'pdf_generated':False,'agreement_is_accuracy':False}
    write_json(TARGET/'verification.json',result);print(json.dumps(result,indent=2));return result


def export(require_complete=True):
    for folder in ('user-prompts','images','inputs','comparisons'): (TARGET/folder).mkdir(parents=True,exist_ok=True)
    write_json(TARGET/'verification.json',{'complete':False,'verification_in_progress':True})
    index,frozen,records=load()
    instruction=(TARGET/'prompts/reasoning.txt').read_text().strip()
    for number,source in frozen.items():
        if source['status']!='ok':continue
        caption=source['document']['parsed'];objects=[{'id':o['id'],'label':o['label']} for o in caption['objects']]
        prompt=reason_prompt(source['frame_id'],objects,caption,instruction)
        for name in PROMPTS:(TARGET/'user-prompts'/f'input-{number:06d}-{name}.txt').write_text(prompt)
    bundles=defaultdict(lambda:defaultdict(dict))
    for r in records:
        source=frozen[r['input_index']];path=destination(source,r['llm'],r['prompt_name']).parent/'responses.json'
        public=public_record(r,source,TARGET/'user-prompts'/f'input-{r["input_index"]:06d}-{r["prompt_name"]}.txt',TARGET/'prompts'/(r['prompt_name']+'-no-examples.txt')) if r['status']!='skipped' else {
            **r,'frozen_input_path':source['path'],'frozen_input_sha256':source['sha256']}
        bundles[path][r['prompt_name']][r['llm_repeat']]=public
    for path,bundle in bundles.items():write_json(path,bundle)
    write_json(TARGET/'responses-index.json',[{'path':str(path.relative_to(TARGET)),
        'sha256':sha256(path.read_bytes()),'positions':sum(len(runs) for runs in bundle.values()),
        'complete_six_positions':set(bundle)==set(PROMPTS) and all(set(runs)==set(SEEDS) for runs in bundle.values())}
        for path,bundle in bundles.items()])
    for path in (OUT/'runs').glob('*/*/*/manifest.json'):
        name='-'.join(path.parts[-4:-1])+'.json'
        manifest_path=TARGET/'provenance/run-manifests'/name
        manifest_path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,manifest_path)
    for path in (OUT/'diagnostics/context-16384').glob('*/*/*/*/*/frame-*.json'):
        archived=TARGET/'diagnostics/context-16384'/path.relative_to(OUT/'diagnostics/context-16384')
        archived.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,archived)
    context_policy=read_json(TARGET/'execution-context-policy.json')
    diagnostic_index=[]
    for item in context_policy['archived_failed_attempts']:
        path=ROOT/item['archived_path']
        relative=path.relative_to(OUT/'diagnostics/context-16384')
        if not path.exists():path=TARGET/'diagnostics/context-16384'/relative
        record=read_json(path)
        assert sha256(path.read_bytes())==item['sha256'] and record['status']=='error'
        number=int(path.stem.split('-')[-1]);source=frozen[number]
        diagnostic_index.append({**item,'input_index':number,'frame_id':source['frame_id'],
            'vlm':source['vlm'],'vlm_repeat':source['vlm_repeat'],
            'published_path':str((TARGET/'diagnostics/context-16384'/relative).relative_to(TARGET)),
            'excluded_from_main_comparisons':True,'original_num_ctx':record['requested_options']['num_ctx'],
            'replacement_num_ctx':32768,'no_successful_response_repeated':True})
    write_json(TARGET/'diagnostics/index.json',diagnostic_index)
    for name in ('runtime.json','runtime-after-context.json','inference-source-before-context.json','checkpoint-paused.json','stage1-preservation.json',
                 'inference-sources-before-parallel.json','inference-sources-parallel.json','runtime-parallel.json','checkpoint-before-parallel.json'):
        path=OUT/'setup'/name
        if path.exists():shutil.copyfile(path,TARGET/'provenance'/name)
    for name in ('gpu-samples.jsonl','finished.json','logs/unit-tests.log','logs/stage-tests.log'):
        path=OUT/name
        if path.exists():shutil.copyfile(path,TARGET/'provenance'/path.name)
    for name in ('docker-image.txt','nvidia-smi.txt','parallel-runtime-evidence.txt','gpu-placement-evidence.txt'):
        path=OUT/'setup'/name
        if path.exists():write_json(TARGET/'provenance'/(path.stem+'.json'),
            {'source_sha256':sha256(path.read_bytes()),'stdout':path.read_text()})
    summary=aggregate(records,frozen);write_json(TARGET/'summary.json',summary);csv_file(TARGET/'summary.csv',summary)
    flat_tables(records,frozen)
    diffs=comparisons(records,frozen);overview=[]
    for kind,rows in diffs.items():
        # JSONL preserves every changed decision/reason without huge redundant CSV cells.
        with (TARGET/'comparisons'/(kind+'.jsonl')).open('w') as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        compact=[{k:v for k,v in r.items() if k not in ('changes','candidate_pairs')} for r in rows]
        csv_file(TARGET/'comparisons'/(kind+'.csv'),compact)
        usable=[r for r in rows if r.get('comparable',r.get('comparable_outputs',False))]
        changes=sum(r.get('class_changes',r.get('candidate_class_changes',0)) or 0 for r in usable)
        objects=sum(r.get('compared_objects',r.get('candidate_name_pairs',0)) for r in usable)
        overview.append({'kind':kind,'pairs':len(rows),'comparable_pairs':len(usable),
            'uncomparable_pairs':len(rows)-len(usable),'object_comparisons':objects,'class_changes':changes,
            'agreement':(objects-changes)/objects if objects else None,
            'matching':'same frozen input, exact typed IDs' if kind in ('llm-repeats','prompts','llms') else 'UNCONFIRMED unique normalized name candidates; not instance identity',
            'accuracy':None})
    csv_file(TARGET/'comparison-summary.csv',overview);write_json(TARGET/'comparison-summary.json',overview)
    review=[{'input_index':r['input_index'],'frame_id':r['frame_id'],'vlm':r['vlm'],'vlm_repeat':r['vlm_repeat'],
        'llm':r['llm'],'prompt':r['prompt_name'],'llm_repeat':r['llm_repeat'],'status':r['status'],
        'reviewer':'','review_class_correctness':'','review_reason_quality':'','review_source_error':'','review_notes':''} for r in records]
    review_path=TARGET/'manual-review-empty.csv'
    if review_path.exists():
        with review_path.open() as stream:previous=list(csv.DictReader(stream))
        if any(value for row in previous for key,value in row.items() if key.startswith('review')):
            # Keep any human edits made during a long-running preview export.
            saved=OUT/'manual-review-preserved'/('manual-review-saved-'+sha256(review_path.read_bytes())[:12]+'.csv')
            saved.parent.mkdir(parents=True,exist_ok=True)
            if not saved.exists():shutil.copyfile(review_path,saved)
    csv_file(review_path,review)
    render(index,frozen,records)
    if require_complete:
        verify()
    else:
        write_json(TARGET/'verification.json',{'complete':False,'positions_so_far':len(records),
            'actual_calls_so_far':sum('request' in r for r in records),
            'statuses_so_far':dict(Counter(r['status'] for r in records)),
            'note':'Preview during GPU execution; final verification has not run.'})
