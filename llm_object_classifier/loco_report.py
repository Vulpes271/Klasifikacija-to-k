"""LOCO publication/evaluation glue around the existing native/text exporters.

Only the offline evaluation below reads labels. No inference helper imports them.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import re
import shutil
import statistics
import subprocess

from .experiment import read_json, write_json, sha256
from .thud_experiment import immutable_json
from .thud_extended_common import ROOT, native_selection
from . import thud_stage1_presentation as native
from . import thud_stage2_presentation as stage
from .thud_extended_report import repetition_consistency, tex, table
from .thud_box_metrics import match_regions
from .vision_prompts import reason_prompt
from .loco_experiment import OUT, PUBLIC, VLM, LLM


def frames():
    return native_selection(OUT/'inputs.json')


def normalize(label):
    return ' '.join(label.lower().replace('_',' ').replace('-',' ').split())


def empty_review(path, rows):
    """Do not lose human drafts when regenerating an explicitly empty form."""
    if path.exists():
        with path.open() as stream:previous=list(csv.DictReader(stream))
        if any(v for r in previous for k,v in r.items() if k.startswith('review')):
            saved=OUT/'manual-review-preserved'/('saved-'+sha256(path.read_bytes())[:16]+'.csv')
            saved.parent.mkdir(parents=True,exist_ok=True)
            if not saved.exists():shutil.copyfile(path,saved)
    native.csv_file(path,rows)


def literal_mentions(text, aliases):
    """Lexical mentions, never detections; longest overlapping term wins."""
    text=normalize(text);candidates=[]
    for label,canonical in aliases.items():
        for match in re.finditer(r'(?<!\w)'+re.escape(label)+r'(?!\w)',text):
            candidates.append((match.start(),match.end(),canonical))
    used=[];classes=set()
    for start,end,label in sorted(candidates,key=lambda x:(x[0],-(x[1]-x[0]))):
        if any(start<b and end>a for a,b in used):continue
        used.append((start,end));classes.add(label)
    return sorted(classes)


def export_vlm():
    selected=frames();roots={m:OUT/'vlm'/m for m in native.MODELS}
    configs=native.audit_protocol(selected,roots)
    VLM.mkdir(parents=True,exist_ok=True)
    records,sources,combined=native.collect(selected,configs,target=VLM,roots=roots,reused=set(),copy_original=False)
    portable=[{**f,'image_path':str(Path(f['image_path']).relative_to(ROOT)),
        'published_original':os.path.relpath(f['image_path'],VLM),'published_directory':native.leaf_name(f)} for f in selected]
    write_json(VLM/'manifest.json',{'experiment':'LOCO industrial','frames':portable,
        'models':native.MODELS,'revisions':native.REVISIONS,'configs':configs,'all_generations_new':True})
    write_json(VLM/'provenance.json',sources)
    summary=native.summarize(records,combined,reused=set())
    write_json(VLM/'summary.json',summary);native.csv_file(VLM/'summary.csv',summary)
    differences=repetition_consistency(records,aliases=read_json(OUT/'setup/class-map.json')['aliases'])
    for r in differences:
        if not all(r['object_list_available']):
            for key in ('name_jaccard','name_intersection','name_union','names_added','names_removed','name_group_details_changed','possible_alias_changes'):r[key]=None
    write_json(VLM/'repeat-differences.json',differences)
    native.csv_file(VLM/'repeat-differences.csv',differences,list(dict.fromkeys(k for r in differences for k in r)))
    metrics=[{'frame_id':r['frame_id'],'model':r['model_name'],'repeat':r['repeat'],'task':r['task'],
        'call_id':r['call_id'],'status':r['status'],'error':r.get('error'),'seed':r['seed'],
        'characters':len(r.get('content','')),'output_tokens':r.get('output_tokens'),
        'object_entries':len(r['view']['objects']) if r['view'].get('object_list_available') else None,
        'elapsed_seconds':r['elapsed_seconds'],'peak_allocated_bytes':r['peak_allocated_bytes'],
        'peak_reserved_bytes':r['peak_reserved_bytes'],'gpu_verified':r['gpu_verified']} for r in records]
    native.csv_file(VLM/'response-metrics.csv',metrics)
    detail=[]
    for model,task in sorted({(r['model_name'],r['task']) for r in records}):
        rs=[r for r in records if r['model_name']==model and r['task']==task]
        objects=[len(r['view']['objects']) for r in rs if r['view'].get('object_list_available')]
        scenes=[r['view']['scene_description'] for r in rs if isinstance(r['view'].get('scene_description'),str) and r['view']['scene_description']]
        detail.append({'model':model,'task':task,'positions':len(rs),'content_available':sum(bool(r.get('content')) for r in rs),
            'literal_list_available':len(objects),'median_object_or_region_entries':statistics.median(objects) if objects else None,
            'median_native_text_characters':statistics.median(len(r.get('content','')) for r in rs),
            'median_scene_words_when_available':statistics.median(len(s.split()) for s in scenes) if scenes else None,
            'note':'Length/list availability only, not quality or confirmed physical object counts; native task formats differ.'})
    write_json(VLM/'detail-summary.json',detail);native.csv_file(VLM/'detail-summary.csv',detail)
    review=[{'frame_id':f['frame_id'],'model':m,'vlm_repeat':r,'reviewer':'','review_description':'',
        'review_objects':'','review_locations_relations':'','review_notes':''} for f in selected for m in native.MODELS for r in stage.SEEDS]
    empty_review(VLM/'manual-review-empty.csv',review)
    for model in native.MODELS:
        for repeat in stage.SEEDS:
            src=roots[model]/repeat;dst=VLM/'provenance'/model/repeat
            dst.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/'manifest.json',dst/'manifest.json')
            for folder in ('source-snapshots','environments'):
                if (src/folder).exists():shutil.copytree(src/folder,dst/folder,dirs_exist_ok=True)
    shutil.copytree(OUT/'vlm/environments',VLM/'provenance/qwen3-vl/environments',dirs_exist_ok=True)
    loads=[]
    for model in native.MODELS:
        paths=list((OUT/'vlm/environments').glob('*.json')) if model=='qwen3-vl' else list((roots[model]).glob('*/environments/*.json'))
        envs=[read_json(p) for p in paths]
        loads.append({'model':model,'load_sessions':len(envs),'load_seconds_sum':sum(e['load_seconds'] for e in envs),
            'load_seconds_each':[e['load_seconds'] for e in envs],
            'parameter_dtypes_each':[e['parameter_dtypes'] for e in envs],
            'runtime':{k:envs[0].get(k) for k in ('python','torch','cuda','transformers','gpu_name','gpu_capability','compiled_arches')},
            'note':'Loading/initialization/environment checks, separate from image inference. Includes pilot and resume loads. No extra warmup generations.'})
    write_json(VLM/'load-summary.json',loads);native.csv_file(VLM/'load-summary.csv',loads)
    tails=[]
    for r in stage.SEEDS:
        envs=[read_json(p) for p in (roots['nvila']/r/'environments').glob('*.json')]
        assert envs and all(e['initialization_seed']==42 for e in envs)
        tails.extend(e['embedding_tail_sha256'] for e in envs)
    assert all(t==tails[0] for t in tails),'NVILA initialization changed between repetitions'
    write_json(VLM/'nvila-initialization-audit.json',{'initialization_seed':42,'all_embedding_tails_equal':True,'embedding_tail_sha256':tails[0],'sessions':len(tails)})
    failures=[]
    for path in (OUT/'setup').glob('*process-exit-*.json'):
        value=read_json(path);failures.append(value)
        write_json(VLM/'provenance/process-failures'/path.name,value)
    for path in (OUT/'logs').glob('*qwen*log'):
        write_json(VLM/'provenance/process-failures'/(path.stem+'.json'),{'source_path':str(path.relative_to(ROOT)),
            'source_sha256':sha256(path.read_bytes()),'stdout':path.read_text(errors='replace')})
    if failures:
        path=OUT/'logs/vlm-full.log'
        write_json(VLM/'provenance/process-failures/vlm-full-log.json',{'source_path':str(path.relative_to(ROOT)),
            'source_sha256':sha256(path.read_bytes()),'stdout':path.read_text(errors='replace')})
    if (OUT/'setup/gpu-sharing-snapshot.json').exists():
        shutil.copyfile(OUT/'setup/gpu-sharing-snapshot.json',VLM/'provenance/gpu-sharing-snapshot.json')
    (VLM/'README.md').write_text('LOCO: 30 slik × 3 VLM × 3 ponovitve; 450 novih native odgovorov.\n\n'
        'Florence uporablja tri ločene izvorne naloge, skupaj en vhod za LLM na ponovitev. '
        'Ostala modela uporabljata obstoječi podrobni prompt. Veljavnost zapisa ni pravilnost. '
        'Surovi odgovori, tudi neveljavni, so nespremenjeni. NVILA/Qwen: model ni vrnil okvirjev. '
        'Ogrevalnih generacij ni; piloti so merjeni r1 položaji. Nalaganje je ločeno v provenance. '
        'Reference LOCO so samo v ločenem offline vrednotenju. Podrobni protokol: ../README.md.\n')
    native.render(portable,records,summary,target=VLM,reused=set(),introduction=
        '<h1>LOCO: novi opisi in zaznave</h1><p>30 slik · 3 VLM · 3 dejanske ponovitve. '
        'Reference niso vhod. Veljaven zapis ni pravilnost; prekrivajoče se Florence regije niso potrjeno različni objekti.</p>')
    assert len(records)==450 and len({r['call_id'] for r in records})==450
    assert len({(r['frame_id'],r['model_name'],r['repeat']) for r in records})==270
    assert all(sha256((VLM/s['published_path']).read_bytes())==s['source_sha256'] for s in sources)
    result={'complete':True,'images':30,'source_positions':270,'native_calls':450,
        'statuses':dict(Counter(r['status'] for r in records)),'physical_warmups':0,
        'all_gpu':all(r['gpu_verified'] for r in records),'unique_call_ids':450,
        'raw_exports_byte_identical':450,'previous_THUD_calls':0,'stability_accuracy':None,
        'responses_with_generated_token_ids':sum(bool(r.get('token_ids')) for r in records),
        'responses_with_text':sum(bool(r.get('content')) for r in records),
        'separate_process_failures':len(failures),
        'unrecovered_inflight_generation_count_unknown':bool(failures),
        'note':'native_calls counts saved native responses, not a claim of zero failed process-level/in-flight attempts. See preserved process-failure evidence.'}
    write_json(VLM/'verification.json',result);print(json.dumps(result,indent=2))


def context_policy():
    """Freeze context sizes once, before all LLM inference; never trim text."""
    index=read_json(LLM/'frozen/index.json');instruction=(LLM/'prompts/reasoning.txt').read_text().strip()
    overrides={};lengths=[]
    for source in index:
        if source['status']!='ok':continue
        doc=read_json(ROOT/source['path'])['parsed']
        text=reason_prompt(source['frame_id'],doc['objects'],doc,instruction)
        length=len(text);context=16384
        for threshold,value in ((30000,32768),(65000,65536),(130000,131072)):
            if length>threshold:context=value
        if context>16384:overrides[source['sha256']]={'num_ctx':context}
        lengths.append({'input_index':source['input_index'],'sha256':source['sha256'],'characters':length,'num_ctx':context})
    immutable_json(LLM/'execution-context-policy.json',{'policy':'Existing 16k default; conservative character thresholds 30k/65k/130k allocate 32k/64k/128k equally for every LLM/prompt/seed. Inputs unchanged; no reference data or outcome selection.',
        'overrides_by_frozen_sha256':overrides,'archived_failed_attempts':[],'input_sizes':lengths})
    (OUT/'llm/setup').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(OUT/'setup/prior-results-sha.json',OUT/'llm/setup/stage1-preservation.json')
    usage=[]
    for model in stage.VLMS:
        rs=[r for r in index if r['vlm']==model]
        usage.append({'vlm':model,'source_positions':len(rs),'usable':sum(r['status']=='ok' for r in rs),
            'unusable':sum(r['status']!='ok' for r in rs),'partial_unvalidated':sum(r['partial_input'] for r in rs),
            'planned_calls':sum(r['status']=='ok' for r in rs)*18,'planned_skips':sum(r['status']!='ok' for r in rs)*18,
            'note':'Transport usability is not visual correctness or completeness.'})
    directory=PUBLIC/'04_povzetki';directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'frozen-input-usability.json',usage);native.csv_file(directory/'frozen-input-usability.csv',usage)
    print('Frozen LLM context distribution:',dict(Counter(r['num_ctx'] for r in lengths)))


def capture_runtime():
    """Collect only this LOCO session, never present old THUD logs as new proof."""
    setup=OUT/'llm/setup';start=read_json(setup/'server-start.json')['at_utc']
    server='loco-ollama'
    log=subprocess.check_output(['docker','logs','--since',start,server],stderr=subprocess.STDOUT,text=True)
    (OUT/'llm/logs').mkdir(parents=True,exist_ok=True)
    (OUT/'llm/logs/ollama.log').write_text(log)
    blobs={}
    for spec in stage.MODELS:
        manifest=read_json(OUT/'llm/runs'/spec['name']/'short/r1/manifest.json')
        modelfile=manifest['model_metadata'][spec['name']]['show']['modelfile']
        blob=re.search(r'^FROM .*?(sha256-[a-f0-9]{64})',modelfile,re.M).group(1)
        blobs[blob]=spec['name']
    active_model=None;slots=None;observed=defaultdict(dict)
    for line in log.splitlines():
        match=re.search(r'--model\s+\S*?(sha256-[a-f0-9]{64})',line)
        if match:active_model=blobs.get(match[1]);slots=None
        match=re.search(r'n_seq_max\s+=\s+(\d+)',line)
        if match:slots=int(match[1])
        match=re.search(r'new slot, n_ctx = (\d+)',line)
        if match and active_model and slots is not None:observed[active_model][match[1]]=slots
    by_context={spec['name']:observed.get(spec['name'],{}) for spec in stage.MODELS}
    actual={m:next(iter(set(v.values()))) if len(set(v.values()))==1 else None for m,v in by_context.items()}
    image=subprocess.check_output(['docker','inspect','-f','{{.Image}}',server],text=True).strip()
    runtime={'session_start_utc':start,'requested_parallel_slots':1,'client_workers':1,'docker_image':image,
        'actual_model_slots':actual,'actual_model_slots_by_context':by_context,
        'model_blob_ids':{name:blob for blob,name in blobs.items()},
        'proof':'Matched the first FROM weight blob in each saved model manifest to --model in the current-session server command, then parsed n_seq_max and each slot n_ctx. Qwen quantizations are tracked separately; no prior THUD lines imported. One slot requested for every model to limit VRAM on the shared GPU.',
        'context_scheduling':'One model and context cohort at a time; same input/options for every prompt/seed.'}
    write_json(setup/'runtime-parallel.json',runtime)
    lines=log.splitlines()
    (setup/'parallel-runtime-evidence.txt').write_text('\n'.join(s for s in lines if any(x in s for x in ('--model','general.architecture','n_seq_max','n_ctx ','new slot','model architecture does not currently support parallel requests')))+'\n')
    (setup/'gpu-placement-evidence.txt').write_text('\n'.join(s for s in lines if any(x in s.lower() for x in ('--model','inference compute','offloaded','offloading','ggml_cuda_init')))+'\n')
    input_warnings=[s for s in lines if any(x in s.lower() for x in
        ('truncating input','truncating prompt','input prompt too long','exceeds context window'))]
    write_json(setup/'server-warnings.json',{'input_truncation_warning_lines':input_warnings,
        'server_warning_lines':[s for s in lines if 'level=WARN' in s or 'level=ERROR' in s],
        'note':'Current LOCO session only. A GPU-discovery context deadline is a watchdog timeout, not evidence of token-context truncation. Backend initialization warnings are separate from saved generation failures.'})
    print('Current LOCO observed model slots:',by_context)


def evaluate():
    """Post-hoc reference matching; reference files never enter model inputs."""
    directory=PUBLIC/'04_povzetki';directory.mkdir(parents=True,exist_ok=True)
    policy=read_json(OUT/'setup/class-map.json')
    assert policy==read_json(ROOT/'configs/loco-class-map.json')
    write_json(directory/'class-map.json',policy)
    references={r['frame_id']:r for r in map(json.loads,(ROOT/'data/loco_test30/objects.jsonl').read_text().splitlines())}
    write_json(directory/'evaluation-provenance.json',{'reference_path':'data/loco_test30/objects.jsonl',
        'reference_sha256':sha256((ROOT/'data/loco_test30/objects.jsonl').read_bytes()),
        'class_map_sha256':sha256((OUT/'setup/class-map.json').read_bytes()),
        'image_sha256':{fid:r['image_sha256'] for fid,r in references.items()},
        'algorithm':'Existing thud_box_metrics maximum-cardinality bipartite within-class matching, IoU >= 0.5',
        'stability_ground_truth_available':False,'references_sent_to_models':False})
    sources=read_json(VLM/'provenance.json');results=[];coverage=[];native_labels=defaultdict(Counter)
    for source in sources:
        row=read_json(VLM/source['published_path']);frame=references[row['frame_id']]
        assert frame['image_sha256']==row['image_sha256']
        task=source['task'];model=source['model']
        view=native.repetition_view({**row,'model_name':model,'task':task})
        labels={normalize(o['label']) for o in view['objects'] if isinstance(o.get('label'),str)}
        mentioned={policy['aliases'][label] for label in labels if label in policy['aliases']}
        annotated={o['label'] for o in frame['objects']}
        coverage.append({'frame_id':row['frame_id'],'model':model,'repeat':source['repeat'],'task':task,
            'schema_status':row['status'],'literal_list_available':view.get('object_list_available',False),
            'annotation_classes_present':sorted(annotated),
            'exact_name_classes_mentioned':sorted(mentioned) if view.get('object_list_available') else None,
            'annotated_classes_with_exact_name_mention':sorted(annotated&mentioned) if view.get('object_list_available') else None,
            'literal_text_mentions_not_detections':literal_mentions(row.get('content',''),policy['aliases']),
            'object_entries':len(view['objects']),'detection_accuracy':None,
            'note':'Exact native-label mentions only, not localized detection, instance recall or correctness. Caption free text is preserved but not converted into a fabricated object list.'})
        if model!='florence' or task not in ('od','dense'):continue
        predicted=[{**o,'label':normalize(o['label'])} for o in view['objects']] if row['status']=='ok' else []
        native_labels[task].update(o['label'] for o in predicted)
        match=match_regions(predicted,frame['objects'],policy)
        for pair in match['pairs']:
            original=view['objects'][pair['prediction_index']]
            pair.update(prediction_native_label=original['label'],prediction_region_id=original['id'],source_task=row['prompt'])
        for label in policy['classes']:
            eligible=[o for o in frame['objects'] if o['label']==label]
            mapped=sum(policy['aliases'].get(o['label'])==label for o in predicted)
            pairs=[p for p in match['pairs'] if p['reference_label']==label]
            results.append({'frame_id':row['frame_id'],'image_sha256':row['image_sha256'],'repeat':source['repeat'],'task':task,'class':label,
                'native_status':row['status'],'reference_count':len(eligible),'mapped_prediction_count':mapped,
                'matched':len(pairs),'false_negative':len(eligible)-len(pairs),'unmatched_mapped_prediction':mapped-len(pairs),
                'precision_within_five_classes':len(pairs)/mapped if mapped else None,
                'recall_within_five_classes':len(pairs)/len(eligible) if eligible else None,
                'full_match_record':match if label==policy['classes'][0] else None,'pairs_for_class':pairs,
                'invalid_prediction_output':row['status']!='ok',
                'note':'Maximum-cardinality one-to-one within-class IoU>=0.5. Unmapped scene labels are not called hallucinations. Failed native outputs count as no accepted predictions; failure is explicit.'})
    write_json(directory/'box-matches.json',results)
    native.csv_file(directory/'box-metrics.csv',[{k:v for k,v in r.items() if k not in ('full_match_record','pairs_for_class')} for r in results])
    write_json(directory/'label-coverage.json',coverage);native.csv_file(directory/'label-coverage.csv',coverage)
    mention_summary=[]
    for model in native.MODELS:
        sources_by_position=defaultdict(list)
        for row in coverage:
            if row['model']==model:sources_by_position[(row['frame_id'],row['repeat'])].append(row)
        for label in policy['classes']:
            eligible=[rs for rs in sources_by_position.values() if label in rs[0]['annotation_classes_present']]
            mention_summary.append({'model':model,'class':label,'reference_class_present_positions':len(eligible),
                'literal_list_available_positions':sum(any(r['literal_list_available'] for r in rs) for rs in eligible),
                'exact_list_label_mentioned_positions':sum(any(label in (r['exact_name_classes_mentioned'] or []) for r in rs) for rs in eligible),
                'literal_fulltext_mentioned_positions':sum(any(label in r['literal_text_mentions_not_detections'] for r in rs) for rs in eligible),
                'note':'Post-hoc literal term coverage using the frozen aliases. Florence tasks combined per image/repetition, not independent duplicates. Not instance recall, localization, correctness or complete semantic coverage; plurals and free phrases are not silently normalized to singular aliases.'})
    write_json(directory/'class-mention-summary.json',mention_summary);native.csv_file(directory/'class-mention-summary.csv',mention_summary)
    repeats=read_json(VLM/'repeat-differences.json');repeat_summary=[]
    for model,task in sorted({(r['model'],r['task']) for r in repeats}):
        rs=[r for r in repeats if r['model']==model and r['task']==task]
        jaccard=[r['name_jaccard'] for r in rs if r['name_jaccard'] is not None]
        repeat_summary.append({'model':model,'task':task,'repeat_pairs':len(rs),
            'equal_native_text_pairs':sum(r['raw_equal'] for r in rs),
            'list_comparable_pairs':len(jaccard),'mean_literal_name_jaccard':statistics.mean(jaccard) if jaccard else None,
            'note':'Three pairings per image, not independent samples. Identical content is not a copied generation: call IDs are distinct. Names and format stability are not object identity or correctness.'})
    write_json(directory/'native-repeat-summary.json',repeat_summary);native.csv_file(directory/'native-repeat-summary.csv',repeat_summary)
    summary=[]
    for task in ('od','dense'):
        for label in policy['classes']:
            rs=[r for r in results if r['task']==task and r['class']==label]
            refs=sum(r['reference_count'] for r in rs);pred=sum(r['mapped_prediction_count'] for r in rs);tp=sum(r['matched'] for r in rs)
            summary.append({'task':task,'class':label,'image_repeat_positions':len(rs),'reference_instances_repeated':refs,
                'mapped_predictions':pred,'matched':tp,'precision':tp/pred if pred else None,'recall':tp/refs if refs else None,
                'note':'Same thirty images repeated three times, not independent samples. Five-class closed-vocabulary matching; no score for descriptions without boxes.'})
    write_json(directory/'box-summary.json',summary);native.csv_file(directory/'box-summary.csv',summary)
    write_json(directory/'native-label-vocabulary.json',{
        task:{'total_native_regions':sum(labels.values()),
              'exact_mapped_regions':sum(n for label,n in labels.items() if label in policy['aliases']),
              'label_counts':dict(labels.most_common()),
              'note':'Frozen exact-label policy: generic box/crate and free region sentences are not silently mapped to the five reference classes. Zero mapped predictions is a vocabulary limitation, not proof of absent visual regions.'}
        for task,labels in native_labels.items()})
    return references,results,coverage


def report():
    # Kept in a separate small report module, sharing existing LaTeX helpers.
    from .loco_examples import build
    evaluate();build()
    directory=PUBLIC/'04_povzetki'
    shutil.copyfile(OUT/'inputs.json',PUBLIC/'inputs.json')
    evidence={'preparation.json','class-map.json','prior-results-sha.json',
        'qwen-process-exit-139.json','qwen-container-exits.json','launcher-exit-2.json','gpu-sharing-snapshot.json'}
    for p in (OUT/'setup').glob('*.json'):
        if p.name in evidence:shutil.copyfile(p,directory/p.name)
    for p in (OUT/'llm/setup').glob('*.json'):
        shutil.copyfile(p,LLM/'provenance'/p.name)
    if (OUT/'gpu-samples.jsonl').exists():shutil.copyfile(OUT/'gpu-samples.jsonl',directory/'gpu-samples.jsonl')
    for p in (OUT/'logs').glob('*tests.log'):
        write_json(directory/(p.stem+'.json'),{'source_sha256':sha256(p.read_bytes()),'stdout':p.read_text()})
    for name in ('llm-full.log','llm-resume-gemma.log','llm-resume-interruption.log'):
        p=OUT/'logs'/name
        write_json(LLM/'provenance'/(p.stem+'.json'),{'source_path':str(p.relative_to(ROOT)),
            'source_sha256':sha256(p.read_bytes()),'stdout':p.read_text(errors='replace')})


def verify():
    from . import thud_stage2_comparison as comparison
    comparison.configure_paths(source=VLM,target=LLM,out=OUT/'llm')
    llm=comparison.verify();vlm=read_json(VLM/'verification.json')
    assert llm['logical_positions']==4860 and vlm['native_calls']==450
    # Verify the portable package itself, including in a fresh checkout without
    # results/loco. Local absolute paths remain provenance, not dependencies.
    selected=read_json(VLM/'manifest.json')['frames']
    assert len(selected)==30 and len({f['frame_id'] for f in selected})==30
    sources=read_json(VLM/'provenance.json');assert len(sources)==450
    rows=[];by_local_path={}
    for source in sources:
        path=VLM/source['published_path']
        assert sha256(path.read_bytes())==source['source_sha256']
        row=read_json(path);rows.append(row)
        by_local_path[source['source_path']]=source['source_sha256']
        assert row['gpu_verified'] and row['token_ids'] and row['content']
        assert row['image_sha256']==next(f['image_sha256'] for f in selected if f['frame_id']==row['frame_id'])
    assert len({r['call_id'] for r in rows})==450
    assert dict(Counter(r['status'] for r in rows))==vlm['statuses']
    counts=Counter((s['model'],s['repeat'],s['task']) for s in sources)
    assert counts=={(m,r,t):30 for m in native.MODELS for r in stage.SEEDS
                    for t in (('caption','od','dense') if m=='florence' else ('description',))}
    for p in (VLM/'provenance/process-failures').glob('*process-exit-*.json'):
        for saved in read_json(p)['saved_records']:
            assert by_local_path[saved['path']]==saved['sha256']
    for source in read_json(LLM/'frozen/index.json'):
        document=read_json(ROOT/source['path'])
        assert document['image_sha256']==next(f['image_sha256'] for f in selected if f['frame_id']==source['frame_id'])
        for raw in document['source_records'].values():
            assert str(raw['path']).startswith(str(VLM.relative_to(ROOT))+'/')
            assert sha256((ROOT/raw['path']).read_bytes())==raw['sha256']
            assert read_json(ROOT/raw['path'])['call_id']==raw['original_call_id']
    runtime=read_json(LLM/'provenance/runtime-parallel.json')
    assert runtime['requested_parallel_slots']==runtime['client_workers']==1
    assert all(values and all(n==1 for n in values.values()) for values in runtime['actual_model_slots_by_context'].values())
    expected_contexts={str(r['num_ctx']) for r in read_json(LLM/'execution-context-policy.json')['input_sizes']}
    assert all(set(values)==expected_contexts for values in runtime['actual_model_slots_by_context'].values())
    observed=read_json(LLM/'provenance/inference-source-observed.json')
    assert all(sha256((ROOT/p).read_bytes())==r['sha256'] for p,r in observed['files'].items())
    checkpoint=read_json(LLM/'provenance/checkpoint-gemma-gpu-loss.json')
    # Portable comparison uses the preserved original-record digests, so the
    # local working files are not needed in a fresh checkout.
    _,_,restored=comparison.load()
    published_digests={r['original_record']:r['original_record_sha256'] for r in restored}
    assert all(published_digests[p]==s for p,s in checkpoint['saved_records'].items())
    interrupted=read_json(LLM/'provenance/checkpoint-turn-interruption.json')
    assert all(published_digests[p]==s for p,s in interrupted['saved_records'].items())
    cancelled=[r for r in restored if r.get('request_reconstructed_after_interruption')]
    assert len(cancelled)==1 and cancelled[0]['status']=='error' and 'response' not in cancelled[0]
    evidence=LLM/'provenance'/Path(cancelled[0]['interruption_evidence_path']).name
    assert sha256(evidence.read_bytes())==cancelled[0]['interruption_evidence_sha256']
    protected_path=OUT/'setup/prior-results-sha.json'
    if not protected_path.exists():protected_path=PUBLIC/'04_povzetki/prior-results-sha.json'
    protected=read_json(protected_path)
    assert all(sha256((ROOT/p).read_bytes())==s for p,s in protected.items())
    from html.parser import HTMLParser
    from urllib.parse import unquote
    links=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):links.extend(v for k,v in attrs if k in ('href','src'))
    count=0
    for path in PUBLIC.rglob('*.html'):
        links.clear();Links().feed(path.read_text())
        for link in links:
            assert not link.startswith(('http:','https:','/')),(path,link)
            assert (path.parent/unquote(link)).is_file(),(path,link)
            count+=1
    for image in selected:
        from PIL import Image
        assert sha256((ROOT/image['image_path']).read_bytes())==image['image_sha256']
        with Image.open(ROOT/image['image_path']) as rgb:rgb.verify()
    overlays=0
    for p in PUBLIC.rglob('*'):
        if p.suffix.lower() in ('.jpg','.png'):
            with Image.open(p) as rgb:rgb.verify()
            overlays+=1
    with (VLM/'manual-review-empty.csv').open() as stream:native_reviews=list(csv.DictReader(stream))
    assert len(native_reviews)==270 and all(not v for r in native_reviews for k,v in r.items() if k.startswith('review'))
    pdf=PUBLIC/'Rezultati_LOCO.pdf'
    for name in ('Rezultati_LOCO','Priloga_LOCO'):
        assert (PUBLIC/(name+'.pdf')).read_bytes().startswith(b'%PDF-')
        assert (PUBLIC/(name+'.tex')).stat().st_size>1000
    with (PUBLIC/'03_primeri/manual-review-empty.csv').open() as stream:review=list(csv.DictReader(stream))
    assert len(review)==10 and all(not v for r in review for k,v in r.items() if k.startswith('review'))
    notes=read_json(PUBLIC/'04_povzetki/qualitative-notes.json')['images']
    selected={r['frame_id']:r for r in selected}
    assert {r['frame_id'] for r in notes}=={r['frame_id'] for r in review}
    for note in notes:
        assert note['image_sha256']==selected[note['frame_id']]['image_sha256']
        for source in note['inspected_sources']:
            assert sha256((ROOT/source['path']).read_bytes())==source['sha256']
    warnings=read_json(LLM/'provenance/server-warnings.json')
    result={'complete':True,'vlm':vlm,'llm':llm,'html_links_verified':count,'previous_files_unchanged':len(protected),
        'server_input_truncation_warnings':len(warnings['input_truncation_warning_lines']),
        'backend_warning_lines':len(warnings['server_warning_lines']),
        'images_readable':30,'exported_images_readable':overlays,'selected_examples':len(review),'pdf':pdf.name,'no_stability_ground_truth':True,
        'previous_THUD_inferences':0,'verified_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(PUBLIC/'verification.json',result);print(json.dumps(result,indent=2))
