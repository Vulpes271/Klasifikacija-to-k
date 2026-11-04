"""ARHIVSKI IZVAJALNIK: samo reprodukcija objavljenih poskusov.

Adapter for independently processing neutral LILocBench RGB views; no pair reasoning."""
import argparse
import csv
import html
import http.client
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import uuid

from .experiment import read_json,write_json,sha256,options,combinations
from .florence_grounding import draw,CAPTION,GROUND
from .thud_extended_common import saved_path
from .thud_experiment import immutable_json
from .vision_prompts import reason_prompt
from .ollama_runtime import runner_http, load_model
from .vision_backends import Backend,request
from .llm_checks import (validate_response,validity_counts,prepare_position,context_after,finish_position)
from .llm_checks import make_signature,check_signature

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results models/lilocbench'
PROMPT40=ROOT/'results models/prompt_comparison40'
CONTAINER='klasifikacija-ollama-1'


def prepare():
    original=read_json(PROMPT40/'manifest.json');bf16=read_json(PROMPT40/'bf16/manifest.json')
    specs={s['name']:s for s in original['config']['models']+bf16['models']}
    names=['qwen35-4b-q4','qwen35-27b-q4','gemma3-27b-q4','qwen35-4b-bf16','qwen35-9b-bf16']
    config={**original['config'],'models':[{k:v for k,v in specs[n].items() if k!='base_url'} for n in names],
        'system_file_text':(PROMPT40/'prompts/new-system.txt').read_text()}
    for k in ('system_file','instruction_file'):config.pop(k,None)
    (OUT/'prompts/users').mkdir(parents=True,exist_ok=True)
    (OUT/'annotated').mkdir(exist_ok=True)
    prompt=OUT/'prompts/system.txt';text=config['system_file_text']
    if prompt.exists():assert prompt.read_text()==text
    else:prompt.write_text(text)
    selection=read_json(OUT/'native-selection.json');frames=[]
    for frame in selection:
        fid=frame['frame_id'];caption=read_json(saved_path(OUT/'native',frame,'caption'));grounding=read_json(saved_path(OUT/'native',frame,'grounding'))
        regions=[]
        if grounding['status']=='ok':
            native=grounding['parsed'][GROUND]
            regions=[{'id':i,'label':label,'bbox_xyxy':box,'source_task':GROUND,'source_region_id':i}
                for i,(label,box) in enumerate(zip(native['labels'],native['bboxes']),1)]
        annotated=OUT/'annotated'/(fid+'.jpg')
        if not annotated.exists():draw(frame['image_path'],regions,annotated)
        objects=[{'id':r['id'],'label':r['label'],
            'observation':f"Florence grounded caption phrase {r['label']!r} at bbox_xyxy {r['bbox_xyxy']} in original pixels. Attachment/contact evidence and physical-instance identity were not established.",
            'attachment_evidence':'not_visible','identifiable':True,'bbox_xyxy':r['bbox_xyxy'],
            'source_task':GROUND,'source_region_id':r['source_region_id']} for r in regions]
        document={'stage':1,'frame_id':fid,'model_name':'florence','model':'microsoft/Florence-2-large',
            'variant':{'id':'caption-grounding-v1'},'image_sha256':frame['image_sha256'],
            'status':'ok' if caption['status']==grounding['status']=='ok' else 'error',
            'parsed':{'scene_description':caption.get('parsed',{}).get(CAPTION,''),'objects':objects},
            'adapter_notes':'Native phrase regions, not verified physical identities. No poses, sequence or pair information in model input.'}
        frozen=OUT/'frozen'/(fid+'.json');immutable_json(frozen,document)
        user=reason_prompt(fid,[{'id':r['id'],'label':r['label']} for r in regions],document['parsed'],config['instruction_file_text'])
        user_path=OUT/'prompts/users'/(fid+'.txt')
        if user_path.exists():assert user_path.read_text()==user
        else:user_path.write_text(user)
        frames.append({'frame_id':fid,'dataset':'lilocbench','environment':'indoor','frozen':str(frozen.relative_to(OUT)),
            'input_sha256':sha256(frozen.read_bytes()),'image_sha256':frame['image_sha256'],
            'user_prompt':str(user_path.relative_to(OUT)),'user_prompt_sha256':sha256(user_path.read_bytes()),
            'caption_status':caption['status'],'grounding_status':grounding['status']})
    manifest={'protocol':'lilocbench-v1','config':config,'inputs':frames,'system_prompt_sha256':sha256(prompt.read_bytes()),
        'source_prompt40_manifest_sha256':sha256((PROMPT40/'manifest.json').read_bytes()),
        'pair_manifest_sha256':sha256((OUT/'pairs.json').read_bytes()),'new_prompt_only':True,'seed':42,
        'expected_llm_positions':len(frames)*5,'poses_used_only_for_selection':True}
    immutable_json(OUT/'manifest.json',manifest);return manifest






def run():
    manifest=prepare();config=manifest['config'];opts=options(config,next(combinations(config)))
    system=(OUT/'prompts/system.txt').read_text()
    for declared in config['models']:
        spec={**declared,'base_url':os.environ.get('THUD_OLLAMA_URL','http://localhost:11434')}
        backend=Backend(spec,config['timeout']);setup=None;failure=None
        tags=request(backend.url+'/api/tags')['models'];digest=next(r['digest'] for r in tags if r['name']==spec['model'])
        metadata=backend.metadata();write_json(OUT/'model-metadata'/(spec['name']+'.json'),{'digest':digest,**metadata})
        # Validate saved signatures before any model load or new generation.
        pending=[]
        for frame in manifest['inputs']:
            path=OUT/'responses'/spec['name']/(frame['frame_id']+'.json');user=(OUT/frame['user_prompt']).read_text()
            row=prepare_position(path,OUT/frame['frozen'],frame['input_sha256'],spec,digest,system,user,opts,metadata['show'].get('template'))
            if row is not None:pending.append((frame,path,user,row))
        if not pending:continue
        try:
            try:
                setup=load_model(backend,opts,digest);write_json(OUT/'setup'/(spec['name']+'.json'),setup)
            except Exception as exc:
                failure=str(exc);write_json(OUT/'setup'/(spec['name']+'.json'),{'status':'error','error':failure,'model_digest':digest})
                # A local tokenizer integration error is fixable without spending a call.
                # Do not create completed positions until it has been resolved.
                raise
            for n,(frame,path,user,row) in enumerate(pending,1):
                document=read_json(OUT/frame['frozen']);row.update(model=spec['model'],model_name=spec['name'],
                    precision=spec['precision'],frame_id=frame['frame_id'],seed=42,prompt_variant='new',
                    call_id=uuid.uuid4().hex,generation_attempted=False)
                if document['status']!='ok' or not document['parsed']['objects']:
                    row.update(status='skipped',error='Invalid or empty native Florence input')
                elif failure:
                    row.update(status='skipped',error=failure)
                else:
                    try:
                        tokens=runner_http(setup['tokenizer_port'],'/tokenize',{'content':system+'\n'+user,'add_special':False})['tokens']
                        estimate=len(tokens)+256
                        row['context_budget_before']={'status':'estimated_sufficient' if estimate+opts['num_predict']<=opts['num_ctx'] else 'estimated_insufficient',
                            'method':'native_model_tokenizer_content_plus_256_chat_margin','exact':False,
                            'content_tokens':len(tokens),'estimated_prompt_tokens':estimate,'reserved_output_tokens':opts['num_predict'],
                            'num_ctx':opts['num_ctx'],'model_digest':digest,
                            'note':'Exact native content tokenization plus conservative overhead for the unchanged two-message text-only chat template; not an exact rendered-chat count.'}
                        if row['context_budget_before']['status']!='estimated_sufficient':
                            row.update(status='skipped',error='Context reservation exceeds configured 8192; no truncation or automatic increase')
                        else:
                            row['generation_attempted']=True
                            row.update(backend.generate(system,user,opts));row['status']='response_received'
                            finish_position(row,document,opts)
                            if not row['signature_check_after']['verified']:failure='Model/request signature mismatch; remaining positions skipped'
                            if row['context_budget_after']['status']!='sufficient':row['context_warning']='Actual reserved context budget insufficient/unverifiable'
                    except Exception as exc:
                        row.update(getattr(exc,'result',{}),status='error',error=str(exc))
                        if not row.get('gpu_verified'):failure='GPU/API/tokenizer failure: '+str(exc)
                write_json(path,row)
                if n%10==0:print(f"{spec['name']}: {n}/{len(pending)} new positions saved",flush=True)
        finally:backend.unload()
        export(manifest)
    export(manifest)


def export(manifest=None):
    manifest=manifest or prepare();pairs=read_json(OUT/'pairs.json')['pairs'];by_id={f['frame_id']:f for f in manifest['inputs']}
    models=manifest['config']['models'];checks=[];per_model={s['name']:[] for s in models};times={s['name']:[] for s in models};csv_rows=[];body=''
    body+='<h1>LILocBench: paired viewpoints / independent image classification</h1><p>Ground-truth poses select camera views only. Local region IDs do not identify the same object across images. Unchanged does not imply fixed; disappearance does not prove motion. No content accuracy or localization performance is evaluated.</p>'
    body+='<p><a href="pairs.json">Pair provenance</a> · <a href="region-classes.csv">Region CSV</a> · <a href="pair-review-empty.csv">Empty pair review</a> · <a href="summary.json">Summary</a> · <a href="LICENSE-DATA.md">Data license</a></p>'
    body+='<p>Data: N. Trekel et al., LILocBench, IROS 2025. <a href="https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/">Official source</a>. Selected data and overlays: <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>.</p>'
    for pair in pairs:
        body+='<section><h2>'+pair['pair_id']+f" — {pair['distance_m']:.3f} m / {pair['orientation_difference_degrees']:.2f}°"+'</h2><div class="pair">'
        for side in ('a','b'):
            fid=pair[side]['frame_id'];frame=by_id[fid];document=read_json(OUT/frame['frozen']);objects=document['parsed']['objects']
            body+='<article><h3>'+fid+' / '+pair[side]['sequence']+'</h3><a href="annotated/'+fid+'.jpg"><img loading="lazy" src="annotated/'+fid+'.jpg"></a><details><summary>Full Florence description</summary><p>'+html.escape(document['parsed']['scene_description'])+'</p></details>'
            row_cache={}
            for spec in models:
                path=OUT/'responses'/spec['name']/(fid+'.json')
                if path.exists():
                    row_cache[spec['name']]=read_json(path)
                    body+='<details><summary>'+html.escape(spec['model']+' — complete response')+'</summary><a href="'+str(path.relative_to(OUT))+'">Raw JSON</a><pre>'+html.escape(row_cache[spec['name']].get('content',''))+'</pre></details>'
            for obj in objects:
                body+='<div class="region"><h4>Region '+str(obj['id'])+' — '+html.escape(obj['label'])+'</h4><p>'+html.escape(str(obj['bbox_xyxy']))+'</p>'
                csv_row={'pair_id':pair['pair_id'],'frame_id':fid,'sequence':pair[side]['sequence'],'input_sha256':frame['input_sha256'],
                    'region_id':json.dumps(obj['id']),'florence_label':obj['label'],'bbox_xyxy':json.dumps(obj['bbox_xyxy'])}
                for spec in models:
                    path=OUT/'responses'/spec['name']/(fid+'.json')
                    if path.exists():
                        row=row_cache[spec['name']];check=validate_response(row,objects)
                        decision=next((x for x in check['usable_items'] if type(x['id']) is type(obj['id']) and x['id']==obj['id']),{})
                        classification=decision.get('classification');reason=decision.get('reason')
                        error='; '.join(check['errors']) or row.get('error','')
                        body+='<details><summary>'+html.escape(spec['model']+' / '+str(classification)+' / '+str(row.get('elapsed_seconds'))+' s')+'</summary><p>'+html.escape(reason or 'Reason not returned')+'</p><p class="error">'+html.escape(error)+'</p><a href="'+str(path.relative_to(OUT))+'">Full raw response</a></details>'
                    else:classification=None;reason=None;check={'schema_valid_response':False};error='Not completed'
                    csv_row[spec['name']+'_classification']=classification;csv_row[spec['name']+'_reason']=reason
                    csv_row[spec['name']+'_whole_valid']=check['schema_valid_response'];csv_row[spec['name']+'_error']=error
                csv_rows.append(csv_row);body+='</div>'
            body+='</article>'
        body+='</div></section>'
    for frame in manifest['inputs']:
        objects=read_json(OUT/frame['frozen'])['parsed']['objects']
        for spec in models:
            path=OUT/'responses'/spec['name']/(frame['frame_id']+'.json')
            if not path.exists():continue
            row=read_json(path);check=validate_response(row,objects);checks.append(check);per_model[spec['name']].append(check)
            if row.get('elapsed_seconds') is not None:times[spec['name']].append(row['elapsed_seconds'])
    with (OUT/'region-classes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]) if csv_rows else ['pair_id','frame_id','region_id']);writer.writeheader();writer.writerows(csv_rows)
    if not (OUT/'pair-review-empty.csv').exists():
        with (OUT/'pair-review-empty.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['pair_id','region_first_image','region_second_image','confirmed_same_identity','moved_unchanged_not_comparable','note']);writer.writeheader()
            writer.writerows({'pair_id':p['pair_id']} for p in pairs)
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>LILocBench pipeline</title><style>body{font:17px Arial;margin:20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%;max-width:700px}.region{border:1px solid #ccc;margin:12px 0;padding:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.error{color:#a00}section{border-top:2px solid #555;margin-top:25px}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style>'+body+'</html>')
    timings={name:{'calls_with_time':len(values),'total_seconds':sum(values),'mean_seconds':statistics.mean(values) if values else None} for name,values in times.items()}
    native_times=[read_json(saved_path(OUT/'native',f,'combined'))['elapsed_seconds'] for f in read_json(OUT/'native-selection.json')]
    summary={'pairs':len(pairs),'images':len(manifest['inputs']),'expected_llm_positions':len(manifest['inputs'])*5,
        **validity_counts(checks),'per_model':{name:validity_counts(values) for name,values in per_model.items()},
        'api_timings_including_load_if_any':timings,'transfer':read_json(OUT/'transfer-summary.json'),
        'timing_scope':'API chat time excludes native token counting and explicit load-only setup; no warmup generations.',
        'florence':{'caption_ok':sum(f['caption_status']=='ok' for f in manifest['inputs']),
            'grounding_ok':sum(f['grounding_status']=='ok' for f in manifest['inputs']),
            'combined_seconds_total':sum(native_times),'combined_seconds_mean':statistics.mean(native_times)},
        'reference_poses_sent_to_models':False,'object_correspondences_annotated':False,'accuracy':None}
    write_json(OUT/'summary.json',summary);print(json.dumps({k:summary[k] for k in ('pairs','images','positions','schema_valid_response','api_error','schema_or_finish_error','skipped')}),flush=True)
    return summary


def verify():
    """Validate only this new dataset, without inference or opening any review DB."""
    import numpy as np
    from scripts.prepare_lilocbench import camera_transform,angle
    from .review_viewer import Results
    from html.parser import HTMLParser
    from PIL import Image
    import zlib
    manifest=read_json(OUT/'manifest.json');pairs=read_json(OUT/'pairs.json')['pairs'];selection=read_json(OUT/'native-selection.json')
    assert len(pairs)<=50 and len(selection)==2*len(pairs)<=100
    assert len({f['image_sha256'] for f in selection})==len(selection)
    assert (OUT/'prompts/system.txt').read_bytes()==(PROMPT40/'prompts/new-system.txt').read_bytes()
    opts=options(manifest['config'],next(combinations(manifest['config'])))
    all_ids=set();main_calls=0;offsets=[];pair_rows=[]
    for i,pair in enumerate(pairs):
        assert pair['distance_m']<=.25+1e-9 and pair['orientation_difference_degrees']<=10+1e-9
        for old in pairs[:i]:
            for side in ('a','b'):
                distance=np.linalg.norm(np.array(pair[side]['camera_position_xyz'])-old[side]['camera_position_xyz'])
                degrees=angle(pair[side]['camera_quaternion_xyzw'],old[side]['camera_quaternion_xyzw'])
                assert distance>=.75-1e-8 or degrees>=25-1e-8
        flat={'pair_id':pair['pair_id'],'distance_m':pair['distance_m'],'rotation_difference_degrees':pair['orientation_difference_degrees']}
        for side in ('a','b'):
            v=pair[side];data=(ROOT/v['local_path']).read_bytes()
            assert v['camera']=='camera_front' and 'camera_front/color/images/' in v['archive_path']
            assert sha256(data)==v['sha256'] and len(data)==v['zip_entry']['bytes'] and zlib.crc32(data)&0xffffffff==v['zip_entry']['crc32']
            assert v['gt_time_offset_seconds']<=.05+1e-5
            assert v['interpolation_gt_timestamps'][1]-v['interpolation_gt_timestamps'][0]<=.1+1e-5
            offsets.append(v['gt_time_offset_seconds'])
            with Image.open(ROOT/v['local_path']) as image:assert image.size==(640,480)
            _,_,cal=camera_transform(v['sequence']);assert cal['optical_forward_in_base'][0]>.9
            for key in ('sequence','camera','timestamp_text','nearest_gt_timestamp','gt_time_offset_seconds','archive_path','sha256'):
                flat[side+'_'+key]=v[key]
            for key in ('robot_position_xyz','robot_quaternion_xyzw','camera_position_xyz','camera_quaternion_xyzw'):
                flat[side+'_'+key]=json.dumps(v[key])
        pair_rows.append(flat)
    with (OUT/'pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pair_rows[0]));writer.writeheader();writer.writerows(pair_rows)
    for frame in manifest['inputs']:
        document=read_json(OUT/frame['frozen']);assert sha256((OUT/frame['frozen']).read_bytes())==frame['input_sha256']
        assert re.fullmatch('view-[0-9]{6}',frame['frame_id'])
        user=(OUT/frame['user_prompt']).read_text();assert sha256(user.encode())==frame['user_prompt_sha256']
        payload=json.loads(user[user.index('{'):]);assert set(payload)=={'frame_id','objects','visual_observations'}
        assert payload['frame_id']==frame['frame_id'] and payload['visual_observations']==document['parsed']
        for spec in manifest['config']['models']:
            row=read_json(OUT/'responses'/spec['name']/(frame['frame_id']+'.json'))
            digest=read_json(OUT/'model-metadata'/(spec['name']+'.json'))['digest']
            expected=make_signature(document,frame['input_sha256'],spec['model'],digest,manifest['config']['system_file_text'],user,opts)
            assert check_signature(row,expected,document,frame['input_sha256'])['verified']
            if row.get('generation_attempted'):
                assert row['call_id'] not in all_ids;all_ids.add(row['call_id']);main_calls+=1
            if row['status']=='response_received':
                assert row['gpu_verified'] and row['gpu']['size_vram']>=row['gpu']['size'] and row['gpu']['context_length']==8192
                assert context_after(row,opts)['status']=='sufficient'
                assert row['response']['prompt_eval_count']<=row['context_budget_before']['estimated_prompt_tokens']
    for frame in selection:
        c=read_json(saved_path(OUT/'native',frame,'caption'));g=read_json(saved_path(OUT/'native',frame,'grounding'))
        assert g['user_prompt']==GROUND+c['parsed'][CAPTION]
        assert g['source_caption']['sha256']==sha256(saved_path(OUT/'native',frame,'caption').read_bytes())
        for row in (c,g):assert row['call_id'] not in all_ids;all_ids.add(row['call_id'])
    with (OUT/'pair-review-empty.csv').open(newline='') as f:
        empty=list(csv.DictReader(f))
        assert len(empty)==len(pairs) and all(not value for row in empty for key,value in row.items() if key!='pair_id')
    class Page(HTMLParser):
        def __init__(self):super().__init__();self.responses=0
        def handle_starttag(self,tag,attrs):
            if tag=='pre':self.responses+=1
            for key,value in attrs:
                if key in ('href','src') and not value.startswith(('http:','https:','#')):assert (OUT/value).is_file(),value
    page=Page();page.feed((OUT/'index.html').read_text());assert page.responses==len(selection)*5
    viewer=Results(OUT);assert viewer.kind=='lilocbench' and len(viewer.frames)==len(selection) and len(viewer.models)==5
    ledger=read_json(OUT/'download-ledger.json');sources=read_json(OUT/'sources.json')['sequences']
    for row in ledger:
        if row['start'] is not None and 'error' not in row:
            total=next(s['archive_bytes'] for s in sources.values() if s['archive']==row['url'])
            assert row['status']==206 and row['content_range']==f"bytes {row['start']}-{row['start']+row['requested_bytes']-1}/{total}"
            assert row['payload_bytes']==row['requested_bytes']
    assert main_calls<=500
    previous=read_json(ROOT/'results models/technical_revision_v1/source-sha256.json')
    for path,digest in previous.items():assert sha256((ROOT/path).read_bytes())==digest,'Changed previous result: '+path
    checkpoint=OUT/'resume-checkpoint.json'
    preserved=read_json(checkpoint)['preserved_response_sha256'] if checkpoint.exists() else {}
    for path,digest in preserved.items():assert sha256((OUT/path).read_bytes())==digest,'Changed completed position: '+path
    report={'pairs':len(pairs),'images':len(selection),'all_pairs_within_initial_threshold':True,
        'max_distance_m':max(p['distance_m'] for p in pairs),'max_orientation_degrees':max(p['orientation_difference_degrees'] for p in pairs),
        'max_nearest_gt_time_offset_seconds':max(offsets),'distinct_rgb_sha256':len(selection),
        'llm_positions':len(selection)*5,'main_generation_attempts':main_calls,'full_responses_in_html':page.responses,
        'html_links_valid':True,'pair_review_human_fields_empty':True,'payload_downloaded_bytes':sum(r['payload_bytes'] for r in ledger),
        'previous_source_files_unchanged':len(previous),'completed_positions_preserved_on_resume':len(preserved),
        'actual_review_databases_opened':False,'accuracy_evaluated':False}
    write_json(OUT/'verification.json',report);print(json.dumps(report),flush=True);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','run','export','verify'))
    args=parser.parse_args()
    if args.phase=='run':run()
    elif args.phase=='export':export()
    elif args.phase=='verify':verify()
    else:prepare()
