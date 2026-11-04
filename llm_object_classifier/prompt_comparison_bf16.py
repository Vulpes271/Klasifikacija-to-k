"""Exactly 160 text-only BF16 positions on the immutable prompt40 inputs."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from html.parser import HTMLParser

from .experiment import read_json, write_json, sha256, options, combinations
from .llm_checks import (fingerprint, make_signature, check_signature, validate_response,
    validity_counts, context_before, context_after, finish_position, prepare_position)
from .thud_experiment import immutable_json
from .vision_backends import Backend, request

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results models/prompt_comparison40'
MODELS=[{'name':'qwen35-4b-bf16','model':'qwen3.5:4b-bf16','backend':'ollama','precision':'BF16','require_gpu':True},
        {'name':'qwen35-9b-bf16','model':'qwen3.5:9b-bf16','backend':'ollama','precision':'BF16','require_gpu':True}]


def prepare(base=BASE):
    base=Path(base);original=read_json(base/'manifest.json');out=base/'bf16';out.mkdir(exist_ok=True)
    assert len(original['inputs'])==40
    inputs=[]
    for frame in original['inputs']:
        frozen=base/frame['frozen'];user=base/frame['user_prompt']
        assert sha256(frozen.read_bytes())==frame['input_sha256']
        assert sha256(user.read_bytes())==frame['user_prompt_sha256']
        assert read_json(frozen)['image_sha256']==frame['image_sha256']
        inputs.append({k:frame[k] for k in ('frame_id','input_sha256','image_sha256')})
    prompts={p:sha256((base/'prompts'/(p+'-system.txt')).read_bytes()) for p in ('old','new')}
    manifest={'protocol':'prompt40-bf16-v1','base_manifest_sha256':sha256((base/'manifest.json').read_bytes()),
        'inputs':inputs,'models':MODELS,'prompt_sha256':prompts,'seed':42,'positions':160,
        'settings':{k:v for k,v in original['config'].items() if k not in ('models','system_file','instruction_file','system_file_text','instruction_file_text')},
        'source':'Original frozen documents and exact saved user/system prompts; no Florence or earlier LLM reruns.'}
    immutable_json(out/'manifest.json',manifest)
    return original,manifest


def response_path(base,spec,variant,frame):
    return Path(base)/'bf16/responses'/spec['name']/variant/(frame['frame_id']+'.json')


def token_profile(backend,model):
    full=request(backend.url+'/api/show',{'model':model,'verbose':True},backend.timeout)
    info=full['model_info'];tokens=info.get('tokenizer.ggml.tokens',[])
    # Ordinary text BPE is shared; conversion-specific audio/vision special tokens
    # are NOT assumed equivalent. No image/audio data is supplied to these calls.
    core={'tokens':tokens[:248000],'merges':info.get('tokenizer.ggml.merges'),
          'pre':info.get('tokenizer.ggml.pre'),'model':info.get('tokenizer.ggml.model')}
    return {'text_bpe_sha256':fingerprint(core),'template_sha256':sha256(full['template'].encode()),
        'token_count':len(tokens),'precision':full['details']['quantization_level'],
        'special_tokens':tokens[248000:],'template':full['template']}


def recorded_budget(base,frame,variant,system,user,opts,profile,source_profile):
    reference=(Path(base)/'baseline/qwen35-4b-q4'/(frame['frame_id']+'.json') if variant=='old' else
               Path(base)/'responses/qwen35-4b-q4/42'/(frame['frame_id']+'.json'))
    row=read_json(reference)
    same=row['input_sha256']==frame['input_sha256'] and row['request']['messages']==[
        {'role':'system','content':system},{'role':'user','content':user}] and row['request']['options']==opts
    compatible=all(profile[k]==source_profile[k] for k in ('text_bpe_sha256','template_sha256'))
    # Do not use the reference estimate for text containing differing special tokens.
    differences=set(profile['special_tokens']) ^ set(source_profile['special_tokens'])
    compatible=compatible and not any(t and t in system+user for t in differences)
    count=row.get('response',{}).get('prompt_eval_count')
    if not same or not compatible or type(count) is not int:
        return context_before(system,user,opts,profile['template'])
    estimate=count+128
    return {'status':'estimated_sufficient' if estimate+opts['num_predict']<=opts['num_ctx'] else 'estimated_insufficient',
        'exact':False,'method':'recorded_matching_text_bpe_and_template_plus_128',
        'estimated_prompt_tokens':estimate,'recorded_reference_prompt_tokens':count,
        'reserved_output_tokens':opts['num_predict'],'num_ctx':opts['num_ctx'],
        'reference_path':str(reference.relative_to(Path(base))), 'reference_sha256':sha256(reference.read_bytes()),
        'text_bpe_sha256':profile['text_bpe_sha256'],'template_sha256':profile['template_sha256'],
        'note':'Conservative estimate from a previous real count, not exact tokenization of the BF16 request; checked against actual counts after every call.'}


def reuse_candidates():
    """Search bytes programmatically; never dump previous answers to the console."""
    roots=[str(ROOT/p) for p in ('results','results models','analiza rezultatov vmesna predstavitev') if (ROOT/p).exists()]
    proc=subprocess.run(['rg','-l','-0','-F','-e','qwen3.5:4b-bf16','-e','qwen3.5:9b-bf16',
        '-g','*.json','-g','!**/models/**','-g','!**/environments/**','-g','!**/source-snapshots/**',*roots],capture_output=True)
    if proc.returncode not in (0,1):raise RuntimeError(proc.stderr.decode())
    result={}
    for raw in proc.stdout.split(b'\0'):
        if not raw:continue
        path=Path(os.fsdecode(raw))
        if 'prompt_comparison40/bf16' in str(path):continue
        try:row=read_json(path)
        except (ValueError,OSError):continue
        if isinstance(row,dict) and row.get('model') in {s['model'] for s in MODELS} and row.get('input_sha256'):
            result.setdefault((row['input_sha256'],row['model']),[]).append(path)
    return result


def gpu_snapshot():
    proc=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True)
    return {'nvidia_smi':proc.stdout.strip(),'error':proc.stderr.strip() if proc.returncode else None}


def run(base=BASE):
    base=Path(base);original,manifest=prepare(base);out=base/'bf16';candidates=reuse_candidates()
    reused=read_json(out/'reused.json') if (out/'reused.json').exists() else {}
    config={**original['config'],'models':manifest['models']};opts=options(config,next(combinations(config)));opts['seed']=42
    source_profile=None
    for declared in manifest['models']:
        spec={**declared,'base_url':os.environ.get('THUD_OLLAMA_URL','http://localhost:11434')}
        backend=Backend(spec,config['timeout']);called=False;failure=None
        try:
            tags=request(backend.url+'/api/tags')['models']
            if not any(t['name']==spec['model'] for t in tags):
                print('Pulling missing model '+spec['model'],flush=True)
                request(backend.url+'/api/pull',{'model':spec['model'],'stream':False},timeout=3600)
                tags=request(backend.url+'/api/tags')['models']
            digest=next(t['digest'] for t in tags if t['name']==spec['model'])
            metadata=backend.metadata();profile=token_profile(backend,spec['model'])
            if profile['precision']!='BF16':raise ValueError('BF16 required; no quantized substitute allowed')
            if source_profile is None:source_profile=token_profile(backend,'qwen3.5:4b-q4_K_M')
            write_json(out/'models'/(spec['name']+'.json'),{'digest':digest,'metadata':metadata,
                'text_tokenizer':{k:v for k,v in profile.items() if k not in ('special_tokens','template')},
                'gpu_before':gpu_snapshot()})
            positions=[]
            for frame in original['inputs']:
                user=(base/frame['user_prompt']).read_text()
                for variant in ('old','new'):
                    system=(base/'prompts'/(variant+'-system.txt')).read_text()
                    budget=recorded_budget(base,frame,variant,system,user,opts,profile,source_profile)
                    positions.append((frame,variant,system,user,budget))
            preflight=out/'preflights'/(spec['name']+'.json')
            if not preflight.exists():
                frame,variant,system,user,budget=max(positions,key=lambda p:p[4].get('estimated_prompt_tokens') or 10**9)
                probe_opts={**opts,'num_predict':16};called=True
                probe={'model':spec['model'],'model_digest':digest,'call_id':uuid.uuid4().hex,
                    'frame_id':frame['frame_id'],'prompt_variant':variant,'input_sha256':frame['input_sha256'],
                    'excluded_from_main_calls':True,'context_budget_before':budget,'gpu_before':gpu_snapshot()}
                try:
                    if budget['status']!='estimated_sufficient':raise ValueError('Preflight input budget insufficient/unverifiable')
                    probe.update(backend.generate(system,user,probe_opts))
                    probe['context_budget_after']=context_after(probe,opts)
                    if not probe['gpu_verified'] or probe['gpu']['context_length']!=opts['num_ctx']:raise ValueError('Full GPU/context verification failed')
                    if probe['gpu']['digest']!=digest:raise ValueError('Loaded model digest differs')
                    if probe['context_budget_after']['status']!='sufficient':raise ValueError('Actual context budget failed')
                    if probe['response']['prompt_eval_count']>budget['estimated_prompt_tokens']:raise ValueError('Recorded token estimate underestimated actual input')
                    probe['status']='gpu_verified'
                except Exception as exc:
                    probe.update(getattr(exc,'result',{}),status='error',error=str(exc))
                write_json(preflight,probe)
            probe=read_json(preflight)
            if probe.get('status')!='gpu_verified' or probe.get('model_digest')!=digest:failure='GPU preflight failed or digest changed; no CPU fallback.'
            done=0
            for frame,variant,system,user,budget in positions:
                target=response_path(base,spec,variant,frame);document=read_json(base/frame['frozen'])
                prepared=prepare_position(target,base/frame['frozen'],frame['input_sha256'],spec,digest,system,user,opts,profile['template'])
                if prepared is None:continue
                prepared['context_budget_before']=budget
                for candidate in candidates.get((frame['input_sha256'],spec['model']),[]):
                    row=read_json(candidate)
                    if validate_response(row,document['parsed']['objects'])['schema_valid_response'] and check_signature(row,prepared['signature'],document,frame['input_sha256'])['verified']:
                        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(candidate,target)
                        reused[str(target.relative_to(out))]={'source':str(candidate.relative_to(ROOT)),'sha256':sha256(candidate.read_bytes())}
                        write_json(out/'reused.json',reused);break
                if target.exists():continue
                row={**prepared,'model':spec['model'],'model_name':spec['name'],'precision':'BF16','seed':42,
                    'frame_id':frame['frame_id'],'prompt_variant':variant,'call_id':uuid.uuid4().hex,'generation_attempted':False}
                if failure or budget['status']!='estimated_sufficient':
                    row.update(status='skipped',error=failure or 'Context budget insufficient/unverifiable')
                else:
                    try:
                        called=True;row['generation_attempted']=True
                        row.update(backend.generate(system,user,opts));row['status']='response_received'
                    except Exception as exc:
                        row.update(getattr(exc,'result',{}),status='error',error=str(exc))
                        if not row.get('gpu_verified'):failure='GPU/API failure; remaining positions skipped without CPU fallback.'
                    finish_position(row,document,opts)
                    if not row['signature_check_after']['verified']:failure='Model/request signature drift; remaining positions skipped.'
                    if row['context_budget_after']['status']!='sufficient':row['context_warning']='Actual reserved context budget is insufficient/unverifiable.'
                write_json(target,row);done+=1
                if done%10==0:print(spec['model']+f': {done}/80 new positions saved',flush=True)
        finally:
            if called:backend.unload()
        export(base)
    verify(base)


def export(base=BASE):
    base=Path(base);original,manifest=prepare(base);out=base/'bf16';checks=[];records=[];new_records=[];body='';new=0;all_answers=0
    reused=read_json(out/'reused.json') if (out/'reused.json').exists() else {}
    for frame in original['inputs']:
        document=read_json(base/frame['frozen']);objects=document['parsed']['objects']
        body+='<section id="'+frame['frame_id']+'"><h2>'+html.escape(frame['frame_id'])+'</h2><img loading="lazy" src="../images/'+frame['frame_id']+'.jpg"><details><summary>Florence description</summary><p>'+html.escape(document['parsed']['scene_description'])+'</p></details><div class="cards">'
        positions=[]
        for spec in original['config']['models']:
            positions.append((spec,'old',42,base/'baseline'/spec['name']/(frame['frame_id']+'.json'),False))
            for seed in ([42,43,44] if frame['repeatability'] else [42]):
                positions.append((spec,'new',seed,base/'responses'/spec['name']/str(seed)/(frame['frame_id']+'.json'),False))
        for spec in manifest['models']:
            for variant in ('old','new'):positions.append((spec,variant,42,response_path(base,spec,variant,frame),True))
        for spec,variant,seed,path,is_bf16 in positions:
            if not path.exists():continue
            row=read_json(path);check=validate_response(row,objects);all_answers+=1
            if is_bf16:
                checks.append(check)
                new+=int(row.get('generation_attempted',False) and str(path.relative_to(out)) not in reused)
            decisions={json.dumps(o['id']):o for o in check['usable_items']}
            for obj in objects:
                decision=decisions.get(json.dumps(obj['id']),{})
                record={'frame_id':frame['frame_id'],'input_sha256':frame['input_sha256'],'region_id':json.dumps(obj['id']),
                    'florence_label':obj['label'],'model':spec['model'],'precision':spec['precision'],'prompt':variant,'seed':seed,
                    'classification':decision.get('classification'),'reason':decision.get('reason'),
                    'whole_response_valid':check['schema_valid_response'],'errors':'; '.join(check['errors']) or row.get('error','')}
                records.append(record)
                if is_bf16:new_records.append(record)
            card='<article><h3>'+html.escape(spec['model']+' / '+variant+f' / seed {seed}')+'</h3><p>'+html.escape(spec['precision'])+' · '+str(row.get('elapsed_seconds'))+' s · '+('schema valid' if check['schema_valid_response'] else 'INVALID / SKIPPED')+'</p><a href="'+html.escape(os.path.relpath(path,out))+'">Raw JSON</a><pre>'+html.escape(row.get('content',''))+'</pre><p>'+html.escape('; '.join(check['errors']) or row.get('error',''))+'</p></article>'
            body+=('<details><summary>Additional repetition</summary>'+card+'</details>') if seed!=42 else card
        body+='</div></section>'
    for name,rows in [('region-classes.csv',new_records),('all-region-classes.csv',records)]:
        if rows:
            with (out/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'index.html').write_text('<!doctype html><html lang="sl"><meta charset="utf-8"><title>All models / prompt40</title><style>body{font:17px Arial;margin:24px}img{max-width:100%;max-height:600px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border:1px solid #ccc;padding:12px}@media(max-width:850px){.cards{grid-template-columns:1fr}}</style><h1>40 frozen Florence inputs / all five LLM models / both prompts</h1><p><a href="../index.html">Original comparison</a> · <a href="all-region-classes.csv">All regions CSV</a> · <a href="region-classes.csv">BF16 CSV</a> · <a href="summary.json">BF16 summary</a>. Schema validity is not content accuracy. Optional manual review: python3 -m llm_object_classifier.review_viewer</p>'+body+'</html>')
    summary={**validity_counts(checks),'expected_positions':160,'saved_positions':len(checks),
        'new_main_calls':new,'reused_responses':len(reused),'preflight_calls':len(list((out/'preflights').glob('*.json'))),
        'all_answers_in_html':all_answers,'all_region_csv_rows':len(records),
        'base_manifest_sha256':manifest['base_manifest_sha256']}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


def verify(base=BASE):
    """Check completeness/provenance without another call or a human-review DB."""
    base=Path(base);original,manifest=prepare(base);out=base/'bf16';opts=options(original['config'],next(combinations(original['config'])));opts['seed']=42
    ids=set();calls=[];contexts=[];expected=[]
    for spec in manifest['models']:
        metadata=read_json(out/'models'/(spec['name']+'.json'));digest=metadata['digest']
        assert metadata['metadata']['show']['details']['quantization_level']=='BF16'
        probe=read_json(out/'preflights'/(spec['name']+'.json'))
        assert probe['status']=='gpu_verified' and probe['gpu_verified'] and probe['gpu']['digest']==digest
        assert probe['gpu']['context_length']==8192 and probe['gpu']['size_vram']>=probe['gpu']['size']
        ids.add(probe['call_id'])
        for frame in original['inputs']:
            document=read_json(base/frame['frozen'])
            for variant in ('old','new'):
                path=response_path(base,spec,variant,frame);row=read_json(path)
                system=(base/'prompts'/(variant+'-system.txt')).read_text();user=(base/frame['user_prompt']).read_text()
                signature=make_signature(document,frame['input_sha256'],spec['model'],digest,system,user,opts)
                assert check_signature(row,signature,document,frame['input_sha256'])['verified']
                if row.get('generation_attempted'):
                    assert row['call_id'] not in ids;ids.add(row['call_id']);calls.append(row['call_id'])
                if row['status']=='response_received':
                    assert row['gpu_verified'] and row['gpu']['size_vram']>=row['gpu']['size'] and row['gpu']['context_length']==8192
                    after=context_after(row,opts);assert after['status']=='sufficient'
                    assert row['response']['prompt_eval_count']<=row['context_budget_before']['estimated_prompt_tokens']
                    contexts.append(after)
                expected.append((frame['frame_id'],spec['model'],variant,42))
    assert len(expected)==160 and len(set(expected))==160 and len({f['image_sha256'] for f in original['inputs']})==40
    # Existing published source checksums include all earlier main/repeat responses.
    previous=ROOT/'results models/technical_revision_v1/source-sha256.json'
    protected=read_json(previous)
    for path,digest in protected.items():assert sha256((ROOT/path).read_bytes())==digest,'Changed old file: '+path
    class Page(HTMLParser):
        def __init__(self):super().__init__();self.answers=0
        def handle_starttag(self,tag,attrs):
            if tag=='pre':self.answers+=1
            for key,value in attrs:
                if key in ('href','src') and not value.startswith('#'):assert (out/value).is_file(),value
    page=Page();page.feed((out/'index.html').read_text());assert page.answers==460
    with (out/'all-region-classes.csv').open(newline='') as f:region_rows=sum(1 for _ in csv.DictReader(f))
    from .review_viewer import Results
    viewer=Results(base);assert len(viewer.frames)==40 and len(viewer.models)==5
    assert sum(len(f['answers']) for f in viewer.frames)==460
    assert viewer.set_id==sha256(('comparison40\0'+manifest['base_manifest_sha256']).encode())
    report={'images':40,'expected_bf16_positions':160,'saved_bf16_positions':len(expected),
        'unique_new_main_call_ids':len(calls),'preflight_calls':2,'full_answers_in_html':page.answers,
        'combined_region_csv_rows':region_rows,'html_links_valid':True,'protected_old_files_unchanged':len(protected),
        'shared_review_namespace':viewer.set_id,'actual_review_database_opened':False,
        'all_received_contexts_sufficient':len(contexts),'quality_evaluated':False}
    write_json(out/'verification.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','run','export'))
    parser.add_argument('--base-dir',type=Path,default=BASE);args=parser.parse_args()
    if args.phase=='run':run(args.base_dir)
    elif args.phase=='export':
        export(args.base_dir);verify(args.base_dir)
    else:prepare(args.base_dir)
