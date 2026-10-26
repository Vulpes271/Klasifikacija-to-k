"""Thin repeat protocol over the existing native and two-stage runners."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from llm_object_classifier import stage1_describe, stage2_reason
from llm_object_classifier.experiment import read_json, write_json, sha256, load_config
from llm_object_classifier.thud_experiment import immutable_json, MODELS
from llm_object_classifier.thud_extended_common import ROOT, frames, snapshot, progress
from llm_object_classifier.vision_backends import Backend, request

OUT=ROOT/'results/ponovitve'
REPORT=ROOT/'reports/PRIMERJAVA_PONOVITEV'
CONFIG=ROOT/'configs/ponovitve'
REPEATS={'r1':42,'r2':43,'r3':44}
NAMES=['florence','nvila']+[s['name'] for s in MODELS]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=sorted((f for f in frames() if f['subset20']),key=lambda f:(f['index'],f['dataset']))
    for f in selected:
        f['source_frame_id']=f['frame_id']
        f['frame_id']=f['dataset']+'/'+f['frame_id']
    immutable_json(OUT/'frames.json',selected)
    inputs=OUT/'inputs.jsonl'
    text=''.join(json.dumps({'frame_id':f['frame_id'],'image_path':os.path.relpath(f['image_path'],OUT),
                           'image_size':f['image_size'],'image_sha256':f['image_sha256']})+'\n' for f in selected)
    if inputs.exists() and inputs.read_text()!=text:
        raise ValueError('Repeat image selection changed')
    if not inputs.exists(): inputs.write_text(text)
    immutable_json(OUT/'plan.json',read_json(CONFIG/'protocol.json'))
    old=OUT/'previous-results-index.json'
    if not old.exists():
        paths=[p for name in ('thud-vmesno','thud-part2','thud-extended')
               for p in (ROOT/'results'/name).rglob('frame-*.json') if 'vendor' not in p.parts]
        immutable_json(old,{str(p.relative_to(ROOT)):sha256(p.read_bytes()) for p in sorted(paths)})
    return selected


def base_url():
    return os.environ['THUD_OLLAMA_URL']


def config(spec,repeat,stage,prompt='short'):
    protocol=read_json(CONFIG/'protocol.json')
    raw={k:protocol[k] for k in ('max_tokens','num_ctx','top_p','top_k','presence_penalty')}
    raw.update(models=[{**spec,'base_url':base_url(),'require_gpu':True}],
               temperatures=[.3],seeds=[REPEATS[repeat]],timeout=600,
               system_file=str(CONFIG/('description-v1.txt' if stage==1 else prompt+'-no-examples.txt')))
    if stage==1: raw['description_schema']='legacy'
    else: raw['instruction_file']=str(CONFIG/'reasoning.txt')
    path=OUT/'configs'/f'{stage}-{spec["name"]}-{prompt}-{repeat}.json'
    immutable_json(path,raw)
    return load_config(path,stage)


def ollama(model,repeat,stage=1,pilot=False,prompt='short'):
    spec=next(s for s in MODELS if s['name']==model)
    destination=OUT/('stage1' if stage==1 else 'stage2')/model
    destination=destination/repeat if stage==1 else destination/prompt/repeat
    complete=destination/'complete.json'
    cfg=config(spec,repeat,stage,prompt)
    if complete.exists():
        previous=read_json(destination/'manifest.json')
        if previous['config']!=cfg: raise ValueError('Completed configuration changed')
        if stage==1 and previous['input_sha256']!=sha256((OUT/'inputs.jsonl').read_bytes()):
            raise ValueError('Completed image input changed')
        if stage==2:
            expected=[{'path':str(p),'sha256':sha256(p.read_bytes())} for p in sorted((OUT/'frozen/qwen35-bf16-r1').glob('frame-*.json'))]
            if previous['source_captions']!=expected: raise ValueError('Completed frozen input changed')
        return
    snapshot(destination/'provenance',cfg,read_json(OUT/'frames.json'))
    args=SimpleNamespace(input=OUT/'inputs.jsonl',captions=OUT/'frozen/qwen35-bf16-r1',
                         output_dir=destination,limit=20,frame_id=None,resume=True,
                         warmup=True,fresh_warmup=True,max_new_calls=2 if pilot else None)
    progress(f'Ponovitve: stage{stage} {model} {prompt} {repeat}, pilot={pilot}')
    try:
        code=(stage1_describe if stage==1 else stage2_reason).run(args,cfg)
    finally:
        Backend(cfg['models'][0],cfg['timeout']).unload()
    if code==1: raise RuntimeError('GPU/model preparation failed; partial results retained')
    rows=list(destination.glob('*/t*/frame-*.json'))
    if not pilot:
        if len(rows)!=20: raise RuntimeError('Incomplete group')
        write_json(complete,{'positions':20,'exit_code':code,'repeat':repeat,'seed':REPEATS[repeat]})


def native(model,repeat,pilot=False):
    destination=OUT/'stage1'/model/repeat
    args=['--selection',str(OUT/'frames.json'),'--output-dir',str(destination),
          '--repeat',repeat,'--seed',str(REPEATS[repeat]),'--limit','2' if pilot else '20']
    if model=='nvila':
        args+=['--system-file',str(CONFIG/'nvila-description-v2.txt'),'--schema','legacy',
               '--temperature','0.3','--max-new-tokens','4096']
    env=['-e','HOME=/tmp','-e','PYTHONFAULTHANDLER=1',
         '-e','THUD_PROGRESS_FILE='+str(OUT/'progress.md'),
         '-e','THUD_SOURCE_COMMIT='+subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()]
    if model=='nvila': env+=['-e',f'PYTHONPATH={ROOT}/results/thud-extended/vendor/VILA:{ROOT}']
    cmd=['docker','run','--rm','--gpus','all','--name','thud-repeat-'+model,
         '--user',f'{os.getuid()}:{os.getgid()}',*env,'-v',f'{ROOT}:{ROOT}','-w',str(ROOT),
         'thud-'+model+':local','python','-m','llm_object_classifier.thud_'+model,*args]
    for attempt in range(2):
        code=subprocess.run(cmd).returncode
        if code==0: return
        write_json(OUT/'process-errors'/f'{model}-{repeat}-{pilot}-{time.time_ns()}-attempt{attempt}.json',
                   {'exit_code':code,'command':cmd,'note':'Resume only missing files; saved errors retained.'})
    raise RuntimeError(f'{model} failed twice; outputs retained')


def freeze():
    source=OUT/'stage1/qwen35-bf16/r1/qwen35-bf16/t0.3-s42'
    index=[]
    for i,f in enumerate(read_json(OUT/'frames.json'),1):
        path=source/f'frame-{i:06d}.json'
        original=read_json(path)
        if original['image_sha256']!=f['image_sha256']: raise ValueError('Frozen input pairing changed')
        # Freeze a portable transport record with the EXACT original description.
        # This same file is published byte-for-byte, so its SHA is verifiable.
        row={k:original[k] for k in ('stage','frame_id','model_name','model','variant',
                                     'status','image_sha256','content') if k in original}
        for key in ('parsed','error'):
            if key in original: row[key]=original[key]
        row.update(source_record=str(path.relative_to(ROOT)),source_record_sha256=sha256(path.read_bytes()))
        if row.get('status')=='ok' and not row['parsed']['objects']:
            row={**row,'status':'error','error':'Empty object list is unusable for classification.'}
        target=OUT/'frozen/qwen35-bf16-r1'/path.name
        immutable_json(target,row)
        index.append({'frame_id':f['frame_id'],'source':str(path.relative_to(ROOT)),
                      'source_sha256':sha256(path.read_bytes()),'frozen_sha256':sha256(target.read_bytes()),
                      'status':row['status']})
    immutable_json(OUT/'frozen/index.json',index)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','pilot','describe','freeze','reason','export','verify','all'],default='all')
    parser.add_argument('--model',choices=NAMES+['all'],default='all')
    parser.add_argument('--repeat',choices=list(REPEATS)+['all'],default='all')
    args=parser.parse_args()
    os.environ['THUD_PROGRESS_FILE']=str(OUT/'progress.md')
    # Text-only reasoning needs the saved selection metadata, never RGB files.
    if args.phase in ('freeze','reason','export','verify') and (OUT/'frames.json').exists():
        selected=read_json(OUT/'frames.json')
    else:
        selected=prepare()
    if args.phase=='prepare': return
    if args.phase=='verify':
        from llm_object_classifier.thud_extended_report import repetitions_verify
        repetitions_verify()
        return
    models=NAMES if args.model=='all' else [args.model]
    repeats=list(REPEATS) if args.repeat=='all' else [args.repeat]
    if args.phase in ('pilot','describe','all'):
        for model in models:
            if args.phase in ('pilot','all'):
                (native if model in ('nvila','florence') else ollama)(model,'r1',pilot=True)
            if args.phase in ('describe','all'):
                for repeat in repeats:
                    (native if model in ('nvila','florence') else ollama)(model,repeat)
    if args.phase in ('freeze','reason','all'):
        freeze()
    if args.phase in ('reason','all'):
        # First two interleaved images cover both datasets; saved pilot is r1.
        ollama('qwen35-bf16','r1',stage=2,prompt='short',pilot=True)
        for model in ('qwen35-bf16','qwen35-q4'):
            for prompt in ('short','long'):
                for repeat in REPEATS:
                    ollama(model,repeat,stage=2,prompt=prompt)
    if args.phase in ('export','all'):
        from llm_object_classifier.thud_extended_report import repetitions_export
        repetitions_export()


if __name__=='__main__': main()
