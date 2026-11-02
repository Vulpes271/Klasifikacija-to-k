"""One text-only call per saved Florence input/model; raw answers for human review."""
import argparse
import html
import os
from pathlib import Path
import shutil

from .description_prompts import validate_description
from .experiment import combinations, load_config, options, read_json, sha256, write_json
from .thud_experiment import immutable_json
from .thud_extended_common import ROOT
from .vision_backends import Backend
from .vision_pipeline import GPURequiredError
from .vision_prompts import reason_prompt

SOURCE=ROOT/'results models/florence_caption_grounding_test10'
OUTPUT=ROOT/'results models/florence_grounding_llm_comparison10'
CONFIG=ROOT/'configs/stage2/florence-grounding-qwen-size.json'


def prepare():
    config=load_config(CONFIG,2)
    for spec in config['models']:
        spec['base_url']=os.environ.get('THUD_OLLAMA_URL',spec['base_url'])
    frames=[]
    for path in sorted((SOURCE/'stage1').glob('frame-*.json')):
        saved=read_json(path);validate_description(saved['parsed'])
        assert saved['stage']==1 and saved['status']=='ok'
        frozen=OUTPUT/'frozen'/path.name;frozen.parent.mkdir(parents=True,exist_ok=True)
        if frozen.exists():assert frozen.read_bytes()==path.read_bytes(),'Changed frozen input'
        else:shutil.copyfile(path,frozen)
        objects=[{'id':o['id'],'label':o['label']} for o in saved['parsed']['objects']]
        prompt=reason_prompt(saved['frame_id'],objects,saved['parsed'],config['instruction_file_text'])
        frames.append({'frame_id':saved['frame_id'],'source':str(path.relative_to(ROOT)),
            'frozen':str(frozen.relative_to(OUTPUT)),'sha256':sha256(path.read_bytes()),'user_prompt':prompt})
    assert len(frames)==10
    immutable_json(OUTPUT/'manifest.json',{'config':config,'inputs':frames,
        'note':'Existing stage1 bytes. No images/reference classes sent. One call per model/input; separate short GPU preflight.'})
    return config,frames


def preflight(config,frames):
    """One short output with the longest actual input and the full context allocation."""
    target=OUTPUT/'preflight.json'
    if target.exists():
        assert read_json(target).get('gpu_verified'),'Previous GPU preflight failed'
        return
    spec=config['models'][0];backend=Backend(spec,config['timeout'])
    write_json(OUTPUT/'models'/(spec['name']+'.json'),backend.metadata())
    frame=max(frames,key=lambda f:len(f['user_prompt']))
    opts=options(config,next(combinations(config)));opts['num_predict']=16
    row={'model':spec['model'],'input_sha256':frame['sha256'],'frame_id':frame['frame_id'],
        'excluded_from_comparison':True,'note':'16 output tokens only; full configured context; incomplete JSON is expected.'}
    try:
        row.update(backend.generate(config['system_file_text'],frame['user_prompt'],opts))
    except Exception as exc:
        row.update(getattr(exc,'result',{}),error=str(exc));write_json(target,row)
        raise
    write_json(target,row)
    assert row['gpu_verified']
    # The actual prompt must leave the full comparison output budget available.
    assert row['response']['prompt_eval_count']+config['max_tokens']<config['num_ctx']
    print('Short preflight: complete GPU allocation verified at num_ctx='+str(config['num_ctx']),flush=True)


def run(config,frames):
    preflight(config,frames)
    opts=options(config,next(combinations(config)))
    for spec in config['models']:
        backend=Backend(spec,config['timeout'])
        write_json(OUTPUT/'models'/(spec['name']+'.json'),backend.metadata())
        try:
            for frame in frames:
                target=OUTPUT/'responses'/spec['name']/(frame['frame_id']+'.json')
                if target.exists():continue  # Preserve every completed attempt, including errors.
                row={'stage':2,'frame_id':frame['frame_id'],'model':spec['model'],
                    'precision':spec['precision'],'input_sha256':frame['sha256'],'frozen_input':frame['frozen']}
                print(spec['model']+' / '+frame['frame_id'],flush=True)
                try:
                    row.update(backend.generate(config['system_file_text'],frame['user_prompt'],opts))
                    row['status']='response_received'  # No interpretation or quality analysis.
                except Exception as exc:
                    row.update(getattr(exc,'result',{}),status='error',error=str(exc))
                    write_json(target,row)
                    if isinstance(exc,GPURequiredError):raise
                write_json(target,row)
        finally:
            backend.unload()  # Compare sequentially; leave space for the next model.


def export(config,frames):
    escape=html.escape
    body='<h1>Florence grounding → Qwen3.5: 27B in 4B</h1><p>Isti shranjeni vhodi in prompt; T=0,3, seme 42, kontekst 8192, največ 2048 izhodnih tokenov, think=false. Ena izvedba na model in sliko. Čas je celoten API klic (vključno z morebitnim nalaganjem). Kakovost odgovorov ni ocenjena.</p>'
    count=0
    for frame in frames:
        fid=frame['frame_id'];image=OUTPUT/'images'/(fid+'.jpg');image.parent.mkdir(exist_ok=True)
        shutil.copyfile(SOURCE/fid/'caption-grounding.predictions.jpg',image)
        body+='<section><h2>'+escape(fid)+'</h2><a href="images/'+fid+'.jpg"><img src="images/'+fid+'.jpg" alt="Obstoječe napovedi Florence"></a><div class="answers">'
        for spec in config['models']:
            path=OUTPUT/'responses'/spec['name']/(fid+'.json');row=read_json(path)
            assert row['input_sha256']==frame['sha256']
            request=row.get('request',{})
            if request:
                assert request['messages']==[{'role':'system','content':config['system_file_text']},{'role':'user','content':frame['user_prompt']}]
                assert request['options']==options(config,next(combinations(config)))
            if row['status']=='response_received':
                assert row['gpu_verified']
                assert row['response']['prompt_eval_count']+config['max_tokens']<config['num_ctx']
            content=row.get('content',row.get('response_text',''))
            path.with_suffix('.txt').write_text(content,encoding='utf-8')
            elapsed=row.get('elapsed_seconds')
            seconds=f'{elapsed:.3f} s' if elapsed is not None else 'ni meritve'
            body+='<article><h3>'+escape(spec['model'])+'</h3><p>'+escape(spec['precision'])+' · '+seconds+' · '+escape(row['status'])+'</p><a href="'+str(path.relative_to(OUTPUT))+'">Celoten surovi zapis</a><pre>'+escape(content)+'</pre>'
            if row.get('error'):body+='<p class="error">'+escape(row['error'])+'</p>'
            body+='</article>';count+=1
        body+='</div></section>'
    assert count==20
    (OUTPUT/'index.html').write_text('<!doctype html><html lang="sl"><meta charset="utf-8"><title>Qwen3.5 27B / 4B: deset Florence vhodov</title><style>body{font-family:Arial;margin:24px auto;padding:12px;max-width:1600px}section{border-top:2px solid #aaa;margin-top:30px}img{max-width:100%;max-height:600px}.answers{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:12px}.error{color:#a00}@media(max-width:800px){.answers{grid-template-columns:1fr}}</style>'+body+'</html>',encoding='utf-8')
    for frame in frames:
        assert sha256((ROOT/frame['source']).read_bytes())==frame['sha256']
    write_json(OUTPUT/'verification.json',{'images':10,'comparison_records':count,
        'short_preflight_calls':1,'stage1_rerun':False,'source_inputs_unchanged':True,
        'interrupted_setup_api_attempts':int((OUTPUT/'setup/preflight-before-gpu-restart.json').exists()),
        'identical_prompts_and_settings':True,'quality_analysis':False})
    print('Saved HTML:',OUTPUT/'index.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','preflight','run','export'))
    args=parser.parse_args();config,frames=prepare()
    if args.phase=='preflight':preflight(config,frames)
    elif args.phase=='run':run(config,frames);export(config,frames)
    elif args.phase=='export':export(config,frames)
