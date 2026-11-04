"""Aktivni pipeline: poljubna RGB mapa → Florence opis/grounding → en besedilni model."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from .experiment import read_json,write_json,sha256,options,finite_number
from .grounding_common import CAPTION,GROUND,draw,freeze,weights_identity,make_document
from .florence_runtime import task_signature,check_native
from .llm_checks import prepare_position,finish_position,validate_response,validity_counts,installed_digest
from .vision_backends import Backend,request
from .vision_prompts import reason_prompt
from .ollama_runtime import runner_http,load_model

ROOT=Path(__file__).resolve().parents[1]
EXTENSIONS={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
SOURCES=['pipeline.py','florence_runtime.py','grounding_common.py','gpu_runtime.py','ollama_runtime.py',
         'llm_checks.py','vision_backends.py','vision_prompts.py','experiment.py']


def load_settings(path):
    """Edino aktivno konfiguracijo razreši glede na koren projekta; prompt ostane nespremenjen."""
    cfg=read_json(path);fl=cfg['florence'];llm=cfg['llm']
    if fl['model']!='microsoft/Florence-2-large' or fl['precision']!='FP16' or fl['attention']!='eager' or fl['do_sample'] is not False:
        raise ValueError('Aktivna pot podpira preverjeni Florence FP16/eager deterministični način')
    for name in ('max_new_tokens','num_beams'):finite_number(fl[name],name,lower=1,integer=True)
    for name in ('max_tokens','num_ctx','timeout'):finite_number(llm[name],name,lower=1,integer=True)
    finite_number(llm['seed'],'seed',lower=-2147483648,integer=True)
    finite_number(llm['temperature'],'temperature',upper=2)
    finite_number(llm['top_p'],'top_p',upper=1);finite_number(llm['top_k'],'top_k',integer=True)
    finite_number(llm['presence_penalty'],'presence_penalty',upper=2)
    if llm['top_p']==0:raise ValueError('top_p mora biti večji od 0')
    fl['model_dir']=str((ROOT/fl['model_dir']).resolve())
    for key in ('system_file','instruction_file'):
        p=(ROOT/llm[key]).resolve();llm[key]=str(p)
        llm[key+'_text']=p.read_text(encoding='utf-8')
        if key=='instruction_file':llm[key+'_text']=llm[key+'_text'].strip()
        if not llm[key+'_text'].strip():raise ValueError(f'Prazen prompt: {p}')
    if llm['selected'] not in llm['alternatives']:raise ValueError('Neznan izbrani besedilni model')
    return cfg


def prepare_run(cfg, output, input_folder=None, florence_from=None):
    """Zamrzne konfiguracijo in originalne slike. Ob nadaljevanju ničesar ne zamenjuje."""
    from PIL import Image
    output=output.resolve();old=read_json(output/'run.json') if (output/'run.json').exists() else None
    source=read_json(florence_from/'run.json') if florence_from else None
    cfg['input']=str((input_folder or (Path(old['config']['input']) if old else Path(source['config']['input']) if source else ROOT/cfg['input'])).resolve())
    cfg['output']=str(output)
    if old:
        if old['config']!=cfg:raise ValueError('Spremenjena konfiguracija ali prompt; uporabi novo izhodno mapo')
        for path,digest in old['source_sha256'].items():
            if sha256((ROOT/path).read_bytes())!=digest:raise ValueError('Spremenjen izvajalnik; uporabi novo izhodno mapo ali samo --stage export')
        for f in old['frames']:
            if sha256((output/f['image_path']).read_bytes())!=f['image_sha256']:raise ValueError('Spremenjena shranjena slika')
        return old
    if output.exists() and any(output.iterdir()):raise ValueError('Nova izhodna mapa mora biti prazna')
    output.mkdir(parents=True,exist_ok=True)
    if source:
        if source['config']['florence']!=cfg['florence'] or source['config']['llm']['seed']!=cfg['llm']['seed']:
            raise ValueError('Shranjeni Florence ima drugačne nastavitve ali seme')
        frames=source['frames'];identity=source['florence_weights']
        shutil.copytree(florence_from/'images',output/'images')
        shutil.copytree(florence_from/'native',output/'native')
        shutil.copyfile(florence_from/'florence-environment.json',output/'florence-environment.json')
    else:
        folder=Path(cfg['input'])
        if not folder.is_dir() or output.is_relative_to(folder):raise ValueError('Vhod mora biti RGB mapa, izhod pa zunaj nje')
        identity=weights_identity(cfg['florence']['model_dir']);frames=[];seen=set()
        for image in sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS):
            digest=sha256(image.read_bytes())
            if digest in seen:raise ValueError(f'Podvojena slika po SHA256: {image}')
            seen.add(digest);fid='image-'+digest[:16];local=Path('images')/(fid+image.suffix.lower())
            with Image.open(image) as opened:size=list(opened.size)
            (output/'images').mkdir(exist_ok=True);shutil.copyfile(image,output/local)
            frames.append({'frame_id':fid,'image_path':str(local),'image_sha256':digest,'image_size':size,'source_path':str(image)})
        if not frames:raise ValueError('V vhodni mapi ni podprtih slik')
    sources={f'llm_object_classifier/{name}':sha256((ROOT/'llm_object_classifier'/name).read_bytes()) for name in SOURCES}
    if source and sources['llm_object_classifier/florence_runtime.py']!=source['source_sha256']['llm_object_classifier/florence_runtime.py']:
        raise ValueError('Neujemanje izvajalnika shranjenega Florence koraka')
    run={'protocol':'florence-qwen-run-v1','config':cfg,'frames':frames,'florence_weights':identity,'source_sha256':sources,
         'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'python':sys.version}
    if source:run['stage1_origin']={'path':str(florence_from),'run_sha256':sha256((florence_from/'run.json').read_bytes())}
    for name in SOURCES:
        target=output/'source'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'llm_object_classifier'/name,target)
    freeze(output/'run.json',run);return run


def prepare_documents(output, run):
    """Opis, fraze, originalni okvirji in izvor nalog postanejo isti zamrznjeni LLM vhod."""
    cfg=run['config'];llm=cfg['llm'];frames=[]
    for frame in run['frames']:
        if sha256((output/frame['image_path']).read_bytes())!=frame['image_sha256']:raise ValueError('Spremenjena shranjena RGB slika')
        fid=frame['frame_id'];cp=output/'native'/fid/'caption.json';gp=output/'native'/fid/'grounding.json'
        if not check_native(cp,task_signature(run,frame,CAPTION)) or not check_native(gp,task_signature(run,frame,GROUND,cp)):
            raise ValueError('Manjka Florence rezultat; najprej izvedi --stage florence')
        doc=make_document(frame,read_json(cp),read_json(gp));frozen=Path('frozen')/(fid+'.json');freeze(output/frozen,doc)
        objects=doc['parsed']['objects'];user=reason_prompt(fid,objects,doc['parsed'],llm['instruction_file_text'])
        user_path=Path('prompts/users')/(fid+'.txt');(output/user_path).parent.mkdir(parents=True,exist_ok=True)
        if (output/user_path).exists() and (output/user_path).read_text()!=user:raise ValueError('Spremenjen uporabniški prompt')
        if not (output/user_path).exists():(output/user_path).write_text(user,encoding='utf-8')
        annotated=output/'annotated'/(fid+'.jpg');annotated.parent.mkdir(exist_ok=True)
        if not annotated.exists():draw(output/frame['image_path'],objects,annotated)
        frames.append({'frame_id':fid,'dataset':'pipeline','environment':'','frozen':str(frozen),
                       'input_sha256':sha256((output/frozen).read_bytes()),'image_sha256':frame['image_sha256'],
                       'image_path':frame['image_path'],'user_prompt':str(user_path)})
    spec={'name':llm['selected'],'model':llm['alternatives'][llm['selected']],'backend':'ollama','precision':'Q4_K_M','require_gpu':True}
    manifest={'protocol':'florence-qwen-v1','seed':llm['seed'],'config':{'models':[spec]},'inputs':frames,
              'run_sha256':sha256((output/'run.json').read_bytes())}
    freeze(output/'manifest.json',manifest)
    system=output/'prompts/system.txt'
    if system.exists() and system.read_text()!=llm['system_file_text']:raise ValueError('Spremenjen sistemski prompt')
    if not system.exists():system.write_text(llm['system_file_text'],encoding='utf-8')
    return manifest


def run_text(output,run,manifest):
    """Izvede samo izbrani model; podpisi se preverijo pred nalaganjem GPU ali novim klicem."""
    cfg=run['config']['llm'];url=cfg['base_url']
    if url=='auto':
        ip=subprocess.check_output(['docker','inspect',cfg['container'],'--format','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'],text=True).strip()
        url=f'http://{ip}:11434'
    spec={**manifest['config']['models'][0],'base_url':url};backend=Backend(spec,cfg['timeout'])
    metadata=backend.metadata();digest=installed_digest(backend);system=cfg['system_file_text']
    freeze(output/'model-metadata.json',{'digest':digest,**metadata})
    opts=options(cfg,{'temperature':cfg['temperature'],'seed':cfg['seed']});pending=[]
    for frame in manifest['inputs']:
        user=(output/frame['user_prompt']).read_text();path=output/'responses'/spec['name']/(frame['frame_id']+'.json')
        row=prepare_position(path,output/frame['frozen'],frame['input_sha256'],spec,digest,system,user,opts,metadata['show'].get('template'))
        if row is not None:pending.append((frame,path,user,row))
    if not pending:print('LLM: vsi podpisi ustrezajo; 0 novih inferenc.',flush=True);return
    try:
        setup=load_model(backend,opts,digest,cfg['container'])
        write_json(output/'environments'/f'llm-{time.time_ns()}.json',setup)
        for frame,path,user,row in pending:
            doc=read_json(output/frame['frozen']);row.update(model=spec['model'],model_name=spec['name'],frame_id=frame['frame_id'],
                seed=cfg['seed'],prompt_variant='new',call_id=uuid.uuid4().hex,generation_attempted=False)
            try:
                if doc['status']!='ok' or not doc['parsed']['objects']:row.update(status='skipped',error='Neveljaven ali prazen Florence vhod')
                else:
                    tokens=runner_http(setup['tokenizer_port'],'/tokenize',{'content':system+'\n'+user,'add_special':False},cfg['container'])['tokens']
                    estimate=len(tokens)+256
                    row['context_budget_before']={'status':'estimated_sufficient' if estimate+opts['num_predict']<=opts['num_ctx'] else 'estimated_insufficient',
                        'method':'native_model_tokenizer_content_plus_256_chat_margin','exact':False,'content_tokens':len(tokens),
                        'estimated_prompt_tokens':estimate,'reserved_output_tokens':opts['num_predict'],'num_ctx':opts['num_ctx']}
                    if estimate+opts['num_predict']>opts['num_ctx']:row.update(status='skipped',error='Nezadosten kontekst; brez krajšanja ali povečanja')
                    else:
                        row['generation_attempted']=True;row.update(backend.generate(system,user,opts));row['status']='response_received'
                        finish_position(row,doc,opts)
                        if not row['signature_check_after']['verified']:raise ValueError('Podpis prejetega odgovora ne ustreza zahtevi')
                        if row['context_budget_after']['status']!='sufficient' or not row['context_budget_after']['context_matches_request']:
                            raise ValueError('Dejanski proračun konteksta ni potrjen')
            except Exception as exc:row.update(getattr(exc,'result',{}),status='error',error=str(exc))
            write_json(path,row)
            print(f"LLM: {frame['frame_id']} / {row['status']}",flush=True)
            if row['status']=='error':raise RuntimeError('Napaka je shranjena; nadaljnji klici ustavljeni: '+row['error'])
    finally:backend.unload()


def export_results(output):
    """Brez GPU, omrežja ali uteži izvozi celotne odgovore ter tehnično veljavnost, ne pravilnosti."""
    manifest=read_json(output/'manifest.json');spec=manifest['config']['models'][0];checks=[];rows=[];records=[];missing=0
    body='<h1>Florence → '+html.escape(spec['model'])+'</h1><p>Regije niso potrjene instance. Veljaven zapis ni dokaz pravilnosti.</p>'
    for frame in manifest['inputs']:
        fid=frame['frame_id'];doc=read_json(output/frame['frozen']);path=output/'responses'/spec['name']/(fid+'.json')
        raw=read_json(path) if path.exists() else {};missing+=not path.exists();check=validate_response(raw,doc['parsed']['objects'])
        if raw:checks.append(check)
        records.append({'frame_id':fid,'input_sha256':frame['input_sha256'],'florence':doc,'llm':raw,'validation':check})
        body+='<section><h2>'+fid+'</h2><img loading="lazy" src="annotated/'+fid+'.jpg"><p>'+html.escape(doc['parsed']['scene_description'])+'</p>'
        error='; '.join(check['errors']+[raw['error']] if raw.get('error') else check['errors'])
        body+='<p class="error">'+html.escape(error or ('Še ni LLM odgovora' if not raw else ''))+'</p><p>Čas: '+str(raw.get('elapsed_seconds'))+' s; '+html.escape(str(raw.get('status','missing')))+'</p>'
        body+='<table><tr><th>ID</th><th>Florence oznaka / bbox</th><th>Razred</th><th>Celotna utemeljitev</th></tr>'
        for obj in doc['parsed']['objects']:
            decision=next((i for i in check['usable_items'] if type(i['id']) is type(obj['id']) and i['id']==obj['id']),{})
            row={'frame_id':fid,'region_id':obj['id'],'label':obj['label'],'bbox_xyxy':json.dumps(obj['bbox_xyxy']),
                 'model':spec['model'],'input_sha256':frame['input_sha256'],'seed':manifest['seed'],
                 'classification':decision.get('classification'),'reason':decision.get('reason'),'schema_valid':check['schema_valid_response'],
                 'status':raw.get('status','missing'),'error':error};rows.append(row)
            body+='<tr>'+''.join('<td>'+html.escape(str(v if v is not None else ''))+'</td>' for v in (obj['id'],obj['label']+' / '+str(obj['bbox_xyxy']),row['classification'],row['reason']))+'</tr>'
        body+='</table><details><summary>Celoten surovi odgovor</summary><pre>'+html.escape(raw.get('content',''))+'</pre></details></section>'
    with (output/'regions.csv').open('w',newline='',encoding='utf-8') as f:
        fields=list(rows[0]) if rows else ['frame_id','region_id','classification','reason'];writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(output/'results.json',records)
    summary={'images':len(manifest['inputs']),'expected_llm_positions':len(manifest['inputs']),'missing_positions':missing,**validity_counts(checks)}
    write_json(output/'summary.json',summary)
    (output/'index.html').write_text('<!doctype html><html lang="sl"><meta charset="utf-8"><title>Pipeline</title><style>body{font:18px Arial;max-width:1400px;margin:24px auto}img{max-width:100%;max-height:800px}pre{white-space:pre-wrap}td,th{border:1px solid #bbb;padding:8px}table{border-collapse:collapse}.error{color:#a00}section{margin-bottom:50px}</style>'+body+'</html>',encoding='utf-8')
    print(json.dumps(summary),flush=True);return summary


def main():
    """En ukaz za oba koraka, posamična koraka ali povsem offline izvoz."""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/pipeline.json')
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--model',choices=('qwen','gemma'))
    p.add_argument('--stage',choices=('all','florence','llm','export'),default='all');p.add_argument('--florence-from',type=Path)
    args=p.parse_args();cfg=load_settings(args.config);output=(args.output or ROOT/cfg['output']).resolve()
    if args.stage=='export':export_results(output);return
    if args.model:cfg['llm']['selected']=args.model
    if args.florence_from and args.stage!='llm':p.error('--florence-from je namenjen samo --stage llm')
    run=prepare_run(cfg,output,args.input,args.florence_from.resolve() if args.florence_from else None)
    try:
        if args.stage in ('all','florence'):
            command=['docker','run','--rm','--gpus','all','--ipc=host','--user',f'{__import__("os").getuid()}:{__import__("os").getgid()}',
                '-e','HOME=/tmp','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','PYTHONDONTWRITEBYTECODE=1',
                '-v',f'{ROOT}:{ROOT}','-v',f'{output}:{output}','-v',f'{cfg["florence"]["model_dir"]}:{cfg["florence"]["model_dir"]}:ro',
                '-w',str(ROOT),cfg['florence']['docker_image'],'python','-m','llm_object_classifier.florence_runtime','--output',str(output)]
            subprocess.run(command,check=True)
        manifest=prepare_documents(output,run)
        if args.stage in ('all','llm'):run_text(output,run,manifest)
    except Exception as exc:
        write_json(output/'errors'/f'{time.time_ns()}.json',{'stage':args.stage,'error':str(exc),'traceback':traceback.format_exc(),
            'details':getattr(exc,'result',{})})
        raise
    finally:
        if (output/'manifest.json').exists():export_results(output)


if __name__=='__main__':main()
