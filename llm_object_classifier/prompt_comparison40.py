"""Frozen 40-image prompt comparison and ten-image repeatability; no vision calls."""
import argparse
from collections import Counter
import csv
import html
import json
from html.parser import HTMLParser
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from .experiment import read_json,write_json,sha256,options,combinations,strict_json
from .florence_grounding_llm import OUTPUT as BASE, SOURCE, table_decisions
from .thud_experiment import immutable_json
from .thud_extended_common import ROOT
from .vision_backends import Backend,request
from .vision_pipeline import GPURequiredError,typed_id
from .schema import validate_frame,validate_classifications
from .llm_checks import (validate_response, validity_counts, installed_digest,
    prepare_position, budget_skip, context_after, finish_position)

OUT=ROOT/'results models/prompt_comparison40'
WORK=ROOT/'results/prompt_comparison40'
PROMPT=ROOT/'configs/prompts/reasoning-evidence-v2.txt'


def copy_exact(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==source.read_bytes(),'Changed frozen/copied file: '+str(target)
    else:shutil.copyfile(source,target)


def text_exact(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==text,'Changed prompt: '+str(path)
    else:path.write_text(text,encoding='utf-8')


def rank(row,salt):
    return sha256((salt+'\0'+row['image_sha256']).encode())


def prepare():
    base=read_json(BASE/'experiment200-manifest.json')
    meta={r['frame_id']:r for r in read_json(BASE/'inputs200.json')}
    candidates=[{**r,'image_sha256':meta[r['frame_id']]['sha256']} for r in base['inputs']]
    groups=[('loco',f'loco-subset-{i}',4) for i in range(1,6)]
    groups += [('thud_agtea_test50','agtea',10),('thud_dense_test50','store',10)]
    selected=[]
    for dataset,env,count in groups:
        pool=[r for r in candidates if r['dataset']==dataset and r['environment']==env]
        selected+=sorted(pool,key=lambda r:rank(r,'prompt40-selection-v1-seed42'))[:count]
    selected.sort(key=lambda r:(r['dataset'],r['environment'],r['frame_id']))
    assert len(selected)==40 and len({r['image_sha256'] for r in selected})==40
    repeated=[]
    for dataset,env,_ in groups:
        pool=[r for r in selected if r['dataset']==dataset and r['environment']==env]
        count=1 if dataset=='loco' else (3 if env=='agtea' else 2)
        repeated+=sorted(pool,key=lambda r:rank(r,'prompt40-repeat10-v1-seed42'))[:count]
    repeat_ids={r['frame_id'] for r in repeated};assert len(repeat_ids)==10
    config=base['config']
    for spec in config['models']:spec['base_url']=os.environ.get('THUD_OLLAMA_URL',spec['base_url'])
    old_system=config['system_file_text'];new_system=PROMPT.read_text().strip()
    text_exact(OUT/'prompts/old-system.txt',old_system)
    text_exact(OUT/'prompts/new-system.txt',new_system)
    text_exact(OUT/'prompts/instruction.txt',config['instruction_file_text'])
    inputs=[]
    for r in selected:
        fid=r['frame_id'];frozen=OUT/'frozen'/(fid+'.json')
        source=BASE/r['frozen'];assert sha256(source.read_bytes())==r['sha256']
        copy_exact(source,frozen)
        user=OUT/'prompts/users'/(fid+'.txt');text_exact(user,r['user_prompt'])
        image=OUT/'images'/(fid+'.jpg');copy_exact(SOURCE/fid/'caption-grounding.predictions.jpg',image)
        for spec in config['models']:
            old=BASE/'responses'/spec['name']/(fid+'.json');row=read_json(old)
            assert row['input_sha256']==r['sha256'] and row['model']==spec['model']
            assert row['request']['messages']==[{'role':'system','content':old_system},{'role':'user','content':r['user_prompt']}]
            assert row['request']['options']==options(config,next(combinations(config)))
            copy_exact(old,OUT/'baseline'/spec['name']/(fid+'.json'))
        inputs.append({'frame_id':fid,'dataset':r['dataset'],'environment':r['environment'],
            'image_sha256':r['image_sha256'],'input_sha256':r['sha256'],
            'source_frozen':str(source.relative_to(ROOT)),'frozen':str(frozen.relative_to(OUT)),
            'user_prompt':str(user.relative_to(OUT)),'user_prompt_sha256':sha256(user.read_bytes()),
            'repeatability':fid in repeat_ids})
    heldout=[{'frame_id':r['frame_id'],'dataset':r['dataset'],'environment':r['environment'],
        'image_sha256':r['image_sha256'],'input_sha256':r['sha256']} for r in candidates if r['frame_id'] not in {s['frame_id'] for s in inputs}]
    assert len(heldout)==160
    immutable_json(OUT/'heldout160.json',heldout)
    manifest={'protocol':'prompt40-v1','config':config,'inputs':inputs,'repeatability_ids':sorted(repeat_ids),
        'selection':'SHA256 rank of salt+NUL+image SHA256; salt prompt40-selection-v1-seed42; 4 per LOCO environment, 10 AGTEA, 10 store.',
        'repeat_selection':'Separate SHA256 rank: prompt40-repeat10-v1-seed42; 1 per LOCO environment, 3 AGTEA, 2 store.',
        'base_manifest_sha256':sha256((BASE/'experiment200-manifest.json').read_bytes()),
        'old_system_sha256':sha256(old_system.encode()),'new_system_sha256':sha256(new_system.encode()),
        'new_prompt_source_sha256':sha256(PROMPT.read_bytes()),'main_seed':42,'repeat_seeds':[43,44],
        'planned_main_calls':120,'planned_repeat_calls':60,'planned_new_calls':180,'reused_old_responses':120,
        'new_warmups_or_preflights':0,'accuracy':None}
    immutable_json(OUT/'manifest.json',manifest)
    WORK.mkdir(parents=True,exist_ok=True)
    if not (WORK/'old-files.json').exists():
        paths=subprocess.check_output(['git','ls-files','-z','--',str(BASE.relative_to(ROOT)),str(SOURCE.relative_to(ROOT))]).decode().split('\0')
        write_json(WORK/'old-files.json',{p:sha256((ROOT/p).read_bytes()) for p in paths if p})
    return manifest


def response_path(spec,frame,seed):
    return OUT/'responses'/spec['name']/str(seed)/(frame['frame_id']+'.json')


def run(manifest):
    config=manifest['config'];system=(OUT/'prompts/new-system.txt').read_text()
    for spec in config['models']:
        backend=Backend(spec,config['timeout'])
        metadata=backend.metadata();digest=installed_digest(backend);called=False
        write_json(OUT/'_technical_checks/metadata'/(spec['name']+'.json'),metadata)
        old=read_json(OUT/'baseline'/spec['name']/(manifest['inputs'][0]['frame_id']+'.json'))
        if digest!=old.get('gpu',{}).get('digest'):raise ValueError('Installed model differs from baseline or baseline digest is unavailable')
        try:
            for frame in manifest['inputs']:
                for seed in ([42,43,44] if frame['repeatability'] else [42]):
                    target=response_path(spec,frame,seed)
                    opts=options(config,next(combinations(config)));opts['seed']=seed
                    prepared=prepare_position(target,OUT/frame['frozen'],frame['input_sha256'],spec,digest,
                        system,(OUT/frame['user_prompt']).read_text(),opts,metadata.get('show',{}).get('template'))
                    if prepared is None:continue
                    row={'frame_id':frame['frame_id'],'model':spec['model'],'model_name':spec['name'],
                        **prepared,
                        'precision':spec['precision'],'seed':seed,'llm_repeat':{42:'r1',43:'r2',44:'r3'}[seed],
                        'call_role':'main' if seed==42 else 'repeatability','call_id':uuid.uuid4().hex,
                        'input_sha256':frame['input_sha256'],'user_prompt_sha256':frame['user_prompt_sha256'],
                        'system_prompt_sha256':manifest['new_system_sha256'],'generation_attempted':False}
                    saved=read_json(OUT/frame['frozen'])
                    if saved['status']!='ok' or not saved['parsed']['objects']:
                        write_json(target,{**row,'status':'skipped','error':'Invalid or empty frozen Florence input'});continue
                    if budget_skip(row):write_json(target,row);continue
                    print(f"{spec['name']} / {frame['frame_id']} / seed {seed}",flush=True)
                    try:
                        row['generation_attempted']=True
                        called=True
                        row.update(backend.generate(system,(OUT/frame['user_prompt']).read_text(),opts))
                        row['status']='response_received'
                    except Exception as exc:
                        row.update(getattr(exc,'result',{}),status='error',error=str(exc));write_json(target,row)
                        if isinstance(exc,GPURequiredError):raise
                    finish_position(row,saved,opts)
                    write_json(target,row)
                    if not row['signature_check_after']['verified']:raise ValueError('Generated response signature mismatch; response preserved: '+str(target))
        finally:
            if called:backend.unload()


def csv_file(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def card(row,label,relative):
    content=row.get('content',row.get('response_text',''));seconds=row.get('elapsed_seconds')
    elapsed=f'{seconds:.3f} s' if seconds is not None else 'brez meritve'
    return '<article><h4>'+html.escape(label)+' · '+elapsed+'</h4><a href="'+relative+'">Surovi zapis</a><pre>'+html.escape(content)+'</pre>'+('<p class="error">'+html.escape(row['error'])+'</p>' if row.get('error') else '')+'</article>'


def schema_note(row,frame_id,objects):
    check=validate_response(row,objects)
    return '; '.join(check['errors']) if check['api_response_received'] else row.get('error',row.get('status','No API response'))


def export(manifest):
    config=manifest['config'];models=config['models'];comparison=[];repeat_rows=[];manual=[];calls=[]
    stats={s['name']:Counter() for s in models};body='';all_display=[]
    for frame in manifest['inputs']:
        fid=frame['frame_id'];saved=read_json(OUT/frame['frozen']);objects=saved['parsed']['objects']
        body+='<section id="'+fid+'"><h2>'+html.escape(fid+' / '+frame['dataset']+' / '+frame['environment'])+'</h2><img loading="lazy" src="images/'+fid+'.jpg"><h3>Opis Florence</h3><p>'+html.escape(saved['parsed']['scene_description'])+'</p>'
        for spec in models:
            name=spec['name'];old_path=OUT/'baseline'/name/(fid+'.json');old=read_json(old_path)
            seeds=[42,43,44] if frame['repeatability'] else [42]
            rows={seed:read_json(response_path(spec,frame,seed)) for seed in seeds};calls.extend(rows.values())
            old_map,old_note=table_decisions(old,objects)
            maps={seed:table_decisions(row,objects)[0] for seed,row in rows.items()}
            body+='<h3>'+html.escape(spec['model']+' · '+spec['precision'])+'</h3><div class="pair">'
            body+=card(old,'Stari prompt · seme 42',str(old_path.relative_to(OUT)));all_display.append(old.get('content',old.get('response_text','')))
            path=response_path(spec,frame,42);body+=card(rows[42],'Novi prompt · r1 · seme 42',str(path.relative_to(OUT)))+'</div>';all_display.append(rows[42].get('content',rows[42].get('response_text','')))
            if frame['repeatability']:
                body+='<details><summary>Novi prompt: dodatni ponovitvi</summary><div class="pair">'
                for seed in (43,44):
                    path=response_path(spec,frame,seed);body+=card(rows[seed],f'Novi prompt · r{seed-41} · seme {seed}',str(path.relative_to(OUT)));all_display.append(rows[seed].get('content',rows[seed].get('response_text','')))
                body+='</div></details>'
            body+='<table><tr><th>ID</th><th>Florence</th><th>Stari</th><th>Novi 42</th><th>Primerjava</th></tr>'
            for obj in objects:
                key=typed_id(obj['id']);a=old_map.get(key);b=maps[42].get(key)
                state='incomplete' if a is None or b is None else ('different' if a!=b else 'same')
                common={'dataset':frame['dataset'],'environment':frame['environment'],'frame_id':fid,
                    'region_id':json.dumps(obj['id']),'florence_label':obj['label'],'model':spec['model']}
                comparison.append({**common,'old_42':a or 'MISSING/INVALID','new_42':b or 'MISSING/INVALID',
                    'new_43':maps.get(43,{}).get(key,''),'new_44':maps.get(44,{}).get(key,''),'comparison':state,'repeatability_image':frame['repeatability']})
                manual.append({**common,'manual_classification':'','old_correct':'','new_correct':'','notes':''})
                body+='<tr class="'+state+'"><td>'+html.escape(str(obj['id']))+'</td><td>'+html.escape(obj['label'])+'</td><td>'+html.escape(a or 'MISSING/INVALID')+'</td><td>'+html.escape(b or 'MISSING/INVALID')+'</td><td>'+state+'</td></tr>'
                if frame['repeatability']:
                    values=[maps[seed].get(key) if rows[seed].get('finish_reason')=='stop' else None for seed in (42,43,44)]
                    valid=all(v is not None for v in values);unchanged=valid and len(set(values))==1
                    stats[name]['expected_regions']+=1;stats[name]['comparable_regions']+=int(valid)
                    stats[name]['unchanged_regions']+=int(unchanged)
                    repeat_rows.append({**common,'seed42':values[0] or 'MISSING/INVALID','seed43':values[1] or 'MISSING/INVALID','seed44':values[2] or 'MISSING/INVALID','comparable':valid,'unchanged':unchanged if valid else ''})
            body+='</table>'
            if old_note:body+='<p class="error">Old: '+html.escape(old_note)+'</p>'
            for seed,row in rows.items():
                _,note=table_decisions(row,objects)
                note='; '.join(filter(None,(note,schema_note(row,fid,objects))))
                if note:body+='<p class="error">'+str(seed)+': '+html.escape(note)+'</p>'
                text_exact(response_path(spec,frame,seed).with_suffix('.txt'),row.get('content',row.get('response_text','')))
        body+='</section>'
    summary=[]
    for spec in models:
        s=stats[spec['name']];n=s['comparable_regions'];unchanged=s['unchanged_regions']
        summary.append({'model':spec['model'],**dict(s),'incomplete_regions':s['expected_regions']-n,
            'changed_regions':n-unchanged,'unchanged_fraction':unchanged/n if n else None,
            'metric':'same class across all three seeds on comparable model-region triples; not accuracy'})
    common=['dataset','environment','frame_id','region_id','florence_label','model']
    csv_file(OUT/'prompt-differences.csv',comparison,common+['old_42','new_42','new_43','new_44','comparison','repeatability_image'])
    csv_file(OUT/'repeatability-regions.csv',repeat_rows,common+['seed42','seed43','seed44','comparable','unchanged'])
    csv_file(OUT/'repeatability-summary.csv',summary,list(summary[0]))
    if not (OUT/'manual-review-empty.csv').exists():csv_file(OUT/'manual-review-empty.csv',manual,common+['manual_classification','old_correct','new_correct','notes'])
    write_json(OUT/'repeatability-summary.json',summary)
    intro='<h1>Stari in novi prompt: 40 slik</h1><p>120 starih odgovorov je ponovno uporabljenih. Novi prompt: 120 glavnih klicev (seme 42) in 60 dodatnih na vnaprej izbranih desetih slikah (43/44). Isti besedilni vhodi, T=0,3, kontekst 8192, največ 2048 tokenov; slike so samo za človekov pregled. Rumena označuje različne razrede, ne napačnosti.</p><p><a href="manifest.json">Manifest</a> · <a href="prompts/old-system.txt">Stari prompt</a> · <a href="prompts/new-system.txt">Novi prompt</a> · <a href="prompt-differences.csv">Razlike CSV</a> · <a href="repeatability-regions.csv">Ponovitve po regijah</a> · <a href="repeatability-summary.csv">Ponovljivost CSV</a> · <a href="response-errors.csv">Napake zapisov</a> · <a href="manual-review-empty.csv">Prazen ročni pregled</a></p><h2>Ponovljivost novega prompta</h2><p>Delež primerljivih regij z enakim razredom pri vseh treh semenih. Manjkajoče/neveljavne klasifikacije imajo ločen imenovalec. Meritev uporablja razrede in ID-je; manjkajoča utemeljitev je posebej napaka sheme, čeprav je razred lahko naveden. Ujemanje ni accuracy ali dokaz pravilnosti.</p><ul>'
    for s in summary:
        value=f"{100*s['unchanged_fraction']:.2f}%" if s['unchanged_fraction'] is not None else 'nedoločeno'
        intro+='<li>'+html.escape(s['model'])+': '+value+f" ({s['unchanged_regions']}/{s['comparable_regions']}; neprimerljivih {s['incomplete_regions']})</li>"
    intro+='</ul><p>'+''.join('<a href="#'+f['frame_id']+'">'+f['frame_id']+'</a> · ' for f in manifest['inputs'])+'</p>'
    (OUT/'index.html').write_text('<!doctype html><html lang="sl"><meta charset="utf-8"><title>Prompt test 40</title><style>body{font-family:Arial;max-width:1700px;margin:24px auto;padding:12px}section{border-top:2px solid #aaa;margin-top:30px}img{max-width:100%;max-height:600px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:12px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}.different{background:#fff0bd}.incomplete,.error{color:#a00}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style>'+intro+body+'</html>',encoding='utf-8')
    verify(manifest,calls,all_display)


def verify(manifest,calls,display):
    assert len(calls)==180 and sum(r['call_role']=='main' for r in calls)==120
    assert len({r['call_id'] for r in calls if r.get('generation_attempted')})==sum(r.get('generation_attempted',False) for r in calls)
    outcomes=Counter();errors=[];expected={(f['frame_id'],s['model'],seed) for f in manifest['inputs'] for s in manifest['config']['models'] for seed in ([42,43,44] if f['repeatability'] else [42])}
    assert {(r['frame_id'],r['model'],r['seed']) for r in calls}==expected
    by_id={f['frame_id']:f for f in manifest['inputs']}
    for row in calls:
        frame=by_id[row['frame_id']];assert row['input_sha256']==frame['input_sha256']
        if row['status']=='skipped':outcomes['skips']+=1;continue
        opts=options(manifest['config'],next(combinations(manifest['config'])));opts['seed']=row['seed']
        assert row['request']['options']==opts
        assert row['request']['messages']==[{'role':'system','content':(OUT/'prompts/new-system.txt').read_text()},{'role':'user','content':(OUT/frame['user_prompt']).read_text()}]
        if row['status']!='response_received':
            outcomes['errors']+=1;errors.append({'frame_id':row['frame_id'],'model':row['model'],'seed':row['seed'],'call_role':row['call_role'],'error_type':'api','error':row.get('error','No response')});continue
        assert row['gpu_verified'] and row['gpu']['size_vram']>=row['gpu']['size'] and row['gpu']['context_length']==8192
        objects=read_json(OUT/frame['frozen'])['parsed']['objects']
        _,note=table_decisions(row,objects);schema=schema_note(row,row['frame_id'],objects)
        valid=validate_response(row,objects)['schema_valid_response']
        outcomes['successes' if valid else 'errors']+=1
        if not valid:errors.append({'frame_id':row['frame_id'],'model':row['model'],'seed':row['seed'],'call_role':row['call_role'],'error_type':'schema_or_finish','error':'; '.join(filter(None,(note,schema))) or str(row.get('finish_reason'))})
    for f in manifest['inputs']:
        assert sha256((OUT/f['frozen']).read_bytes())==f['input_sha256']==sha256((ROOT/f['source_frozen']).read_bytes())
        assert sha256((OUT/f['user_prompt']).read_bytes())==f['user_prompt_sha256']
    old=read_json(WORK/'old-files.json')
    for path,digest in old.items():assert sha256((ROOT/path).read_bytes())==digest,'Changed old file: '+path
    csv_file(OUT/'response-errors.csv',errors,['frame_id','model','seed','call_role','error_type','error'])
    class Page(HTMLParser):
        def __init__(self):super().__init__();self.pre=False;self.answers=[];self.ids=set();self.links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='pre':self.pre=True;self.answers.append('')
            for key,value in attrs:
                if key=='id':self.ids.add(value)
                if key in ('href','src'):
                    if value.startswith('#'):self.links.append(value[1:])
                    else:assert (OUT/value).is_file(),value
        def handle_endtag(self,tag):
            if tag=='pre':self.pre=False
        def handle_data(self,data):
            if self.pre:self.answers[-1]+=data
    page=Page();page.feed((OUT/'index.html').read_text());assert page.answers==display and set(page.links)<=page.ids
    checks=[validate_response(r,read_json(OUT/by_id[r['frame_id']]['frozen'])['parsed']['objects']) for r in calls]
    write_json(OUT/'verification.json',{'images':40,'repeatability_images':10,'heldout_images':160,**validity_counts(checks),
        'new_positions':180,'main_positions':120,'repeat_positions':60,'reused_old_responses':120,
        'actual_new_attempts':sum(r.get('generation_attempted',False) for r in calls),
        'successes':outcomes['successes'],'errors':outcomes['errors'],'skips':outcomes['skips'],
        'api_errors':sum(c['api_error'] for c in checks),'schema_or_finish_errors':sum(c['schema_or_finish_error'] for c in checks),
        'old_files_unchanged':len(old),'full_answers_in_html':len(display),'html_links_valid':True,
        'identical_frozen_inputs':True,'quality_evaluation':False,'accuracy':None})
    print('Verified prompt test:',dict(outcomes),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','run','export'));args=parser.parse_args()
    manifest=prepare()
    if args.phase=='run':run(manifest);export(manifest)
    elif args.phase=='export':export(manifest)
