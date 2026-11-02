"""One text-only call per saved Florence input/model; raw answers for human review."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil

from .description_prompts import validate_description
from .experiment import combinations, load_config, options, read_json, sha256, write_json, strict_json
from .thud_experiment import immutable_json
from .thud_extended_common import ROOT
from .vision_backends import Backend
from .vision_pipeline import GPURequiredError, typed_id
from .vision_prompts import reason_prompt

SOURCE=ROOT/'results models/florence_caption_grounding_test10'
OUTPUT=ROOT/'results models/florence_grounding_llm_comparison10'
CONFIG=ROOT/'configs/stage2/florence-grounding-qwen-size.json'


def prepare(gemma_only=False,loco30=False,experiment200=False):
    if experiment200:
        previous=read_json(OUTPUT/'loco30-manifest.json');config=previous['config']
        for spec in config['models']:spec['base_url']=os.environ.get('THUD_OLLAMA_URL',spec['base_url'])
        metadata=read_json(OUTPUT/'inputs200.json');by_id={r['frame_id']:r for r in metadata}
        frames=[]
        for f in previous['inputs']:
            assert sha256((OUTPUT/f['frozen']).read_bytes())==f['sha256']
            frames.append({**f,'dataset':by_id[f['frame_id']]['dataset'],'environment':by_id[f['frame_id']]['environment']})
        for index,r in enumerate(metadata[30:],30):
            path=SOURCE/'stage1'/f'frame-{index+1:06d}.json';saved=read_json(path)
            assert saved['frame_id']==r['frame_id'] and saved['image_sha256']==r['sha256']
            if saved['status']=='ok':validate_description(saved['parsed'])
            frozen=OUTPUT/'frozen'/path.name
            if frozen.exists():assert frozen.read_bytes()==path.read_bytes(),'Changed frozen input'
            else:shutil.copyfile(path,frozen)
            objects=[{'id':o['id'],'label':o['label']} for o in saved['parsed']['objects']]
            frames.append({'frame_id':r['frame_id'],'dataset':r['dataset'],'environment':r['environment'],
                'source':str(path.relative_to(ROOT)),'frozen':str(frozen.relative_to(OUTPUT)),
                'sha256':sha256(path.read_bytes()),'user_prompt':reason_prompt(r['frame_id'],objects,saved['parsed'],config['instruction_file_text'])})
        immutable_json(OUTPUT/'experiment200-manifest.json',{'config':config,'inputs':frames,
            'base_manifest_sha256':sha256((OUTPUT/'loco30-manifest.json').read_bytes()),
            'note':'Reuse matching 30-image results; freeze all new inputs before calls; missing calls only, no warmups.'})
        return config,frames
    if loco30:
        previous=read_json(OUTPUT/'manifest.json');gemma=read_json(OUTPUT/'gemma3-manifest.json')
        assert previous['inputs']==gemma['inputs']
        config={**previous['config'],'models':previous['config']['models']+gemma['config']['models']}
        for spec in config['models']:
            spec['base_url']=os.environ.get('THUD_OLLAMA_URL',spec['base_url'])
        frames=list(previous['inputs'])
        for frame in frames:assert sha256((OUTPUT/frame['frozen']).read_bytes())==frame['sha256']
        paths=sorted((SOURCE/'stage1').glob('frame-*.json'))
        assert len(paths)==30
        for path in paths[10:]:
            saved=read_json(path)
            if saved['status']=='ok':validate_description(saved['parsed'])
            frozen=OUTPUT/'frozen'/path.name
            if frozen.exists():assert frozen.read_bytes()==path.read_bytes(),'Changed frozen input'
            else:shutil.copyfile(path,frozen)
            objects=[{'id':o['id'],'label':o['label']} for o in saved['parsed']['objects']]
            frames.append({'frame_id':saved['frame_id'],'source':str(path.relative_to(ROOT)),
                'frozen':str(frozen.relative_to(OUTPUT)),'sha256':sha256(path.read_bytes()),
                'user_prompt':reason_prompt(saved['frame_id'],objects,saved['parsed'],config['instruction_file_text'])})
        assert len({f['frame_id'] for f in frames})==30
        immutable_json(OUTPUT/'loco30-manifest.json',{'config':config,'inputs':frames,
            'base_manifest_sha256':sha256((OUTPUT/'manifest.json').read_bytes()),
            'gemma_manifest_sha256':sha256((OUTPUT/'gemma3-manifest.json').read_bytes()),
            'note':'First ten inputs and thirty LLM answers reused unchanged. Only missing calls; no new preflights or warmups.'})
        return config,frames
    if gemma_only:
        # Read the published frozen inputs/prompts, not a new Florence export.
        previous=read_json(OUTPUT/'manifest.json');frames=previous['inputs']
        assert len(frames)==10
        for frame in frames:
            assert sha256((OUTPUT/frame['frozen']).read_bytes())==frame['sha256']
        spec={'name':'gemma3-27b-q4','backend':'ollama','model':'gemma3:27b-it-q4_K_M',
              'precision':'Q4_K_M','require_gpu':True,
              'base_url':os.environ.get('THUD_OLLAMA_URL',previous['config']['models'][0]['base_url'])}
        config={**previous['config'],'models':[spec]}
        immutable_json(OUTPUT/'gemma3-manifest.json',{'config':config,'inputs':frames,
            'base_manifest_sha256':sha256((OUTPUT/'manifest.json').read_bytes()),
            'differences':[
                'Same API messages and sampler settings; native Gemma chat template/stop tokens differ (see models metadata).',
                'Gemma has no thinking capability; think=false is retained in the API request, without a Gemma reasoning-mode toggle.'
            ]})
        return config,frames
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
    target=OUTPUT/('gemma3-preflight.json' if config['models'][0]['model'].startswith('gemma3:') else 'preflight.json')
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


def run(config,frames,extension=False):
    if not extension:preflight(config,frames)
    opts=options(config,next(combinations(config)))
    for spec in config['models']:
        if all((OUTPUT/'responses'/spec['name']/(f['frame_id']+'.json')).exists() for f in frames):continue
        backend=Backend(spec,config['timeout'])
        write_json(OUTPUT/'models'/((f'extension{len(frames)}-' if extension else '')+spec['name']+'.json'),backend.metadata())
        try:
            for frame in frames:
                target=OUTPUT/'responses'/spec['name']/(frame['frame_id']+'.json')
                if target.exists():continue  # Preserve every completed attempt, including errors.
                row={'stage':2,'frame_id':frame['frame_id'],'model':spec['model'],
                    'precision':spec['precision'],'input_sha256':frame['sha256'],'frozen_input':frame['frozen']}
                saved=read_json(OUTPUT/frame['frozen'])
                if saved['status']!='ok' or not saved['parsed']['objects']:
                    write_json(target,{**row,'status':'skipped','error':'Invalid or empty saved Florence input; no LLM call.'})
                    continue
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


def table_decisions(row,objects):
    """Literal classes by typed local ID. Never fill missing/ambiguous IDs."""
    if row['status']!='response_received':return {},row.get('error',row['status'])
    try:
        value=strict_json(row['content'])
        if not isinstance(value,dict) or not isinstance(value.get('objects'),list):
            raise ValueError('Missing JSON objects list')
        allowed={typed_id(o['id']) for o in objects};result={};duplicates=set();notes=[]
        for item in value['objects']:
            if not isinstance(item,dict):raise ValueError('Non-object prediction')
            key=typed_id(item.get('id'))
            if key not in allowed:
                notes.append('Unexpected ID: '+repr(item.get('id')));continue
            if key in result:duplicates.add(key)
            label=item.get('classification')
            result[key]=label if label in ('fixed','movable','uncertain') else None
        for key in duplicates:result[key]=None
        if duplicates:notes.append('Duplicate IDs; their classes are not assigned')
        if any(result.get(k) is None for k in allowed):notes.append('Missing or invalid classes/IDs')
        if row.get('finish_reason') in ('length','max_tokens'):notes.append('Output token limit reached')
        return result,'; '.join(notes)
    except (ValueError,KeyError,TypeError) as exc:
        return {},'Unusable table JSON: '+str(exc)


def region_table(frame,objects,rows,models):
    """Requested side-by-side classes, with disagreements distinct from data errors."""
    parsed=[table_decisions(row,objects) for row in rows]
    body='<table><tr><th>Nabor</th><th>Okolje</th><th>ID regije</th><th>Oznaka Florence</th>'
    body+=''.join('<th>'+html.escape(s['model'])+'</th>' for s in models)+'<th>Primerjava</th></tr>'
    records=[]
    for obj in objects:
        classes=[values.get(typed_id(obj['id'])) for values,_ in parsed]
        differing=len({c for c in classes if c is not None})>1
        incomplete=any(c is None for c in classes)
        state=('različni razredi' if differing else 'enaki razredi') if not incomplete else ('različni razredi; nepopolno' if differing else 'nepopolno')
        style='different' if differing else ('incomplete' if incomplete else '')
        dataset=frame.get('dataset','loco');environment=frame.get('environment','loco-subset-'+frame['frame_id'].split('-')[1][1:])
        body+='<tr class="'+style+'"><td>'+html.escape(dataset)+'</td><td>'+html.escape(environment)+'</td><td>'+html.escape(json.dumps(obj['id']))+'</td><td>'+html.escape(obj['label'])+'</td>'
        cells=[c if c is not None else 'MANJKA / NEVELJAVNO' for c in classes]
        body+=''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'<td>'+state+'</td></tr>'
        records.append({'dataset':dataset,'environment':environment,'frame_id':frame['frame_id'],'region_id':json.dumps(obj['id']),
            'florence_label':obj['label'],**{s['name']:c for s,c in zip(models,cells)},'comparison':state})
    body+='</table>'
    for spec,(_,note) in zip(models,parsed):
        if note:body+='<p class="error">'+html.escape(spec['model']+': '+note)+'</p>'
    return body,records,sum(bool(note) for _,note in parsed)


def export(config,frames):
    escape=html.escape
    body='<h1>Florence grounding → Qwen3.5 in Gemma 3</h1><p>Isti shranjeni vhodi in prompt; T=0,3, seme 42, kontekst 8192, največ 2048 izhodnih tokenov, think=false. Ena izvedba na model in sliko. Čas je celoten API klic (vključno z morebitnim nalaganjem). Kakovost odgovorov ni ocenjena.</p>'
    if any(s['model'].startswith('gemma3:') for s in config['models']):
        body+='<p>Gemma nima načina thinking; uporabljen je isti think=false API zahtevek. Njena izvorna pogovorna predloga in zaključni tokeni se razlikujejo. <a href="gemma3-manifest.json">Zabeležene razlike</a>.</p>'
    body+='<p>Rumene vrstice označujejo različne vrnjene razrede. MANJKA / NEVELJAVNO pomeni neuporaben zapis, ne razreda uncertain. Regije se lahko prekrivajo ali opisujejo skupine; tabela ni ocena pravilnosti. <a href="region-classes.csv">Vse regije CSV</a>.</p>'
    if len(frames)==30:body+='<p>Prvih deset primerov je ohranjenih. <a href="loco30-manifest.json">Zamrznjeni vhodi in nastavitve za vseh 30 slik</a>.</p>'
    if len(frames)>30:body+='<p>Prvih 30 primerov je ohranjenih. <a href="inputs200.json">Seznam RGB vhodov</a> · <a href="experiment200-manifest.json">Zamrznjeni opisi in nastavitve</a>.</p>'
    intro=body;body='';count=0;records=[];all_rows=[];table_errors=0
    overview='<table><tr><th>Nabor</th><th>Okolje</th><th>Slika</th><th>Florence: oba koraka (s)</th>'+''.join('<th>'+escape(s['model'])+' (s)</th>' for s in config['models'])+'</tr>'
    for frame in frames:
        fid=frame['frame_id'];image=OUTPUT/'images'/(fid+'.jpg');image.parent.mkdir(exist_ok=True)
        shutil.copyfile(SOURCE/fid/'caption-grounding.predictions.jpg',image)
        combined=read_json(SOURCE/fid/'combined.raw.json');caption=read_json(OUTPUT/frame['frozen'])
        body+='<section id="'+fid+'"><h2>'+escape(fid)+'</h2><p>'+escape(frame.get('dataset','loco')+' / '+frame.get('environment',''))+'</p><a href="images/'+fid+'.jpg"><img loading="lazy" src="images/'+fid+'.jpg" alt="Napovedi Florence"></a>'
        body+='<details><summary>Celoten opis Florence in izvorni podatki</summary><p>'+escape(caption['parsed']['scene_description'])+'</p><a href="../florence_caption_grounding_test10/'+fid+'/index.html">Opis, regije in surovi odgovori Florence</a></details><div class="answers">'
        image_rows=[]
        for spec in config['models']:
            path=OUTPUT/'responses'/spec['name']/(fid+'.json');row=read_json(path)
            assert row['input_sha256']==frame['sha256']
            request=row.get('request',{})
            if request:
                assert request['messages']==[{'role':'system','content':config['system_file_text']},{'role':'user','content':frame['user_prompt']}]
                assert request['options']==options(config,next(combinations(config)))
            if row['status']=='response_received':
                assert row['gpu_verified']
                # Preserve responses even if the configured window was tight.
                # Actual token counts and finish reason remain in the raw record.
            content=row.get('content',row.get('response_text',''))
            path.with_suffix('.txt').write_text(content,encoding='utf-8')
            elapsed=row.get('elapsed_seconds')
            seconds=f'{elapsed:.3f} s' if elapsed is not None else 'ni meritve'
            body+='<article><h3>'+escape(spec['model'])+'</h3><p>'+escape(spec['precision'])+' · '+seconds+' · '+escape(row['status'])+'</p><a href="'+str(path.relative_to(OUTPUT))+'">Celoten surovi zapis</a><pre>'+escape(content)+'</pre>'
            if row.get('error'):body+='<p class="error">'+escape(row['error'])+'</p>'
            body+='</article>';count+=1;image_rows.append(row)
        table,table_rows,errors=region_table(frame,caption['parsed']['objects'],image_rows,config['models'])
        body+='</div>'+table+'</section>';records.extend(table_rows);table_errors+=errors;all_rows.extend(image_rows)
        times=[combined.get('elapsed_seconds')]+[r.get('elapsed_seconds') for r in image_rows]
        overview+='<tr><td>'+escape(frame.get('dataset','loco'))+'</td><td>'+escape(frame.get('environment',''))+'</td><td><a href="#'+fid+'">'+fid+'</a></td>'+''.join('<td>'+(f'{t:.3f}' if t is not None else 'napaka/preskok')+'</td>' for t in times)+'</tr>'
    assert count==len(frames)*len(config['models'])
    with (OUTPUT/'region-classes.csv').open('w',newline='',encoding='utf-8') as output:
        writer=csv.DictWriter(output,fieldnames=['dataset','environment','frame_id','region_id','florence_label']+[s['name'] for s in config['models']]+['comparison'],lineterminator='\n')
        writer.writeheader();writer.writerows(records)
    (OUTPUT/'index.html').write_text('<!doctype html><html lang="sl"><meta charset="utf-8"><title>LLM: '+str(len(frames))+' Florence vhodov</title><style>body{font-family:Arial;margin:24px auto;padding:12px;max-width:1800px}section{border-top:2px solid #aaa;margin-top:30px}img{max-width:100%;max-height:600px}.answers{display:grid;grid-template-columns:repeat('+str(len(config['models']))+',minmax(0,1fr));gap:20px}article{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:12px}table{border-collapse:collapse;margin:15px 0}td,th{border:1px solid #aaa;padding:7px;text-align:left}.different{background:#fff0bd;font-weight:bold}.incomplete,.error{color:#a00}@media(max-width:800px){.answers{grid-template-columns:1fr}}</style>'+intro+overview+'</table>'+body+'</html>',encoding='utf-8')
    for frame in frames:
        assert sha256((ROOT/frame['source']).read_bytes())==frame['sha256']
    write_json(OUTPUT/'verification.json',{'images':len(frames),'comparison_records':count,
        'short_preflight_calls':sum((OUTPUT/name).exists() for name in ('preflight.json','gemma3-preflight.json')),
        'first_ten_rerun':False,'source_inputs_unchanged':True,
        'completed_responses':sum(r['status']=='response_received' and r.get('response',{}).get('done',False) for r in all_rows),
        'api_errors':sum(r['status']=='error' for r in all_rows),'skipped':sum(r['status']=='skipped' for r in all_rows),
        'output_limit_responses':sum(r.get('finish_reason') in ('length','max_tokens') for r in all_rows),
        'table_records_with_format_or_id_notes':table_errors,
        'fully_gpu_responses':sum(r.get('gpu_verified',False) for r in all_rows),
        'interrupted_setup_api_attempts':int((OUTPUT/'setup/preflight-before-gpu-restart.json').exists()),
        'identical_prompts_and_settings':True,'quality_analysis':False})
    print('Saved HTML:',OUTPUT/'index.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','preflight','run','export'))
    parser.add_argument('--gemma-only',action='store_true',help='Only new Gemma calls; reuse all frozen Qwen inputs and answers')
    parser.add_argument('--loco30',action='store_true',help='Extend all three models to 30 images; no new warmups/preflights')
    parser.add_argument('--experiment200',action='store_true',help='Reuse 30 old results and classify only the new LOCO/THUD inputs')
    args=parser.parse_args()
    if args.loco30 and (args.gemma_only or args.phase=='preflight'):parser.error('Use --loco30 with prepare, run or export only')
    if args.experiment200 and (args.loco30 or args.gemma_only or args.phase=='preflight'):parser.error('Use --experiment200 alone with prepare/run/export')
    config,frames=prepare(args.gemma_only,args.loco30,args.experiment200)
    if args.phase=='preflight':preflight(config,frames)
    elif args.phase in ('run','export'):
        if args.phase=='run':run(config,frames,extension=args.loco30 or args.experiment200)
        if args.gemma_only:
            config={**config,'models':read_json(OUTPUT/'manifest.json')['config']['models']+config['models']}
        export(config,frames)
