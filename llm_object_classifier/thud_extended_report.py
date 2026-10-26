"""Offline native-model comparison; never executes models or changes raw files.

All latency denominators include failed outputs. Annotation matching is explicit
and saved pair by pair; it is annotated-region coverage, not full scene accuracy.
"""
from collections import Counter
import csv
import html
import json
from pathlib import Path
import shutil
import statistics
import base64
import re
import subprocess

from .experiment import read_json, write_json, sha256
from .thud_box_metrics import match_regions
from .thud_extended_common import ROOT, OUT, REPORT, DATASETS, frames, saved_path
from .thud_florence import TASKS
from .thud_metrics import percentile


def write_csv(path, rows, fields):
    """Use LF for versioned report tables; preserve all literal cell content."""
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def native_rows(model, task):
    return [read_json(p) for p in sorted((OUT/model).glob(f'*/{task}/frame-*.json'))]


def combined_rows():
    rows=native_rows('florence','combined')
    for row in rows:
        row['status']='ok' if all(read_json(saved_path(OUT/'florence',row,t))['status']=='ok' for t in TASKS) else 'error'
    return rows


def optional_setup(name):
    path=OUT/'setup'/name
    return read_json(path) if path.exists() else None


def stats(rows):
    times = [r['elapsed_seconds'] for r in rows if isinstance(r.get('elapsed_seconds'), (int,float))]
    return {'positions': len(rows), 'ok': sum(r.get('status')=='ok' for r in rows),
            'errors': sum(r.get('status')=='error' for r in rows),
            'gpu_verified': sum(r.get('gpu_verified',False) for r in rows),
            'mean_s': statistics.mean(times) if times else None,
            'median_s': statistics.median(times) if times else None,
            'p90_s': percentile(times,.9), 'sum_s': sum(times),
            'output_tokens': sum(r.get('output_tokens',r.get('response',{}).get('eval_count',0)) or 0 for r in rows),
            'max_allocated_bytes': max((r.get('peak_allocated_bytes',0) for r in rows), default=0),
            'max_reserved_bytes': max((r.get('peak_reserved_bytes',0) for r in rows), default=0),
            'max_ollama_vram_bytes': max((r.get('gpu',{}).get('size_vram',0) for r in rows),default=0)}


def evaluate_boxes(selected):
    policy = read_json(ROOT/'configs/thud-florence-matching.json')
    all_matches = []
    summary = []
    for dataset in DATASETS:
        references = [json.loads(line) for line in (ROOT/'data'/dataset/'objects.jsonl').read_text().splitlines() if line.strip()]
        for task in ('od','dense'):
            rows = []
            for frame in selected:
                if frame['dataset'] != dataset:
                    continue
                path = saved_path(OUT/'florence',frame,task)
                native = read_json(path) if path.exists() else {'status':'missing'}
                value = native.get('parsed',{}).get(TASKS[task],{}) if native['status']=='ok' else {}
                predictions = [{'label': label, 'bbox_xyxy': box} for label,box in zip(value.get('labels',[]),value.get('bboxes',[]))]
                ref = references[frame['index']]
                # Verify reference/frame pairing by actual RGB SHA, not duplicate local IDs.
                image_path = ROOT/'data'/dataset/ref['image_path']
                if sha256(image_path.read_bytes()) != frame['image_sha256']:
                    raise ValueError('Reference/image pairing changed')
                matching = match_regions(predictions,ref['objects'],policy)
                rows.append({'dataset':dataset,'task':task,'frame_id':frame['frame_id'],
                             'image_sha256':frame['image_sha256'],'native_status':native['status'],**matching})
            all_matches.extend(rows)
            fields = ['matched','eligible_reference_regions','excluded_reference_regions',
                      'prediction_regions','mapped_prediction_regions']
            sums = {k: sum(r[k] for r in rows) for k in fields}
            summary.append({'dataset':dataset,'task':task,**sums,
                'annotated_region_coverage_iou50': sums['matched']/sums['eligible_reference_regions'] if sums['eligible_reference_regions'] else None,
                'mean_matched_iou': statistics.mean(p['iou'] for r in rows for p in r['pairs']) if sums['matched'] else None})
    write_json(REPORT/'extended-box-matches.json', {'policy':policy,'results':all_matches})
    write_json(REPORT/'extended-box-summary.json', summary)
    write_csv(REPORT/'extended-box-summary.csv',summary,list(summary[0]))
    return summary


def previous_comparison():
    result = []
    for model in ('qwen35-bf16','qwen35-q4','gemma3-q4'):
        rows = [read_json(p) for ds in DATASETS for p in sorted((ROOT/'results/thud-part2'/f'rich-{model}-{ds}'/model/'t0-s42').glob('frame-*.json'))]
        result.append({'model':model,'task':'rich-v2 (existing, reused)',**stats(rows)})
    return result


def audit_old():
    index = read_json(OUT/'previous-results-index.json')
    changed = [name for name,digest in index.items() if not (ROOT/name).exists() or sha256((ROOT/name).read_bytes()) != digest]
    if changed:
        raise ValueError('Existing results changed: '+str(changed[:5]))
    return {'previous_response_files_checked':len(index),'changed':changed}


def tex(value):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in str(value))


def literal_scene(row):
    """Quote a model string even when its enclosing JSON failed validation."""
    scene=row.get('parsed',{}).get('scene_description')
    if scene:
        return scene
    content=row.get('content','')
    match=re.search(r'"scene_description"\s*:\s*',content)
    if match:
        try:
            value,_=json.JSONDecoder().raw_decode(content[match.end():])
            if isinstance(value,str):
                return value
        except ValueError:
            pass
    return row.get('error','')


def table(headers, rows, columns):
    return '\\begin{center}\\small\\begin{tabular}{'+columns+'}\\toprule\n'+' & '.join(headers)+r' \\ \midrule'+'\n'+ '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\\end{center}\n'


def reasoning_summary():
    summary=[]
    decisions=[]
    for model in ('nvila','florence'):
        paths=sorted((OUT/'reason'/model).glob('*/*/t*/frame-*.json'))
        records=[read_json(p) for p in paths]
        measured=[r for r in records if r.get('response')]
        warmups=[read_json(p) for p in sorted((OUT/'reason'/model).glob('*/*/warmup.json'))]
        counts=Counter(o['classification'] for r in records if r['status']=='ok' for o in r['parsed']['objects'])
        summary.append({'caption_model':model,'positions':len(records),
                        **dict(Counter(r['status'] for r in records)),
                        'classifications':dict(counts),
                        'measured_calls':stats(measured),
                        'max_prompt_eval_tokens':max((r['response'].get('prompt_eval_count',0) for r in measured),default=0),
                        'warmup_calls':len(warmups),
                        'warmup_gpu_verified':sum(r.get('gpu_verified',False) for r in warmups),
                        'warmup_seconds':sum(r.get('elapsed_seconds',0) for r in warmups)})
        for p,r in zip(paths,records):
            for o in r.get('parsed',{}).get('objects',[]):
                decisions.append({'caption_model':model,'dataset':p.parents[2].name,
                                  'frame_id':r['frame_id'],'object_id':o['id'],
                                  'classification':o['classification'],'reason':o['reason'],
                                  'source_record':str(p.relative_to(ROOT))})
    write_json(REPORT/'extended-reasoning-summary.json',summary)
    timing=[{'caption_model':r['caption_model'],**r['measured_calls'],
             'max_prompt_eval_tokens':r['max_prompt_eval_tokens']} for r in summary]
    write_csv(REPORT/'extended-reasoning-timing.csv',timing,list(timing[0]))
    write_json(REPORT/'extended-decisions.json',decisions)
    if decisions:
        write_csv(REPORT/'extended-decisions.csv',decisions,list(decisions[0]))
    return summary


def make_review(selected):
    """Same20, photos/reference overlays and full literal output for human review."""
    from .thud_extended_common import draw_boxes
    review=[]
    pages=['<!doctype html><meta charset="utf-8"><title>THUD native model review</title>',
           '<style>body{font:15px sans-serif;max-width:1450px;margin:auto}img{max-width:32%}pre{white-space:pre-wrap;background:#eee;padding:1em}details{border:1px solid #bbb;margin:8px;padding:8px}</style>',
           '<h1>NVILA / Florence: same 20 photographs</h1><p>References below are review-only. No annotations entered inference. Pending review is not confirmed ground truth.</p>']
    notes=read_json(REPORT/'extended-manual-notes.json') if (REPORT/'extended-manual-notes.json').exists() else {}
    for frame in selected:
        if not frame['subset20']:
            continue
        ds=frame['dataset']; i=frame['index']; key=f'{ds}/{i}'
        refs=[json.loads(line) for line in (ROOT/'data'/ds/'objects.jsonl').read_text().splitlines() if line.strip()][i]
        ref_image=OUT/'reference-overlays'/ds/f'frame-{i+1:06d}.jpg'
        draw_boxes(frame['image_path'], {'reference':{'labels':[o['label'] for o in refs['objects']],
                    'bboxes':[o['bbox_xyxy'] for o in refs['objects']]}}, 'reference', ref_image)
        pages.append('<h2>'+html.escape(f'{ds} / {refs["frame_id"]} / {frame["frame_id"]}')+'</h2>')
        for caption,path in [('Original',Path(frame['image_path'])),('THUD references',ref_image),
                 ('Florence OD',OUT/'florence/overlays'/ds/'od'/f'frame-{i+1:06d}.jpg'),
                 ('Florence dense',OUT/'florence/overlays'/ds/'dense'/f'frame-{i+1:06d}.jpg')]:
            if path.exists():
                mime='image/png' if path.suffix=='.png' else 'image/jpeg'
                pages.append('<img title="'+html.escape(caption)+'" src="data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()+'">')
        for model in ('nvila','florence','qwen35-bf16','qwen35-q4','gemma3-q4'):
            if model in ('nvila','florence'):
                paths=[saved_path(OUT/model,frame,t) for t in (['rich'] if model=='nvila' else list(TASKS))]
            else:
                paths=[ROOT/'results/thud-part2'/f'rich-{model}-{ds}'/model/'t0-s42'/f'frame-{i+1:06d}.json']
            for path in paths:
                if path.exists():
                    row=read_json(path)
                    display={k:row[k] for k in ('status','error','content','parsed','candidate_parsed','elapsed_seconds','output_tokens') if k in row}
                    pages.append('<details><summary>'+html.escape(model+' / '+path.parent.name)+'</summary><pre>'+html.escape(json.dumps(display,ensure_ascii=False,indent=2))+'</pre></details>')
            if model in ('nvila','florence'):
                reason_paths=sorted((OUT/'reason'/model/ds).glob('*/t*/frame-*.json'))
                reason=next((read_json(p) for p in reason_paths if read_json(p)['frame_id']==frame['frame_id']),None)
                if reason:
                    pages.append('<details><summary>Qwen BF16 reasoning: '+model+'</summary><pre>'+html.escape(json.dumps(reason,ensure_ascii=False,indent=2))+'</pre></details>')
                review.append({'dataset':ds,'frame_id':frame['frame_id'],'image_sha256':frame['image_sha256'],
                               'model':model,'review_status':'pending','missed_objects':'','hallucinated_objects':'',
                               'locations_relations':'','attachment_evidence':'','note':notes.get(key,'')})
        pages.append('<p>Partial visual note: '+html.escape(notes.get(key,'pending'))+'</p>')
    (OUT/'manual-review.html').write_text('\n'.join(pages))
    write_csv(REPORT/'extended-manual-review.csv',review,list(review[0]))


def rejected_overlays():
    """Illustrate malformed predictions without repairing/reusing them in metrics."""
    from PIL import Image, ImageDraw
    for name,task in TASKS.items():
        if name=='caption':
            continue
        for row in native_rows('florence',name):
            if row['status']!='error' or task not in row.get('parsed',{}):
                continue
            value=row['parsed'][task]
            image=Image.open(row['image_path']).convert('RGB')
            draw=ImageDraw.Draw(image)
            draw.rectangle([0,0,image.width,24],fill='black')
            draw.text((4,5),'REJECTED native prediction; original coordinates, no repair',fill='red')
            for box,label in zip(value.get('bboxes',[]),value.get('labels',[])):
                x1,y1,x2,y2=box
                # Line segments preserve the original corner order even if inverted.
                draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)],fill='red',width=2)
                draw.text((x1,max(25,y1)),str(label),fill='red',stroke_width=1,stroke_fill='black')
            target=OUT/'florence/overlays'/row['dataset']/(name+'-rejected')/f"frame-{row['index']+1:06d}.jpg"
            target.parent.mkdir(parents=True,exist_ok=True)
            image.save(target)


def save_provenance():
    models={name:read_json(OUT/f'setup/{name}-api.txt') for name in ('nvila','florence')}
    log=OUT/'logs/tests-final.log'
    text=log.read_text() if log.exists() else ''
    tested=re.search(r'Ran (\d+) tests in ([\d.]+)s',text)
    snapshots={name:[{'path':str(p.relative_to(OUT)),'sha256':sha256(p.read_bytes()),
                     'git_commit':read_json(p)['git_commit']}
                    for p in sorted((OUT/name/'source-snapshots').glob('*.json'))]
               for name in ('nvila','florence','reason')}
    write_json(REPORT/'extended-provenance.json',{
        'model_revisions':{name:{'repo':data['id'],'revision':data['sha']} for name,data in models.items()},
        'report_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'report_generator_sha256':sha256(Path(__file__).read_bytes()),
        'inference_source_snapshots':snapshots,
        'inference_manifests':{name:read_json(OUT/name/'manifest.json') for name in models},
        'model_files':optional_setup('model-files.json'),
        'docker_images':[
            {k:r[k] for k in ('Id','RepoTags','Created','Architecture')}
            for r in (json.loads(line) for line in (OUT/'setup/docker-images.jsonl').read_text().splitlines())
        ] if (OUT/'setup/docker-images.jsonl').exists() else [],
        'vila_patch':(OUT/'setup/vila-inference.patch').read_text(),
        'raw_outputs':str(OUT),
        'tests':{'count':int(tested[1]) if tested else None,
                 'seconds':float(tested[2]) if tested else None,
                 'passed':text.strip().endswith('OK'),
                 'log':str(log.relative_to(ROOT)), 'log_sha256':sha256(log.read_bytes()) if log.exists() else None},
        'generation_revision':optional_setup('nvila-generation-revision.json'),
        'measurement_notes':optional_setup('measurement-notes.json'),
        'florence_process_recovery':optional_setup('florence-process-recovery.json'),
        'nvila_launcher_recovery':optional_setup('nvila-launcher-recovery.json'),
        'gpu_sampling_audit':optional_setup('gpu-sampling-audit.json')})


def make_latex(rows, boxes, reason, selected):
    number=lambda x: '--' if x is None else f'{x:.2f}'
    labels={'nvila':'NVILA-8B FP16','florence':'Florence-2-large FP16',
            'qwen35-bf16':'Qwen BF16 (staro)','qwen35-q4':'Qwen Q4 (staro)','gemma3-q4':'Gemma Q4 (staro)'}
    timing=[]
    for r in rows:
        timing.append([tex(labels[r['model']]),tex(r['task'].replace(' (existing, reused)','')),
                       f"{r['ok']}/{r['positions']}",
                       number(r['mean_s']),number(r['median_s']),number(r['p90_s'])])
    environments=read_json(REPORT/'extended-model-metadata.json')
    memory=[]
    for model in ('nvila','florence'):
        e=environments[model][-1] if environments[model] else {}
        measured=[r for r in rows if r['model']==model]
        memory.append([tex(labels[model]),tex(', '.join(e.get('parameter_dtypes',{}))),
                       number(max((r['max_allocated_bytes'] for r in measured),default=0)/2**30),
                       number(max((r['max_reserved_bytes'] for r in measured),default=0)/2**30),
                       number(e.get('load_seconds'))])
    combined_row=next(r for r in rows if r['model']=='florence' and r['task']=='combined')
    pages=[r'''\section*{Nadaljevanje: NVILA-8B in Florence-2-large na RTX 5090}
Nova primerjava uporablja istih 100 RGB slik in isti razpršeni podnabor 20 slik. Referenčne vrste in okvirji niso vhod modelov. Prejšnjih uspešnih inferenc Qwen/Gemma ne ponavljamo; v tabelah uporabimo njihove shranjene izhode 2. dela. Rezultati so povezani pogledi dveh prostorov.
\subsection*{Dejanski izolirani zagon in razlike nalog}
Oba nova modela sta bila zagnana v ločenih Docker okoljih: Python 3.11, PyTorch 2.7.1+cu128, CUDA 12.8, Transformers 4.46.0. Binarne datoteke vsebujejo sm\_120; vsak parameter je preverjeno na GPU. NVILA ima 8,09 milijarde parametrov, Florence približno 0,776 milijarde. Oba dejansko računata v FP16; to ni Qwenova BF16 referenca ali kvantizacija Q4.

NVILA uporablja uradno VILA kodo, pripeto na 57ecbfb904b9, in originalne uteži NVILA-8B. Neposredni izvajalnik odstrani omejitev starega API seznama dovoljenih modelov. Python gostitelja se ne uporablja. Starega torch 2.3/CUDA 12.2 FlashAttention ne namestimo: Qwen2 in SigLIP uporabita SDPA. Učne/FP8 uvoze in neuporabljene vizualne arhitekture izločimo; dinamični S2 ter projektor in uteži ostanejo. Natančen popravek in vse napake so shranjeni. Model ni preimenovan v 15B.

NVILA prejme isti rich-v2 prompt in slikovne bajte kot 2. del, greedy T=0, seed 42, največ 8192 izhodnih tokenov in kontekst 16384. Top-p/top-k sta pri greedy neaktivna. Kazen prisotnosti iz Ollame ni podprta; ločeno uporabimo repetition\_penalty=1,05 iz konfiguracije NVILA uteži. Začetni diagnostični pilot z ovojevo vrednostjo 1,0 je dosegel limit in ponavljal zapise; ohranjen je ločeno, brez tihega popravka.

Florence prejme samo tri izvirne kontrolne tokene: \texttt{<MORE\_DETAILED\_CAPTION>}, \texttt{<OD>}, \texttt{<DENSE\_REGION\_CAPTION>}. Ne dobi Qwenovega prompta ali sheme. Uporabimo zahtevani osnovni \texttt{microsoft/Florence-2-large}, ne različice \texttt{-ft}. Nastavitve so največ 1024 tokenov, num\_beams=3, do\_sample=false, seed 42; pozornost eager. Njegov predprocesor in ločljivost se razlikujeta od dinamičnega S2 in Ollame. Razlike modelov, števila parametrov, dolžin in nalog preprečujejo arhitekturno ali vzročno razlago hitrosti.
\subsection*{Časi vseh odgovorov, tudi zavrnjenih; brez ogrevanja}
'''+table(['Model','Naloga','Velj./vsi','Povp. s','Med. s','p90 s'],timing,r'p{3.4cm}p{3.1cm}rrrr')+r'''
Florence combined je neposredno merjen celotni postopek treh zaporednih nalog na sliko, s predprocesiranjem, razčlenjevanjem in zapisi JSON; izris slik je izključen. To ni samo čas ene od nalog. '''+tex(f"Skupni merjeni čas kombiniranega postopka za {combined_row['positions']} slik je {combined_row['sum_s']:.2f} s.")+r''' Surovi tokeni in ID-ji so ohranjeni. Števila tokenov različnih tokenizerjev niso neposredno primerljiva. Veljaven zapis ne pomeni pravilnega opisa.
''']
    warmups={model: [read_json(p) for p in sorted((OUT/model/'warmups').rglob('*.json'))] for model in ('nvila','florence')}
    warmtext='; '.join(f"{labels[m]}: {len(rs)} klicev, skupaj {sum(r.get('elapsed_seconds',0) for r in rs):.1f} s" for m,rs in warmups.items())
    matchrows=[[tex('trgovina' if r['dataset']==DATASETS[0] else 'agtea'),tex(r['task']),
                str(r['prediction_regions']),str(r['mapped_prediction_regions']),
                f"{r['matched']}/{r['eligible_reference_regions']}",number(r['mean_matched_iou'])] for r in boxes]
    reasonrows=[]
    for r in reason:
        c=r['classifications']
        reasonrows.append([tex(r['caption_model']),f"{r.get('ok',0)}/{r.get('error',0)}/{r.get('skipped',0)}",
                           str(c.get('fixed',0)),str(c.get('movable',0)),str(c.get('uncertain',0))])
    reason_timing='; '.join(f"{r['caption_model']}: {r['measured_calls']['positions']} merjenih klicev, "
                           f"povprečje {number(r['measured_calls']['mean_s'])} s, "
                           f"največ {r['max_prompt_eval_tokens']} vhodnih tokenov" for r in reason)
    pages.append(r'''\section*{Pomnilnik, lokalizacija in besedilna presoja}
'''+table(['Model','Dejanski tipi','Alloc. GiB','Reserv. GiB','Nalag. s'],memory,r'p{4cm}p{3.8cm}rrr')+r'''
Allocated/reserved sta procesna vrhova PyTorch, ne isti podatek kot Ollama size\_vram. Skupna poraba iz nvidia-smi vključuje namizje in tuje procese; surovi časovni vzorci so shranjeni. Nalaganje je ločeno; tabela kaže zadnjo sejo. Fizično ogrevanje po vsakem nalaganju je izključeno iz glavnih 100 slik: '''+tex(warmtext)+r'''. Začetna diagnostika NVILA je dodatna in ločena. Shranjeni prvi merjeni odgovori se ob nadaljevanju ponovno uporabijo.
\subsection*{Okvirji: dokumentirano ujemanje z anotiranimi regijami}
Florence napove xyxy v pikah izvirne slike. Validator preveri končnost, vrstni red in meje; okvirjev ne obrezuje ali popravlja. Izrisanih je do 200 slik (OD in dense za vsako sliko). Ujemanje je po isti sliki/SHA256 in vnaprej določenem slovarju imen (npr. person/people/man/woman $\rightarrow$ People); brez podnizov ali samodejnega ugibanja samostalnika v dolgi frazi. Pri IoU $\geq 0,5$ uporabimo največje enolično dvodelno ujemanje. Vsak par, imeni in oba okvirja so v extended-box-matches.json; slovar, izključitve in algoritem so v Git.
'''+table(['Nabor','Naloga','Nap. regije','Mapirana imena','Ujem./ref.','Povp. IoU'],matchrows,r'p{2.1cm}p{1.5cm}rrrr')+r'''
To je pokritost anotiranih regij v omejenem slovarju, ne popolna natančnost detekcije. Products/Fruits in druge vrste zunaj slovarja so izključene; skupinska oznaka ne pomeni ene instance. Nepovezana napoved ni avtomatično halucinacija, ker reference niso izčrpne. Dolga dense fraza lahko pravilno opiše regijo, a nima dobesednega razrednega ujemanja. Zato ne poročamo mAP, skupnega precision/recall/F1 ali caption accuracy. Qwen/Gemma in NVILA v tej primerjavi nimajo napovedanih okvirjev; prostorskega besedila ne pretvorimo v IoU.
\subsection*{Drugi korak: isti Qwen BF16 na istih 20 slikah}
NVILA veljavni rich opisi ostanejo v celoti nespremenjeni. Pri zavrnjenem rich izhodu ločen transport literal-native-text-v1 ohrani celoten surovi odgovor kot sceno, iz prvega strogo razčlenjenega JSON predponskega objekta pa vzame samo dobesedna imena in celotne izvirne objekte kot opazovanja. Izvirni odgovor ostane zavrnjen, ne postane veljaven rich JSON; nič besedila se ne zavrže. Če ni takega seznama ali so ključi podvojeni, je preskok. To je ločen besedilni preizkus, ne primerjava popolnoma iste vstopne sheme.

Florencejev opis ter OD oznake in dense fraze pretvorimo dobesedno: vsaka izvirna regija dobi lokalni ID in izvirno oznako/frazo ter okvir v besedilu. Regij med nalogama ne združujemo kot isto instanco. Ne dodamo materialov, stikov, namestitev ali odnosov. Transportni not\_visible pomeni nepodan/neveljaven dokaz namestitve; identifiable=true označi modelovo imenovano regijo, ne ročne potrditve. NVILA izvirne veljavne oznake dokazov ohranimo; ostale izvirne trditve ostanejo v opazovanju kot modelove trditve. Izhodi in izvorni SHA so zamrznjeni; drugi korak ne bere ali pošilja slik.
'''+table(['Opisovalnik','OK/napaka/preskok','fixed','movable','uncertain'],reasonrows,r'p{3cm}p{4cm}rrr')+r'''
Razvrščanje uporabi prejšnji short-no-examples prompt, T=0,3, seed 42. Opisi brez veljavnega dobesednega prenosa ostanejo preskočeni; napake razvrščanja niso popravljene. '''+tex(reason_timing)+r'''. Vseh 36 merjenih klicev in štiri izključena ogrevanja imajo potrjen GPU; kontekst je 16384. Različna granularnost regij pomeni, da števila razredov niso accuracy ali strinjanje istih instanc. Referenc stabilnosti ni; razlogi so modelove trditve, ne potrditev dejanskega pritrjevanja.
''')
    notes=read_json(REPORT/'extended-manual-notes.json') if (REPORT/'extended-manual-notes.json').exists() else {}
    cases=[f for f in selected if f['index'] in (0,5)]
    for offset in range(len(cases)):
        page=r'\section*{Dejanski primeri: fotografija in Florence regije}'
        for frame in cases[offset:offset+1]:
            ds=frame['dataset']; index=frame['index']; key=f'{ds}/{index}'
            page+=r'\subsection*{'+tex(('Trgovina' if ds==DATASETS[0] else 'agtea')+f' / indeks {index+1}')+'}\n'
            picture_paths=[]
            for task in ('od','dense'):
                source=OUT/'florence/overlays'/ds/task/f'frame-{index+1:06d}.jpg'
                target=REPORT/'figures'/f'extended-{ds}-{index+1}-{task}.jpg'
                if source.exists():
                    shutil.copyfile(source,target)
                    picture_paths.append(target.relative_to(REPORT).as_posix())
            original=REPORT/'figures'/f'extended-{ds}-{index+1}-rgb.png'
            shutil.copyfile(frame['image_path'],original)
            page+=r'\begin{center}\includegraphics[width=.32\linewidth]{'+original.relative_to(REPORT).as_posix()+'}'
            for path in picture_paths:
                page+=r'\hfill\includegraphics[width=.32\linewidth]{'+path+'}'
            page+=r'\\{\scriptsize Levo: RGB; sredina: OD; desno: DENSE\_REGION\_CAPTION. Okvirji so napovedi, ne reference.}\end{center}'+'\n'
            snippets=[]
            for model in ('nvila','florence','qwen35-bf16','qwen35-q4','gemma3-q4'):
                path=(saved_path(OUT/model,frame,'rich' if model=='nvila' else 'caption') if model in ('nvila','florence') else ROOT/'results/thud-part2'/f'rich-{model}-{ds}'/model/'t0-s42'/f'frame-{index+1:06d}.json')
                if path.exists():
                    r=read_json(path)
                    text=(r.get('parsed',{}).get(TASKS['caption'],'') if model=='florence' else literal_scene(r)) or r.get('error','')
                    text=' '.join(text.split())
                    snippets.append([tex(labels[model]),tex(r.get('status','')),tex(text[:210]+('...' if len(text)>210 else ''))])
            page+=table(['Model','Zapis','Dejanski izsek / napaka'],snippets,r'p{3.2cm}p{1.1cm}p{11cm}')
            page+=r'\textbf{Lastni delni vizualni pregled:} '+tex(notes.get(key,'Pregled še ni potrjen; glej lokalni HTML.'))+'\n'
            reason_examples=[]
            for model in ('nvila','florence'):
                matches=[read_json(p) for p in sorted((OUT/'reason'/model/ds).glob('*/t*/frame-*.json'))]
                record=next((r for r in matches if r['frame_id']==frame['frame_id']),None)
                if record and record['status']=='ok':
                    caption=read_json(Path(record['caption_source']))['parsed']
                    by_id={o['id']:o['label'] for o in caption['objects']}
                    for obj in record['parsed']['objects'][:2]:
                        reason_examples.append([tex(model),tex(by_id[obj['id']]),tex(obj['classification']),tex(obj['reason'])])
                elif record:
                    reason_examples.append([tex(model),'--',tex(record['status']),tex(record.get('error',''))])
            if reason_examples:
                page+=r'\subsection*{Dejanske besedilne presoje novih opisov}'
                page+=table(['Opis','Dobesedni objekt','Razred','Modelov razlog'],reason_examples,r'p{1.6cm}p{4.2cm}p{1.8cm}p{7.6cm}')
                page+=r'Različna imena niso potrjeno iste instance. To so modelove presoje; brez referenc stabilnosti jih ne štejemo kot accuracy.'+'\n'
        pages.append(page)
    tests=read_json(REPORT/'extended-provenance.json')['tests']
    test_note=f"Celotna zbirka: {tests['count']} testov, {tests['seconds']:.3f} s, " + ('vsi uspešni.' if tests['passed'] else 'glej shranjeni dnevnik napak.')
    pages.append(r'''\section*{Sklep nadaljevanja in ponovljivost}
\subsection*{Konkretne ohranjene napake in rešitve}
Florence se je po 37 shranjenih slikah končal s procesnim izhodom 132 (SIGILL), brez Python izjeme. Nadaljevanje z vključeno sledjo faulthandler je dokončalo vseh 100 slik brez ponovitve že shranjenih odgovorov. Vzrok ni potrjen; ponovnega zagona ne predstavljamo kot dokazanega popravka odvisnosti. En dense izhod agtea, indeks 24, vsebuje obrnjen okvir [0,48; 31,05; 87,84; 0,27]. Celoten izhod ostane zavrnjen; ločen rdeč diagnostični izris ohrani prvotni vrstni red koordinat. Ne obrnemo ga v veljaven okvir in ne vključimo v ujemanje.

NVILA začetno ogrevanje in merjeni pilot z repetition\_penalty=1,0 oba dosežeta 8192 tokenov s ponavljanjem JSON. Pri 1,05 je prvi merjeni odgovor kratek, vendar se zanke pri nekaterih drugih slikah še pojavijo. Pogoste so tudi dodatno besedilo za JSON, manjkajoča polja in napačna sintaksa. To so izmerjene omejitve formata modela, ne več nedelujoč GPU. Celotni odgovori, tudi zavrnitve in odmevi navodil, ostanejo shranjeni. Besedilni transport za razvrščanje ne spremeni teh rezultatov validacije.

Po uspešnem zaključku NVILA kontejnerja (izhod 0, vseh 100 odgovorov) je lupinski zaganjalnik javil sintaktično napako. Datoteka je bila med izvajanjem urejena; to je verjetna, ne dokazana razlaga. Razvrščanje je bilo nato uspešno izvedeno ločeno. Novi zaganjalnik najprej razčleni celotno funkcijo. GPU dnevnik ima 374 veljavnih vzorcev in poškodovan zaključek 125 ničelnih bajtov po meritvah; izvirnik ostane shranjen, veljavne vrstice so izvožene ločeno. Vrhovi VRAM v tabelah izhajajo iz inferenčnih zapisov.

Obe zahtevani novi okolji sta pripravljeni za dejansko slikovno inferenco na RTX 5090. Priročnost formata, uspešen GPU zagon in veljavnost izhoda so ločene ugotovitve od vizualne kakovosti. Primerjava vključuje tudi zavrnjene odgovore in razliko med splošnim opisom, detekcijo in regionalnim opisovanjem. Celoten pregled istih 20 fotografij z vsemi starimi/novimi odgovori, referencami in presojami je lokalno v manual-review.html. Preglednica 40 novih model/slika vrstic ostane pending; štirje prikazani primeri so samo delni pregled, ne izmišljena celotna referenca.

Surovi novi izhodi: \texttt{results/thud-extended/}; tam so originalni tokeni, razčlenjeni odgovori, konfiguracije, izvorne kode/SHA, pripete uteži, Docker dnevniki, diagnostika in vsi izrisi. Obstoječi odgovori so preverjeni proti indeksu SHA256 pred nadaljevanjem. Git vsebuje kodo, Dockerfile, navodila, agregate, izbrane slike, PDF in LaTeX. Točne revizije modelov in dejanski tipi parametrov so v extended-model-metadata.json in manifestih; dolgi izhodi ostanejo lokalno.
'''+tex(test_note)+r'''
\begin{verbatim}
bash scripts/thud_extended.sh prepare
bash scripts/thud_extended.sh nvila 1
bash scripts/thud_extended.sh nvila 100
bash scripts/thud_extended.sh florence 1
bash scripts/thud_extended.sh florence 100
bash scripts/thud_extended.sh reason
bash scripts/thud_extended.sh report
python3 -m unittest discover -s tests -v
\end{verbatim}
Izvajalnika zahtevata GPU brez CPU nadomestila. Vsako novo nalaganje fizično ogrejeta; že shranjene merjene odgovore (tudi napake) nadaljujeta brez ponovitev. Spremenjeni vhodi/nastavitve zahtevajo ločen poskus. Za vseobsegajoč zagon je na voljo all. Samo faza report ne izvaja inferenc. Neposredni NVILA izvajalnik ne odpira slikovnega API, zato ne trdimo, da stari strežnik deluje.

Naslednji korak naj bo ročni pregled pokritosti posameznih instanc in dejanskih podpor/stikov na istih 20 slikah. Namenske naloge Florence ne zagotovijo ustrezne granularnosti samo s tem, da vrnejo okvirje; NVILA ne postane najboljši model samo zaradi več parametrov ali več besed. Izbira mora upoštevati kakovost in napake skupaj s stroškom. Stabilnosti ne moremo potrditi z eno fotografijo.
\vfill{\footnotesize Uradni viri: \href{https://huggingface.co/Efficient-Large-Model/NVILA-8B}{NVILA-8B}, \href{https://github.com/NVlabs/VILA/tree/57ecbfb904b99b563db69ef7916f6390eecea516}{pripeta NVIDIA VILA koda}, \href{https://huggingface.co/microsoft/Florence-2-large}{Florence model in izvirne naloge}, \href{https://pytorch.org/get-started/previous-versions/}{PyTorch CUDA različice}. Preverjeno 21. 10. 2026.}
''')
    (REPORT/'extended.tex').write_text('\n\\newpage\n'.join(pages))


def main():
    selected = frames()
    rows = [{'model':'nvila','task':'rich',**stats(native_rows('nvila','rich'))}]
    rows += [{'model':'florence','task':name,**stats(native_rows('florence',name))} for name in TASKS]
    combined = combined_rows()
    rows.append({'model':'florence','task':'combined',**stats(combined)})
    rows += previous_comparison()
    write_json(REPORT/'extended-aggregates.json', rows)
    write_csv(REPORT/'extended-aggregates.csv',rows,list(rows[0]))
    boxes = evaluate_boxes(selected)
    reason = reasoning_summary()
    errors = [{'path':str(p.relative_to(ROOT)),'error':read_json(p).get('error')}
              for model in ('nvila','florence') for p in sorted((OUT/model).glob('*/*/frame-*.json'))
              if read_json(p).get('status')=='error']
    write_json(REPORT/'extended-errors.json',errors)
    verification = audit_old()
    verification['new_stage1_positions'] = {f"{r['model']}/{r['task']}":r['positions'] for r in rows[:4]}
    verification['all_stage1_complete'] = all(r['positions']==100 for r in rows[:4])
    verification['gpu_all_measured'] = all(r['gpu_verified']==100 for r in rows[:4])
    verification['content_and_token_ids_present'] = all(
        isinstance(r.get('content'),str) and bool(r.get('token_ids'))
        for model,task in [('nvila','rich')]+[('florence',t) for t in TASKS]
        for r in native_rows(model,task))
    write_json(REPORT/'extended-verification.json',verification)
    environments = {model:[read_json(p) for p in sorted((OUT/model/'environments').glob('*.json'))] for model in ('nvila','florence')}
    write_json(REPORT/'extended-model-metadata.json',environments)
    # Full original observations are kept locally; report contains compact tables.
    summary={'stage1':rows,'box_metrics':boxes,'verification':verification,
             'reasoning':reason,
             'process_failures':[read_json(p) for p in sorted((OUT/'setup').glob('*process-exit-*.json'))],
             'denominator_note':'Tables count final image/task positions after resume, including saved schema errors. One SIGILL process failure is separately recorded; its in-flight task/elapsed time are unknown, so no fabricated response or latency is included. This is not a zero-failure claim about physical attempts.',
             'note':'JSON/native parse validity is not visual correctness. No stability ground truth. Large raw results remain local.'}
    write_json(REPORT/'extended-summary.json',summary)
    save_provenance()
    rejected_overlays()
    make_review(selected)
    make_latex(rows,boxes,reason,selected)
    print(json.dumps(verification,indent=2))



# The repeat export below is offline: it only reads the new results/ponovitve.
# It shares the existing validators, matching, image drawing and table helpers.
def repetition_view(record):
    """Literal display extraction; never repair a source or feed a candidate to LLM."""
    task=record['task']
    if record['model_name']=='florence':
        value=record.get('parsed',{}).get(TASKS[task])
        if task=='caption': return {'scene_description':value or '', 'objects':[], 'object_list_available':False,'extraction':'native caption'}
        if not isinstance(value,dict): return {'scene_description':'','objects':[],'extraction':'no parsed native regions'}
        return {'scene_description':'','object_list_available':True,'objects':[{'id':i,'label':label,'bbox_xyxy':box}
                for i,(label,box) in enumerate(zip(value.get('labels',[]),value.get('bboxes',[])),1)],
                'extraction':'native regions; invalid coordinates remain rejected'}
    value=record.get('parsed',record.get('candidate_parsed'))
    document_count=1 if isinstance(value,dict) else 0
    unconsumed=0 if isinstance(value,dict) else None
    policy='validated description' if record['status']=='ok' else 'unvalidated syntactic candidate'
    if not isinstance(value,dict):
        try:
            content=record.get('content','').lstrip()
            from .experiment import strict_json
            documents=[];offset=0
            while offset<len(content):
                offset+=len(content[offset:])-len(content[offset:].lstrip())
                if offset==len(content): break
                try:
                    _,end=json.JSONDecoder().raw_decode(content[offset:])
                    doc=strict_json(content[offset:offset+end])
                except ValueError: break
                documents.append(doc);offset+=end
            if not documents: raise ValueError('No strict JSON document')
            document_count=len(documents);unconsumed=len(content)-offset
            # A display view only: keep objects from all literal JSON documents.
            # The original remains invalid and its complete trailing text remains.
            value={'scene_description':next((d['scene_description'] for d in documents if isinstance(d,dict) and isinstance(d.get('scene_description'),str)),'')}
            lists=[d['objects'] for d in documents if isinstance(d,dict) and isinstance(d.get('objects'),list)]
            if lists: value['objects']=[o for objects in lists for o in objects]
            policy=f'{len(documents)} literal JSON document(s) for display only; {len(content)-offset} trailing characters retained in raw'
        except (ValueError,TypeError): value={'scene_description':record.get('content','')}; policy='raw text only'
    if not isinstance(value,dict): value={}
    return {'scene_description':value.get('scene_description',''),
            'objects':[o for o in value.get('objects',[]) if isinstance(o,dict)] if isinstance(value.get('objects',[]),list) else [],
            'object_list_available':isinstance(value.get('objects'),list),
            'parsed_json_documents':document_count,'unconsumed_characters':unconsumed,
            'relations':value.get('relations'), 'extraction':policy}


def repetition_compact(value, key='', path=()):
    """Remove transport images/debug paths, never shorten model text or prompts."""
    if isinstance(value,dict):
        return {k:repetition_compact(v,k,path+(k,)) for k,v in value.items()
                if not (k=='traceback' and not path)
                and not (k=='images' and path==('request','messages'))
                and not (k=='image_url' and path==('request','messages','content'))}
    if isinstance(value,list): return [repetition_compact(v,key,path) for v in value]
    if isinstance(value,str) and not set(path)&{'parsed','candidate_parsed','response','view'} and key in {'image_path','image_path_resolved','caption_source','source_record','system_file','instruction_file'}:
        if value.startswith(str(ROOT)+'/'): return value[len(str(ROOT))+1:]
    return value


def repetition_consistency(records):
    """Name-group changes, not instance identities; exact label boxes matched separately."""
    from itertools import combinations
    aliases=read_json(ROOT/'configs/thud-florence-matching.json')['aliases']
    groups={}
    for r in records: groups.setdefault((r['frame_id'],r['model_name'],r['task']),{})[r['repeat']]=r
    result=[]
    for (fid,model,task),runs in sorted(groups.items()):
        for a,b in combinations(('r1','r2','r3'),2):
            if a not in runs or b not in runs: continue
            left,right=runs[a],runs[b]
            views=[repetition_view(r) for r in (left,right)]
            named=[]
            for v in views:
                d={}
                for obj in v['objects']:
                    label=obj.get('label')
                    if isinstance(label,str): d.setdefault(' '.join(label.lower().split()),[]).append(obj)
                named.append(d)
            x,y=map(set,named); union=x|y
            changes=[{'normalized_label':name,
                      'left_objects':named[0][name],'right_objects':named[1][name]}
                     for name in sorted(x&y) if [{k:v for k,v in o.items() if k!='id'} for o in named[0][name]] !=
                                                [{k:v for k,v in o.items() if k!='id'} for o in named[1][name]]]
            row={'frame_id':fid,'model':model,'task':task,'left':a,'right':b,
                 'left_status':left['status'],'right_status':right['status'],
                 'raw_equal':left.get('content')==right.get('content'),
                 'scene_equal':views[0]['scene_description']==views[1]['scene_description'],
                 'left_objects':len(views[0]['objects']),'right_objects':len(views[1]['objects']),
                 'object_list_available':[v.get('object_list_available',True) for v in views],
                 'name_intersection':len(x&y),'name_union':len(union),
                 'name_jaccard':len(x&y)/len(union) if union else None,
                 'names_removed':sorted(x-y),'names_added':sorted(y-x),
                 'possible_alias_changes':[{'left':s,'right':t,'canonical_hint':aliases[s]} for s in sorted(x-y) for t in sorted(y-x)
                                           if s in aliases and t in aliases and aliases[s]==aliases[t]],
                 'name_group_details_changed':changes,
                 'structured_relations_available':all(isinstance(v.get('relations'),list) for v in views),
                 'relations_equal':views[0].get('relations')==views[1].get('relations') if all(isinstance(v.get('relations'),list) for v in views) else None}
            if model=='florence' and task in ('od','dense') and left['status']==right['status']=='ok':
                labels=x|y
                refs=[{**o,'label':' '.join(o['label'].lower().split())} for o in views[1]['objects']]
                row['box_matching']=match_regions(views[0]['objects'],refs,{'aliases':{s:s for s in labels},'iou_threshold':.5})
            result.append(row)
    return result


def repetition_reason_consistency(records):
    """Compare only identical frozen file SHA and typed local object IDs."""
    from itertools import combinations
    groups={}
    for r in records: groups.setdefault(r['frame_id'],[]).append(r)
    result=[]
    for fid,rows in sorted(groups.items()):
        for left,right in combinations(rows,2):
            changes=[k for k in ('repeat','model_name','prompt_name') if left[k]!=right[k]]
            if len(changes)!=1: continue
            same=left['caption_sha256']==right['caption_sha256']
            valid=same and left['status']==right['status']=='ok'
            row={'frame_id':fid,'dimension':changes[0],
                 'left':{k:left[k] for k in ('model_name','prompt_name','repeat','status')},
                 'right':{k:right[k] for k in ('model_name','prompt_name','repeat','status')},
                 'frozen_sha256':left['caption_sha256'],'same_frozen_input':same,
                 'comparable':valid,'objects':[]}
            if valid:
                by_id={(type(o['id']),o['id']):o for o in right['parsed']['objects']}
                for obj in left['parsed']['objects']:
                    other=by_id.get((type(obj['id']),obj['id']))
                    if other is None: raise ValueError('Typed IDs differ in valid classifications')
                    row['objects'].append({'id':obj['id'],'id_type':type(obj['id']).__name__,
                        'left_class':obj['classification'],'right_class':other['classification'],
                        'class_changed':obj['classification']!=other['classification'],
                        'left_reason':obj['reason'],'right_reason':other['reason']})
            row['changed_objects']=sum(o['class_changed'] for o in row['objects'])
            row['compared_objects']=len(row['objects'])
            result.append(row)
    return result


def repetitions_export():
    """Publish all 20 textual results and ten portable, independently readable cases."""
    from itertools import product
    base=ROOT/'results/ponovitve'; target=ROOT/'reports/PRIMERJAVA_PONOVITEV'
    target.mkdir(parents=True,exist_ok=True)
    selected=read_json(base/'frames.json')
    records=[]
    combined=[]
    for p in sorted((base/'stage1').glob('*/*/*/*/frame-*.json')):
        r=read_json(p); model=p.parts[-5]; repeat=p.parts[-4]; task=p.parent.name
        if model=='florence' and task=='combined':
            combined.append({'model_name':model,'task':'combined','repeat':repeat,**r}); continue
        if model!='florence': task='description'
        r.update(model_name=model,repeat=repeat,task=task,local_record=str(p.relative_to(ROOT)),
                 original_record_sha256=sha256(p.read_bytes()))
        if model=='nvila':
            # Pinned AUTO chat template and extract_media concatenate the image
            # marker with the entire instruction. Raw component fields stay intact.
            user='<image>'+r['system_prompt']+'\n\n'+r['user_prompt']
            native_text='<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n'
            if r.get('native_chat_prompt_text',native_text)!=native_text: raise ValueError('Native chat reconstruction differs from runtime')
            r['native_prompt_provenance']='runtime capture' if 'native_chat_prompt_text' in r else 'reconstructed from pinned tokenizer template and extract_media'
            r.update(native_system_prompt='You are a helpful assistant',native_user_prompt=user,native_chat_prompt_text=native_text)
        r['view']=repetition_view(r)
        names=[o.get('label','') for o in r['view']['objects'] if isinstance(o.get('label'),str)]
        normalized=[' '.join(s.lower().split()) for s in names]
        r['view']['name_cautions']={'repeated_names':{s:n for s,n in Counter(normalized).items() if n>1},
            'plural_or_group_wording':[s for s in names if ' '.join(s.lower().split()) in {'people','products','goods','chairs','tables','shelves','bottles','plants','windows','lights'}],
            'note':'Wording flags only. Do not infer instance count or merge synonymous/plural names.'}
        records.append(r)
    reason=[]
    for p in sorted((base/'stage2').glob('*/*/*/*/*/frame-*.json')):
        r=read_json(p)
        r.update(model_name=p.parts[-6],prompt_name=p.parts[-5],repeat=p.parts[-4],
                 local_record=str(p.relative_to(ROOT)),original_record_sha256=sha256(p.read_bytes()))
        frozen=Path(r['caption_source'])
        if sha256(frozen.read_bytes())!=r['caption_sha256']: raise ValueError('Frozen input changed')
        if r.get('response'):
            if any(m.get('images') for m in r['request']['messages']): raise ValueError('Image leaked into reasoner')
        reason.append(r)
    frozen_sha={read_json(p)['frame_id']:sha256(p.read_bytes()) for p in (base/'frozen/qwen35-bf16-r1').glob('frame-*.json')}
    if any(r['caption_sha256']!=frozen_sha.get(r['frame_id']) for r in reason):
        raise ValueError('Reasoners did not use the same Qwen BF16 r1 frozen input')
    for r in combined:
        tasks=[s for s in records if s['frame_id']==r['frame_id'] and s['repeat']==r['repeat'] and s['model_name']=='florence']
        r['status']='ok' if len(tasks)==3 and all(s['status']=='ok' for s in tasks) else 'error'
        if not r.get('complete_three_task_measurement'):
            r['elapsed_seconds']=None  # Never present a partial resume as three-task wall time.
    summaries=[]
    for stage,rows,keys in [('stage1',records,['model_name','task','repeat']),
                            ('combined',combined,['model_name','task','repeat']),
                            ('stage2',reason,['model_name','prompt_name','repeat'])]:
        groups={}
        for r in rows: groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
        for values,rs in sorted(groups.items()):
            values=dict(zip(keys,values))
            measured=[r for r in rs if r.get('gpu_verified')]
            summaries.append({'stage':stage,**values,**stats(rs),
                              'skipped':sum(r.get('status')=='skipped' for r in rs),
                              'actual_generations':len(measured) if stage!='combined' else 0,
                              'syntax_valid':sum(r.get('syntax_valid',False) for r in rs) if values.get('model_name')!='florence' else None,
                              'literal_objects':sum(len(r.get('view',{}).get('objects',[])) for r in rs) if stage=='stage1' else None,
                              'nonempty_literal_object_lists':sum(bool(r.get('view',{}).get('objects')) for r in rs) if stage=='stage1' else None,
                              'schema_note':'Native task validity for Florence; legacy description validation otherwise.' if stage=='stage1' else ''})
    write_json(target/'summary.json',summaries)
    headers=list(dict.fromkeys(k for r in summaries for k in r))
    write_csv(target/'summary.csv',[{k:r.get(k,'') for k in headers} for r in summaries],headers)
    consistent=repetition_consistency(records)
    reason_pairs=repetition_reason_consistency(reason)
    write_json(target/'consistency.json',{'stage1':consistent,'stage2':reason_pairs,
        'note':'Normalized exact names only; groups/aliases are not automatically equal. Name groups are not cross-run instance identities. Agreement is not correctness.'})
    flat=[]
    for r in consistent:
        flat.append({k:r[k] for k in ('frame_id','model','task','left','right','raw_equal','left_objects','right_objects','name_intersection','name_union','name_jaccard')})
    if flat: write_csv(target/'consistency.csv',flat,list(flat[0]))
    reason_flat=[{'frame_id':r['frame_id'],'dimension':r['dimension'],'comparable':r['comparable'],
                 'same_frozen_input':r['same_frozen_input'],'compared_objects':r['compared_objects'],
                 'changed_objects':r['changed_objects'],'left':json.dumps(r['left']),'right':json.dumps(r['right'])} for r in reason_pairs]
    if reason_flat: write_csv(target/'reasoning-consistency.csv',reason_flat,list(reason_flat[0]))
    # Apply the already fixed reference matching policy, separately per repetition.
    policy=read_json(ROOT/'configs/thud-florence-matching.json'); boxes=[]
    refs={}
    for f in selected:
        refs[f['frame_id']]=[json.loads(s) for s in (ROOT/'data'/f['dataset']/'objects.jsonl').read_text().splitlines() if s.strip()][f['index']]
    for r in records:
        if r['model_name']=='florence' and r['task'] in ('od','dense'):
            boxes.append({'frame_id':r['frame_id'],'task':r['task'],'repeat':r['repeat'],'status':r['status'],
                          **match_regions(r['view']['objects'] if r['status']=='ok' else [],refs[r['frame_id']]['objects'],policy)})
    write_json(target/'reference-matching.json',{'policy':policy,'results':boxes,
        'note':'Coverage of annotated regions in the existing vocabulary; not full precision/recall or mAP. A matched box does not verify a caption or stability.'})
    reference_summary=[]
    for ds,task,repeat in product(DATASETS,('od','dense'),('r1','r2','r3')):
        rs=[r for r in boxes if r['frame_id'].startswith(ds+'/') and r['task']==task and r['repeat']==repeat]
        if rs:
            reference_summary.append({'dataset':ds,'task':task,'repeat':repeat,
                **{k:sum(r[k] for r in rs) for k in ('matched','eligible_reference_regions','excluded_reference_regions','prediction_regions')},
                'mean_matched_iou':statistics.mean(p['iou'] for r in rs for p in r['pairs']) if any(r['pairs'] for r in rs) else None})
    write_json(target/'reference-summary.json',reference_summary)
    if reference_summary: write_csv(target/'reference-summary.csv',reference_summary,list(reference_summary[0]))
    for name,rows in [('stage1-responses',records),('stage2-responses',reason)]:
        (target/(name+'.jsonl')).write_text(''.join(json.dumps(repetition_compact(r),ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    diagnostic=[]
    for p in sorted((base/'diagnostics').rglob('*.json')):
        if not (p.name.startswith('frame-') or 'warmups' in p.parts): continue
        row=read_json(p)
        diagnostic.append({'local_source':str(p.relative_to(base)),**repetition_compact(row)})
    if diagnostic:
        (target/'diagnostic-nvila.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in diagnostic))
    frozen_dir=target/'frozen-inputs'; frozen_dir.mkdir(exist_ok=True)
    for p in sorted((base/'frozen/qwen35-bf16-r1').glob('frame-*.json')): shutil.copyfile(p,frozen_dir/p.name)
    prompts=target/'prompts'; prompts.mkdir(exist_ok=True)
    for p in (ROOT/'configs/ponovitve').glob('*.txt'): shutil.copyfile(p,prompts/p.name)
    unique={}
    for r in records+reason:
        if r.get('status')=='skipped': continue
        system=r.get('system_prompt',''); user=r.get('user_prompt',r.get('prompt',''))
        digest=sha256(json.dumps([system,user],ensure_ascii=False).encode())
        unique.setdefault(digest,{'sha256':digest,'system_prompt':system,'user_prompt':user,
                                 **{k:r[k] for k in ('native_system_prompt','native_user_prompt','native_chat_prompt_text','native_prompt_provenance') if k in r}})
    (prompts/'actual-requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in unique.values()))
    write_json(prompts/'native-tasks.json',TASKS)
    tokenizer_config=read_json(ROOT/'results/thud-extended/models/nvila/llm/tokenizer_config.json')
    (prompts/'nvila-tokenizer-chat-template.txt').write_text(tokenizer_config['chat_template'])
    old=read_json(base/'previous-results-index.json')
    changed=[p for p,h in old.items() if sha256((ROOT/p).read_bytes())!=h]
    if changed: raise ValueError('Old responses changed')
    keys={(r['frame_id'],r['model_name'],r['task'],r['repeat']) for r in records}
    expected={(f['frame_id'],m,t,r) for f in selected for m in ('florence','nvila','qwen35-bf16','qwen35-q4','gemma3-q4')
              for t in (TASKS if m=='florence' else ('description',)) for r in ('r1','r2','r3')}
    expected_reason={(f['frame_id'],m,p,r) for f in selected for m in ('qwen35-bf16','qwen35-q4') for p in ('short','long') for r in ('r1','r2','r3')}
    actual_reason={(r['frame_id'],r['model_name'],r['prompt_name'],r['repeat']) for r in reason}
    if len(keys)!=len(records) or len(actual_reason)!=len(reason): raise ValueError('Duplicate positions')
    call_ids=[r.get('call_id',r.get('response',{}).get('created_at')) for r in records+reason if r.get('gpu_verified')]
    if None in call_ids or len(call_ids)!=len(set(call_ids)): raise ValueError('Repeated/missing physical call identity')
    plan=read_json(base/'plan.json')
    public_frames=[{**{k:v for k,v in f.items() if k!='image_path'},'image_path':str(Path(f['image_path']).relative_to(ROOT))} for f in selected]
    configs={str(p.relative_to(base)):repetition_compact(read_json(p)) for p in sorted((base/'configs').glob('*.json'))}
    native_configs={str(p.relative_to(base)):read_json(p)['config'] for m,r in product(('nvila','florence'),('r1','r2','r3'))
                    for p in [base/'stage1'/m/r/'manifest.json'] if p.exists()}
    environments=[]
    for p in sorted((base/'stage1').glob('*/*/environments/*.json')):
        env=read_json(p)
        environments.append({'model':p.parts[-4],'repeat':p.parts[-3],
                             **{k:env[k] for k in ('python','torch','cuda','transformers','gpu_name','gpu_capability',
                                  'parameter_devices','parameter_dtypes','parameter_count','load_seconds') if k in env}})
        for key in ('initialization_seed','embedding_tail_sha256'):
            if key in env: environments[-1][key]=env[key]
    warmups=[]
    for stage in ('stage1','stage2'):
        for p in sorted((base/stage).glob('**/warmups/**/*.json')):
            row=read_json(p)
            warmups.append({'path':str(p.relative_to(base)), 'stage':stage,
                            'seconds':row.get('elapsed_seconds'),'gpu_verified':row.get('gpu_verified',False),
                            'ollama_load_seconds':row.get('response',{}).get('load_duration',0)/1e9,
                            'excluded_from_quality':True,'error':row.get('error')})
    write_json(target/'warmups.json',warmups)
    # Native manifests contain prompt text/config, no weights or vendor code.
    manifest={'protocol':plan,'frames':public_frames,'ollama_configs':configs,'native_configs':native_configs,
              'native_environments':environments,'physical_warmups':len(warmups),
              'model_revisions':{m:read_json(ROOT/f'results/thud-extended/setup/{m}-api.txt')['sha'] for m in ('nvila','florence')},
              'ollama_digests':{r['model_name']:r['gpu']['digest'] for r in records+reason if r.get('gpu')},
              'previous_responses_unchanged':len(old),'missing_stage1':sorted(expected-keys),'missing_stage2':sorted(expected_reason-actual_reason),
              'actual_stage1_generations':sum(bool(r.get('gpu_verified')) for r in records),
              'actual_stage2_generations':sum(bool(r.get('gpu_verified')) for r in reason),
              'same_frozen_qwen_r1_verified':True,
              'stage2_skipped':sum(r['status']=='skipped' for r in reason),'unique_physical_call_ids':len(call_ids),
              'export_omissions':['Image bytes/base64 in transport requests (source RGB SHA retained).','Python traceback debug text; original remains local.'],
              'accuracy_note':'No stability ground truth; format validity and agreement do not establish accuracy.',
              'timing_note':'GPU is shared with other processes. Loading and physical warmups are separate; do not infer model-only speed differences.',
              'raw_local':'results/ponovitve/','export_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    if (base/'nvila-prompt-revision.json').exists(): manifest['nvila_prompt_revision']=read_json(base/'nvila-prompt-revision.json')
    if (base/'setup/nvila-initialization-correction.json').exists(): manifest['nvila_initialization_correction']=read_json(base/'setup/nvila-initialization-correction.json')
    if (base/'setup/nvila-r3-segfault.json').exists(): manifest['nvila_r3_process_failure']=read_json(base/'setup/nvila-r3-segfault.json')
    manifest['process_failures']=[{'source':str(p.relative_to(base)),
                                  'exit_code':read_json(p)['exit_code'],
                                  'note':read_json(p).get('note'),
                                  'original_command_sha256':sha256(json.dumps(read_json(p).get('command'),ensure_ascii=False).encode())}
                                 for p in sorted((base/'process-errors').glob('*.json'))]
    manifest['diagnostic_measured_generations']=sum(not r.get('excluded_from_quality',False) for r in diagnostic)
    manifest['diagnostic_warmups']=sum(bool(r.get('excluded_from_quality',False)) for r in diagnostic)
    tests_path=base/'logs/tests-final.log'
    if tests_path.exists():
        test_text=tests_path.read_text(); match=re.search(r'Ran (\d+) tests in ([\d.]+)s',test_text)
        manifest['tests']={'count':int(match[1]) if match else None,'passed':test_text.strip().endswith('OK'),
                           'seconds':float(match[2]) if match else None,'log_sha256':sha256(tests_path.read_bytes())}
    write_json(target/'manifest.json',manifest)
    if not manifest['missing_stage1']:
        repetition_cases(target,base,selected,records,reason,consistent,refs)
    repetition_readme(target,manifest,summaries,consistent,reason_pairs)
    print(json.dumps({k:manifest[k] for k in ('actual_stage1_generations','actual_stage2_generations','stage2_skipped','unique_physical_call_ids')},indent=2))


def repetition_cases(target,base,selected,records,reason,consistent,refs):
    """Select five cases per dataset by explicit diagnostics, then retain that choice."""
    from .thud_extended_common import draw_boxes
    selection_file=target/'case-selection.json'
    if selection_file.exists():
        chosen=read_json(selection_file)
    else:
        chosen=[]
        for ds in DATASETS:
            candidates=[]
            for frame in selected:
                if frame['dataset']!=ds: continue
                fid=frame['frame_id']
                pairs=[r for r in consistent if r['frame_id']==fid]
                native=[r for r in records if r['frame_id']==fid]
                values=[r['name_jaccard'] for r in pairs if r['name_jaccard'] is not None and r['model']!='florence']
                failures=sum(r['status']!='ok' for r in native)
                density=max((len(r['view']['objects']) for r in native if r['task']=='od'),default=0)
                stable_flo=all(r['raw_equal'] and r['left_status']==r['right_status']=='ok'
                               for r in pairs if r['model']=='florence')
                score=(1-statistics.mean(values) if values else 1)+failures*.15+density*.01
                candidates.append({'frame_id':fid,'dataset':ds,'index':frame['index'],'diagnostic_score':score,
                                   'format_failures':failures,'max_od_regions':density,'florence_stable':stable_flo})
            # One stable native example per room, followed by four difficult/divergent cases.
            stable=sorted(candidates,key=lambda r:(not r['florence_stable'],r['format_failures'],-r['max_od_regions'],r['index']))[0]
            stable={**stable,'selection_reason':'Stable deterministic native results; visually check successful localization/caption separately.'}
            difficult=[{**r,'selection_reason':'High scripted divergence/failure/density score; inspect occlusion, unsupported claims and format errors.'}
                       for r in sorted(candidates,key=lambda r:(-r['diagnostic_score'],r['index'])) if r['frame_id']!=stable['frame_id']][:4]
            chosen.extend([stable]+difficult)
        write_json(selection_file,chosen)
    notes=read_json(target/'manual-review.json') if (target/'manual-review.json').exists() else {}
    pages=['<!doctype html><html lang="sl"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
           '<title>THUD: tri ponovitve, deset primerov</title>',
           '<style>body{font:16px system-ui;max-width:1550px;margin:2em auto;padding:0 1em;color:#18222b}a{color:#005b96}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1em;max-height:65vh;overflow:auto}.runs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1em}.run{border:1px solid #bcc7d0;border-radius:5px;padding:1em}details{margin:.8em 0}summary{cursor:pointer;font-weight:bold}article{margin-top:3em;border-top:3px solid #557}.notice{padding:1em;background:#fff0d2}.references{border:3px dashed #a33;padding:.5em}nav{display:flex;flex-wrap:wrap;gap:1em}li{margin-bottom:.5em}@media(max-width:900px){.runs{grid-template-columns:1fr}}</style>',
           '<style>.case-layout{display:grid;grid-template-columns:minmax(240px,320px) minmax(0,1fr);gap:1em}.case-layout aside{position:sticky;top:1em;align-self:start}@media(max-width:1000px){.case-layout{display:block}.case-layout aside{position:static}}</style>',
           '<h1>THUD: trije sveži izhodi na isti fotografiji</h1>',
           '<p>10 podrobnih primerov od 20 slik (5 trgovina + 5 agtea). Odpri ta index.html neposredno v brskalniku po prenosu repozitorija. Podatki vseh 20: <a href="stage1-responses.jsonl">VLM</a>, <a href="stage2-responses.jsonl">LLM</a>, <a href="summary.csv">meritve</a>, <a href="consistency.json">primerjave</a>.</p>',
           '<p class="notice">Veljaven JSON in strinjanje nista pravilnost. Lokalni ID med modeli ni identiteta istega predmeta. Florence okvirji so njegove napovedi; referenčni izris je posebej označen. Drugi modeli nimajo lastnih okvirjev. Brez referenc stabilnosti ni accuracy.</p>',
           '<nav>'+''.join(f'<a href="#case-{i}">{i:02d}: {html.escape(c["dataset"].replace("thud_", ""))} / {c["index"]+1}</a>' for i,c in enumerate(chosen,1))+'</nav>']
    def pre(value): return '<pre>'+html.escape(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre>'
    for i,case in enumerate(chosen,1):
        frame=next(f for f in selected if f['frame_id']==case['frame_id'])
        slug=f'{i:02d}-'+('trgovina' if frame['dataset']==DATASETS[0] else 'agtea')+f'-{frame["index"]+1:02d}'
        directory=target/'cases'/slug;directory.mkdir(parents=True,exist_ok=True)
        original=directory/'original.png'
        if not original.exists(): shutil.copyfile(frame['image_path'],original)
        stage1=[repetition_compact(r) for r in records if r['frame_id']==frame['frame_id']]
        stage2=[repetition_compact(r) for r in reason if r['frame_id']==frame['frame_id']]
        write_json(directory/'stage1.json',stage1);write_json(directory/'stage2.json',stage2)
        note=notes.get(frame['frame_id'],{'review_status':'pending','selection_reason':case['selection_reason']})
        write_json(directory/'review.json',note)
        refpath=directory/'reference-only.jpg'
        ref=refs[frame['frame_id']]
        draw_boxes(frame['image_path'],{'reference':{'labels':[o['label'] for o in ref['objects']],
                  'bboxes':[o['bbox_xyxy'] for o in ref['objects']]}},'reference',refpath)
        relative=directory.relative_to(target).as_posix()
        pages+=['<article id="case-'+str(i)+'"><h2>'+html.escape(f'{i:02d} — {frame["frame_id"]}, izvorni indeks {frame["index"]+1}')+'</h2>',
                '<div class="case-layout"><aside><a href="'+relative+'/original.png"><img src="'+relative+'/original.png" alt="Izvirni RGB"></a><p>Izvirni RGB. Klik za polno ločljivost. Fotografija ostane ob rezultatih.</p></aside><div>',
                '<p><strong>Razlog izbire:</strong> '+html.escape(note.get('selection_reason',case['selection_reason']))+'</p><details><summary>Lastna podrobna vizualna presoja in pregled LLM</summary>'+pre(note)+'</details>',
                '<p><a href="'+relative+'/stage1.json">Celotni VLM izhodi</a> · <a href="'+relative+'/stage2.json">Celotni LLM izhodi</a> · <a href="'+relative+'/review.json">Ročni pregled</a></p>',
                '<details class="references"><summary>REFERENCE — samo pregled/vrednotenje; niso vhod VLM</summary><img src="'+relative+'/reference-only.jpg">'+pre(ref['objects'])+'</details>']
        for model in ('florence','nvila','qwen35-bf16','qwen35-q4','gemma3-q4'):
            pages.append('<details'+(' open' if model=='florence' else '')+'><summary>VLM: '+model+'</summary><div class="runs">')
            for repeat in ('r1','r2','r3'):
                pages.append('<section class="run"><h3>'+repeat+' / seed '+str({'r1':42,'r2':43,'r3':44}[repeat])+'</h3>')
                for row in (r for r in stage1 if r['model_name']==model and r['repeat']==repeat):
                    pages.append('<h4>'+html.escape(row['task']+' — '+row['status'])+'</h4>')
                    if model=='florence' and row['task'] in ('od','dense'):
                        name=row['task']+'-'+repeat+'.jpg'; image=directory/name
                        if row['status']=='ok':
                            draw_boxes(frame['image_path'],row['parsed'],TASKS[row['task']],image)
                            pages.append('<p>NAPOVED Florence '+html.escape(row['task'])+'</p><a href="'+relative+'/'+name+'"><img src="'+relative+'/'+name+'"></a>')
                        else: pages.append('<p>Zavrnjeni okvirji; niso popravljeni ali izrisani kot veljavna napoved.</p>')
                    elif model!='florence': pages.append('<p>Izvirna fotografija zgoraj + dobesedni seznam spodaj; lastnih okvirjev ni.</p>')
                    pages.append('<p>'+html.escape(f"Čas {row.get('elapsed_seconds',0):.3f} s; prikaz: {row['view']['extraction']}")+'</p>')
                    if row['view']['scene_description']: pages.append(pre(row['view']['scene_description']))
                    pages.append('<ol>'+''.join('<li>'+pre(o)+'</li>' for o in row['view']['objects'])+'</ol>')
                    pages.append('<details><summary>Celotni surovi odgovor, prompti in podatki</summary>'+pre(row)+'</details>')
                pages.append('</section>')
            pages.append('</div></details>')
        pages.append('<h3>Besedilni LLM — isti zamrznjeni Qwen BF16 opis iz r1</h3>')
        for model in ('qwen35-bf16','qwen35-q4'):
            for prompt in ('short','long'):
                rows=[r for r in stage2 if r['model_name']==model and r['prompt_name']==prompt]
                pages.append('<details><summary>'+html.escape(model+' / '+prompt+'-no-examples')+'</summary>')
                if rows:
                    pages.append('<details><summary>Celotni dejanski sistemski/uporabniški prompt</summary>'+pre({'system_prompt':rows[0]['system_prompt'],'user_prompt':rows[0].get('user_prompt','Source description skipped'),'caption_sha256':rows[0]['caption_sha256']})+'</details>')
                pages.append('<div class="runs">')
                for r in rows:
                    pages.append('<section class="run"><h4>'+html.escape(r['repeat']+' — '+r['status'])+'</h4>'+pre({k:r[k] for k in ('parsed','candidate_parsed','error','elapsed_seconds','caption_sha256') if k in r})+
                                 '<details><summary>Celotni odgovor in sled</summary>'+pre(r)+'</details></section>')
                pages.append('</div></details>')
        pages.append('</div></div></article>')
    pages.append('</html>')
    (target/'index.html').write_text('\n'.join(pages))
    write_json(target/'case-index.json',[{'directory':str(p.relative_to(target))} for p in sorted((target/'cases').iterdir()) if p.is_dir()])


def repetition_readme(target,manifest,summaries,consistent,reason_pairs):
    stage1=[r for r in summaries if r['stage']=='stage1']
    table=['| Model / naloga | r1 OK/vsi | r2 OK/vsi | r3 OK/vsi |',
           '| --- | ---: | ---: | ---: |']
    for model,task in sorted({(r['model_name'],r['task']) for r in stage1}):
        cells=[]
        for repeat in ('r1','r2','r3'):
            r=next((r for r in stage1 if r['model_name']==model and r['task']==task and r['repeat']==repeat),None)
            cells.append(f"{r['ok']}/{r['positions']}" if r else 'manjka')
        table.append('| '+model+' / '+task+' | '+' | '.join(cells)+' |')
    flo=[r for r in consistent if r['model']=='florence']
    nvila=[r for r in stage1 if r['model_name']=='nvila']
    grouped=[]
    for dimension in ('repeat','prompt_name','model_name'):
        rows=[r for r in reason_pairs if r['dimension']==dimension]
        grouped.append(f"- `{dimension}`: {sum(r['comparable'] for r in rows)}/{len(rows)} veljavnih parov klicev; "
                       f"{sum(r['changed_objects'] for r in rows)}/{sum(r['compared_objects'] for r in rows)} različnih razredov na primerjanih objektih.")
    text=f'''# Tri ponovitve slikovnih opisov in besedilnega razvrščanja

Uporabili smo istih 20 THUD slik (10 trgovina + 10 agtea) kot v prejšnjem poskusu.
To so povezani pogledi dveh prostorov, ne 20 neodvisnih prizorov.
Cilj je pogledati, kaj se spremeni med semeni, in ločeno preveriti vpliv kratkega/dolgega
navodila ter BF16/Q4 besedilnega modela na isti shranjeni opis. To je paket meritev
in primerov, iz katerega lahko pripravimo poročilo; novega dolgega PDF ne izdelujemo.

Odpri **[index.html](index.html)** neposredno v brskalniku po prenosu repozitorija.
Vse slike, opisi in podatki desetih primerov so relativno povezani in delujejo brez strežnika.
Na GitHubu je HTML izvorna datoteka; za prikaz prenesi repozitorij ali mapo s primeri.
Deset primerov (5 + 5) izberejo kazalniki razhajanja, napak in gostote, vključno z dvema
stabilnima Florence primeroma. Lastna vizualna presoja je v `manual-review.json` in vsaki
mapi primera; modelove trditve ostanejo ločene od te presoje.

## Dejanski obseg

- VLM: **{manifest['actual_stage1_generations']} svežih GPU generacij / 420 položajev**.
- LLM: **{manifest['actual_stage2_generations']} svežih GPU generacij / 240 položajev**, **{manifest['stage2_skipped']} preskokov** zaradi neuporabnega Qwen BF16 r1 vira.
- Diagnostični NVILA poskusi: {manifest['diagnostic_measured_generations']} dodatnih merjenih odgovorov in {manifest['diagnostic_warmups']} zaključenih izključenih ogrevanj, ohranjeni posebej; ne štejejo v glavno primerjavo.
- Glavni izbor ni prepis starega cache: fizične generacije imajo enolične ID oziroma strežniške čase. Pilot istega protokola ostane del r1.
- Predhodnih **{manifest['previous_responses_unchanged']}** shranjenih odgovorov je preverjeno nespremenjenih. Ogrevanja in nalaganja so ločena v `warmups.json` in `manifest.json`.

{chr(10).join(table)}

OK pomeni skladnost z ustreznim izhodnim formatom, **ne vsebinske pravilnosti**.
NVILA ima dobesedno razčlenjen neprazen seznam objektov v **{sum(r['nonempty_literal_object_lists'] for r in nvila)}/{sum(r['positions'] for r in nvila)}** odgovorih,
tudi kadar celoten zapis ni veljaven po shemi. Te vsebine ne zavržemo.
NVILA več JSON dokumentov/odvečno besedilo ostane napaka celotnega odgovora. Za prikaz
skripta izlušči samo dobesedne objekte iz strogo prebranih JSON dokumentov; ničesar ne
popravlja v veljaven opis. Preostalo besedilo in celotni original ostaneta dostopna.
Neobstoječ razčlenjen seznam ni dokaz, da na sliki ni objektov.

## Protokol in meje primerjave

Qwen BF16/Q4 in Gemma Q4 uporabljajo isti novi `description-v1.txt`, T=0.3,
semena 42/43/44 za r1/r2/r3, kontekst 16384, največ 4096 novih tokenov,
top-p=0.8, top-k=20, presence penalty=1.5. JSON transport je obstoječa Ollama.
NVILA uporablja vsebinsko primerljivo, a izrecneje strukturirano navodilo
`nvila-description-v2.txt`; prvi novi pilot je vrnil prosti tekst in je ohranjen
v `diagnostic-nvila.jsonl`. NVILA je FP16, SDPA, T=0.3, ista semena,
4096 novih tokenov, repetition penalty=1.05; presence penalty ni podprt.
Zaradi različnih promptov/formatov/preprocesiranja razlik ne pripisujemo samo modelu.
Pri NVILA je inicializacija dodatnih tokenov vedno seed42; le generiranje uporablja
42/43/44. Med revizijo odkriti neprimerljivi r2 (in prekinjeno nalaganje r3) sta
ohranjena v diagnostiki, r2/r3 sta izvedena s pravilno ločenima semenoma. R1 je že
uporabil init seed42. Popravljena nalaganja hranijo SHA256 zadnjih 64 vrstic vhodnih
in izhodnih embeddingov; `nvila_initialization_correction` dokumentira popravek.
Pri ponovnem r3 je proces po ogrevanju odpovedal s SIGSEGV. Napaka in neznani čas
nedokončanega klica sta zabeležena; nadaljevanje ni popravilo ali ponovilo shranjenega odgovora.
Vzrok sesutja ni potrjen. Native navodilo se po VILA API pošlje v uporabniškem
sporočilu; dodatna polja `native_*` ohranijo dejanski chat format in njegov privzeti
sistem, raw komponenti `system_prompt/user_prompt` pa ostaneta nespremenjeni.

Florence je izvirni `microsoft/Florence-2-large`, FP16, beam=3,
do_sample=false, max_new_tokens=1024. T=0.3 zanj ni uporabljen in seme pri tem
režimu ni vir vzorčenja; tri dejanske generacije so zahtevane tudi ob enakem odgovoru.
Prejme samo native naloge caption/OD/dense, ne Qwenove JSON sheme.
Ima **{sum(r['raw_equal'] for r in flo)}/{len(flo)}** parov z dobesedno enakim odgovorom.

Drugi korak prejme samo **isti zamrznjeni Qwen BF16 r1 opis**, brez slike in referenc.
Datoteke `frozen-inputs/` so objavljene bajt za bajtom, s preverljivim SHA256.
BF16 in Q4, kratki in dolgi prompt brez primerov ter semena 42/43/44 uporabljajo
isti vhod. Napake in preskoki ostanejo v imenovalcih; ne nadomeščamo jih z drugim virom.

{chr(10).join(grouped)}

To so razlike med napovedmi po istem tipiziranem ID, ne accuracy stabilnosti.
Referenc stabilnosti ni. Imena primerjamo po malih črkah in presledkih; sinonimov,
množin ali skupinskih regij ne enačimo samodejno. Spremembe lastnosti so spremembe
celotnih dobesednih zapisov v skupini istega imena, ne potrjeno iste fizične instance.
Odnosi v besedilu ostanejo v opazovanju; odsotnost strukturiranega polja ni odsotnost odnosa.
Florence okvirji imajo enolično IoU ujemanje. Reference in že določen slovar se uporabljajo
samo pri vrednotenju: `reference-matching.json` hrani vse pare, `reference-summary.*` agregate.
Pokritost anotiranih regij ni celoten precision/recall/mAP; neujemanje ni nujno halucinacija.

GPU je med meritvami v souporabi. Časi vključujejo dejanski obseg posameznega izvajalnika;
Florence combined meri vse tri naloge skupaj brez izrisa. CUDA allocated/reserved in
Ollama size_vram sta različni meritvi. Iz teh časov ne sklepamo o čistem vplivu arhitekture.

## Ponovitev in podatki

Uporabljena so obstoječa Docker okolja in lokalni modeli, brez ponovne namestitve:

```bash
bash scripts/thud_ponovitve.sh prepare
bash scripts/thud_ponovitve.sh all
# Ločeno: pilot/describe --model florence|nvila|qwen35-bf16|qwen35-q4|gemma3-q4
# Describe omogoča tudi --repeat r1|r2|r3; pilot se ohrani kot r1.
bash scripts/thud_ponovitve.sh freeze
bash scripts/thud_ponovitve.sh reason
bash scripts/thud_ponovitve.sh export
bash scripts/thud_ponovitve.sh verify
python3 -m unittest tests.test_separate_stages tests.test_thud_extended tests.test_thud_repetitions tests.test_stage_scripts tests.test_thud_selections -v
```

Zaganjalnik nadaljuje samo manjkajoče klice, ohrani napake in zavrne spremenjene
konfiguracije ali vhode. Vsako novo nalaganje fizično ogreje. GPU modeli tečejo zaporedno.
Veliki lokalni izhodi so `results/ponovitve/`; stare meritve ostanejo v svojih mapah.
Konfiguracije/prompti so `configs/ponovitve/`; točne poslane prompte hrani tudi `prompts/`.

Za **vseh 20** so objavljeni `stage1-responses.jsonl`, `stage2-responses.jsonl`,
zamrznjeni vhodi, manifest, summary in consistency. Deset map `cases/` vsebuje original,
Florence izrise z ID/oznakami za OD in dense vsake ponovitve, ločen referenčni izris,
celotne odgovore/podatke obeh korakov in ročno opombo. Drugi VLM nimajo lastnih okvirjev;
prikazujejo se original in oštevilčeni dobesedni seznami. Dodatno grounding ni izveden.

Iz kompaktnih Git kopij so odstranjena transportna slikovna polja/base64, gostiteljski
argumenti diagnostičnih Docker ukazov (ohranjen SHA256 izvirnika) in Python
tracebacki, laboratorijske poti v strukturnih poljih so relativne. **Besedilo odgovora,
promptov, opisov, objektov in utemeljitev ni skrajšano.** Originalni lokalni zapisi ostanejo
nespremenjeni; `original_record_sha256` jih poveže s kopijami. Nobene uteži ali vendor
knjižnice niso del tega paketa. `verification.json` dokumentira končne preverbe.
'''
    (target/'README.md').write_text(text)


def repetitions_verify(require_review=True):
    """Check the complete exported experiment, local originals and relative HTML assets."""
    from html.parser import HTMLParser
    base=ROOT/'results/ponovitve';target=ROOT/'reports/PRIMERJAVA_PONOVITEV'
    manifest=read_json(target/'manifest.json')
    if manifest['missing_stage1'] or manifest['missing_stage2']: raise ValueError('Missing required positions')
    if len(manifest['frames'])!=20: raise ValueError('Expected same20')
    tails=[e['embedding_tail_sha256'] for e in manifest['native_environments']
           if e['model']=='nvila' and e.get('embedding_tail_sha256')]
    if len(tails)<2 or len({json.dumps(t,sort_keys=True) for t in tails})!=1:
        raise ValueError('Corrected native initialization fingerprints differ')
    for frame in manifest['frames']:
        if sha256((ROOT/frame['image_path']).read_bytes())!=frame['image_sha256']: raise ValueError('RGB changed')
    records=[json.loads(line) for name in ('stage1','stage2') for line in (target/(name+'-responses.jsonl')).read_text().splitlines()]
    for row in records:
        seed=row.get('seed',row.get('variant',{}).get('seed'))
        if seed!={'r1':42,'r2':43,'r3':44}[row['repeat']]: raise ValueError('Wrong repeat sampling seed')
        original=read_json(ROOT/row['local_record'])
        if original.get('content')!=row.get('content'): raise ValueError('Response text shortened')
        for key in ('system_prompt','user_prompt'):
            if original.get(key)!=row.get(key): raise ValueError('Prompt changed in export')
        if sha256((ROOT/row['local_record']).read_bytes())!=row['original_record_sha256']: raise ValueError('Raw original changed')
        if str(ROOT) in json.dumps(row,ensure_ascii=False): raise ValueError('Absolute laboratory path in public record')
        if row.get('stage')==2:
            source=Path(row['caption_source']).name
            if sha256((target/'frozen-inputs'/source).read_bytes())!=row['caption_sha256']: raise ValueError('Public frozen SHA differs')
    cases=read_json(target/'case-selection.json')
    if len(cases)!=10 or sorted(Counter(c['dataset'] for c in cases).values())!=[5,5]: raise ValueError('Case selection must be five plus five')
    notes=read_json(target/'manual-review.json') if (target/'manual-review.json').exists() else {}
    if require_review and any(notes.get(c['frame_id'],{}).get('review_status')!='reviewed' for c in cases): raise ValueError('Ten detailed reviews not complete')
    links=[]
    class Assets(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href') and value and not value.startswith('#'): links.append(value)
    Assets().feed((target/'index.html').read_text())
    for link in links:
        if link.startswith(('/','http:','https:','data:')) or not (target/link).is_file(): raise ValueError('Nonportable/missing asset '+link)
    for directory in (target/'cases').iterdir():
        rows=read_json(directory/'stage1.json')
        if len(rows)!=21 or len(read_json(directory/'stage2.json'))!=12: raise ValueError('Case is incomplete')
        frame=next(f for f in manifest['frames'] if f['frame_id']==rows[0]['frame_id'])
        if sha256((directory/'original.png').read_bytes())!=frame['image_sha256']: raise ValueError('Case original changed')
    old=read_json(base/'previous-results-index.json')
    if any(sha256((ROOT/p).read_bytes())!=digest for p,digest in old.items()): raise ValueError('Old result modified')
    result={'passed':True,'stage1_positions':420,'stage2_positions':240,'same20':True,'cases':10,'case_split':[5,5],
            'all_case_positions_present':True,'unshortened_answers_and_prompts_verified':len(records),
            'relative_html_assets_checked':len(links),'published_frozen_sha_verified':True,
            'repeat_seeds_verified':True,'corrected_initial_embedding_fingerprints_equal':True,
            'previous_originals_unchanged':len(old),'reviews_complete':require_review,
            'note':'Format errors are measured outcomes, not failed experiment invariants.'}
    write_json(target/'verification.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    main()
