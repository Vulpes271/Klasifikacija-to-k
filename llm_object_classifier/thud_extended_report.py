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

from .experiment import read_json, write_json, write_csv, sha256
from .thud_box_metrics import match_regions
from .thud_extended_common import ROOT, OUT, REPORT, DATASETS, frames, saved_path
from .thud_florence import TASKS
from .thud_metrics import percentile


def native_rows(model, task):
    return [read_json(p) for p in sorted((OUT/model).glob(f'*/{task}/frame-*.json'))]


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


def table(headers, rows, columns):
    return '\\begin{center}\\small\\begin{tabular}{'+columns+'}\\toprule\n'+' & '.join(headers)+r' \\ \midrule'+'\n'+ '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\\end{center}\n'


def reasoning_summary():
    summary=[]
    decisions=[]
    for model in ('nvila','florence'):
        paths=sorted((OUT/'reason'/model).glob('*/*/t*/frame-*.json'))
        records=[read_json(p) for p in paths]
        counts=Counter(o['classification'] for r in records if r['status']=='ok' for o in r['parsed']['objects'])
        summary.append({'caption_model':model,'positions':len(records),
                        **dict(Counter(r['status'] for r in records)),
                        'classifications':dict(counts)})
        for p,r in zip(paths,records):
            for o in r.get('parsed',{}).get('objects',[]):
                decisions.append({'caption_model':model,'dataset':p.parents[2].name,
                                  'frame_id':r['frame_id'],'object_id':o['id'],
                                  'classification':o['classification'],'reason':o['reason'],
                                  'source_record':str(p.relative_to(ROOT))})
    write_json(REPORT/'extended-reasoning-summary.json',summary)
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


def make_latex(rows, boxes, reason, selected):
    number=lambda x: '--' if x is None else f'{x:.2f}'
    labels={'nvila':'NVILA-8B FP16','florence':'Florence-2-large FP16',
            'qwen35-bf16':'Qwen BF16 (staro)','qwen35-q4':'Qwen Q4 (staro)','gemma3-q4':'Gemma Q4 (staro)'}
    timing=[]
    for r in rows:
        timing.append([tex(labels[r['model']]),tex(r['task'].replace(' (existing, reused)','')),
                       f"{r['ok']}/{r['positions']}" if r['task']!='combined' else str(r['positions']),
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
Razvrščanje uporabi prejšnji short-no-examples prompt, T=0,3, seed 42. Zavrnjeni opisi ostanejo preskočeni; napake razvrščanja niso popravljene. Različna granularnost regij pomeni, da števila razredov niso accuracy ali strinjanje istih instanc. Referenc stabilnosti ni; razlogi so modelove trditve, ne potrditev dejanskega pritrjevanja.
''')
    notes=read_json(REPORT/'extended-manual-notes.json') if (REPORT/'extended-manual-notes.json').exists() else {}
    cases=[f for f in selected if f['index'] in (0,5)]
    for offset in (0,2):
        page=r'\section*{Dejanski primeri: fotografija in Florence regije}'
        for frame in cases[offset:offset+2]:
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
                    text=(r.get('parsed',{}).get(TASKS['caption'],'') if model=='florence' else r.get('parsed',{}).get('scene_description','')) or r.get('error','')
                    text=' '.join(text.split())
                    snippets.append([tex(labels[model]),tex(r.get('status','')),tex(text[:210]+('...' if len(text)>210 else ''))])
            page+=table(['Model','Zapis','Dejanski izsek / napaka'],snippets,r'p{3.2cm}p{1.1cm}p{11cm}')
            page+=r'\textbf{Lastni delni vizualni pregled:} '+tex(notes.get(key,'Pregled še ni potrjen; glej lokalni HTML.'))+'\n'
        pages.append(page)
    pages.append(r'''\section*{Sklep nadaljevanja in ponovljivost}
\subsection*{Konkretne ohranjene napake in rešitve}
Florence se je po 37 shranjenih slikah končal s procesnim izhodom 132 (SIGILL), brez Python izjeme. Nadaljevanje z vključeno sledjo faulthandler je dokončalo vseh 100 slik brez ponovitve že shranjenih odgovorov. Vzrok ni potrjen; ponovnega zagona ne predstavljamo kot dokazanega popravka odvisnosti. En dense izhod agtea, indeks 24, vsebuje obrnjen okvir [0,48; 31,05; 87,84; 0,27]. Celoten izhod ostane zavrnjen; ločen rdeč diagnostični izris ohrani prvotni vrstni red koordinat. Ne obrnemo ga v veljaven okvir in ne vključimo v ujemanje.

NVILA začetno ogrevanje in merjeni pilot z repetition\_penalty=1,0 oba dosežeta 8192 tokenov s ponavljanjem JSON. Pri 1,05 je prvi merjeni odgovor kratek, vendar se zanke pri nekaterih drugih slikah še pojavijo. Pogoste so tudi dodatno besedilo za JSON, manjkajoča polja in napačna sintaksa. To so izmerjene omejitve formata modela, ne več nedelujoč GPU. Celotni odgovori, tudi zavrnitve in odmevi navodil, ostanejo shranjeni. Besedilni transport za razvrščanje ne spremeni teh rezultatov validacije.

Obe zahtevani novi okolji sta pripravljeni za dejansko slikovno inferenco na RTX 5090. Priročnost formata, uspešen GPU zagon in veljavnost izhoda so ločene ugotovitve od vizualne kakovosti. Primerjava vključuje tudi zavrnjene odgovore in razliko med splošnim opisom, detekcijo in regionalnim opisovanjem. Celoten pregled istih 20 fotografij z vsemi starimi/novimi odgovori, referencami in presojami je lokalno v manual-review.html. Preglednica 40 novih model/slika vrstic ostane pending; štirje prikazani primeri so samo delni pregled, ne izmišljena celotna referenca.

Surovi novi izhodi: \texttt{results/thud-extended/}; tam so originalni tokeni, razčlenjeni odgovori, konfiguracije, izvorne kode/SHA, pripete uteži, Docker dnevniki, diagnostika in vsi izrisi. Obstoječi odgovori so preverjeni proti indeksu SHA256 pred nadaljevanjem. Git vsebuje kodo, Dockerfile, navodila, agregate, izbrane slike, PDF in LaTeX. Točne revizije modelov in dejanski tipi parametrov so v extended-model-metadata.json in manifestih; dolgi izhodi ostanejo lokalno.
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
    combined = native_rows('florence','combined')
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
             'note':'JSON/native parse validity is not visual correctness. No stability ground truth. Large raw results remain local.'}
    write_json(REPORT/'extended-summary.json',summary)
    rejected_overlays()
    make_review(selected)
    make_latex(rows,boxes,reason,selected)
    print(json.dumps(verification,indent=2))


if __name__ == '__main__':
    main()
