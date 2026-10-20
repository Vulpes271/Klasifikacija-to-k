"""Generate compact Slovenian LaTeX, tables, plots and review HTML from saved runs.

No results are typed into the report. Raw failed responses remain local. The
20-image review sheet is explicitly unfinished until a human supplies judgments.
Run with python3 -m llm_object_classifier.thud_report, then compile_thud_report.sh.
"""
from __future__ import annotations
import base64
import csv
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import shutil
import os
import statistics
import difflib
from .experiment import read_json, write_json, write_csv, sha256, strict_json
from .image_dataset import load_images
from .thud_experiment import OUT, REPORT, ROOT, DATASETS, MODELS
from .thud_metrics import aggregate, name_overlap


def tex(value):
    table={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_', '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    normalized=str(value).replace('\u2011','-').replace('\u202f',' ').replace('\u2212','-')
    return ''.join(table.get(c,c) for c in normalized)


def number(value, digits=1):
    return '--' if value is None else f'{value:.{digits}f}'


def table(headers, rows, widths=None):
    cols=widths or ('l'+'r'*(len(headers)-1))
    return '\\begin{center}\\small\\begin{tabular}{'+cols+'}\\toprule\n'+ ' & '.join(headers)+r' \\ \midrule'+'\n'+ '\n'.join(' & '.join(str(v) for v in row)+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\\end{center}\n'


def collect():
    jobs={}
    for path in sorted(OUT.iterdir()):
        if not path.is_dir() or not path.name.startswith(('baseline-','extra-','reason-','pilot-')):
            continue
        if '-attempt' in path.name:
            continue
        actual=Path(read_json(path/'resumed-attempt.json')['path']) if (path/'resumed-attempt.json').exists() else path
        records=[read_json(p) for p in sorted(actual.glob('*/t*/frame-*.json'))]
        if not records:
            continue
        warmups=[read_json(p) for p in actual.glob('*/warmup.json')]
        jobs[path.name]={'warmups':warmups,'records':records,'aggregate':aggregate(records),'complete':(path/'complete.json').exists(),
                         'end_to_end_seconds':read_json(path/'complete.json').get('end_to_end_seconds') if (path/'complete.json').exists() else None}
    return jobs


def baseline(jobs,ds,model,mode):
    return jobs.get(f'baseline-{model}-{mode}-{ds}',{}).get('records',[])


def matched_row(rows, digest):
    return next((r for r in rows if r.get('image_sha256')==digest),None)


def review_svg(frame):
    """Single embedded photograph with optional original reference boxes."""
    w,h=frame['image_size']
    source='data:'+frame['mime']+';base64,'+base64.b64encode(frame['image']).decode()
    elements=[f'<svg viewBox="0 0 {w} {h}" style="max-width:900px;width:100%"><image href="{source}" width="{w}" height="{h}"/><g class="regions" style="display:none">']
    for obj in frame['regions']:
        x1,y1,x2,y2=obj['bbox_xyxy']
        label=html.escape(str(obj['id'])+': '+obj['label'])
        elements.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#ffca28" stroke-width="2"/><text x="{x1+2}" y="{max(y1+15,15)}" font-size="14" fill="#ffca28" stroke="#111" stroke-width="3" paint-order="stroke">{label}</text>')
    elements.append('</g></svg>')
    toggle="<label><input type=checkbox onchange=\"this.closest('section').querySelector('.regions').style.display=this.checked?'block':'none'\"> Prikaži referenčne okvirje (skupinske oznake niso instance)</label><br>"
    return toggle+''.join(elements)


def create_review(jobs, frames, audit):
    rows=[]
    name_rows=[]
    doc=['<!doctype html><meta charset="utf-8"><title>THUD ročni pregled — OSNUTEK</title><style>body{font:15px sans-serif;max-width:1200px;margin:auto}img{max-width:700px}pre{white-space:pre-wrap}details{border:1px solid #bbb;padding:8px}section{margin:30px 0}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:5px}</style><h1>THUD: ročni pregled 20 istih slik</h1><p>OSNUTEK. Polja v manual-review.csv so prazna, dokler niso dejansko preverjena. Oznake so nepopolne in lahko skupinske. Prekrivanje imen ni pravilnost. ID velja znotraj slike.</p>']
    for ds in DATASETS:
        for i in audit[ds]['subset_indices_zero_based']:
            f=frames[ds][i]
            fid=f['record']['frame_id']
            doc.append(f'<section><h2>{html.escape(ds)} / {html.escape(fid)}</h2>{review_svg(f)}<details><summary>Referenčne regije in okvirji</summary><pre>{html.escape(json.dumps(f["regions"],indent=2))}</pre></details>')
            for mode in ('discovery','supplied'):
                for spec in MODELS:
                    r=matched_row(baseline(jobs,ds,spec['name'],mode),f['image_sha256'])
                    if r is None:
                        continue
                    row={'dataset':ds,'source_frame_id':fid,'model':spec['name'],'mode':mode,
                         'image_sha256':f['image_sha256'],'image_path':f['path'],'status':r['status'],
                         'review_status':'pending','confirmed_hits':'','missed':'','wrong_or_hallucinated':'',
                         'locations_correct':'','relations_correct':'','visible_evidence_correct':'','reviewer_notes':''}
                    rows.append(row)
                    doc.append(f'<details><summary>{mode} / {spec["name"]} / {r["status"]}</summary><pre>{html.escape(json.dumps(r.get("parsed",{"error":r.get("error"),"content":r.get("content")}),indent=2,ensure_ascii=False))}</pre></details>')
                    if mode=='discovery':
                        labels=[o['label'] for o in r.get('parsed',{}).get('objects',[])]
                        name_rows.append({'dataset':ds,'frame_id':fid,'model':spec['name'],'status':r['status'],
                                          **name_overlap(labels,[o['label'] for o in f['regions']])})
            neutral_id=audit[ds]['frames'][i]['discovery_id']
            for name,job in jobs.items():
                if not name.startswith(('extra-','reason-')) or not name.endswith(ds):
                    continue
                record=next((r for r in job['records'] if r['frame_id']==neutral_id),None)
                if record:
                    doc.append(f'<details><summary>Dodatni test: {html.escape(name)} / {record["status"]}</summary><pre>{html.escape(json.dumps(record.get("parsed",{"error":record.get("error"),"content":record.get("content")}),indent=2,ensure_ascii=False))}</pre></details>')
            doc.append('</section>')
    (OUT/'manual-review.html').write_text('\n'.join(doc))
    # Merge review fields by immutable image/model/mode key; never lose human edits.
    path=REPORT/'manual-review.csv'
    key=lambda r:(r.get('image_sha256'),r.get('model'),r.get('mode'))
    manual_fields=['review_status','confirmed_hits','missed','wrong_or_hallucinated',
                   'locations_correct','relations_correct','visible_evidence_correct','reviewer_notes']
    old={}
    if path.exists():
        with path.open(newline='') as stream:
            old={key(r):r for r in csv.DictReader(stream)}
    for row in rows:
        if key(row) in old:
            for field in manual_fields:
                row[field]=old[key(row)].get(field,row[field])
    write_csv(path,rows,list(rows[0]) if rows else ['review_status'])
    write_json(REPORT/'name-overlap.json',name_rows)
    write_csv(REPORT/'name-overlap.csv',name_rows,list(name_rows[0]) if name_rows else ['status'])


def save_run_summary(jobs):
    """Regenerate the compact total, keeping pilot and warmups separate."""
    groups={}
    for prefix,label in [('pilot-','pilot'),('baseline-','baseline'),('extra-','additional_descriptions'),('reason-','reasoning')]:
        groups[label]=aggregate([r for name,job in jobs.items() if name.startswith(prefix) for r in job['records']])
    warmups=[r for job in jobs.values() for r in job['warmups']]
    write_json(REPORT/'run-summary.json',{'groups':groups,
        'measured_gpu_calls':sum(g['attempted'] for g in groups.values()),
        'verified_measured_gpu_calls':sum(g['gpu_verified_calls'] for g in groups.values()),
        'warmup_calls_excluded':len(warmups),'verified_warmup_calls':sum(r.get('gpu_verified',False) for r in warmups),
        'raw_outputs':str(OUT),'initial_cpu_diagnostic':str(OUT/'diagnostic-cpu-initial-pilot'),
        'second_part_executed':False})


def save_rejected_candidates():
    """Keep parsed rejected JSON separately; never turn a failure into a caption."""
    rejected=[]
    for source in sorted(OUT.glob('*/*/t*/frame-*.json')):
        if not source.relative_to(OUT).parts[0].startswith(('baseline-','extra-','reason-','pilot-')):
            continue
        row=read_json(source)
        if row.get('status')!='error':
            continue
        candidate=None
        parse_error=None
        try:
            candidate=strict_json(row.get('content',''))
        except (TypeError,ValueError) as exc:
            parse_error=str(exc)
        rejected.append({'source':str(source),'source_sha256':sha256(source.read_bytes()),
                         'status':'rejected','schema_error':row.get('error'),
                         'parsed_candidate':candidate,'parse_error':parse_error})
    (OUT/'rejected-candidates.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rejected))
    write_json(REPORT/'rejected-index.json',[{k:v for k,v in row.items() if k!='parsed_candidate'} for row in rejected])


def reasoning_comparisons(jobs):
    comparisons=[]
    for ds in DATASETS:
        reference=jobs.get(f'reason-long-no-examples-t0.3-s42-{ds}',{}).get('records',[])
        ref={r['caption_sha256']:r for r in reference}
        for job,data in jobs.items():
            if not job.startswith('reason-') or not job.endswith(ds):continue
            for row in data['records']:
                other=ref.get(row['caption_sha256'])
                if row.get('status')!='ok' or not other or other.get('status')!='ok':continue
                by_id={(type(o['id']),o['id']):o for o in other['parsed']['objects']}
                caption=read_json(Path(row['caption_source']))
                labels={(type(o['id']),o['id']):o['label'] for o in caption['parsed']['objects']}
                for obj in row['parsed']['objects']:
                    key=(type(obj['id']),obj['id'])
                    base=by_id[key]
                    comparisons.append({'job':job,'frame_id':row['frame_id'],'caption_sha256':row['caption_sha256'],
                                        'id':obj['id'],'label':labels[key], 'classification':obj['classification'],
                                        'baseline_classification':base['classification'],'changed':obj['classification']!=base['classification'],
                                        'reason':obj['reason'],'baseline_reason':base['reason']})
    write_json(REPORT/'reasoning-comparisons.json',comparisons)
    write_csv(REPORT/'reasoning-comparisons.csv',comparisons,list(comparisons[0]) if comparisons else ['job'])
    return comparisons


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'figures').mkdir(exist_ok=True)
    from .thud_verify import verify
    verification=verify()
    jobs=collect()
    save_run_summary(jobs)
    save_rejected_candidates()
    audit=read_json(REPORT/'data-audit.json')
    frames={ds:load_images(ROOT/'data'/ds/'objects.jsonl',50) for ds in DATASETS}
    stats=[{'job':name,'complete':j['complete'],'end_to_end_seconds':j['end_to_end_seconds'],
            'warmup_calls':len(j['warmups']),'warmup_seconds':sum(r.get('elapsed_seconds',0) for r in j['warmups']),
            'warmup_load_seconds':sum(r.get('response',{}).get('load_duration',0)/1e9 for r in j['warmups']),**j['aggregate']} for name,j in jobs.items()]
    write_json(REPORT/'aggregates.json',stats)
    write_csv(REPORT/'aggregates.csv',stats,list(stats[0]) if stats else ['job'])
    create_review(jobs,frames,audit)
    comparisons=reasoning_comparisons(jobs)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/thud-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[]; means=[]
    for mode in ('discovery','supplied'):
        for spec in MODELS:
            rows=[r for ds in DATASETS for r in baseline(jobs,ds,spec['name'],mode)]
            a=aggregate(rows)
            labels.append(spec['name']+' / '+mode); means.append(a['mean_s'] or 0)
    fig,ax=plt.subplots(figsize=(8,2.7));ax.barh(labels,means,color=['#235c87']*3+['#b2682b']*3)
    ax.set_xlabel('Povprečni čas merjenega klica [s]');ax.invert_yaxis();fig.tight_layout()
    fig.savefig(REPORT/'figures/latency.png',dpi=180);plt.close(fig)
    baseline_rows=[r for name,j in jobs.items() if name.startswith('baseline-') for r in j['records']]
    extra_rows=[r for name,j in jobs.items() if name.startswith('extra-') for r in j['records']]
    reason_rows=[r for name,j in jobs.items() if name.startswith('reason-') and not name.startswith('reason-q4-') for r in j['records']]
    q4_reason_rows=[r for name,j in jobs.items() if name.startswith('reason-q4-') for r in j['records']]
    environment=read_json(REPORT/'environment.json')
    tests=read_json(REPORT/'tests.json') if (REPORT/'tests.json').exists() else {}
    validation_sentence=f'Preverjanje kode: {tests.get("tests","--")} testov ({tests.get("result","še ni izvedeno")}); revizija {verification["complete_jobs"]}/{verification["expected_jobs"]} skupin, {len(verification["invariant_errors"])} kršitev nespremenljivih vhodov/GPU. Podrobnosti: tests.json in verification.json.'
    snapshots=sorted((OUT/'source-snapshots').glob('all-*.json'))
    commit=read_json(snapshots[0])['git_commit'] if snapshots else environment['git_commit']['stdout'].strip()
    total=aggregate(baseline_rows)
    pages=[]
    model_table=[]
    model_metadata={}
    for spec in MODELS:
        manifests=list(OUT.glob(f'baseline-{spec["name"]}-*/manifest.json')) or list(OUT.glob(f'pilot-{spec["name"]}-*/manifest.json'))
        info=read_json(manifests[0])['model_metadata'].get(spec['name'],{}) if manifests else {}
        model_metadata[spec['name']]=info
        projector=info.get('show',{}).get('projector_info',{}).get('general.file_type')
        precision={1:'F16',32:'BF16',15:r'Q8\_0'}.get(projector,'ni poročano' if projector is None else str(projector))
        rows=[r for name,j in jobs.items() if name.startswith('baseline-'+spec['name']+'-') for r in j['records']]
        a=aggregate(rows)
        model_table.append([tex(spec['model']),tex(spec['precision']),precision,number(a['max_vram_bytes']/2**30)])
    write_json(REPORT/'model-metadata.json',model_metadata)
    pages.append(r'''\begin{center}{\LARGE THUD: opis slike in stabilnost objektov}\\[5pt]
{\large Vmesno poročilo za predstavitev 21. 10. 2026}\\[4pt]
\textbf{Osnutek za poznejši ročni pregled in dopolnitev}\end{center}
\section*{Cilj in postopek}
Robot za dolgoročno karto potrebuje opis vidnih objektov in njihovih odnosov ter ločeno presojo, ali se lahko položaj spremeni med obiski. Trenutna negibnost na fotografiji ni dokaz trajnosti.
\begin{center}\fbox{izbor slik $\rightarrow$ opis/detekcija $\rightarrow$ besedilna presoja $\rightarrow$ vrednotenje $\rightarrow$ poročilo}\end{center}
Prvi korak uporablja sliko, drugi samo zamrznjeni JSON opis. Razredi so \texttt{fixed}, \texttt{movable}, \texttt{uncertain}; izhod vsebuje kratko besedno utemeljitev, ne skritega notranjega razmišljanja.
\section*{Podatki in dve ločeni nalogi}
'''+table(['Nabor','Slike','Regije'],[['Trgovina (dense)',50,678],['agtea',50,1021]])+r'''
Poti, mere in SHA256 vseh slik so preverjeni. To so povezani pogledi istih prostorov, ne 100 neodvisnih prizorov. Oznake vsebujejo vrste in okvirje, ne razredov stabilnosti. \texttt{Products} je lahko skupina; ID velja samo znotraj slike.
\begin{itemize}\item \textbf{Samostojno zaznavanje (discovery):} ločen manifest vsebuje le nevtralni ID slike, slikovno pot, mere in SHA256. Model ne prejme referenčnih imen, regijskih ID, okvirjev ali anotacijskih poti.
\item \textbf{Opisovanje danih regij (supplied):} model prejme anotirane regije, vrne njihove iste ID in imena. Kopirano ime ni detekcijski zadetek.\end{itemize}
\section*{Modeli in nastavitve}
'''+table(['Točen model','Uteži','Projektor','VRAM [GiB]'],model_table, 'lllr')+r'''
Točni tagi: \texttt{qwen3.5:4b-bf16}, \texttt{qwen3.5:4b-q4\_K\_M}, \texttt{gemma3:4b-it-q4\_K\_M}. Tagi in slikovna podpora so preverjeni v uradnem katalogu ter z \texttt{/api/show}. Tudi Qwen slikovna projektorja imata različni natančnosti (F16/BF16). BF16 metapodatki navajajo 33 blokov in nextn\_predict\_layers=1, Q4 pa 32 blokov. Zato to ni strogo izoliran poseg samo v natančnost istih pretvorjenih uteži; vpliv dodatne napovedne plasti ni posebej preverjen. Družin pri različnih natančnostih ne primerjamo kot čiste arhitekturne razlike.
VRAM je največja dodelitev modelu v shranjenih merjenih odgovorih API, ne celotna zasedenost kartice. Projektor: GGUF file type iz metapodatkov, F16=1 in BF16=32. Osnova: podroben prompt brez vsebinskih primerov, $T=0$, seed 42, kontekst 16384, izhod do 8192 tokenov, top-p 0.8, top-k 20, presence penalty 1.5. Enaki izvorni bajti slike; notranje predprocesiranje modelov se lahko razlikuje.
''')
    timing=[]
    base_stats=[s for s in stats if s['job'].startswith('baseline-')]
    warm_seconds=sum(s['warmup_seconds'] for s in base_stats)
    warm_load_seconds=sum(s['warmup_load_seconds'] for s in base_stats)
    measured_load_seconds=sum(s['load_seconds'] for s in base_stats)
    end_seconds=sum(s['end_to_end_seconds'] or 0 for s in base_stats)
    for ds in DATASETS:
        for mode in ('discovery','supplied'):
            for spec in MODELS:
                a=aggregate(baseline(jobs,ds,spec['name'],mode))
                timing.append(['trgovina' if ds==DATASETS[0] else 'agtea','samost.' if mode=='discovery' else 'regije',tex(spec['name']),f'{a["ok"]}/{a["attempted"]}',number(a['median_s']),number(a['p90_s']),number(a['mean_s']),number(a['mean_output_tokens'],0),number(a['generation_tokens_s'])])
    q4_note=f' Dodatna Q4 ponovitev: {len(q4_reason_rows)}/20.' if q4_reason_rows else ''
    pilot=read_json(REPORT/'pilot-estimate.json') if (REPORT/'pilot-estimate.json').exists() else {}
    pages.append(r'\section*{Dejanski obseg in hitrost}'+f'Osnovna primerjava: {len(baseline_rows)}/600 merjenih odgovorov, {total["ok"]} odgovorov ustreza shemi, {total["errors"]} napak; skladen zapis JSON ima {total["json_parse_valid_calls"]} odgovorov. Dodatni prvi korak: {len(extra_rows)}/80; drugi korak: {len(reason_rows)}/120.{q4_note} Ogrevalni klici so shranjeni ločeno in izključeni iz teh imenovalcev.\n'+table(['Nabor','Način','Model','OK/vsi','Med.','p90','Povp.','Izh. tok.','tok/s'],timing,'lllrrrrrr')+r'''
\begin{center}\includegraphics[width=.94\linewidth]{figures/latency.png}\end{center}
Izh. tok. je povprečno število izhodnih tokenov. Časi so sekunde na sliko/klic; p90 uporablja linearno interpolacijo. Čas vključuje vse odgovore z izmerjenim časom, tudi zavrnjeni JSON. Napake prenosa lahko nimajo časa; število takih klicev je v agregatih. Tok/s je vsota izhodnih tokenov deljena z vsoto API časa generiranja. Nalaganje in tokeni so ločeno ohranjeni; end-to-end čas je čas celotne skupine, vključno z ogrevanjem, preverjanjem in zapisovanjem.
'''+f'Osnovni del: skupni end-to-end čas {number(end_seconds/60)} min; ogrevalni klici {number(warm_seconds)} s, od tega API nalaganje {number(warm_load_seconds)} s; API nalaganje v merjenih klicih {number(measured_load_seconds)} s. Nalaganje je podmnožica časa zahteve, zato teh časov ne seštevamo.\n'+f'Pilot: {pilot.get("pilot_calls","--")} merjenih klicev; linearna ocena osnovnega dela pred polnim zagonom je bila {number((pilot.get("estimated_baseline_seconds") or 0)/3600)} h. Ocena ni meritev celotnega izvajanja.\n'+r'''
RTX 5090 (32607 MiB), Docker Compose in projektna Ollama. Začetni stari kontejner je izgubil GPU: CPU diagnostika je ohranjena ločeno, kontejner je bil ponovno ustvarjen z GPU override. Veljavni klici preverijo VRAM; delni CPU offload se zavrne. Drug uporabnikov proces je hkrati obremenjeval GPU; ni bil ustavljen. To omejuje primerjavo hitrosti. Različno dolgi odgovori in API predpomnilnik predpon/slikovnega kodiranja dodatno vplivajo na čase; svež HTTP klic ne zagotavlja hladnega predpomnilnika.
''')
    extra_table=[]
    subset_base=[]
    for ds in DATASETS:
        allowed={frames[ds][i]['image_sha256'] for i in audit[ds]['subset_indices_zero_based']}
        subset_base += [r for r in baseline(jobs,ds,'qwen35-bf16','discovery') if r['image_sha256'] in allowed]
    groups=[('podroben, T=0, s42',subset_base)]
    for key,label in [('short-t0','kratek, T=0, s42'),('long-t03','podroben, T=.3, s42'),('long-t07','podroben, T=.7, s42'),('long-t03-s43','podroben, T=.3, s43')]:
        groups.append((label,[r for ds in DATASETS for r in jobs.get(f'extra-{key}-{ds}',{}).get('records',[])]))
    for label,rs in groups:
        a=aggregate(rs);extra_table.append([label,f'{a["ok"]}/{a["attempted"]}',number(a['median_s']),a['objects']])
    reasoning_table=[]
    reason_specs=[('long-examples',.3,42),('long-no-examples',.3,42),('short-no-examples',.3,42),('long-no-examples',0,42),('long-no-examples',.7,42),('long-no-examples',.3,43)]
    for prompt,t,seed in reason_specs:
        names=[f'reason-{prompt}-t{t}-s{seed}-{ds}' for ds in DATASETS]
        rs=[r for name in names for r in jobs.get(name,{}).get('records',[])]
        a=aggregate(rs);cl=Counter(o['classification'] for r in rs if r['status']=='ok' for o in r['parsed']['objects'])
        cmp=[c for c in comparisons if c['job'] in names]
        changed=sum(c['changed'] for c in cmp)
        reasoning_table.append([tex(prompt)+f' / {t} / {seed}',f'{a["ok"]}/{a["attempted"]}',cl['fixed'],cl['movable'],cl['uncertain'],f'{changed}/{len(cmp)}'])
    if q4_reason_rows:
        a=aggregate(q4_reason_rows)
        cl=Counter(o['classification'] for r in q4_reason_rows if r['status']=='ok' for o in r['parsed']['objects'])
        cmp=[c for c in comparisons if c['job'].startswith('reason-q4-')]
        reasoning_table.append(['Q4 / long-no-examples / .3 / 42',f'{a["ok"]}/{a["attempted"]}',cl['fixed'],cl['movable'],cl['uncertain'],f'{sum(c["changed"] for c in cmp)}/{len(cmp)}'])
    prompt_success={}
    for prompt in ('long-examples','long-no-examples','short-no-examples'):
        rs=[r for ds in DATASETS for r in jobs.get(f'reason-{prompt}-t0.3-s42-{ds}',{}).get('records',[])]
        prompt_success[prompt]=aggregate(rs)
    findings='Pri T=.3, seed 42: '+', '.join(f'{name} {a["ok"]}/{a["attempted"]}' for name,a in prompt_success.items())+'. To meri izpolnitev sheme, ne pravilnosti odločitev.'
    candidate='kratki prompt brez primerov' if prompt_success['short-no-examples']['ok']>=prompt_success['long-examples']['ok'] else 'dolgi prompt s primeri'
    reason_errors=[r.get('error','') for r in reason_rows+q4_reason_rows if r['status']=='error']
    other_reason_errors=len(reason_errors)-sum('missing reason' in e or 'Duplicate JSON key' in e for e in reason_errors)
    error_findings=f'V besedilnem delu je {sum("missing reason" in e for e in reason_errors)} odgovorov brez obvezne utemeljitve in {sum("Duplicate JSON key" in e for e in reason_errors)} s podvojenim ključem ter {other_reason_errors} z drugimi kršitvami sheme. Vsi ostanejo zavrnjeni s shranjenim surovim odgovorom; podvojenega ključa ne razrešimo z izbiro zadnje vrednosti.'
    pages.append(r'''\section*{Vpliv prompta, temperature in semena}
Vnaprej je izbranih 10 enakomerno razpršenih slik iz vsakega nabora (indeksi v \texttt{data-audit.json}). Vsi dodatni testi uporabijo isti izbor; dodatni opisovalni testi merijo samostojno zaznavanje. Kratek in podroben opisovalni prompt imata isto izhodno shemo in obvezne podatke, brez vsebinskih primerov. Osnovni podrobni T=0 se ponovno uporabi na izbranih slikah.
'''+table(['Opisovalnik BF16','OK/vsi','Med. [s]','Objekti'],extra_table,'lrrr')+r'''
Število objektov je opis izhoda, ne merilo pravilnosti. Temperatura in seme se spreminjata ločeno; ni polnega kartezičnega produkta. Pri T=0 seme nima zagotovljenega vpliva.
\subsection*{Besedilna presoja istih zamrznjenih opisov}
Vhodi so iz osnovnega samostojnega Qwen BF16. SHA256 datotek se zapiše za vsak klic; drugi korak ne odpre slike. Dolga prompta se razlikujeta samo po odstranitvi dveh blokov vsebinskih primerov. Definicije in shema ostanejo iste. Kratka različica ohrani isti pomen. Datoteke in razlike so shranjene v Git.
'''+table(['Prompt / T / seed','OK/vsi','F','M','U','Spremembe'],reasoning_table,'lrrrrr')+tex(findings)+r'''
Vrstice so BF16, dodatna vrstica Q4 je ponovitev istega prompta in opisov. F/M/U so števila odločitev fixed/movable/uncertain v veljavnih odgovorih. Spremembe so primerjave istega tipiziranega ID na istem SHA256 opisa proti dolgemu promptu brez primerov pri T=.3, seed 42. Imenovalec vsebuje samo pare veljavnih odgovorov; neuspešni klici so posebej v OK/vsi. Strinjanje ni pravilnost. Izbira dolgega prompta brez primerov za temperaturne teste je vnaprejšnja metodološka izbira zaradi ohranjenih definicij, ne dokaz boljše kakovosti.
\subsection*{Kakovost: meje samodejnih meritev}
\texttt{name-overlap.csv} poroča le presek unikatnih imen po pretvorbi v male črke in normalizaciji presledkov, brez sinonimov ali ujemanja instanc. Izpuščena drugačna imena niso avtomatično zgrešeni objekti, neoznačeni vidni objekti pa niso avtomatično halucinacije. Brez napovedanih okvirjev ni detekcijskega mAP; brez referenc stabilnosti ni accuracy. \texttt{manual-review.html} prikazuje iste slike, referenčne okvirje in vse izhode. Ročno nepreverjena polja v CSV ostanejo označena \texttt{pending}.
''')
    cases=[]
    for ds in DATASETS:
        indices=audit[ds]['subset_indices_zero_based']
        dense=max(indices,key=lambda i:len(frames[ds][i]['regions']))
        indices_cases=[indices[0], dense if dense!=indices[0] else indices[-1]]
        for i in indices_cases:
            cases.append((ds,i))
    case_meta=[]
    manual=read_json(REPORT/'manual-notes.json') if (REPORT/'manual-notes.json').exists() else {}
    for page_index in range(2):
        content=r'\section*{Dejanske slike in primerjava modelov}'
        for ds,i in cases[page_index*2:page_index*2+2]:
            f=frames[ds][i];fid=f['record']['frame_id'];key=ds+'/'+fid
            image_name=f'figures/{ds}-{fid}.png'
            shutil.copyfile(f['path'],REPORT/image_name)
            case_meta.append({'dataset':ds,'frame_id':fid,'index':i,'image':image_name,'sha256':f['image_sha256']})
            content+=r'\subsection*{'+tex(('Trgovina' if ds==DATASETS[0] else 'agtea')+' / '+fid)+r'}\begin{center}\includegraphics[height=4.1cm,width=.9\linewidth,keepaspectratio]{'+image_name+r'}\end{center}'
            entries=[]
            case_errors=[]
            for spec in MODELS:
                disc=matched_row(baseline(jobs,ds,spec['name'],'discovery'),f['image_sha256'])
                supplied=matched_row(baseline(jobs,ds,spec['name'],'supplied'),f['image_sha256'])
                for mode_name, record in [('samostojno',disc),('regije',supplied)]:
                    if record and record.get('status')=='error':
                        case_errors.append(spec['name']+' / '+mode_name+': '+record.get('error','')[:130])
                excerpt=disc.get('parsed',{}).get('scene_description',disc.get('error','')) if disc else 'Še ni izvedeno.'
                excerpt=' '.join(excerpt.split())
                excerpt=excerpt[:192]+'...' if len(excerpt)>195 else excerpt
                entries.append([tex(spec['name']),tex(f'{disc["status"]}: {len(disc.get("parsed",{}).get("objects",[]))}' if disc else '--'),tex(supplied['status'] if supplied else '--'),tex(excerpt)])
            content+=table(['Model','Detekcija','Regije','Dejanski izsek opisa (model)'],entries,r'p{2.2cm}p{1.5cm}p{1.2cm}p{10cm}')
            if case_errors:
                content+=r'\textbf{Dejanska zavrnitev:} '+tex('; '.join(case_errors))+'\n'
            content+=r'\textbf{Referenčna imena:} '+tex(', '.join(Counter(o['label'] for o in f['regions'])))+'. '
            content+=r'\textbf{Lastna presoja:} '+tex(manual.get(key,'OSNUTEK: prizor še ni ročno potrjen; tabela prikazuje modelove trditve.'))+'\n'
        pages.append(content)
    write_json(REPORT/'cases.json',case_meta)
    categories=[('Vrata',('door',)),('Slike',('picture','painting','artwork','photograph')),
                ('Police',('shelf','shelves','shelving','bookshelf')),('Pohištvo',('chair','table','desk','sofa','cabinet','furniture'))]
    example_table=[]
    for category,words in categories:
        candidates=[c for c in comparisons if any(w in c['label'].lower() for w in words)]
        if category=='Pohištvo':
            candidates=[c for c in candidates if not any(w in c['label'].lower() for w in ('people','person','seated'))]
        if category=='Vrata':
            candidates=[c for c in candidates if not any(w in c['label'].lower() for w in ('cooler','refrigerat','fridge'))]
        # Avoid reference-vs-itself; prefer a real change, then a matched agreement.
        candidates=[c for c in candidates if 'reason-long-no-examples-t0.3-s42-' not in c['job']]
        c=next((c for c in candidates if c['changed']),candidates[0] if candidates else None)
        if c is None:
            example_table.append([category,'--','Ni samostojnega predmeta te vrste v primerjanih veljavnih opisih.'])
        else:
            variant=c['job'].removeprefix('reason-').split('-thud_')[0]
            scene='trgovina' if c['job'].endswith(DATASETS[0]) else 'agtea'
            source=scene+' / '+c['frame_id']+' / ID '+str(c['id'])+' / '+variant
            label=category+': '+c['label']
            decision=tex(c['baseline_classification'])+r' $\rightarrow$ '+tex(c['classification'])
            example_table.append([tex(label),decision,tex(c['reason'])+r'\newline{\scriptsize '+tex(source)+'}'])
    pages.append(r'''\section*{Utemeljitve, napake in omejitve}
Naslednji izseki so besedni razlogi modela; niso lastna potrditev vidnih dokazov. Primeri so izbrani iz vrat, slik, polic in pohištva, kjer so te vrste dejansko vrnjene. Hladilnih vrat ne enačimo z vrati prostora. Če slik/umetniških del opisovalnik ne izloči, njihovega razvrščanja ne prikazujemo kot izmerjenega. Celotni pari z imenom poskusa in SHA256 so v \texttt{reasoning-comparisons.csv}.
'''+table(['Objekt',r'Osnova $\rightarrow$ različica','Dejanski modelov razlog'],example_table,r'p{3.3cm}p{3.1cm}p{8.7cm}')+r'''
'''+tex(error_findings)+r'''
\subsection*{NVILA 8B}
Dodatni model ni izveden. Omejena začetna diagnostika: uradna validacijska shema ne vključuje \texttt{NVILA-8B}, izolirani venv pa je odpovedal zaradi manjkajočega ensurepip. Uteži niso prenesene in slikovni API ni preverjen; 8B ni preimenovan v 15B. Celotne ovire in viri so v \texttt{nvila-diagnostic.json}.
\subsection*{Ponovljivost in izvor}
'''+tex(validation_sentence)+r' Izhodiščni Git commit: \texttt{'+tex(commit[:12])+r'}. Različice, gonilnik in slika Docker so v \texttt{environment.json}; točna natančnost, slikovni projektor in zmožnosti v manifestih posameznih poskusov. Konfiguracije vsebujejo vse nastavitve, shranjeni odgovori pa prompte, SHA256 slike/opisa, čas, možnosti, GPU stanje in surovi API odgovor.'+r'''
\begin{verbatim}
bash scripts/thud_experiment.sh --phase all
python3 -m llm_object_classifier.thud_q4_repeat
python3 -m llm_object_classifier.thud_verify --require-complete
python3 -m llm_object_classifier.thud_report
bash scripts/compile_thud_report.sh
python3 -m unittest discover -s tests -v
\end{verbatim}
Zaganjalnik uporablja projektno Ollamo (privzeti omrežni naslov lahko nastavimo z \texttt{--base-url}). Predpogoji: zdrava GPU Ollama, preneseni trije tagi, Python s Pillow. Vsaka dokončana skupina dobi oznako; nadaljevanje preverja nespremenljivost vhodov in ne prepisuje odgovorov. Prekinjeni poskusi drugega koraka ostanejo kot ločeni poskusi.
Veliki surovi izhodi: \texttt{results/thud-vmesno/}. Majhen paket PDF/LaTeX, slike, agregati, izbor, preglednica in metodologija: \texttt{reports/THUD\_VMESNO/}. Stari NYU poskusi ostanejo ločeni.
\subsection*{Naslednji utemeljeni poskus}
'''+tex(f'Za naslednji ročni pregled izberemo {candidate} kot kandidata: primerjava sheme zgoraj podpira njegov operativni preizkus, ne dokazuje pravilnosti. Preveriti je treba iste objekte in razhajanja proti dolgemu promptu s primeri, predvsem pri policah in pohištvu. Hitrost ponoviti ob izključni uporabi GPU. Šele po primerljivi ročno potrjeni kakovosti izbrati končni prompt. Nato po izrecnem naročilu začeti 2. del z bogatejšimi odnosi in vidnimi dokazi; NVILA ostaja ločena priprava.')+r'''
''')
    header=r'''\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\input{glyphtounicode}
\pdfgentounicode=1
\usepackage[slovene]{babel}
\usepackage[margin=1.5cm]{geometry}
\usepackage{graphicx,booktabs,hyperref,array}
\hypersetup{colorlinks=true,urlcolor=blue}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{2em}
\setlength{\parskip}{4pt}
\renewcommand{\arraystretch}{1.16}
\begin{document}
'''
    sources=r'''\vfill{\footnotesize Uradni viri: \href{https://ollama.com/library/qwen3.5:4b-bf16}{Qwen BF16}, \href{https://ollama.com/library/qwen3.5:4b-q4_K_M}{Qwen Q4}, \href{https://ollama.com/library/gemma3:4b-it-q4_K_M}{Gemma Q4}, \href{https://huggingface.co/Efficient-Large-Model/NVILA-8B}{NVILA uteži}, \href{https://github.com/NVlabs/VILA/blob/main/server.py}{NVILA strežnik}. Preverjeno 20. 10. 2026.}
'''
    (REPORT/'porocilo.tex').write_text(header+('\n\\newpage\n'.join(pages))+sources+'\n\\end{document}\n')
    print(f'Generated {len(pages)} intended pages; {len(baseline_rows)} baseline, {len(extra_rows)} extra, {len(reason_rows)} core reasoning responses, {len(q4_reason_rows)} optional Q4.')

if __name__=='__main__':main()
