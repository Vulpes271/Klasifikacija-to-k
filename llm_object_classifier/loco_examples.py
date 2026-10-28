"""Ten illustrative LOCO cases and a plain editable LaTeX report.

All thirty images remain in the shared exporters; this selection is only for
reading, never a new quality denominator or a filled-in human review.
"""
from collections import Counter
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

from .experiment import read_json, write_json
from . import thud_stage2_comparison as comparison
from . import thud_stage1_presentation as native
from .thud_extended_common import draw_boxes
from .thud_extended_report import tex, table
from .loco_experiment import ROOT, OUT, PUBLIC, VLM, LLM


def choose_cases(records, box_rows):
    scores={}
    for frame in read_json(VLM/'manifest.json')['frames']:
        fid=frame['frame_id'];rs=[r for r in box_rows if r['frame_id']==fid]
        scores[fid]={'frame':frame,'matched':sum(r['matched'] for r in rs),
            'reference_count_repeated':sum(r['reference_count'] for r in rs),
            'llm_errors':sum(r['status']=='error' for r in records if r['frame_id']==fid)}
    # Fixed visual variety; outcomes cannot silently replace the chosen images.
    selection=read_json(ROOT/'configs/loco-examples.json')
    chosen=[{**scores[r['frame_id']],'selection_reason':r['reason'],
             'selection_policy':selection['policy']} for r in selection['images']]
    assert Counter(r['frame']['frame_id'].split('-')[1] for r in chosen)=={f's{i}':2 for i in range(1,6)}
    assert len({r['frame']['frame_id'] for r in chosen})==10
    return chosen


def contrast(frame_id, frozen, records):
    candidates=[]
    for n,s in frozen.items():
        if s['frame_id']!=frame_id or s['status']!='ok':continue
        rows=[r for r in records if r['input_index']==n and r['llm_repeat']=='r1']
        for obj in s['document']['parsed']['objects']:
            key=comparison.typed(obj['id']);answers=[]
            for r in rows:
                predicted=comparison.decisions(r)
                answer=predicted.get(key) if predicted is not None else None
                answers.append({'llm':r['llm'],'prompt':r['prompt_name'],'llm_repeat':r['llm_repeat'],
                    'seed':42,'status':r['status'],'classification':answer['classification'] if answer else None,
                    'reason':answer['reason'] if answer else None,'error':r.get('error')})
            labels={r['classification'] for r in answers if r['classification']}
            candidates.append(((len(labels),sum(a['status']=='ok' for a in answers),s['vlm']=='qwen3-vl'),
                {'input_index':n,'source_vlm':s['vlm'],'vlm_repeat':s['vlm_repeat'],'frozen_sha256':s['sha256'],
                 'object_id_type':type(obj['id']).__name__,
                 'distinct_accepted_classes':len(labels),'classification_difference_observed':len(labels)>1,
                 'object_id':obj['id'],'label':obj['label'],'observation':obj['observation'],'answers':answers,
                 'selection_note':'One supplied ID selected for contrasting accepted classes; seed42 illustration only. All LLM/VLM repeats remain in the full browser. This is not a correctness judgment.'}))
    return max(candidates,key=lambda r:r[0])[1] if candidates else None


def reference_image(frame, refs, destination):
    """Separate explicitly labelled GT; this image never enters inference."""
    from PIL import Image, ImageDraw
    task='<REFERENCE_LOCO>'
    draw_boxes(ROOT/frame['image_path'],{task:{'bboxes':[o['bbox_xyxy'] for o in refs['objects']],
        'labels':[o['label']+' / ref-ID '+str(o['id']) for o in refs['objects']]}},task,destination)
    with Image.open(destination) as rgb:
        draw=ImageDraw.Draw(rgb);draw.rectangle((0,0,740,25),fill='black')
        draw.text((8,7),'REFERENCA LOCO - 5 razredov; ni vhod modelov',fill='white');rgb.save(destination)


def build():
    comparison.configure_paths(source=VLM,target=LLM,out=OUT/'llm')
    _,frozen,records=comparison.load()
    assert len(records)==4860
    box_rows=read_json(PUBLIC/'04_povzetki/box-matches.json')
    refs={r['frame_id']:r for r in map(json.loads,(ROOT/'data/loco_test30/objects.jsonl').read_text().splitlines())}
    selected=choose_cases(records,box_rows)
    notes={r['frame_id']:r for r in read_json(PUBLIC/'04_povzetki/qualitative-notes.json')['images']}
    directory=PUBLIC/'03_primeri';directory.mkdir(parents=True,exist_ok=True)
    cases=[];appendix=[];index='<h1>LOCO: deset izbranih primerov</h1><p>Po dva različna primera iz vsakega okolja. Izbor je namenjen branju; agregati uporabljajo vseh 30 slik. Strinjanje ni pravilnost.</p><nav><a href="../index.html">Celoten paket</a></nav>'
    for i,choice in enumerate(selected,1):
        frame=choice['frame'];fid=frame['frame_id'];case_dir=directory/fid;case_dir.mkdir(exist_ok=True)
        reference_image(frame,refs[fid],case_dir/'reference-LOCO.jpg')
        write_json(case_dir/'reference-LOCO.json',refs[fid])
        source=VLM/frame['published_directory']/'florence/r1/od.predictions.jpg'
        predicted=False
        if source.exists():
            import shutil
            shutil.copyfile(source,case_dir/'florence-od-r1.predictions.jpg');predicted=True
        selected_contrast=contrast(fid,frozen,records)
        assert notes[fid]['image_sha256']==frame['image_sha256']
        case={**choice,'index':i,'contrast':selected_contrast,'predicted_overlay_available':predicted,'qualitative_notes':notes[fid]}
        cases.append(case)
        body=f'<p><a href="../index.html">← Deset primerov</a></p><h1>{fid}</h1>'
        body+='<p>Izbor: '+native.escaped(choice['selection_reason'])+'. Ocena človeka ni izpolnjena.</p>'
        body+='<h2>Opombe pomočnika ob pregledu, ne ročne ocene</h2><p>'+native.escaped(notes[fid]['visible_successes'])+'</p><p>'+native.escaped(notes[fid]['limitations_and_checks'])+'</p>'
        body+='<p>'+native.escaped(notes[fid]['native_regions_inspection'])+'</p>'
        body+='<p>Referenca LOCO: pet razredov; ne popoln popis scene, brez stabilnosti.</p><img src="reference-LOCO.jpg" alt="Referenca LOCO, samo za naknadni pregled">'
        if predicted:body+='<p>Florence OD r1: samo napovedani okvirji; številke so lokalne regije, ne referenčni ID.</p><img src="florence-od-r1.predictions.jpg" alt="Florence napovedi">'
        body+='<p>NVILA / Qwen: model ni vrnil okvirjev. Florence OD/dense se lahko prekrivata.</p>'
        body+='<h2>Celotni odzivi vseh ponovitev</h2>'
        for vlm in native.MODELS:
            for repeat in ('r1','r2','r3'):
                n,s=next((n,s) for n,s in frozen.items() if s['frame_id']==fid and s['vlm']==vlm and s['vlm_repeat']==repeat)
                name=f'{fid}--{vlm}--{repeat}.html'
                body+=f'<h3>{vlm} / VLM-{repeat}</h3><p><a href="../../02_llm/inputs/{name}">Vsi trije LLM, oba prompta, vse tri LLM ponovitve, celotni razlogi in surovi odzivi</a></p>'
                leaf=VLM/frame['published_directory']/vlm/repeat
                tasks=('caption','od','dense') if vlm=='florence' else ('description',)
                for task in tasks:
                    row=read_json(leaf/(task+'.raw.json'));result=read_json(leaf/(task+'.result.json'))
                    body+='<details><summary>'+task+' · '+row['status']+' · celoten opis/seznam in raw</summary>'+comparison.pre(result['view'])+comparison.pre(row.get('content',''))+'</details>'
                    appendix.append({'frame_id':fid,'vlm':vlm,'vlm_repeat':repeat,'task':task,'status':row['status'],
                        'call_id':row['call_id'],'raw_path':str((leaf/(task+'.raw.json')).relative_to(ROOT)),
                        'content':row.get('content',''),'view':result['view'],'error':row.get('error')})
        body+='<h2>Primerjava istega ID in SHA, LLM-r1</h2>'+comparison.pre(selected_contrast)
        (case_dir/'index.html').write_text(native.document(fid,body))
        index+=f'<p><a href="{fid}/index.html">Primer {i}: {fid}</a> · {native.escaped(choice["selection_reason"])}</p>'
    (directory/'index.html').write_text(native.document('LOCO izbrani primeri',index))
    write_json(directory/'selection.json',cases);write_json(directory/'full-selected-native-responses.json',appendix)
    from .loco_report import empty_review
    empty_review(directory/'manual-review-empty.csv',[{'example':r['index'],'frame_id':r['frame']['frame_id'],
        'reviewer':'','review_description_quality':'','review_localization':'','review_stability_reasons':'','review_notes':''} for r in cases])
    latex(cases,appendix)
    root='<h1>Industrijski preizkus LOCO</h1><p>Vseh 30 slik, novi native VLM odzivi in razvrščanje zamrznjenih opisov. Brez ponovitev THUD. Brez referenc v vhodih. Skladnost ni pravilnost.</p><nav>'
    for href,label in (('01_vlm/index.html','Prvi del: vsi VLM odzivi'),('02_llm/index.html','Drugi del: vse LLM odločitve'),
        ('03_primeri/index.html','Deset primerov'),('04_povzetki/box-summary.csv','Okvirji: pet razredov'),
        ('04_povzetki/qualitative-notes.md','Opombe ob pregledu desetih slik'),
        ('04_povzetki/class-mention-summary.csv','Dobesedne omembe petih razredov'),
        ('02_llm/comparison-summary.csv','Razlike LLM/promptov/ponovitev'),('Rezultati_LOCO.pdf','Kratko PDF poročilo'),
        ('Priloga_LOCO.pdf','Celotna besedila izbranih primerov'),('README.md','Protokol in ukazi'),('verification.json','Preverjanje popolnosti')):
        root+=f'<p><a href="{href}">{label}</a></p>'
    (PUBLIC/'index.html').write_text(native.document('Industrija LOCO',root+'</nav>'))


def literal(value):
    """LaTeX display preserves non-ASCII identity as explicit codepoints if needed."""
    result=[];run=0
    for char in str(value):
        if 32<=ord(char)<128 or char in '\n\tčČšŠžŽ':
            result.append(tex(char))
            if char in ',:>':result.append(r'\allowbreak{}')
        else:result.append(r'\texttt{[U+'+f'{ord(char):04X}'+']}')
        run=0 if char.isspace() else run+1
        if run>=32:
            result.append(r'\allowbreak{}');run=0
    return ''.join(result).replace('\n',r'\par'+'\n')


PREAMBLE=r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage[slovene]{babel}
\usepackage{lmodern,graphicx,booktabs,array,longtable,hyperref,geometry}
\geometry{margin=18mm}\hypersetup{hidelinks}\setlength{\emergencystretch}{3em}\sloppy
\begin{document}
'''


def latex(cases,appendix):
    summary=read_json(VLM/'summary.json');llm=read_json(LLM/'summary.json');verification=read_json(LLM/'verification.json')
    box=read_json(PUBLIC/'04_povzetki/box-summary.json');difference=read_json(LLM/'comparison-summary.json')
    warnings=read_json(LLM/'provenance/server-warnings.json')
    number=lambda value:'--' if value is None else f'{value:.2f}'
    pages=[PREAMBLE,r'\section*{Industrijski preizkus LOCO}',
        tex(datetime.now(ZoneInfo('Europe/Ljubljana')).strftime('%d. %m. %Y')),
        '\n30 izvirnih slik iz petih okolij, po šest na okolje. To je namenski zahtevnejši izbor, ne neodvisen naključen vzorec. '
        'Vsi rezultati so novi; THUD inferenc nismo ponavljali. Uporabljene so lokalne uteži in obstoječa izolirana okolja. '
        'Referenčne oznake niso bile vhod modelov. Podatki imajo pet logističnih razredov, ne popolnega popisa scene, in nimajo referenčne stabilnosti.\n',
        'Inferenca je izvedena na RTX 5090; vsak native parameter je preverjen na CUDA. Modeli tečejo zaporedno v ločenih obstoječih okoljih. Revizije modelov, dejanski tipi parametrov in različice Python/PyTorch/CUDA/Transformers so shranjene za vsako sejo.\n',
        tex('VLM: '+', '.join(native.MODELS.values()))+'\n',
        r'\subsection*{Dejanski obseg}',tex(f"VLM: 450 native odgovorov za 270 izvornih položajev. LLM: {verification['logical_positions']} položajev, {verification['main_calls']} API poskusov; statusi {verification['statuses']}. Število prejetih odgovorov je {verification['main_completed_response_records']}. Preskoki so posebej evidentirani; napak ne popravljamo s ponovnim generiranjem."),
        '\nVeljaven JSON oziroma native parse ni pravilnost zaznave ali razvrstitve. Surovi odgovori, tudi neveljavni in delni, so v celoti objavljeni. '
        'Qwen se je po 16 shranjenih opisih procesno prekinil s SIGSEGV (139), brez Python izjeme. Ti zapisi so ostali nespremenjeni; nadaljevanje je obdelalo samo manjkajoče z istimi parametri in vključeno sledjo napak. '
        'Prekinitev je ločeno objavljena. Točen obseg in čas nezapisane generacije nista obnovljiva, zato 450 shranjenih odgovorov ni trditev o nič procesnih napakah. Vzrok prekinitve ni potrjen. '
        'Dodatni Python procesi na deljenem GPU lahko vplivajo na čase; to ne dokazuje vzroka SIGSEGV.\n',
        'GPU vsebnik nadaljevanja je zaključil z 0. Ločeno dokumentirana naknadna napaka lupine (2), povzročena z urejanjem aktivnega zaganjalnika, je odpravljena brez ponavljanja generacij.\n',
        'Pri Gemmi je vsebnik izgubil dostop do GPU (NVML Unknown Error), čeprav je bil GPU viden gostitelju. Šest odgovorov brez potrditve GPU je ohranjenih kot napake. Ponovni zagon samo LOCO vsebnika je obnovil GPU; nadaljevanje je preskočilo vseh 3247 takrat shranjenih položajev. Nobena napaka ni ponovno generirana. Osnovni vzrok ni potrjen; prekinitev je dokumentirana ločeno.\n',
        'Prekinitev pogovora je pozneje ustavila odjemalca med Gemma long/r2 vhodom 36. Strežnik je potrdil preklic HTTP 500. Položaj ostane napaka brez odgovora; iz zamrznjenih datotek je izrecno rekonstruiran samo zahtevek, brez ponovitve. Po ponovni izgubi NVML je bil LOCO vsebnik obnovljen še enkrat; prednaložitev iste Gemme brez prompta je vrnila prazen odziv done-reason=load, zato ni dodatna besedilna generacija. Vhod 69 je ohranil svoj izvirni odgovor, njegov odjemalski čas pa vključuje operativni premor.\n',
        r'\subsection*{VLM: čas in VRAM}',table(['Model','Naloga','N','Velj.','Med. s','Vrh GiB'],
        [[tex(r['model']),tex(r['task']),r['n'],r['schema_or_native_parse_ok'] if r['task']!='combined' else '--',number(r['median_seconds']),number(r['max_allocated_GiB'])] for r in summary],r'p{2.8cm}p{2.7cm}rrrr'),
        'Florence uporablja native caption, OD in dense: deterministično, tri žarke, največ 1024 tokenov. Kombinirani čas je izmerjen za tri zaporedne naloge na sliko, brez izrisovanja. '
        'NVILA FP16 in Qwen3-VL BF16 uporabljata isti podrobni prompt, T=0,3, semena 42/43/44, največ 4096 tokenov, top-p=0,8, top-k=20 in repetition-penalty=1,05. '
        'Modeli imajo različne naloge, predprocesiranje in tokenizerje. Nalaganje je ločeno v provenance; dodatnih ogrevalnih generacij ni. Prvi merjeni klic zato lahko vključuje hladno izvajanje. '
        'PyTorch allocated/reserved sta procesna vrhova; Ollama model VRAM je trenutna alokacija, ne vrh. Vzorci celotnega GPU so ločeni.\n',
        r'\subsection*{LLM: enaki zamrznjeni vhodi}',
        tex('Besedilni modeli: qwen3.5:4b-bf16, qwen3.5:4b-q4_K_M in gemma3:4b-it-q4_K_M. Natančni digesti, strežnik in uteži so shranjeni v provenance.')+'\n',
        'Vsak od 270 izvorov je zamrznjen s SHA256 in lokalnimi ID-ji. Florence je en skupni dokument s sledljivimi OD/dense regijami; prekrivanje ne potrjuje različnih fizičnih objektov. '
        'NVILA uporablja samo obstoječe dobesedno izluščenje; celoten neveljavni tekst je ohranjen, delni vhodi označeni, nezanesljivi seznami preskočeni. '
        'LLM ne prejme slike ali referenčnih razredov. Oba nespremenjena prompta sta brez vsebinskih primerov in uporabljata iste definicije fixed/movable/uncertain. '
        'Fixed pomeni strukturen/nameščen element, movable prestavljiv predmet, uncertain premalo dokazov. Odpiranje nameščenih vrat ni prestavitev celih vrat.\n',
        'LLM ponovitve so ločene od VLM ponovitev. T=0,3, semena 42/43/44, top-p=0,8, top-k=20, presence-penalty=1,5, največ 4096 tokenov, think=false, JSON. '
        'Kontekstni razredi so določeni pred inferenco za isti vhod enako pri vseh LLM in promptih; dolgih opisov ne krajšamo. Dejanska mesta in konteksti so v runtime dokazih. '
        'Število regij, dolžina opisa in omejitev izhoda vplivajo na čas in napake sheme; različni izvori niso enako zahtevne klasifikacijske naloge. '
        'HTTP časi lahko vključujejo čakanje; mean-decode čas je ločen. Seštevanje prekrivajočih HTTP časov ni trajanje poskusa.\n']
    pages.append('Časovne tabele vključujejo tudi neuspele poskuse, vključno s šestimi Gemma odgovori brez potrditve GPU. Ti niso sprejete GPU klasifikacije. Prekinitev in nadaljevanje pomenita, da celotnega stenskega časa ne predstavljamo kot neprekinjen benchmark.\n')
    pages.append(tex(f"Strežniški dnevnik: {len(warnings['server_warning_lines'])} vrstic opozoril/napak, od tega {len(warnings['input_truncation_warning_lines'])} opozoril o krajšanju vhoda. Celotne vrstice so v server-warnings.json; inicializacijska opozorila so ločena od generacijskih API poskusov.")+'\n')
    timing=[]
    for model in ('qwen35-bf16','qwen35-q4','gemma3-q4'):
        for prompt in ('short','long'):
            rows=[r for r in llm if r['llm']==model and r['prompt']==prompt]
            timing.append([tex(model),tex(prompt),sum(r['ok'] for r in rows),sum(r['error'] for r in rows),sum(r['skipped'] for r in rows),
                number(sum((r['mean_seconds'] or 0)*r['timed_calls'] for r in rows)/max(1,sum(r['timed_calls'] for r in rows))),
                number(max(r['max_reported_model_vram_bytes'] for r in rows)/2**30)])
    pages += [table(['LLM','Prompt','Velj.','Napake','Preskok','Povp. HTTP s','Max. alok. GiB'],timing,r'p{2.6cm}p{1.5cm}rrrrr'),
        r'\subsection*{Razlike odločitev, ne točnost}',
        table(['Primerjava','Pari','Primerljivi','Pari zapisov','Spremembe','Strinjanje'],
            [[tex(r['kind']),r['pairs'],r['comparable_pairs'],r['object_comparisons'],r['class_changes'],number(100*r['agreement'])+r'\%' if r['agreement'] is not None else '--'] for r in difference],r'p{2.7cm}rrrrr'),
        'Prve tri vrstice primerjajo isti zamrznjeni SHA in tipizirane ID-je. Med različnimi VLM in njihovimi ponovitvami so pari samo edinstvena enaka normalizirana imena: NEPOTRJENA identiteta, ne isti fizični objekt. '
        'Neveljavni odzivi niso prisilno sprejete delne klasifikacije. Imenovalci položajev, neprimerljivih parov in objektnih primerjav so ločeni. Strinjanje ni pravilnost in pari niso neodvisna opazovanja.\n',
        r'\subsection*{Okvirji: pet referenčnih razredov}',
        table(['Naloga','Razred','Napovedi','Ujemanja','Reference','P','R'],
            [[tex(r['task']),tex(r['class']),r['mapped_predictions'],r['matched'],r['reference_instances_repeated'],number(r['precision']),number(r['recall'])] for r in box],r'p{1.1cm}p{3.5cm}rrrrr'),
        'Vnaprej zamrznjeni slovar uporablja samo točno normalizirana imena. Ujemanje je največje enolično znotraj razreda pri IoU vsaj 0,5; vsak par z obema okvirjema in referenčnim ID je shranjen. '
        'Reference v tabeli se ponovijo trikrat, niso nove neodvisne instance. Nedefinirana precision brez preslikanih napovedi ostane prazna. Neoznačen regal, stena ali človek ni avtomatično halucinacija. '
        'Prostih opisov brez okvirjev ne ocenjujemo kot detektorja. Pokritost imen/omemb je ločena besedilna mera brez lokalizacijske pravilnosti.\n']
    vocabulary=read_json(PUBLIC/'04_povzetki/native-label-vocabulary.json')
    detail=read_json(VLM/'detail-summary.json')
    repeat_summary=read_json(PUBLIC/'04_povzetki/native-repeat-summary.json')
    pages += [r'\subsection*{Ponovljivost VLM}',
        table(['Model','Naloga','Pari','Enak izhod','Primerljiva imena','Povp. Jaccard'],
            [[tex(r['model']),tex(r['task']),r['repeat_pairs'],r['equal_native_text_pairs'],r['list_comparable_pairs'],number(r['mean_literal_name_jaccard'])] for r in repeat_summary],r'p{2.8cm}p{2.6cm}rrrr'),
        'Tri parjenja ponovitev na sliko niso neodvisni vzorci. Enak izhod pomeni enako native besedilo, brez časov in ID-jev. Call ID-ji so različni; Florence je determinističen. Primerjava imen ni potrditev fizične identitete ali pravilnosti.\n']
    pages += [r'\subsection*{Obseg opisov in seznamov}',
        table(['Model','Naloga','Seznam N/90','Med. zapisi','Med. znaki'],
            [[tex(r['model']),tex(r['task']),r['literal_list_available'],number(r['median_object_or_region_entries']),number(r['median_native_text_characters'])] for r in detail],r'p{2.8cm}p{2.6cm}rrr'),
        'Dolžina ni kakovost. Število regij ni potrjeno število predmetov. Delni NVILA seznami so dobesedni, ne preverjeni; caption brez seznama ni prazna scena.\n']
    mentions=read_json(PUBLIC/'04_povzetki/class-mention-summary.json')
    pages += [table(['Model','Prisotnost razredov','Omemba v seznamu','Omemba v celotnem tekstu'],
        [[tex(model),sum(r['reference_class_present_positions'] for r in mentions if r['model']==model),
          sum(r['exact_list_label_mentioned_positions'] for r in mentions if r['model']==model),
          sum(r['literal_fulltext_mentioned_positions'] for r in mentions if r['model']==model)] for model in native.MODELS],r'p{2.8cm}rrr'),
        'Vsota razred/slika/ponovitev: vnaprejšnji slovar in točne dobesedne omembe, Florence naloge združene na izvor. To ni instančni recall, lokalizacija ali točnost. Množin in opisnih stavkov po ogledu rezultatov ne pretvarjamo v singularna imena. Tabela po petih razredih in vsi imenovalci so v class-mention-summary.csv.\n']
    pages.append(tex('Florence je vrnil '+str(vocabulary['od']['total_native_regions'])+' OD in '+str(vocabulary['dense']['total_native_regions'])+' dense regij (vsota treh ponovitev). Nobena celotna oznaka se ni točno ujemala z zamrznjenim slovarjem petih razredov; zato ni preslikanih napovedi, precision ni definirana, recall tega strogega postopka je nič. To ne dokazuje, da model ni lokaliziral vidnih regij. Splošnih box/crate in stavčnih opisov po ogledu rezultatov nismo preimenovali v referenčne razrede. Vsa izvirna imena in okvirji so ohranjeni.')+'\n')
    for case in cases:
        fid=case['frame']['frame_id'];prefix='03_primeri/'+fid+'/'
        pages.extend([r'\clearpage\section*{Primer '+str(case['index'])+': '+tex(fid)+'}',
            tex(case['selection_reason'])+'. Dva kontrastna primera na okolje; ročne ocene niso izpolnjene.\n',
            r'\begin{center}'])
        if case['predicted_overlay_available']:pages.append(r'\includegraphics[width=.48\linewidth]{'+prefix+'florence-od-r1.predictions.jpg}')
        pages += [r'\includegraphics[width=.48\linewidth]{'+prefix+'reference-LOCO.jpg}',r'\end{center}',
            'Levo: Florence OD r1, samo napovedi. Desno: ločeno označena referenca LOCO. Barve označujejo regije, ne zanesljivosti. NVILA in Qwen nimata napovedanih okvirjev.\n',
            'Celotni opisi in seznami vseh treh modelov ter njihovih ponovitev so v ločeni prilogi in HTML; noben odgovor ni skrajšan v objavljenih podatkih. '
            'Florence lahko isti predmet opiše večkrat; število regij ni število potrjenih fizičnih objektov.\n']
        pages += [r'\paragraph{Opombe ob slikovnem pregledu, ne referenčne ocene.}',
            tex(case['qualitative_notes']['visible_successes'])+' '+tex(case['qualitative_notes']['limitations_and_checks'])+' '+tex(case['qualitative_notes']['native_regions_inspection'])+'\n']
        c=case['contrast']
        if c:
            pages += [r'\paragraph{Isti vhod in isti ID.}',tex(f"VLM {c['source_vlm']}/{c['vlm_repeat']}, vhod {c['input_index']}, ID {c['object_id']}, oznaka {c['label']}. LLM-r1, seed42. Primerjane so odločitve istega ID; to ni ocena pravilnosti."),
                table(['LLM / prompt','Razred oziroma status','Utemeljitev ali izrecna napaka'],
                [[tex(a['llm']+' / '+a['prompt']),tex(a['classification'] or a['status']),literal(a['reason'] if a['reason'] else 'NAPAKA VALIDACIJE: '+str(a['error']))] for a in c['answers']],r'p{4cm}p{2.7cm}p{8.2cm}'),
                r'\paragraph{Izvorno opazovanje za ta ID.}',literal(c['observation'])]
    pages += [r'\clearpage\section*{Ponovitev in datoteke}',
        'Odprite index.html lokalno v prenesenem repozitoriju. '+tex('01_vlm')+' hrani vse native odgovore in prompte, '+tex('02_llm')+' vse zamrznjene vhode, odzive in primerjave, '+tex('03_primeri')+' deset izbranih slik in prazen obrazec, '+tex('04_povzetki')+' dokumentirano vrednotenje. '
        'Uteži, vsebniki, virtualna okolja in veliki podvojeni delovni izhodi ostanejo lokalno v results/loco. V javnem paketu so vsi celotni besedilni odzivi.\n',
        r'\begin{verbatim}'+'\nbash scripts/loco_experiment.sh prepare\nbash scripts/loco_experiment.sh vlm-pilots\nbash scripts/loco_experiment.sh vlm\nbash scripts/loco_experiment.sh vlm-export\nbash scripts/loco_experiment.sh freeze\nbash scripts/loco_experiment.sh pilot\nbash scripts/loco_experiment.sh run\nbash scripts/loco_experiment.sh export\nbash scripts/loco_experiment.sh report\nbash scripts/loco_experiment.sh pdf\nbash scripts/loco_experiment.sh verify\n'+r'\end{verbatim}',
        'Nadaljevanje preskoči shranjene uspehe in napake ter zavrne spremembo konfiguracije ali SHA vhodov. Preverjanje vključuje celotno mrežo, prompte, dejanske nastavitve, povezave, prazne ročne obrazce in nespremenjenost starih rezultatov. '
        'Človekov pregled pokritosti, lokacij in dejanskih podpor/pritrditve ostaja potreben; ne izmišljamo referenčne stabilnosti. LOCO vir/licenca in izbor so v '+tex('data/loco_test30')+' (CC0-1.0). '
        r'Uradni vir: \url{https://github.com/tum-fml/loco}, preverjeno 28. 10. 2026. Navedba vira: Mayershofer, Holm, Molter in Fottner, LOCO: Logistics Objects in Context, ICMLA 2020.\par',r'\end{document}']
    (PUBLIC/'Rezultati_LOCO.tex').write_text('\n'.join(pages))
    appendix_latex(appendix)


def appendix_latex(appendix):
    """Native text is independent of the later LLM completion/analysis."""
    appendix_text=[PREAMBLE,r'\section*{Priloga LOCO: celotna besedila desetih primerov}',
        'Vse tri VLM ponovitve vseh treh modelov. Izvirne native naloge so ločene. Neveljaven JSON ostane neveljaven. '
        'Bajtno nespremenjeni surovi zapisi in tokeni so v 01-vlm, vse LLM ponovitve z razlogi pa v 02-llm in HTML. '
        'Nepodprti znaki v tem PDF so izrecno zapisani kot U+ kodne točke; izvirno besedilo ni spremenjeno v JSON/HTML.\n']
    for row in appendix:
        appendix_text += [r'\subsection*{'+tex(row['frame_id']+' / '+row['vlm']+' / '+row['vlm_repeat']+' / '+row['task'])+'}',
            tex('Status: '+row['status']+'. Call ID: '+row['call_id']+'. Napaka: '+str(row.get('error') or '--')),r'\begin{quote}\footnotesize',literal(row['content']),r'\end{quote}']
        if row['vlm']=='florence' and row['task'] in ('od','dense'):
            appendix_text += [r'\paragraph{Celotni native zapisi v pikah izvirne slike.}',literal(json.dumps(row['view'],ensure_ascii=False,indent=2))]
    appendix_text.append(r'\end{document}')
    (PUBLIC/'Priloga_LOCO.tex').write_text('\n'.join(appendix_text))
