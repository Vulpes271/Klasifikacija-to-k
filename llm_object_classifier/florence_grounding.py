"""Ten-image export for the existing native Florence caption-to-grounding runner."""
import argparse
import html
from pathlib import Path
import shutil

from .experiment import read_json,write_json,sha256
from .thud_experiment import immutable_json
from .thud_extended_common import ROOT,saved_path

WORK=ROOT/'results/florence-caption-grounding-test10'
PUBLIC=ROOT/'results models/florence_caption_grounding_test10'
CONFIG=ROOT/'configs/stage1/florence-caption-grounding.json'
CAPTION='<MORE_DETAILED_CAPTION>'
GROUND='<CAPTION_TO_PHRASE_GROUNDING>'
REMAINING_WORK=ROOT/'results/florence-caption-grounding-loco20'
WORK200=ROOT/'results/experiment200/florence'


def prepare():
    from PIL import Image
    selected=[]
    for i,item in enumerate(read_json(ROOT/'configs/loco-examples.json')['images']):
        fid=item['frame_id'];path=ROOT/'data/loco_test30/images'/(fid+'.jpg')
        with Image.open(path) as image:size=list(image.size)
        selected.append({'dataset':'florence_grounding10','index':i,'frame_id':fid,
            'source_frame_id':fid,'image_path':str(path),'image_sha256':sha256(path.read_bytes()),
            'image_size':size,'subset20':False})
    immutable_json(WORK/'inputs.json',selected)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(CONFIG,PUBLIC/'settings.json')
    write_json(PUBLIC/'inputs.json',[{**f,'image_path':str(Path(f['image_path']).relative_to(ROOT))} for f in selected])
    print('Prepared ten existing LOCO images, two per environment, no reference labels.')


def prepare_remaining():
    """Only the other twenty RGB files; keep the original ten indices intact."""
    from PIL import Image
    first=read_json(PUBLIC/'inputs.json')[:10]
    old_ids={f['frame_id'] for f in first}
    images=sorted((ROOT/'data/loco_test30/images').glob('*.jpg'))
    assert len(images)==30 and len(old_ids)==10
    selected=[]
    for index,path in enumerate((p for p in images if p.stem not in old_ids),10):
        with Image.open(path) as image:size=list(image.size)
        selected.append({'dataset':'florence_grounding_loco20','index':index,'frame_id':path.stem,
            'source_frame_id':path.stem,'image_path':str(path),'image_sha256':sha256(path.read_bytes()),
            'image_size':size,'subset20':False})
    assert len(selected)==20
    assert read_json(CONFIG)==read_json(PUBLIC/'settings.json')
    immutable_json(REMAINING_WORK/'inputs.json',selected)
    print('Prepared only the remaining twenty LOCO RGB images, without annotations.')


def prepare200():
    """Build a reference-free RGB manifest; preserve all thirty existing IDs."""
    import json
    from PIL import Image
    old=read_json(PUBLIC/'inputs.json')[:30]
    loco=read_json(ROOT/'data/loco_test100/selection.json')
    by_id={r['frame_id']:r for r in loco};records=[]
    for f in old:
        r=by_id[f['frame_id']];records.append({k:r[k] for k in ('frame_id','dataset','environment','image_path','sha256','image_size')})
    used_ids={r['frame_id'] for r in records}
    records.extend({k:r[k] for k in ('frame_id','dataset','environment','image_path','sha256','image_size')} for r in loco if r['frame_id'] not in used_ids)
    for dataset in ('thud_agtea_test50','thud_dense_test50'):
        metadata=[json.loads(line) for line in (ROOT/'data'/dataset/'objects.jsonl').read_text().splitlines()]
        scenes={r['frame_id']:r['scene'] for r in metadata}
        discovery=ROOT/'results/thud-part2/inputs'/(dataset+'-discovery-all.jsonl')
        for line in discovery.read_text().splitlines():
            r=json.loads(line);path=(discovery.parent/r['image_path']).resolve()
            records.append({'frame_id':dataset+'--'+r['frame_id'],'dataset':dataset,
                'environment':scenes[path.name.split('.color')[0]],'image_path':str(path.relative_to(ROOT)),
                'sha256':r['image_sha256'],'image_size':r['image_size']})
    assert len(records)==len(loco)+100
    assert len({r['frame_id'] for r in records})==len({r['sha256'] for r in records})==len(records)
    for r in records:assert sha256((ROOT/r['image_path']).read_bytes())==r['sha256']
    immutable_json(ROOT/'results models/florence_grounding_llm_comparison10/inputs200.json',records)
    selected=[{'dataset':'grounding200','index':i,'frame_id':r['frame_id'],'source_frame_id':r['frame_id'],
        'image_path':str(ROOT/r['image_path']),'image_sha256':r['sha256'],'image_size':r['image_size'],'subset20':False}
        for i,r in enumerate(records) if i>=30]
    immutable_json(WORK200/'inputs.json',selected)
    print('Prepared RGB inputs:',len(records),'new Florence images:',len(selected),flush=True)


def draw(image_path, regions, target):
    """Original pixels/coordinates; the full phrases remain in the adjacent legend."""
    from PIL import Image,ImageDraw,ImageFont
    image=Image.open(image_path).convert('RGB');pen=ImageDraw.Draw(image)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',max(12,image.width//100))
    except OSError:font=ImageFont.load_default()
    for row in regions:
        color=('#ff3232','#00c3ff','#ffe600','#48ff80','#ff70ff')[(row['id']-1)%5]
        pen.rectangle(row['bbox_xyxy'],outline=color,width=max(2,image.width//400))
        label=row['label'];short=label if len(label)<=55 else label[:52]+'...'
        pen.text((row['bbox_xyxy'][0],max(0,row['bbox_xyxy'][1]-18)),f"{row['id']}: {short}",
            font=font,fill=color,stroke_width=1,stroke_fill='black')
    if not regions:pen.text((8,8),'NO VALID GROUNDED REGIONS',font=font,fill='red',stroke_width=1,stroke_fill='white')
    image.save(target)


def document(title,body):
    return '<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font-family:Arial;max-width:1600px;margin:24px auto;padding:12px;line-height:1.5}.images{display:flex;gap:12px}.images section{width:50%}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb;text-align:left}</style>'+body


def export(remaining=False,experiment200=False):
    work=WORK200 if experiment200 else (REMAINING_WORK if remaining else WORK)
    selected=read_json(work/'inputs.json')
    rows=read_json(PUBLIC/'results.json')['results'][:(30 if experiment200 else 10)] if remaining or experiment200 else []
    for frame in selected:
        fid=frame['frame_id'];folder=PUBLIC/fid;folder.mkdir(parents=True,exist_ok=True)
        cp=saved_path(work/'native',frame,'caption');gp=saved_path(work/'native',frame,'grounding')
        caption=read_json(cp);grounding=read_json(gp)
        original='original'+Path(frame['image_path']).suffix
        shutil.copyfile(frame['image_path'],folder/original)
        shutil.copyfile(cp,folder/'caption.raw.json');shutil.copyfile(gp,folder/'grounding.raw.json')
        shutil.copyfile(saved_path(work/'native',frame,'combined'),folder/'combined.raw.json')
        text=caption.get('parsed',{}).get(CAPTION,'')
        (folder/'caption.txt').write_text(text,encoding='utf-8')
        (folder/'caption.native.txt').write_text(caption.get('content',''),encoding='utf-8')
        (folder/'grounding.native.txt').write_text(grounding.get('content',''),encoding='utf-8')
        regions=[]
        if grounding['status']=='ok':
            value=grounding['parsed'][GROUND]
            regions=[{'id':i,'label':label,'bbox_xyxy':box,'source_task':GROUND,'source_region_id':i}
                for i,(label,box) in enumerate(zip(value['labels'],value['bboxes']),1)]
        write_json(folder/'regions.json',{'status':grounding['status'],'image_size':frame['image_size'],
            'coordinate_space':'original image pixels, xyxy','regions':regions,'error':grounding.get('error'),
            'note':'Phrase regions can overlap or denote groups; not verified physical instances or exhaustive detections.'})
        draw(frame['image_path'],regions,folder/'caption-grounding.predictions.jpg')
        old=ROOT/'analiza rezultatov vmesna predstavitev/industrija LOCO/01_vlm'/('loco_test30--'+fid)/'florence/r1'
        if not experiment200:
            shutil.copyfile(old/'od.predictions.jpg',folder/'previous-od.predictions.jpg')
            shutil.copyfile(old/'od.raw.json',folder/'previous-od.raw.json')
        # Literal transport for the existing second step; not_visible means
        # attachment evidence was not supplied, not that attachment is absent.
        objects=[{'id':r['id'],'label':r['label'],
            'observation':f"Florence grounded caption phrase {r['label']!r} at bbox_xyxy {r['bbox_xyxy']} in original pixels. Attachment/contact evidence and physical-instance identity were not established.",
            'attachment_evidence':'not_visible','identifiable':True,'bbox_xyxy':r['bbox_xyxy'],
            'source_task':GROUND,'source_region_id':r['source_region_id']} for r in regions]
        write_json(PUBLIC/'stage1'/f"frame-{frame['index']+1:06d}.json",{'stage':1,'frame_id':fid,
            'model_name':'florence','model':'microsoft/Florence-2-large','variant':{'id':'caption-grounding-v1'},
            'image_sha256':frame['image_sha256'],'status':'ok' if caption['status']==grounding['status']=='ok' else 'error',
            'parsed':{'scene_description':text,'objects':objects},
            'source_records':{'caption':str((folder/'caption.raw.json').relative_to(ROOT)),
                'grounding':str((folder/'grounding.raw.json').relative_to(ROOT))},
            'adapter_notes':'Native regions, not verified physical objects. No LLM/reference labels used.'})
        rows.append({'frame_id':fid,'caption_status':caption['status'],'grounding_status':grounding['status'],
            'grounded_regions':len(regions),'previous_od_regions':None if experiment200 else len(read_json(old/'od.raw.json')['parsed']['<OD>']['bboxes']),
            'caption_call_id':caption.get('call_id'),'grounding_call_id':grounding.get('call_id')})
        body=f'<p><a href="../index.html">← Vse slike</a></p><h1>{fid}</h1><h2>Celoten novi opis</h2><p>'+html.escape(text)+'</p>'
        if not experiment200:body+='<h2>Prejšnji OD (shranjen)</h2><img src="previous-od.predictions.jpg">'
        body+='<h2>Opis → phrase grounding</h2><a href="caption-grounding.predictions.jpg"><img src="caption-grounding.predictions.jpg"></a>'
        body+='<p><a href="'+original+'">Original</a> · <a href="caption.raw.json">Caption raw</a> · <a href="grounding.raw.json">Grounding raw</a> · <a href="regions.json">Koordinate</a></p><p>Številke so lokalni ID regije. Dolge slikovne oznake so skrajšane; spodaj so celotne fraze. Napovedane regije se lahko prekrivajo ali opisujejo skupine.</p><table><tr><th>ID</th><th>Celotna fraza</th><th>bbox xyxy</th></tr>'
        for r in regions:body+=f'<tr><td>{r["id"]}</td><td>'+html.escape(r['label'])+'</td><td>'+html.escape(str(r['bbox_xyxy']))+'</td></tr>'
        if grounding['status']!='ok':body+='<tr><td colspan="3">'+html.escape(grounding.get('error',''))+'</td></tr>'
        (folder/'index.html').write_text(document(fid,body+'</table>'))
    write_json(PUBLIC/'results.json',{'images':len(rows),'profile':read_json(CONFIG),'results':rows,
        'quality_metrics':None,'note':'Region counts are output volume, not detection recall or correctness; no references supplied.'})
    body=f'<h1>Florence: podroben opis → označevanje fraz ({len(rows)} slik)</h1><p>Dva native klica na sliko, brez LLM ali referenc. Stari OD je samo kopiran za primerjavo. Grounding označuje fraze iz opisa; ne zagotavlja vseh objektov ali instanc.</p><table><tr><th>Slika</th><th>Prejšnji OD: regije</th><th>Novo grounding: regije</th><th>Status</th></tr>'
    for r in rows:body+=f'<tr><td><a href="{r["frame_id"]}/index.html">{r["frame_id"]}</a></td><td>{r["previous_od_regions"]}</td><td>{r["grounded_regions"]}</td><td>{r["grounding_status"]}</td></tr>'
    (PUBLIC/'index.html').write_text(document('Florence grounding: test 10',body+'</table>'))
    provenance=PUBLIC/'provenance'/('experiment200' if experiment200 else ('loco20' if remaining else ''))
    if not experiment200:
        shutil.copytree(work/'native/environments',provenance/'environments',dirs_exist_ok=True)
        shutil.copytree(work/'native/source-snapshots',provenance/'source-snapshots',dirs_exist_ok=True)
    shutil.copyfile(work/'native/manifest.json',provenance/'manifest.json')
    shutil.copyfile(work/'logs/inference.log',provenance/'inference.log')
    shutil.copyfile(work/'progress.md',provenance/'progress.md')
    if remaining or experiment200:
        first=read_json(PUBLIC/'inputs.json')[:(30 if experiment200 else 10)]
        write_json(PUBLIC/'inputs.json',first+[{**f,'image_path':str(Path(f['image_path']).relative_to(ROOT))} for f in selected])
        if not experiment200:(PUBLIC/'README-loco30.md').write_text('30 slik LOCO: prvih deset ohranjenih, samo preostalih dvajset je novih. Isti profil; 40 novih native klicev, brez ogrevanja ali ponovitev. Ukaz: bash scripts/stage1_florence.sh --loco30. Nadaljevanje preskoči shranjene rezultate, vključno z napakami.\n')
    else:
        write_readme()
    verify_export()
    print('Saved full descriptions and annotated images:',len(rows),PUBLIC)


def write_readme():
    (PUBLIC/'README.md').write_text('''# Florence: detailed caption → phrase grounding, 10 slik

Nastavitve: `configs/stage1/florence-caption-grounding.json`.
MORE_DETAILED_CAPTION → CAPTION_TO_PHRASE_GROUNDING dobi natanko celoten novi
opis. Prej je bil opis že podroben, OD/dense pa sta delovala ločeno od njega.
Limit izhoda je 2048 namesto 1024; tri žarke, deterministično, FP16/eager,
izvirna obdelava slike iz checkpointa. Koordinate so v originalnih pikah.

Ukaz: `bash scripts/stage1_florence.sh`. Obstoječih deset LOCO slik, dve na
okolje. Dvajset novih native generacij, brez ponovitev, ogrevalnih generacij,
LLM ali PDF. Stari poskusi se ne spreminjajo. Nadaljevanje preskoči shranjeno.

Odprite index.html. Vsaka mapa: original, celoten opis in raw odgovora,
tokeni/nastavitve/čas/VRAM, novi okvirji in prejšnji OD za primerjavo.
`stage1/` vsebuje literalni transport za poznejši drugi korak, ki NI izveden.

Opis lahko izpusti predmete. Regije lahko opisujejo skupine ali se prekrivajo;
njihovo število ni število potrjenih fizičnih objektov ali kakovostna metrika.
Ni ročnega dodajanja/popravljanja okvirjev ali uporabe referenčnih oznak.

Uradna naloga: https://huggingface.co/microsoft/Florence-2-large#caption-to-phrase-grounding
''')


def verify_export():
    """Check provenance and file completeness, without another model call."""
    from PIL import Image,ImageDraw
    frames=read_json(PUBLIC/'inputs.json');call_ids=[]
    preview=Image.new('RGB',(1400,500*((min(30,len(frames))+1)//2)),'white');pen=ImageDraw.Draw(preview)
    for i,frame in enumerate(frames):
        folder=PUBLIC/frame['frame_id']
        caption=read_json(folder/'caption.raw.json');grounding=read_json(folder/'grounding.raw.json')
        original='original'+Path(frame['image_path']).suffix
        assert sha256((folder/original).read_bytes())==frame['image_sha256']
        if grounding.get('source_caption'):
            assert grounding['user_prompt']==GROUND+caption['parsed'][CAPTION]
            assert grounding['source_caption']['call_id']==caption['call_id']
            assert grounding['source_caption']['sha256']==sha256((folder/'caption.raw.json').read_bytes())
        for row in (caption,grounding):
            if row.get('call_id'):
                assert row['gpu_verified'];call_ids.append(row['call_id'])
        for filename in (original,'caption-grounding.predictions.jpg'):
            with Image.open(folder/filename) as image:
                assert list(image.size)==frame['image_size']
                image.verify()
        if i<30:
            with Image.open(folder/'caption-grounding.predictions.jpg') as image:
                image.thumbnail((690,460));preview.paste(image,((i%2)*700,(i//2)*500))
            pen.text(((i%2)*700+5,(i//2)*500+465),frame['frame_id'],fill='black')
    assert len(set(call_ids))==len(call_ids)
    preview.save(PUBLIC/'preview.jpg')
    write_json(PUBLIC/'verification.json',{'images':len(frames),'native_gpu_calls':len(call_ids),'unique_call_ids':len(set(call_ids)),
        'full_caption_was_grounding_input':True,'annotation_images':len(frames),'llm_calls':0,
        'warmup_generations':0,'quality_verified':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','export','prepare-remaining','export-remaining','prepare200','export200'))
    args=parser.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='prepare-remaining':prepare_remaining()
    elif args.phase=='prepare200':prepare200()
    elif args.phase=='export200':export(experiment200=True)
    else:export(remaining=args.phase=='export-remaining')
