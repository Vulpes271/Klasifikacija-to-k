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


def export():
    rows=[]
    for frame in read_json(WORK/'inputs.json'):
        fid=frame['frame_id'];folder=PUBLIC/fid;folder.mkdir(parents=True,exist_ok=True)
        cp=saved_path(WORK/'native',frame,'caption');gp=saved_path(WORK/'native',frame,'grounding')
        caption=read_json(cp);grounding=read_json(gp)
        shutil.copyfile(frame['image_path'],folder/'original.jpg')
        shutil.copyfile(cp,folder/'caption.raw.json');shutil.copyfile(gp,folder/'grounding.raw.json')
        shutil.copyfile(saved_path(WORK/'native',frame,'combined'),folder/'combined.raw.json')
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
            'grounded_regions':len(regions),'previous_od_regions':len(read_json(old/'od.raw.json')['parsed']['<OD>']['bboxes']),
            'caption_call_id':caption.get('call_id'),'grounding_call_id':grounding.get('call_id')})
        body=f'<p><a href="../index.html">← Deset slik</a></p><h1>{fid}</h1><h2>Celoten novi opis</h2><p>'+html.escape(text)+'</p>'
        body+='<div class="images"><section><h2>Prejšnji OD (shranjen)</h2><a href="previous-od.predictions.jpg"><img src="previous-od.predictions.jpg"></a></section><section><h2>Opis → phrase grounding (nov)</h2><a href="caption-grounding.predictions.jpg"><img src="caption-grounding.predictions.jpg"></a></section></div>'
        body+='<p><a href="original.jpg">Original</a> · <a href="caption.raw.json">Caption raw</a> · <a href="grounding.raw.json">Grounding raw</a> · <a href="regions.json">Koordinate</a></p><p>Številke so lokalni ID regije. Dolge slikovne oznake so skrajšane; spodaj so celotne fraze. Napovedane regije se lahko prekrivajo ali opisujejo skupine.</p><table><tr><th>ID</th><th>Celotna fraza</th><th>bbox xyxy</th></tr>'
        for r in regions:body+=f'<tr><td>{r["id"]}</td><td>'+html.escape(r['label'])+'</td><td>'+html.escape(str(r['bbox_xyxy']))+'</td></tr>'
        if grounding['status']!='ok':body+='<tr><td colspan="3">'+html.escape(grounding.get('error',''))+'</td></tr>'
        (folder/'index.html').write_text(document(fid,body+'</table>'))
    write_json(PUBLIC/'results.json',{'images':len(rows),'profile':read_json(CONFIG),'results':rows,
        'quality_metrics':None,'note':'Region counts are output volume, not detection recall or correctness; no references supplied.'})
    body='<h1>Florence: podroben opis → označevanje fraz</h1><p>Deset slik, dva nova native klica na sliko, brez LLM ali referenc. Stari OD je samo kopiran za primerjavo. Grounding označuje fraze iz opisa; ne zagotavlja vseh objektov ali instanc.</p><table><tr><th>Slika</th><th>Prejšnji OD: regije</th><th>Novo grounding: regije</th><th>Status</th></tr>'
    for r in rows:body+=f'<tr><td><a href="{r["frame_id"]}/index.html">{r["frame_id"]}</a></td><td>{r["previous_od_regions"]}</td><td>{r["grounded_regions"]}</td><td>{r["grounding_status"]}</td></tr>'
    (PUBLIC/'index.html').write_text(document('Florence grounding: test 10',body+'</table>'))
    shutil.copytree(WORK/'native/environments',PUBLIC/'provenance/environments',dirs_exist_ok=True)
    shutil.copytree(WORK/'native/source-snapshots',PUBLIC/'provenance/source-snapshots',dirs_exist_ok=True)
    shutil.copyfile(WORK/'native/manifest.json',PUBLIC/'provenance/manifest.json')
    shutil.copyfile(WORK/'logs/inference.log',PUBLIC/'provenance/inference.log')
    shutil.copyfile(WORK/'progress.md',PUBLIC/'provenance/progress.md')
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
    verify_export()
    print('Saved ten full descriptions and annotated images:',PUBLIC)


def verify_export():
    """Check provenance and file completeness, without another model call."""
    from PIL import Image,ImageDraw
    frames=read_json(PUBLIC/'inputs.json');call_ids=[]
    preview=Image.new('RGB',(1400,2500),'white');pen=ImageDraw.Draw(preview)
    assert len(frames)==10
    for i,frame in enumerate(frames):
        folder=PUBLIC/frame['frame_id']
        caption=read_json(folder/'caption.raw.json');grounding=read_json(folder/'grounding.raw.json')
        assert caption['status']==grounding['status']=='ok'
        assert sha256((folder/'original.jpg').read_bytes())==frame['image_sha256']
        assert grounding['user_prompt']==GROUND+caption['parsed'][CAPTION]
        assert grounding['source_caption']['call_id']==caption['call_id']
        assert grounding['source_caption']['sha256']==sha256((folder/'caption.raw.json').read_bytes())
        assert caption['gpu_verified'] and grounding['gpu_verified']
        call_ids.extend((caption['call_id'],grounding['call_id']))
        for filename in ('original.jpg','caption-grounding.predictions.jpg','previous-od.predictions.jpg'):
            with Image.open(folder/filename) as image:
                assert list(image.size)==frame['image_size']
                image.verify()
        with Image.open(folder/'caption-grounding.predictions.jpg') as image:
            image.thumbnail((690,460));preview.paste(image,((i%2)*700,(i//2)*500))
        pen.text(((i%2)*700+5,(i//2)*500+465),frame['frame_id'],fill='black')
    assert len(set(call_ids))==20
    preview.save(PUBLIC/'preview.jpg')
    write_json(PUBLIC/'verification.json',{'images':10,'native_gpu_calls':20,'unique_call_ids':20,
        'full_caption_was_grounding_input':True,'new_annotation_images':10,'llm_calls':0,
        'warmup_generations':0,'quality_verified':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','export'))
    args=parser.parse_args();(prepare if args.phase=='prepare' else export)()
