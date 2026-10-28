"""Four-image, eight-call bbox export using the existing isolated VLM runners."""
import argparse
import html
import math
from pathlib import Path
import shutil

from .experiment import read_json, write_json, sha256, strict_json
from .thud_experiment import immutable_json
from .thud_extended_common import ROOT, draw_boxes

WORK = ROOT/'results/bbox-comparison'
PUBLIC = ROOT/'results models/bbox_comparison'
FRAMES = ('loco-s1-33243','loco-s2-34608','loco-s3-36540','loco-s4-31835')
INSTRUCTION = ROOT/'configs/prompts/loco-bbox.txt'


def bbox_prompt(instruction, frame):
    width,height=frame['image_size']
    return (instruction.strip()+f'\n\nOriginal image dimensions: width={width} pixels, height={height} pixels. '
        f'Use only this image. Coordinates must satisfy 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}.')


def validate_bbox(value, size):
    """Format/bounds validation only. No repair, rescaling, clipping or guessing."""
    if not isinstance(value,dict) or set(value)!={'objects'} or not isinstance(value['objects'],list):
        raise ValueError('Expected exactly {"objects": [...]}')
    seen=set()
    for obj in value['objects']:
        if not isinstance(obj,dict) or set(obj)!={'id','label','bbox'}:
            raise ValueError('Each object must contain exactly id, label and bbox')
        identity=obj['id']
        if type(identity) not in (int,str) or (isinstance(identity,str) and not identity.strip()):
            raise ValueError('Invalid object id')
        key=(type(identity),identity)
        if key in seen:raise ValueError('Duplicate object id')
        seen.add(key)
        if not isinstance(obj['label'],str) or not obj['label'].strip():raise ValueError('Empty label')
        box=obj['bbox']
        if not isinstance(box,list) or len(box)!=4 or not all(type(x) in (int,float) and math.isfinite(x) for x in box):
            raise ValueError('bbox must be four finite pixel coordinates')
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=size[0] and 0<=y1<y2<=size[1]):raise ValueError('Out-of-image or inverted bbox')
    return value


def prepare():
    from PIL import Image
    selection=[]
    for i,fid in enumerate(FRAMES):
        image=ROOT/'data/loco_test30/images'/(fid+'.jpg')
        with Image.open(image) as rgb:size=list(rgb.size)
        selection.append({'dataset':'loco_bbox4','index':i,'frame_id':fid,'source_frame_id':fid,
            'image_path':str(image),'image_sha256':sha256(image.read_bytes()),'image_size':size,'subset20':False})
    immutable_json(WORK/'inputs.json',selection)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    write_json(PUBLIC/'inputs.json',[{**f,'image_path':str(Path(f['image_path']).relative_to(ROOT))} for f in selection])
    shutil.copyfile(INSTRUCTION,PUBLIC/'prompt.txt')
    for f in selection:
        fid=f['frame_id'];source=ROOT/'analiza rezultatov vmesna predstavitev/industrija LOCO/01_vlm'/('loco_test30--'+fid)/'florence/r1'
        for folder in ('originals','florence','new_predictions','responses/'+fid):(PUBLIC/folder).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(f['image_path'],PUBLIC/'originals'/(fid+'.jpg'))
        shutil.copyfile(source/'od.predictions.jpg',PUBLIC/'florence'/(fid+'.od.predictions.jpg'))
        shutil.copyfile(source/'od.raw.json',PUBLIC/'responses'/fid/'florence.od.raw.json')
    print('Prepared exactly four RGB images; Florence r1 OD copied, no Florence inference.')


def export():
    from PIL import Image,ImageDraw
    rows=[]
    for frame in read_json(WORK/'inputs.json'):
        fid=frame['frame_id']
        for model,task in (('qwen3-vl','description'),('nvila','rich')):
            path=WORK/model/'r1/loco_bbox4'/task/f"frame-{frame['index']+1:06d}.json"
            row=read_json(path);folder=PUBLIC/'responses'/fid
            shutil.copyfile(path,folder/(model+'.raw.json'))
            # Intentionally may be invalid JSON: preserve the complete emitted text.
            (folder/(model+'.response.json')).write_text(row.get('content',''),encoding='utf-8')
            accepted=row.get('parsed',{'objects':[]}) if row['status']=='ok' else {'objects':[]}
            extraction='native JSON' if row['status']=='ok' else None
            error=row.get('error')
            text=row.get('content','').strip()
            # A single complete markdown fence is a transport wrapper, not a
            # coordinate correction. Never repair/truncate the JSON inside it.
            if row['status']!='ok' and text.startswith(('```json\n','```\n')) and text.endswith('\n```') and row.get('output_tokens',4096)<4096:
                try:
                    body=text.split('\n',1)[1][:-3]
                    accepted=validate_bbox(strict_json(body),frame['image_size'])
                    extraction='verbatim JSON inside one complete markdown fence'
                    error=None
                except ValueError as exc:error=str(exc)
            state='invalid' if extraction is None else ('ok' if accepted['objects'] else 'empty')
            write_json(folder/(model+'.parsed.json'),{'status':state,'objects':accepted['objects'],
                'error':error,'native_format_error':row.get('error'),'extraction':extraction,'coordinate_space':'original image pixels',
                'note':'Schema/bounds validity is not localization accuracy. Invalid output draws no boxes.'})
            target=PUBLIC/'new_predictions'/(fid+'.'+model+'.jpg')
            objects=accepted['objects'];native={'<BBOX>':{'bboxes':[o['bbox'] for o in objects],
                'labels':[str(o['id'])+' '+o['label'] for o in objects]}}
            draw_boxes(frame['image_path'],native,'<BBOX>',target)
            if state!='ok':
                with Image.open(target) as rgb:
                    ImageDraw.Draw(rgb).text((8,8),model+': '+state.upper()+' - NO BOXES DRAWN',fill='red',stroke_width=2,stroke_fill='white')
                    rgb.save(target)
            rows.append({'frame_id':fid,'model':model,'status':state,'accepted_boxes':len(objects),
                'call_id':row['call_id'],'error':error,'extraction':extraction,'image':str(target.relative_to(PUBLIC)),
                'raw':str((folder/(model+'.raw.json')).relative_to(PUBLIC)),
                'response':str((folder/(model+'.response.json')).relative_to(PUBLIC))})
    write_json(PUBLIC/'results.json',{'new_inferences':len(rows),'warmup_generations':0,'repeats':1,
        'florence_inferences':0,'llm_calls':0,'manual_coordinate_changes':False,'results':rows})
    body='<h1>Primerjava bbox: štiri slike</h1><p>Osem novih inferenc, brez ponovitev. Florence OD r1 je prejšnji rezultat. Neveljaven/prazen izhod pomeni prikaz brez okvirjev; veljavnost ni dokaz pravilne lokalizacije.</p>'
    for fid in FRAMES:
        body+=f'<h2>{fid}</h2><p><a href="originals/{fid}.jpg">Original</a></p><div class="row">'
        for model in ('florence','qwen3-vl','nvila'):
            if model=='florence':image='florence/'+fid+'.od.predictions.jpg';note='existing OD r1';link='responses/'+fid+'/florence.od.raw.json'
            else:
                item=next(r for r in rows if r['frame_id']==fid and r['model']==model)
                image=item['image'];note=item['status']+' / '+str(item['accepted_boxes'])+' bbox';link=item['response']
            body+=f'<section><h3>{model}: {html.escape(note)}</h3><a href="{image}"><img src="{image}"></a><p><a href="{link}">Celoten originalni odgovor</a></p></section>'
        body+='</div>'
    (PUBLIC/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Bbox comparison</title><style>body{font-family:Arial;margin:20px}.row{display:flex;gap:12px}section{width:33%}img{width:100%}</style>'+body)
    (PUBLIC/'README.md').write_text('# Prostorska lokalizacija: 4 slike × 2 modela\n\n'
        'Natanko osem novih generacij, brez ogrevalnih generacij, ponovitev, LLM ali PDF. '
        'Ista navodila zahtevajo originalne slikovne pike in objekte id/label/bbox. '
        'T=0.3, seed=42, top-p=0.8, top-k=20, repetition-penalty=1.05, max-new-tokens=4096. '
        'Obstoječa lokalna Qwen BF16 in NVILA FP16 okolja; Florence OD r1 samo kopiran.\n\n'
        '`new_predictions/` vsebuje osem novih prikazov. Pri neveljavnem/praznem rezultatu '
        'je prikazan original z izrecnim napisom NO BOXES DRAWN; noben okvir ni izmišljen. '
        '`florence/` vsebuje štiri stare OD prikaze, `originals/` štiri originale. '
        '`responses/` ohrani celotno emitirano besedilo v .response.json (tudi neveljavno), '
        'raw metapodatke, čas/VRAM/tokene in sprejete rezultate. Koordinate niso ročno '
        'popravljene, preračunane ali omejene na rob slike. Celoten JSON v enem '
        'zaključenem markdown bloku se dobesedno prebere, brez spreminjanja njegove vsebine. '
        'Izvirni ovoj in formatna napaka ostaneta v raw/response datotekah. '
        'Ni kakovostnih metrik ali GT.\n\n'
        'Odprite index.html. Ponovljiv ukaz: `bash scripts/bbox_comparison.sh`; '
        'shranjeni položaji se preskočijo, tudi če je izhod neveljaven.\n')
    for model in ('qwen3-vl','nvila'):
        source=WORK/model/'r1'
        for name in ('manifest.json','environments','source-snapshots'):
            path=source/name;target=PUBLIC/'provenance'/model/name
            if path.is_dir():shutil.copytree(path,target,dirs_exist_ok=True)
            elif path.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    if (WORK/'environments').exists():shutil.copytree(WORK/'environments',PUBLIC/'provenance/qwen3-vl/environments',dirs_exist_ok=True)
    for path in (WORK/'logs').glob('*.log'):
        write_json(PUBLIC/'provenance'/(path.stem+'.log.json'),{'source':str(path.relative_to(ROOT)),'stdout':path.read_text(errors='replace')})
    print('Saved eight model outputs/images; statuses:',[(r['model'],r['frame_id'],r['status']) for r in rows])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','export'))
    args=parser.parse_args();(prepare if args.phase=='prepare' else export)()
