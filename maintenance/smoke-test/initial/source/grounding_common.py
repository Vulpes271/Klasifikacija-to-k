"""Dobesedni prenos Florence regij in izris v izvornih slikovnih pikah."""
CAPTION='<MORE_DETAILED_CAPTION>'
GROUND='<CAPTION_TO_PHRASE_GROUNDING>'

def draw(image_path, regions, target):
    """Izriše vrnjene koordinate; celotne fraze ostanejo v JSON/HTML legendi."""
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


def freeze(path, value):
    """Atomsko shrani nov zapis; ob spremembi obstoječega zahteva novo izhodno mapo."""
    from .experiment import read_json, write_json
    from pathlib import Path
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError(f'Spremenjen podpis/vhod: {path}. Uporabi novo izhodno mapo.')
    else:write_json(path,value)


def weights_identity(folder):
    """Preveri dejanske lokalne datoteke uteži in procesorja, brez prenosa modela."""
    import hashlib
    from pathlib import Path
    from .llm_checks import fingerprint
    folder=Path(folder)
    if not (folder/'model.safetensors').is_file():raise ValueError(f'Manjkajo lokalne uteži Florence: {folder}')
    files={}
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or any(p.startswith('.') or p=='__pycache__' for p in path.relative_to(folder).parts) or path.suffix=='.pyc':continue
        with path.open('rb') as stream:files[str(path.relative_to(folder))]=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'sha256':fingerprint(files),'files':files}


def make_document(frame, caption, grounding):
    """Celoten opis in native fraze prenese dobesedno; regije niso potrjene instance."""
    regions=[]
    if grounding.get('status')=='ok':
        value=grounding['parsed'][GROUND]
        regions=[{'id':i,'label':label,'bbox_xyxy':box,'source_task':GROUND,'source_region_id':i}
                 for i,(label,box) in enumerate(zip(value['labels'],value['bboxes']),1)]
    objects=[{**r,'observation':f"Florence grounded caption phrase {r['label']!r} at bbox_xyxy {r['bbox_xyxy']} in original pixels. Attachment/contact evidence and physical-instance identity were not established.",
              'attachment_evidence':'not_visible','identifiable':True} for r in regions]
    return {'stage':1,'frame_id':frame['frame_id'],'model_name':'florence','model':'microsoft/Florence-2-large',
            'variant':{'id':'caption-grounding-v1'},'image_sha256':frame['image_sha256'],
            'status':'ok' if caption.get('status')==grounding.get('status')=='ok' else 'error',
            'parsed':{'scene_description':caption.get('parsed',{}).get(CAPTION,''),'objects':objects},
            'source_records':{'caption':f"native/{frame['frame_id']}/caption.json",'grounding':f"native/{frame['frame_id']}/grounding.json"},
            'adapter_notes':'Native phrase regions, not verified physical instances. No reference annotations supplied.'}
