"""Optional offline postprocessing of stored Florence/LLM regions (no reclassification)."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def label_key(label):
    # Only remove grammatical articles. No synonyms, singularization or part/whole merging.
    return re.sub(r'^(?:a|an|the)\s+', '', label.strip().lower()).strip(' .')


def near_equal(a, b, min_iou=0.95, max_edge_fraction=0.02):
    """Both IoU and every edge must agree; containment alone is insufficient."""
    ix=max(0, min(a[2],b[2])-max(a[0],b[0]))
    iy=max(0, min(a[3],b[3])-max(a[1],b[1]))
    area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1])
    union=area_a+area_b-ix*iy
    if union<=0 or ix*iy/union<min_iou:return False
    limits=[max_edge_fraction*min(a[2]-a[0],b[2]-b[0]),
            max_edge_fraction*min(a[3]-a[1],b[3]-b[1])]*2
    return all(abs(x-y)<=limit for x,y,limit in zip(a,b,limits))


def filter_regions(objects, scene, decisions, min_iou=0.95, max_edge_fraction=0.02):
    """Complete-link grouping prevents chains of gradually drifting boxes.

    Identity evidence is deliberately narrow: same article-normalized phrase (or explicit singular box anaphora), explicitly
    present in the caption. The generated bbox observation is not independent evidence.
    All original records and per-model classes survive even when their classes disagree.
    """
    # A narrow caption-confirmed anaphora, never a global adjective/synonym removal.
    box_mentions=re.findall(r'\b(?:a|an|one)\s+(?:(?:large|white|black|cardboard)\s+)*box\b',scene,re.I)
    box_reference=len(box_mentions)==1 and bool(re.search(r'\bthe box\b',scene,re.I))
    def identity(label):
        key=label_key(label)
        if box_reference and key in ('box','cardboard box','large cardboard box'):return 'box'
        return key
    groups=[]
    for obj in objects:
        key=identity(obj['label'])
        supported=bool(key and re.search(r'(?<!\w)'+re.escape(key)+r'(?!\w)',scene,re.I))
        source={'object':obj,'llm':decisions[obj['id']]}
        match=None
        if supported:
            for group in groups:
                if key==identity(group['label']) and all(near_equal(obj['bbox_xyxy'],s['object']['bbox_xyxy'],min_iou,max_edge_fraction) for s in group['sources']):
                    match=group;break
        if match is None:
            groups.append({'region_id':obj['id'],'label':obj['label'],'bbox_xyxy':obj['bbox_xyxy'],
                           'source_ids':[obj['id']],'sources':[source],
                           'merge_evidence':None})
        else:
            match['source_ids'].append(obj['id']);match['sources'].append(source)
            match['merge_evidence']='same caption-supported identity or explicit box anaphora; IoU >= threshold; all edges agree pairwise'
    # A relationship is metadata only; the logo keeps its own region and mask.
    # Only the explicit singular box-with-logo construction is supported for now.
    evidence=re.search(r'\b(?:a |the )?(?:large )?(?:cardboard )?box with the logo\b[^!?\n]{0,160}?\bon it\b',scene,re.I)
    if evidence:
        for part in groups:
            if label_key(part['label'])!='logo':continue
            a=part['bbox_xyxy'];parents=[]
            for whole in groups:
                if label_key(whole['label']) not in ('box','cardboard box','large cardboard box'):continue
                b=whole['bbox_xyxy']
                if b[0]<=a[0] and b[1]<=a[1] and b[2]>=a[2] and b[3]>=a[3]:parents.append(whole)
            if len(parents)==1:
                part['part_of']={'region_id':parents[0]['region_id'],'source_ids':parents[0]['source_ids'],
                                 'description_evidence':evidence.group().strip()}
    return groups



def save_preview(image, mask, region, target, model='qwen35-27b-q4'):
    """Color by stored Qwen classes; conflicting or missing classes stay red."""
    import numpy as np
    from PIL import Image
    classes=[s['llm']['models'].get(model,{}).get('classification') for s in region['sources']]
    color=[30,220,60] if classes and all(c=='fixed' for c in classes) else [255,50,20]
    overlay=np.array(image)
    overlay[mask]=(overlay[mask]*0.5+np.array(color)*0.5).astype(np.uint8)
    Image.fromarray(overlay).save(target)


def run_segmentation(source, output, frame_ids, settings, root):
    """Read stored results only; write full-resolution independent binary PNG masks."""
    import numpy as np
    from PIL import Image
    source=Path(source).resolve();output=Path(output).resolve()
    if not frame_ids or len(frame_ids)!=len(set(frame_ids)):
        raise ValueError('Specify distinct --frames explicitly')
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be separate from stored input results')
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty output directory')
    checkpoint=(root/settings['checkpoint']).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(f'SAM 2 checkpoint missing: {checkpoint}; see docs/box-segmentation.md')
    manifest_path=source/'manifest.json';table_path=source/'region-classes.csv'
    manifest=json.loads(manifest_path.read_text());frames={f['frame_id']:f for f in manifest['inputs']}
    if any(fid not in frames for fid in frame_ids):raise ValueError('Unknown stored frame ID')
    with table_path.open(newline='',encoding='utf-8') as stream:table=list(csv.DictReader(stream))
    models=[m['name'] for m in manifest['config']['models']]
    prepared=[];inputs={str(manifest_path):digest(manifest_path),str(table_path):digest(table_path)}
    for fid in frame_ids:
        frame=frames[fid];frozen=(source/frame['frozen']).resolve()
        if not frozen.is_relative_to(source):raise ValueError('Frozen path outside source')
        if digest(frozen)!=frame['input_sha256']:raise ValueError(f'Changed frozen input: {fid}')
        doc=json.loads(frozen.read_text());objects=doc['parsed']['objects']
        if doc['status']!='ok':raise ValueError(f'Invalid stored Florence: {fid}')
        candidates=[p for p in (source/'images').glob(fid+'.*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}]
        if len(candidates)!=1:raise ValueError(f'Expected one original image: {fid}')
        image_path=candidates[0]
        if digest(image_path)!=frame['image_sha256']:raise ValueError(f'Changed original image: {fid}')
        image=Image.open(image_path).convert('RGB');w,h=image.size;decisions={}
        ids=[o['id'] for o in objects]
        if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate original IDs: {fid}')
        for obj in objects:
            box=obj['bbox_xyxy']
            if len(box)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in box) or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
                raise ValueError(f'Invalid original bbox: {fid}/{obj["id"]}')
            rows=[r for r in table if r['frame_id']==fid and r['region_id']==str(obj['id'])]
            if len(rows)!=1:raise ValueError(f'Expected one stored class row: {fid}/{obj["id"]}')
            row=rows[0]
            if row['input_sha256']!=frame['input_sha256'] or json.loads(row['bbox_xyxy'])!=box or row['florence_label']!=obj['label']:
                raise ValueError(f'Stored class row differs from frozen region: {fid}/{obj["id"]}')
            decisions[obj['id']]={'models':{model:{'classification':row[model+'_classification'] or None,
                'reason':row[model+'_reason'],'whole_valid':row[model+'_whole_valid'],
                'error':row[model+'_error']} for model in models},'csv_row':row}
        regions=filter_regions(objects,doc['parsed']['scene_description'],decisions,
                               settings['min_iou'],settings['max_edge_fraction'])
        prepared.append((fid,image_path,image,regions))
        inputs[str(frozen)]=digest(frozen);inputs[str(image_path)]=digest(image_path)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device='cuda' if torch.cuda.is_available() else 'cpu'
    predictor=SAM2ImagePredictor(build_sam2(settings['model_config'],str(checkpoint),device=device,
                                          apply_postprocessing=False))
    output.mkdir(parents=True,exist_ok=True)
    report={'protocol':'stored-box-segmentation-v1','source':str(source),'settings':settings,
            'model_sha256':digest(checkpoint),'implementation_sha256':digest(__file__),'device':device,'torch':torch.__version__,
            'inputs_sha256':inputs,'coordinate_system':'original image pixels, xyxy; PNG 0=background 255=foreground',
            'classification_policy':'all original classes preserved per source ID and model; missing classes remain null',
            'frames':[]}
    with torch.inference_mode():
        for fid,image_path,image,regions in prepared:
            predictor.set_image(np.array(image));folder=output/'masks'/fid;folder.mkdir(parents=True)
            preview=output/'previews'/fid;preview.mkdir(parents=True)
            for region in regions:
                masks,scores,_=predictor.predict(box=np.asarray(region['bbox_xyxy'],dtype=np.float32),multimask_output=False)
                mask=np.asarray(masks[0],dtype=bool)
                if mask.shape!=(image.height,image.width):raise ValueError('Mask is not in original coordinates')
                local=Path('masks')/fid/f'region-{region["region_id"]:04d}.png'
                Image.fromarray(mask.astype(np.uint8)*255).save(output/local)
                save_preview(image,mask,region,preview/f'region-{region["region_id"]:04d}.jpg',
                             settings.get('preview_model','qwen35-27b-q4'))
                region.update(mask_path=str(local),mask_sha256=digest(output/local),mask_area_pixels=int(mask.sum()),
                              sam_score=float(scores[0]),mask_status='ok' if mask.any() else 'empty')
            report['frames'].append({'frame_id':fid,'image_path':str(image_path),'image_size':list(image.size),
                                     'source_region_count':sum(len(r['source_ids']) for r in regions),'regions':regions})
            print(f'{fid}: {sum(len(r["source_ids"]) for r in regions)} original regions -> {len(regions)} masks',flush=True)
    # Only source integrity is checked here, never localization or classification correctness.
    if any(digest(p)!=value for p,value in inputs.items()):raise ValueError('Stored inputs changed during segmentation')
    (output/'segmentation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report
