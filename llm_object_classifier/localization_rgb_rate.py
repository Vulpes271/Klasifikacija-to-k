"""One local cached batch comparing unfiltered AMCL, Qwen ~1 Hz and Qwen ~5 Hz."""
import argparse
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from .box_segmentation import digest
from .lilocbench_data import extract_rgb
from .localization_pilot import ROOT,write,pose_table,project_scan,transform_tree,mask_layers,evaluate
from .localization_full import infer,build_masks,run_amcl,plot_full
from .localization_comparison import load_estimates
import yaml


def read(path):return json.loads(Path(path).read_text())


def added_grid_targets(start,end,hz,retained):
    """Retained images replace nearest grid slots; never discard an original frame."""
    grid=np.arange(start,end,1/hz)
    occupied={int(np.argmin(abs(grid-t))) for t in retained}
    return [float(t) for i,t in enumerate(grid) if i not in occupied]


def partition(gt,valid,boundary):
    return {'whole':np.ones(len(gt),bool),'room':valid&(gt[:,1]<boundary),
            'corridor':valid&(gt[:,1]>=boundary)}


def run(config):
    cfg=read(config);pilot=read(ROOT/cfg['pilot_config']);reference=ROOT/cfg['reference_section']
    output=ROOT/cfg['output'];cache=ROOT/cfg['cache'];data=ROOT/cfg['data']
    output.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    base_protocol=read(reference/'protocol.json');base_cfg=base_protocol['configuration']
    for key in ['sequence','start','end','plot_bounds_xyxy','seed']:
        if cfg[key]!=base_cfg[key]:raise ValueError(f'Cached section differs: {key}')
    if pilot['amcl']!=base_protocol['amcl'] or cfg['seed']!=42:raise ValueError('AMCL/seed differ from cached test')
    original=np.load(reference/'replay-A/scans.npz');one=np.load(reference/'replay-B/scans.npz')
    for key in ['times','ranges','odom','odom_end','occupancy']:
        if not np.array_equal(original[key],one[key],equal_nan=True):raise ValueError('Cached 1-Hz inputs differ')
    if digest(reference/'replay-A/input.json')!=digest(reference/'replay-B/input.json') or digest(reference/'replay-A/amcl.yaml')!=digest(reference/'replay-B/amcl.yaml'):
        raise ValueError('Cached initialization/parameters differ')
    full=read(ROOT/cfg['full_config']);full_source=ROOT/full['output'];all_eval=np.load(full_source/'changed-replay/evaluation.npz')
    all_times=np.load(full_source/'changed-replay/scans.npz')['times'];selected=(all_times>=cfg['start'])&(all_times<=cfg['end'])
    times=original['times'];gt=all_eval['gt'][selected];has_gt=all_eval['has_gt'][selected]
    if not np.array_equal(times,all_times[selected]):raise ValueError('Cached scan times differ')
    wanted=set(base_protocol['rgb_ids']);full_status=read(ROOT/cfg['full_cache']/'mask-status.json')
    retained=[item for item in full_status if item['frame_id'] in wanted]
    if len(retained)!=51 or len(retained)!=len(wanted):raise ValueError('Original 51 RGB frames are not all present')
    targets=added_grid_targets(cfg['start'],cfg['end'],cfg['rgb_hz'],[item['timestamp'] for item in retained])
    protocol={'configuration':cfg,'amcl':pilot['amcl'],'initialization_sha256':digest(reference/'replay-A/input.json'),
              'retained_rgb_ids':sorted(wanted),'additional_target_timestamps':targets,'scans':len(times),
              'partition':'GT y < -5.0 m: room; y >= -5.0 m: corridor, selected before 5-Hz results',
              'selected_before_dense_run':True,'baseline_reuse':'reference A = unfiltered; reference B = Qwen 1 Hz'}
    path=output/'protocol.json'
    if path.exists() and read(path)!=protocol:raise ValueError('Changed protocol; use a new output')
    if not path.exists():write(path,protocol)
    # Only extra frames enter the existing Florence/Qwen/SAM pipeline.
    plan_path=cache/'rgb-plan.json'
    if not plan_path.exists():
        images=extract_rgb(data,cfg['sequence'],targets);new=[];seen=set(wanted)
        folder=cache/'rgb-missing';folder.mkdir(exist_ok=True)
        cached_by_sha={item['image_sha256']:item for item in full_status}
        for target,image in zip(targets,images):
            sha=digest(image)
            if sha in cached_by_sha:
                item=dict(cached_by_sha[sha])
            else:
                fid='image-'+sha[:16];item={'target':target,'timestamp':float(image.stem),'frame_id':fid,
                    'path':str(image),'image_sha256':sha,'origin':'new-pipeline'}
                destination=folder/(fid+'.png')
                if not destination.exists():shutil.copyfile(image,destination)
            if item['frame_id'] in seen:continue
            seen.add(item['frame_id']);new.append(item)
        # Existing full-cache items are kept separately; no duplicate inference.
        write(cache/'extra-reused.json',[i for i in new if i['image_sha256'] in cached_by_sha])
        write(plan_path,[i for i in new if i['image_sha256'] not in cached_by_sha])
    work={**full,'data':cfg['data'],'cache':cfg['cache'],'output':cfg['output'],'pipeline_config':cfg['pipeline_config']}
    expected_digest=read(full_source/'metrics.json')['filter']['model_digest']
    if read(plan_path):
        infer(work,pilot)
        if read(cache/'inference/model-metadata.json')['digest']!=expected_digest:raise ValueError('Qwen weights differ from the cached 1-Hz test')
        new_status=build_masks(work,pilot)
    else:new_status=[]
    dense=sorted([*retained,*read(cache/'extra-reused.json'),*new_status],key=lambda i:i['timestamp'])
    if not wanted.issubset({i['frame_id'] for i in dense}):raise ValueError('Dense plan dropped original frames')
    write(output/'rgb-plan-5hz.json',dense)
    reports={};layers={}
    for item in dense:
        if not item['classification_valid'] or not item.get('mask_report'):continue
        report=Path(item['mask_report'])
        if report not in reports:reports[report]={f['frame_id']:f for f in read(report)['frames']}
        layers[item['frame_id']]=mask_layers(report.parent,reports[report][item['frame_id']],pilot['classification_model'])
    candidates=[i for i in dense if i['frame_id'] in layers];stamps=np.array([i['timestamp'] for i in candidates])
    baseline=read(reference/'replay-A/input.json');seq=data/cfg['sequence'];ot,op,_=pose_table(seq/'dingo_velocity_controller/odom.txt')
    lookup=transform_tree(yaml.safe_load((seq/'transformations.yaml').read_text()))
    base_scan=lookup('base_link','scan_front_link');base_camera=lookup('base_link','camera_front_color_optical_frame')
    intrinsic=yaml.safe_load((seq/'camera_front/color/intrinsics.yaml').read_text())
    removed=np.zeros(original['ranges'].shape,bool);coverage5=np.zeros(len(times),bool)
    for i,t in enumerate(times):
        if not len(stamps):continue
        nearest=int(np.argmin(abs(stamps-(t+baseline['laser']['scan_time']/2))))
        if abs(stamps[nearest]-(t+baseline['laser']['scan_time']/2))>pilot['mask_max_dt']:continue
        item=candidates[nearest];uv,visible=project_scan(original['ranges'][i],t,item['timestamp'],baseline['laser'],base_scan,base_camera,intrinsic,ot,op,pilot['pose_max_gap'])
        pixels=np.floor(uv[visible]).astype(int);indices=np.flatnonzero(visible)
        removed[i,indices]=layers[item['frame_id']][pixels[:,1],pixels[:,0]];coverage5[i]=True
    stamps1=np.array([i['timestamp'] for i in retained if i['classification_valid'] and i.get('mask_report')])
    coverage1=np.array([bool(len(stamps1) and abs(stamps1-(t+baseline['laser']['scan_time']/2)).min()<=pilot['mask_max_dt']) for t in times])
    previous=read(reference/'summary.json')['metrics']
    if coverage1.sum()!=previous['qwen']['mask_valid_scans'] or one['removed'].sum()!=previous['qwen']['removed_hits']:
        raise ValueError('Cached 1-Hz coverage/removal differs from published result')
    if np.any(coverage1&~coverage5):raise ValueError('Dense plan lost original temporal coverage')
    replay=output/'changed-replay';replay.mkdir(exist_ok=True)
    np.savez_compressed(replay/'scans.npz',**{**dict(original),'removed':removed})
    shutil.copyfile(reference/'replay-A/input.json',replay/'input.json');shutil.copyfile(reference/'replay-A/amcl.yaml',replay/'amcl.yaml')
    run_amcl(work,pilot,'C')  # only the missing dense variant runs
    estimates={'A':load_estimates(reference/'A/poses.json',times),'B':load_estimates(reference/'B/poses.json',times),
               'C':load_estimates(output/'C/poses.json',times)}
    first=np.flatnonzero(removed.any(axis=1));prefix=int(first[0]) if len(first) else len(times)
    if not np.array_equal(estimates['A'][:prefix],estimates['C'][:prefix],equal_nan=True):raise ValueError('Dense run differs before first filtered hit')
    coverages={'A':np.zeros(len(times),bool),'B':coverage1,'C':coverage5}
    removals={'A':np.zeros(removed.shape,bool),'B':one['removed'],'C':removed}
    regions=partition(gt,has_gt,cfg['room_below_y_m']);sections={};success={};branches={}
    for name,region in regions.items():
        results={}
        for branch in 'ABC':
            summary,_,_,ok=evaluate(gt,estimates[branch],has_gt&region,data/'reference-code/evaluate.py')
            summary['times_without_gt']=int((~has_gt&region).sum())
            n=int(region.sum());summary.update(scans=n,mask_valid_scans=int((coverages[branch]&region).sum()),
                mask_coverage_percent=100*float(coverages[branch][region].mean()) if n else 0,
                removed_hits=int(removals[branch][region].sum()))
            results[branch]=summary
            if name=='whole':success[branch]=ok
        sections[name]=results
    for branch,key in [('A','unfiltered'),('B','qwen')]:
        if sections['whole'][branch]['successful_poses']!=previous[key]['successful_poses']:raise ValueError('Reused success metric changed')
    for branch in 'ABC':branches[branch]={'times':times,'gt':gt,'has_gt':has_gt,'est':estimates[branch]}
    plot_full(output,data,branches,success,bounds=cfg['plot_bounds_xyxy'],
        titles={'A':'AMCL brez filtra','B':'Qwen ~1 Hz','C':'Qwen ~5 Hz'},filename='trajectories_qwen_1hz_5hz.png',
        heading='Same 51-second section — nearest single-image mask, ±50 ms, seed 42, no voting')
    metrics={'configuration':cfg,'sections':sections,'rgb_1hz':len(retained),'rgb_5hz':len(dense),
             'additional_rgb':len(dense)-len(retained),'new_inference_rgb':len(read(plan_path)),
             'valid_rgb_5hz':len(candidates),'model':cfg['qwen'],'qwen_digest':expected_digest,'retained_all_original_frames':True,
             'reused_unfiltered_and_1hz_amcl':True,'identical_prefix_dense_scans':prefix,
             'room_corridor_boundary_y_m':cfg['room_below_y_m'],
             'five_minus_one_success_percentage_points':{name:s['C']['success_percent']-s['B']['success_percent'] for name,s in sections.items()}}
    write(output/'metrics.json',metrics);write(output/'summary.json',metrics)
    identities={str(p.relative_to(ROOT)):digest(p) for p in [config,reference/'A/poses.json',reference/'B/poses.json',
        reference/'replay-A/input.json',reference/'replay-A/amcl.yaml',plan_path,output/'protocol.json',output/'C/poses.json']}
    for item in dense:
        identities[str(Path(item['path']).relative_to(ROOT))]=digest(item['path'])
        if item.get('mask_report'):
            report=Path(item['mask_report']);identities[str(report.relative_to(ROOT))]=digest(report)
    write(output/'provenance.json',{'sha256':identities,'all_original_images_retained':True})
    publish(output,cfg,metrics)
    print(json.dumps(metrics,indent=2));return metrics


def publish(output,cfg,m):
    public=ROOT/'docs/lilocbench-rgb-rate';public.mkdir(exist_ok=True)
    for name in ['metrics.json','protocol.json','trajectories_qwen_1hz_5hz.png']:shutil.copyfile(output/name,public/name)
    labels={'A':'AMCL brez filtra','B':'Qwen ~1 Hz','C':'Qwen ~5 Hz'};scope={'whole':'Celoten odsek','room':'Soba','corridor':'Hodnik'}
    lines=['# Qwen: približno 1 Hz proti 5 Hz — isti odsek','',
      'Odsek `lt_changes_0`, **1733414160–1733414211**, 765 nespremenjenih skenov. Model `qwen3.5:27b-q4_K_M`, isti Florence/SAM, prompti, projekcija, filter, ±50 ms, začetna poza, odometrija, AMCL parametri in seme 42. Posamezna slika, brez glasovanja in tuninga.','',
      '![AMCL brez filtra, Qwen 1 Hz in Qwen 5 Hz](trajectories_qwen_1hz_5hz.png)','',
      '| Območje | Različica | Uspešne / ocenljive | Uspeh [%] | Pokritost skenov z masko [%] | Odstranjeni zadetki |',
      '|---|---|---:|---:|---:|---:|']
    for name in ['whole','room','corridor']:
        for b in 'ABC':
            s=m['sections'][name][b];lines.append(f'| {scope[name]} | {labels[b]} | {s["successful_poses"]}/{s["evaluable_poses"]} | {s["success_percent"]:.2f} | {s["mask_coverage_percent"]:.2f} | {s["removed_hits"]} |')
    whole=m['sections']['whole'];gain=m['five_minus_one_success_percentage_points']['whole']
    baseline_gap=whole['C']['success_percent']-whole['A']['success_percent']
    lines+=['',f'**5 Hz minus 1 Hz:** celoten odsek {gain:+.2f} odstotne točke; soba {m["five_minus_one_success_percentage_points"]["room"]:+.2f}; hodnik {m["five_minus_one_success_percentage_points"]["corridor"]:+.2f}. V primerjavi z nefiltriranim AMCL je razlika pri 5 Hz {baseline_gap:+.2f} odstotne točke.',
      '',f'Vseh {m["rgb_1hz"]} izvirnih slik je ohranjenih. Pri 5 Hz je skupaj {m["rgb_5hz"]} slik, dodanih {m["additional_rgb"]}; z lokalnim pipelineom je obdelanih le {m["new_inference_rgb"]} manjkajočih. Stare slike nadomestijo najbližja mesta mreže 0,2 s, zato vzorčenje ni povsem enakomerno. Veljavno klasifikacijo in maske ima {m["valid_rgb_5hz"]} slik.',
      'Zaradi več slik se lahko spremeni izbrana najbližja maska; odstranjeni zadetki niso nujno nadmnožica zadetkov pri 1 Hz. Pokritost časovnih mask pa ohrani vse veljavne povezave pri 1 Hz.',
      '', 'Soba/hodnik sta pred rezultati razdeljena z mejo **GT y = −5,0 m**: soba y < −5,0; hodnik y ≥ −5,0. To je ponovljiva geometrijska delitev pri vratih, ne ročna oznaka predmetov. Vsak sken z GT pripada natanko enemu delu.',
      'Uspeh zahteva **<10 cm in <4° hkrati**. Manjkajoča poza pri veljavnem GT šteje kot neuspeh. Vsi skeni ostanejo; odstranitev pomeni NaN, ne največjega dosega. Neveljavne klasifikacije ne filtrirajo. GT samo inicializira prvo pozo, nato ocenjuje in riše; brez poravnave.',
      '', 'Nefiltrirani AMCL in Qwen 1 Hz sta preverjeno ponovno uporabljena iz predhodnega odsečnega testa. Izveden je samo en nov AMCL zagon pri 5 Hz. Modra: ocena AMCL; GT zelen pri skupnem uspehu, drugod rdeč. Isti izrez in merilo v vseh panelih.',
      '', 'Omejitev: en 51-sekundni odsek in eno seme; veljavnost klasifikacij ter redka/neenakomerna časovna pokritost vplivata na izid. Rezultat ni dokaz splošnega izboljšanja. Višja frekvenca ne spremeni napačne klasifikacije in ne uporablja glasovanja.',
      '', '```bash','python3 -m llm_object_classifier.localization_rgb_rate --config configs/lilocbench-rgb-rate.json','```','',
      '[Metrike](metrics.json), [protokol](protocol.json), [konfiguracija](../../configs/lilocbench-rgb-rate.json). Surovi izpisi/cache ostajajo lokalno v `data/lilocbench-rgb-rate/` in `results/lilocbench-rgb-rate/`.',
      '', 'Zemljevid: [LILocBench, Trekel et al.](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); sestavljena slika pod isto licenco.']
    (public/'README.md').write_text('\n'.join(lines)+'\n')
    vals='; '.join(f'{labels[b]}: uspeh {whole[b]["success_percent"]:.2f} %, maske {whole[b]["mask_coverage_percent"]:.2f} %, odstranjenih {whole[b]["removed_hits"]}' for b in 'ABC')
    regional='; '.join(f'{scope[n]} uspeh A/1Hz/5Hz: '+ '/'.join(f'{m["sections"][n][b]["success_percent"]:.2f} %' for b in 'ABC') for n in ['room','corridor'])
    handoff=f'Metoda: lt_changes_0 1733414160–1733414211, 765 skenov; Qwen lokalno, Florence/SAM cache, posamezna slika ±50 ms, ista inicializacija/AMCL/seme 42, brez glasovanja. {vals}. {regional}. Glavna ugotovitev: 5 Hz proti 1 Hz spremeni skupni uspeh za {gain:+.2f} odstotne točke, glede na nefiltrirani AMCL pa za {baseline_gap:+.2f}. Omejitev: en kratek odsek in eno seme, neenakomerne in ne vedno veljavne maske, brez dokaza splošne veljavnosti. Zemljevid: https://raw.githubusercontent.com/Vulpes271/Klasifikacija-to-k/lilocbench-full/docs/lilocbench-rgb-rate/trajectories_qwen_1hz_5hz.png'
    (public/'handoff.txt').write_text(handoff+'\n');(output/'handoff.txt').write_text(handoff+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/lilocbench-rgb-rate.json')
    args=p.parse_args();output=ROOT/read(args.config)['output']
    try:run(args.config.resolve())
    except Exception as exc:write(output/'batch-error.json',{'error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
