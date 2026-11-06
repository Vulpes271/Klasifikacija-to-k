"""Three-way AMCL comparison: approximate static reference A, changed original B, filtered C."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
import yaml

from .box_segmentation import digest
from .lilocbench_data import prepare_data,extract_rgb
from .localization_pilot import (ROOT,write,pose_table,interpolate,evaluate,docker_command,
                                  transform_tree,scan_points,mask_layers)


def spatial_mask(poses,bounds):
    """Half-open rectangles assign the shared office/corridor boundary exactly once."""
    x0,y0,x1,y1=bounds
    return np.isfinite(poses).all(axis=1)&(poses[:,0]>=x0)&(poses[:,0]<x1)&(poses[:,1]>=y0)&(poses[:,1]<y1)


def ground_truth(path,times,max_gap):
    stamps,poses,raw=pose_table(path);gt,valid=interpolate(stamps,poses,times,max_gap)
    if valid.any():gt[valid,2]=Slerp(stamps,Rotation.from_quat(raw[:,4:8]))(times[valid]).as_euler('ZYX')[:,0]
    return gt,valid


def load_estimates(path,times):
    rows=json.loads(path.read_text())
    if len(rows)!=len(times) or any(abs(r['timestamp']-t)>1e-6 for r,t in zip(rows,times)):
        raise ValueError(f'Pose times do not match scans: {path}')
    return np.array([r['pose'] if r['pose'] is not None else [np.nan]*3 for r in rows])


def reference_run(cfg,pilot,data,output,map_replay):
    ref=cfg['reference'];seq=prepare_data(data,ref['sequence'],ROOT/pilot['stored_results'])
    times=[];scans=[]
    with (seq/'laser_scan_front/ranges.txt').open() as stream:
        for line in stream:
            t=float(line.split(maxsplit=1)[0])
            if ref['start']<=t<=ref['end']:
                values=np.fromstring(line,sep=' ');times.append(values[0]);scans.append(values[1:])
    times=np.asarray(times);ranges=np.asarray(scans)
    if not len(times) or np.any(np.diff(times)<=0):raise ValueError('Missing/non-monotonic reference scan interval')
    ot,op,_=pose_table(seq/'dingo_velocity_controller/odom.txt')
    laser=yaml.safe_load((seq/'laser_scan_front/laser_specs.yaml').read_text())
    odom,good=interpolate(ot,op,times,pilot['pose_max_gap'])
    odom_end,good_end=interpolate(ot,op,times+laser['scan_time'],pilot['pose_max_gap'])
    if not (good&good_end).all():raise ValueError('Reference odometry is incomplete')
    gt,valid=ground_truth(seq/'gt_poses.txt',times,pilot['pose_max_gap'])
    if not valid[0]:raise ValueError('Reference initialization lacks ground truth')
    transforms=yaml.safe_load((seq/'transformations.yaml').read_text())
    scan_transform=next(r for r in transforms if r['parent']=='base_link' and r['child']=='scan_front_link')
    replay=output/'reference-replay';replay.mkdir(exist_ok=True)
    baseline=json.loads((map_replay/'input.json').read_text())
    occupancy=np.load(map_replay/'scans.npz')['occupancy']
    np.savez_compressed(replay/'scans.npz',times=times,ranges=ranges,removed=np.zeros(ranges.shape,bool),
                        odom=odom,odom_end=odom_end,occupancy=occupancy)
    write(replay/'input.json',{'laser':laser,'map':baseline['map'],'scan_transform':scan_transform,
                              'initial_pose':gt[0].tolist(),'seed':pilot['seed']})
    (replay/'amcl.yaml').write_text(yaml.safe_dump({'amcl':{'ros__parameters':pilot['amcl']}}))
    identity=hashlib.sha256()
    for array in [times,ranges,odom,odom_end,occupancy]:identity.update(array.tobytes())
    for path in [replay/'input.json',replay/'amcl.yaml',Path(__file__).with_name('amcl_replay.py'),Path(__file__).with_name('amcl_seed.c')]:
        identity.update(path.read_bytes())
    signature={'sha256':identity.hexdigest()};sigpath=replay/'signature.json'
    if sigpath.exists() and json.loads(sigpath.read_text())!=signature:raise ValueError('Reference replay changed; use a new output')
    write(sigpath,signature)
    if not (output/'A/poses.json').exists():
        command=docker_command(pilot['ros_image']);command[3:3]=['--network','none','-e','ROS_DOMAIN_ID=91','-e','ROS_LOCALHOST_ONLY=1']
        command+=['bash','-c','source /opt/ros/jazzy/setup.bash && exec python3 -m llm_object_classifier.amcl_replay "$@"','pilot',
                  '--input',str(replay/'input.json'),'--branch','A','--output',str(output/'A')]
        subprocess.run(command,check=True)
    return {'times':times,'gt':gt,'has_gt':valid,'est':load_estimates(output/'A/poses.json',times),
            'mask_valid':np.zeros(len(times),bool),'sequence':ref['sequence']}


def differences(c,other):
    return {'success_percentage_points':c['success_percent']-other['success_percent'],
            'position_rmse_cm':c['position_rmse_cm']-other['position_rmse_cm'],
            'yaw_rmse_deg':c['yaw_rmse_deg']-other['yaw_rmse_deg']}


def make_figure(output,cfg,pilot,data,branches,success,common,reference_image,changed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    metadata=yaml.safe_load(next((data/'map').glob('*.yaml')).read_text())
    image=np.array(Image.open(data/'map'/Path(metadata['image']).name).convert('L'))
    h,w=image.shape;res=metadata['resolution'];ox,oy,_=metadata['origin']
    extent=[ox,ox+w*res,oy,oy+h*res];x0,y0,x1,y1=cfg['plot_bounds_xyxy']
    fig=plt.figure(figsize=(14,11));grid=fig.add_gridspec(2,3,height_ratios=[1.65,1])
    def background(ax):
        ax.imshow(image,cmap='gray',vmin=0,vmax=255,extent=extent,origin='upper')
        ax.set(xlim=(x0,x1),ylim=(y0,y1),xlabel='x [m]',ylabel='y [m]');ax.set_aspect('equal')
    titles={'A':'A: static_0 (approximate reference)','B':'B: lt_changes_0 original','C':'C: same lt_changes_0, movable filtered'}
    for j,b in enumerate('ABC'):
        ax=fig.add_subplot(grid[0,j]);background(ax);d=branches[b];gt=d['gt'];valid=d['has_gt']
        segments=np.stack([gt[:-1,:2],gt[1:,:2]],axis=1);in_region=common[b][:-1]&common[b][1:]
        ax.add_collection(LineCollection(segments[valid[:-1]&valid[1:]&~in_region],colors='#aaaaaa',linewidths=1))
        ax.add_collection(LineCollection(segments[in_region],colors=np.where(success[b][1:][in_region],'#159447','#df3025'),linewidths=3))
        ax.plot(d['est'][:,0],d['est'][:,1],color='#1858c9',lw=1,zorder=4)
        for section in cfg['sections']:
            bx,by,ex,ey=section['bounds_xyxy'];ax.add_patch(Rectangle((bx,by),ex-bx,ey-by,fill=False,ls=':',lw=.8,edgecolor='#666666'))
        ax.set_title(titles[b],fontsize=10)
        ax.legend(handles=[Line2D([],[],color='#159447',lw=3,label='GT success: <10 cm AND <4°'),
                           Line2D([],[],color='#df3025',lw=3,label='GT failure / missing estimate'),
                           Line2D([],[],color='#1858c9',label='AMCL estimate'),
                           Line2D([],[],color='#aaaaaa',label='GT outside evaluated rectangles')],fontsize=6.8,loc='lower right')
    ax=fig.add_subplot(grid[1,0]);ax.imshow(Image.open(reference_image));ax.axis('off');ax.set_title('A RGB: selected boxes absent',fontsize=11)
    frame=cfg['rgb_changed_frame'];source=ROOT/pilot['stored_results'];original=np.array(Image.open(source/'images'/(frame+'.png')).convert('RGB'))
    report_path=ROOT/pilot['existing_masks']/'segmentation.json';report=json.loads(report_path.read_text())
    mask_frame=next(f for f in report['frames'] if f['frame_id']==frame)
    mask=mask_layers(report_path.parent,mask_frame,pilot['classification_model'])
    colored=original.copy();colored[mask]=(colored[mask]*.5+np.array([255,50,20])*.5).astype(np.uint8)
    ax=fig.add_subplot(grid[1,1]);ax.imshow(colored);ax.axis('off');ax.set_title('B/C RGB: added boxes; red = C filter mask',fontsize=10)
    # Diagnostic projection uses GT placement, never the AMCL output.
    pair=next(p['b'] for p in json.loads((source/'pairs.json').read_text())['pairs'] if p['b']['frame_id']==frame)
    laser=json.loads((changed/'replay/input.json').read_text())['laser'];arrays=np.load(changed/'replay/scans.npz')
    i=int(np.argmin(abs(arrays['times']+laser['scan_time']/2-pair['timestamp'])))
    ranges=arrays['ranges'][i];removed=arrays['removed'][i]
    records=yaml.safe_load((data/'lt_changes_0/transformations.yaml').read_text());base_scan=transform_tree(records)('base_link','scan_front_link')
    finite=np.isfinite(ranges)&(ranges>laser['range_min'])&(ranges<laser['range_max'])
    points=scan_points(np.where(finite,ranges,0),laser,base_scan);pose=branches['C']['gt'][i];c=np.cos(pose[2]);s=np.sin(pose[2])
    xy=points[:,:2]@np.array([[c,s],[-s,c]])+pose[:2]
    ax=fig.add_subplot(grid[1,2]);background(ax)
    ax.scatter(xy[finite&~removed,0],xy[finite&~removed,1],s=3,c='#00a0c6',label='retained')
    ax.scatter(xy[removed,0],xy[removed,1],s=7,c='#e33320',label=f'removed ({removed.sum()})')
    ax.scatter(*pose[:2],s=20,c='#1858c9',marker='x')
    ax.legend(fontsize=7);ax.set_title('C LiDAR: diagnostic placement with GT',fontsize=10)
    fig.suptitle('One shared spatial section: corridor + office entry\nA is approximate; only C−B isolates the filter effect',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(output/'comparison.png',dpi=145);plt.close(fig)


def run(config):
    cfg=json.loads(config.read_text());pilot=json.loads((ROOT/cfg['pilot_config']).read_text())
    data=ROOT/cfg['data'];output=ROOT/cfg['output'];changed=ROOT/cfg['changed_results'];output.mkdir(parents=True,exist_ok=True)
    selection_path=output/'selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
        if selection['configuration']!=cfg:raise ValueError('Spatial/time selection changed; use a new output')
    else:
        selection={'selected_before_reference_run':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'B_C_preexisting':True,'basis':'GT trajectories and matched RGBs only; no static_0 AMCL metrics inspected','configuration':cfg}
        write(selection_path,selection)
    # Check the existing B/C pair and common AMCL configuration before reusing it.
    old_selection=json.loads((changed/'selection.json').read_text())
    if old_selection['config']!=pilot or (old_selection['start'],old_selection['end'])!=(cfg['changed']['start'],cfg['changed']['end']):
        raise ValueError('B/C source has different settings/interval')
    arrays=np.load(changed/'replay/scans.npz');times=arrays['times'];gt,has_gt=ground_truth(data/'lt_changes_0/gt_poses.txt',times,pilot['pose_max_gap'])
    poses=list(csv.DictReader((changed/'poses.csv').open()))
    mask_valid=np.array([r['mask_valid']=='True' for r in poses])
    if len(mask_valid)!=len(times) or any(abs(float(r['timestamp'])-t)>1e-6 for r,t in zip(poses,times)):raise ValueError('B/C mask timestamps mismatch')
    rgb=extract_rgb(data,cfg['reference']['sequence'],[cfg['rgb_reference_timestamp']])[0]
    branches={'B':{'times':times,'gt':gt,'has_gt':has_gt,'mask_valid':mask_valid,'sequence':'lt_changes_0',
                   'est':load_estimates(changed/'A/poses.json',times)},
              'C':{'times':times,'gt':gt,'has_gt':has_gt,'mask_valid':mask_valid,'sequence':'lt_changes_0',
                   'est':load_estimates(changed/'B/poses.json',times)}}
    expected=json.loads((changed/'replay/input.json').read_text())
    if expected['seed']!=pilot['seed'] or not np.allclose(expected['initial_pose'],gt[0],atol=1e-10):
        raise ValueError('B/C initialization/seed differs from the selected run')
    if yaml.safe_load((changed/'replay/amcl.yaml').read_text())['amcl']['ros__parameters']!=pilot['amcl']:
        raise ValueError('B/C AMCL parameters differ')
    first_removed=np.flatnonzero(arrays['removed'].any(axis=1))
    prefix=int(first_removed[0]) if len(first_removed) else len(times)
    if not np.array_equal(branches['B']['est'][:prefix],branches['C']['est'][:prefix],equal_nan=True):
        raise ValueError('B/C estimates differ before the first filtered scan')
    for old_branch in ['A','B']:
        runtime=json.loads((changed/old_branch/'runtime.json').read_text())
        if runtime['seed']!=pilot['seed'] or runtime['frames']!=len(times):raise ValueError('B/C replay runtime mismatch')
    branches['A']=reference_run(cfg,pilot,data,output,changed/'replay')
    evaluator=data/'reference-code/evaluate.py';results={};global_success={};common={}
    sections=[*cfg['sections'],{'name':'combined','label':'Skupni odsek skupaj'}]
    for section in sections:
        summaries={}
        for b,d in branches.items():
            region=np.zeros(len(d['times']),bool)
            for rectangle in (cfg['sections'] if section['name']=='combined' else [section]):region|=spatial_mask(d['gt'],rectangle['bounds_xyxy'])
            valid=d['has_gt']&region
            summary,pos,yaw,success=evaluate(d['gt'],d['est'],valid,evaluator)
            summary['times_without_gt']=int((~d['has_gt']).sum());summary['outside_section']=int((d['has_gt']&~region).sum())
            n=int(valid.sum())
            if not n:raise ValueError(f'No shared-section poses: {b}/{section["name"]}')
            summary['mask_valid_scans']=int((d['mask_valid']&valid).sum());summary['mask_valid_scans_percent']=100*summary['mask_valid_scans']/n
            summaries[b]=summary
            if section['name']=='combined':global_success[b]=success;common[b]=valid
        results[section['name']]={'label':section['label'],'A':summaries['A'],'B':summaries['B'],'C':summaries['C'],
                                  'C_minus_B':differences(summaries['C'],summaries['B']),
                                  'C_minus_A_approximate':differences(summaries['C'],summaries['A'])}
    metrics={'comparability':cfg['comparability'],'configuration':cfg,'sections':results,
             'B_C_same_scans_odometry_initialization_seed':True,'B_C_identical_prefix_scans':prefix,'B_C_source':str(changed.relative_to(ROOT)),
             'mask_policy':'single nearest RGB, max 50 ms; original cached Qwen movable classes; no temporal voting',
             'removed_hits_all_B_C':int(arrays['removed'].sum()),'mask_valid_scans_all_B_C':int(mask_valid.sum()),
             'total_scans':{b:len(d['times']) for b,d in branches.items()}}
    write(output/'metrics.json',metrics)
    make_figure(output,cfg,pilot,data,branches,global_success,common,rgb,changed)
    with (output/'poses.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['branch','sequence','timestamp','gt_x','gt_y','estimated_x','estimated_y','gt_yaw','estimated_yaw','in_common_section','success','mask_valid'])
        for b in 'ABC':
            d=branches[b]
            for i,t in enumerate(d['times']):writer.writerow([b,d['sequence'],f'{t:.9f}',*d['gt'][i,:2],*d['est'][i,:2],d['gt'][i,2],d['est'][i,2],common[b][i],global_success[b][i],d['mask_valid'][i]])
    paths=[config,Path(__file__),Path(__file__).with_name('amcl_replay.py'),Path(__file__).with_name('amcl_seed.c'),
           data/'map/map_office.yaml',data/'map/map_office.png',rgb,ROOT/pilot['stored_results']/'images'/(cfg['rgb_changed_frame']+'.png'),
           changed/'replay/scans.npz',changed/'replay/input.json',changed/'replay/amcl.yaml',changed/'A/poses.json',changed/'B/poses.json']
    paths+=[data/'static_0/transformations.yaml',data/'static_0/laser_scan_front/laser_specs.yaml',ROOT/cfg['pilot_config'],evaluator]+list((data/'static_0').glob('*.txt'))+[data/'static_0/laser_scan_front/ranges.txt',data/'static_0/dingo_velocity_controller/odom.txt']
    write(output/'provenance.json',{'sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths},'selection_record':selection})
    # Explicitly authorized small public result: table + one composite PNG, no raw dataset/cache.
    public=ROOT/'docs/lilocbench-abc';public.mkdir(exist_ok=True)
    shutil.copyfile(output/'comparison.png',public/'comparison.png');shutil.copyfile(output/'metrics.json',public/'metrics.json')
    lines=['# AMCL: A / B / C — prvi skupni prostorski odsek','',
           '**A je približna referenca.** Poti se razlikujeta; spodaj so ocenjene samo vnaprej določene iste pravokotne prostorske regije. '+cfg['comparability'],'',
           '**Predmeti:** '+cfg['selected_objects'],'',
           'A: `static_0` 1733376760–1733376800; B/C: `lt_changes_0` 1733414028–1733414058. Izbira temelji na GT poti in RGB, pred zagonom A. B/C sta nespremenjena prejšnja pilota. B = prejšnji A; C = prejšnji B.','',
           '![Primerjava A/B/C, RGB in odstranjenih zadetkov](comparison.png)','',
           '| Prostor | Zagon | Uspešne / ocenljive poze | Uspeh [%] | RMSE položaj [cm] | RMSE yaw [°] | Veljavna maska [%] |',
           '|---|---|---:|---:|---:|---:|---:|']
    for section in sections:
        r=results[section['name']]
        for b in 'ABC':
            s=r[b];lines.append(f'| {section["label"]} | {b} | {s["successful_poses"]}/{s["evaluable_poses"]} | {s["success_percent"]:.2f} | {s["position_rmse_cm"]:.3f} | {s["yaw_rmse_deg"]:.3f} | {s["mask_valid_scans_percent"]:.2f} |')
    lines+=['','| Prostor | Razlika | Uspeh [odstotne točke] | RMSE položaj [cm] | RMSE yaw [°] |','|---|---|---:|---:|---:|']
    for section in sections:
        r=results[section['name']]
        for name,title in [('C_minus_B','C–B: učinek filtra'),('C_minus_A_approximate','C–A: približna referenca')]:
            d=r[name];lines.append(f'| {section["label"]} | {title} | {d["success_percentage_points"]:+.2f} | {d["position_rmse_cm"]:+.3f} | {d["yaw_rmse_deg"]:+.3f} |')
    lines+=['','Uspeh zahteva hkrati položaj **<10 cm** in absolutni yaw **<4°**. Manjkajoča ocena ob GT šteje kot neuspeh. Sive poti so zunaj ocenjevanih regij. A ne uporablja mask; B prikazuje razpoložljivost istih časovno veljavnih mask kot C, vendar jih ne uporabi.',
            '', 'C–B je parjena primerjava istih skenov in časov. C–A je opisna razlika med različnima vožnjama; različna pot, trajanje in prisotni ljudje preprečujejo sklep, da je vsa razlika posledica izbranih škatel. Na sliki je zanesljivo vidna odsotnost/prisotnost škatel, ne potrjena nespremenjenost vseh drugih predmetov.',
            '', 'AMCL 1.3.12, isti uradni zemljevid, nastavitve in seme 42. Prva GT poza vsake vožnje služi inicializaciji, kasnejši GT samo oceni/prikazu. Brez poravnave trajektorij. B/C ohranita isto odometrijo in inicializacijo; odstranjenih je 789 zadetkov v celotnem intervalu B/C. Veljavna maska obstaja na 11/450 skenih. Rdeči LiDAR zadetki v spodnjem panelu so **diagnostično postavljeni z GT**, ne z AMCL oceno.',
            '', 'Ponovitev (cache se ponovno uporabi; za nov A spremeni `output` v kopiji konfiguracije):','',
            '```bash','python3 -m llm_object_classifier.localization_comparison --config configs/lilocbench-abc.json','```','',
            'Nastavitve in prostorske meje: [konfiguracija](../../configs/lilocbench-abc.json). [Strojno berljive metrike](metrics.json). Celotni lokalni izhodi: `results/lilocbench-abc/`; podatki/cache ostajajo lokalno.','',
            'Viri: [LILocBench](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/), [uradni evaluator](https://github.com/PRBonn/LILocBench). RGB in referenčni zemljevid: Trekel et al., LILocBench (2025), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); sestavljena slika je objavljena pod isto licenco.']
    (public/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,indent=2))
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/lilocbench-abc.json')
    args=p.parse_args();run(args.config.resolve())


if __name__=='__main__':main()
