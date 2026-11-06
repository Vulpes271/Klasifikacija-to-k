"""Full-sequence AMCL A/B/C using the existing local Florence/Qwen/SAM pipeline."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import yaml

from .box_segmentation import digest
from .lilocbench_data import prepare_data,extract_rgb
from .localization_pilot import (ROOT,write,pose_table,interpolate,docker_command,
                                mask_layers,project_scan,transform_tree,evaluate)
from .localization_comparison import ground_truth,load_estimates,differences


def read(path):return json.loads(Path(path).read_text())


def visited_cells(gt,valid,size,origin):
    return {tuple(row) for row in np.floor((gt[valid,:2]-origin)/size).astype(int)}


def cell_membership(gt,valid,cells,size,origin):
    out=np.zeros(len(gt),bool)
    out[valid]=[tuple(row) in cells for row in np.floor((gt[valid,:2]-origin)/size).astype(int)]
    return out


def load_full_scans(seq,pilot):
    values=np.loadtxt(seq/'laser_scan_front/ranges.txt');times=values[:,0];ranges=values[:,1:]
    if np.any(np.diff(times)<=0):raise ValueError('Non-monotonic LiDAR times')
    laser=yaml.safe_load((seq/'laser_scan_front/laser_specs.yaml').read_text())
    ot,op,_=pose_table(seq/'dingo_velocity_controller/odom.txt')
    # Preserve ALL scans. At recording boundaries only, use the nearest wheel-odom
    # sample for at most pose_max_gap; never extrapolate GT for evaluation.
    odom,valid=interpolate(ot,op,times,pilot['pose_max_gap'])
    odom_end,valid_end=interpolate(ot,op,times+laser['scan_time'],pilot['pose_max_gap'])
    edge_counts={}
    for name,query,array,good in [('odom',times,odom,valid),('odom_end',times+laser['scan_time'],odom_end,valid_end)]:
        for index in np.flatnonzero(~good):
            nearest=int(np.argmin(abs(ot-query[index])))
            if abs(ot[nearest]-query[index])>pilot['pose_max_gap']:raise ValueError('Wheel odometry missing beyond allowed gap')
            array[index]=op[nearest];array[index,2]=(array[index,2]+np.pi)%(2*np.pi)-np.pi
        edge_counts[name]=int((~good).sum())
    gt,valid_gt=ground_truth(seq/'gt_poses.txt',times,pilot['pose_max_gap'])
    first=np.flatnonzero(valid_gt)
    if not len(first):raise ValueError('No ground truth for initialization')
    # Use first available synchronized GT once, even if one leading scan precedes GT.
    initial=gt[first[0]].tolist()
    return dict(times=times,ranges=ranges,odom=odom,odom_end=odom_end,gt=gt,has_gt=valid_gt,
                laser=laser,initial=initial,initial_gt_index=int(first[0]),odom_edge_counts=edge_counts)


def prepare(cfg,pilot):
    data=ROOT/cfg['data'];cache=ROOT/cfg['cache'];output=ROOT/cfg['output']
    cache.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    stored=ROOT/pilot['stored_results'];sequences={}
    for seq in ['static_0','lt_changes_0']:
        sequences[seq]=load_full_scans(prepare_data(data,seq,stored),pilot)
    metadata=yaml.safe_load(next((data/'map').glob('*.yaml')).read_text())
    image=np.array(Image.open(data/'map'/Path(metadata['image']).name).convert('L'))
    probability=image/255. if metadata['negate'] else 1-image/255.
    occupancy=np.full(image.shape,-1,dtype=np.int8);occupancy[probability>metadata['occupied_thresh']]=100;occupancy[probability<metadata['free_thresh']]=0
    origin=np.array(metadata['origin'][:2]);size=cfg['common_cell_size_m']
    shared=visited_cells(sequences['static_0']['gt'],sequences['static_0']['has_gt'],size,origin)&visited_cells(sequences['lt_changes_0']['gt'],sequences['lt_changes_0']['has_gt'],size,origin)
    protocol={'configuration':cfg,'amcl':pilot['amcl'],'seed':pilot['seed'],
              'whole_recordings':{seq:{'scans':len(d['times']),'start':float(d['times'][0]),'end':float(d['times'][-1]),
                                      'initial_gt_index':d['initial_gt_index'],'odom_edge_counts':d['odom_edge_counts']} for seq,d in sequences.items()},
              'common_cells':sorted([[int(v) for v in c] for c in shared]),'common_cell_origin':origin.tolist(),
              'common_policy':'intersection of 0.5-m axis-aligned cells visited by GT scan poses; fixed before localization; A/C retain their own timestamps and dwell distribution',
              'mask_policy':'nearest individual RGB within pilot 50-ms midpoint window; no temporal voting; retain all scans',
              'selected_before_full_runs':True}
    path=output/'protocol.json'
    if path.exists() and read(path)!=protocol:raise ValueError('Full-run protocol changed; use a new output/cache')
    write(path,protocol)
    for seq,d in sequences.items():
        replay=output/('reference-replay' if seq=='static_0' else 'changed-replay');replay.mkdir(exist_ok=True)
        np.savez_compressed(replay/'evaluation.npz',gt=d['gt'],has_gt=d['has_gt'],
                            common=cell_membership(d['gt'],d['has_gt'],shared,size,origin))
        if not (replay/'scans.npz').exists():
            np.savez_compressed(replay/'scans.npz',times=d['times'],ranges=d['ranges'],odom=d['odom'],odom_end=d['odom_end'],
                                removed=np.zeros(d['ranges'].shape,bool),occupancy=np.flipud(occupancy))
        transform=next(r for r in yaml.safe_load((data/seq/'transformations.yaml').read_text()) if r['parent']=='base_link' and r['child']=='scan_front_link')
        write(replay/'input.json',{'laser':d['laser'],'map':{**metadata,'width':image.shape[1],'height':image.shape[0]},
                                'scan_transform':transform,'initial_pose':d['initial'],'seed':pilot['seed']})
        (replay/'amcl.yaml').write_text(yaml.safe_dump({'amcl':{'ros__parameters':pilot['amcl']}}))
    return sequences


def run_amcl(cfg,pilot,branch):
    output=ROOT/cfg['output'];replay=output/('reference-replay' if branch=='A' else 'changed-replay');dest=output/branch
    if (dest/'poses.json').exists():return
    command=docker_command(pilot['ros_image']);command[3:3]=['--network','none','-e','ROS_DOMAIN_ID=91','-e','ROS_LOCALHOST_ONLY=1']
    command+=['bash','-c','source /opt/ros/jazzy/setup.bash && exec python3 -m llm_object_classifier.amcl_replay "$@"','full',
              '--input',str(replay/'input.json'),'--branch','B' if branch=='C' else 'A','--output',str(dest)]
    subprocess.run(command,check=True)


def select_rgb(cfg,pilot):
    cache=ROOT/cfg['cache'];plan=cache/'rgb-plan.json'
    if plan.exists():return read(plan)
    source=ROOT/pilot['stored_results'];pairs=read(source/'pairs.json')['pairs']
    old=[p['b'] for p in pairs if p['b']['sequence']=='lt_changes_0']
    scans=np.load(ROOT/cfg['output']/'changed-replay/scans.npz');times=scans['times']
    targets=np.arange(times[0],times[-1]+.000001,1/cfg['rgb_hz'])
    chosen=[];fetch=[];used=set()
    for t in targets:
        candidates=[f for f in old if abs(f['timestamp']-t)<=cfg['cached_rgb_max_offset_s'] and f['frame_id'] not in used]
        if candidates:
            f=min(candidates,key=lambda f:abs(f['timestamp']-t));used.add(f['frame_id'])
            chosen.append({'target':float(t),'timestamp':f['timestamp'],'frame_id':f['frame_id'],'path':str(source/'images'/(f['frame_id']+'.png')),
                           'image_sha256':f['sha256'],'origin':'stored-lilocbench'})
        else:fetch.append(float(t))
    if fetch:
        paths=extract_rgb(ROOT/cfg['data'],'lt_changes_0',fetch)
        inputs=cache/'rgb-missing';inputs.mkdir(exist_ok=True)
        for t,path in zip(fetch,paths):
            sha=digest(path);fid='image-'+sha[:16];target=inputs/(fid+'.png')
            if not target.exists():shutil.copyfile(path,target)
            chosen.append({'target':t,'timestamp':float(path.stem),'frame_id':fid,'path':str(path),'image_sha256':sha,'origin':'new-pipeline'})
    chosen.sort(key=lambda f:f['timestamp'])
    write(plan,chosen);print(f'RGB plan: {len(chosen)} images, reused {len(chosen)-len(fetch)}, missing {len(fetch)}',flush=True)
    return chosen


def infer(cfg,pilot):
    cache=ROOT/cfg['cache'];plan=select_rgb(cfg,pilot)
    if any(f['origin']=='new-pipeline' for f in plan):
        completed=cache/'inference';manifest=completed/'manifest.json'
        if manifest.exists() and all((completed/'responses/qwen'/(f['frame_id']+'.json')).exists() for f in read(manifest)['inputs']):
            # Reuse completed inference offline, including terminal skips/errors.
            # The standard pipeline verifies immutable config, runner and RGB hashes.
            from .pipeline import load_settings,prepare_run
            prepare_run(load_settings(ROOT/cfg['pipeline_config']),completed,cache/'rgb-missing')
            for frame in read(manifest)['inputs']:
                if digest(completed/frame['frozen'])!=frame['input_sha256']:raise ValueError('Changed frozen inference input')
            print('Inference cache complete: 0 new Florence/LLM calls',flush=True)
            return
        subprocess.run(['python3','-m','llm_object_classifier.pipeline','--config',str(ROOT/cfg['pipeline_config']),
                        '--input',str(cache/'rgb-missing'),'--output',str(cache/'inference')],check=True,cwd=ROOT)



def build_masks(cfg,pilot):
    """Adapt existing pipeline outputs to the stored-region segmentation interface."""
    from .llm_checks import validate_response
    data=ROOT/cfg['data'];cache=ROOT/cfg['cache'];source=ROOT/pilot['stored_results'];plan=read(cache/'rgb-plan.json')
    canonical=pilot['classification_model'];reports=[ROOT/pilot['existing_masks']/'segmentation.json',
        *sorted((data/'mask-cache').rglob('segmentation.json')),*sorted((cache/'masks').glob('segmentation.json'))]
    existing={}
    for path in reports:
        if path.exists():
            for frame in read(path)['frames']:existing.setdefault(frame['frame_id'],(path.parent,frame))
    bundle=cache/'segmentation-source';bundle.mkdir(exist_ok=True);(bundle/'images').mkdir(exist_ok=True);(bundle/'frozen').mkdir(exist_ok=True)
    inputs=[];rows=[];status=[]
    for item in plan:
        fid=item['frame_id'];old=item['origin']=='stored-lilocbench';folder=source if old else cache/'inference'
        frozen=folder/'frozen'/(fid+'.json');doc=read(frozen)
        raw_path=folder/'responses'/(canonical if old else 'qwen')/(fid+'.json')
        raw=read(raw_path) if raw_path.exists() else {};check=validate_response(raw,doc['parsed']['objects'])
        valid=doc['status']=='ok' and check['schema_valid_response']
        status.append({**item,'classification_valid':bool(valid),'florence_status':doc['status'],
                       'llm_status':raw.get('status','missing'),'llm_errors':check['errors'],
                       'frozen_sha256':digest(frozen),'response_sha256':digest(raw_path) if raw_path.exists() else None})
        if doc['status']!='ok' or fid in existing:continue
        image=Path(item['path'])
        if digest(image)!=item['image_sha256']:raise ValueError('Changed selected RGB')
        shutil.copyfile(image,bundle/'images'/(fid+'.png'));shutil.copyfile(frozen,bundle/'frozen'/(fid+'.json'))
        inputs.append({'frame_id':fid,'frozen':f'frozen/{fid}.json','input_sha256':digest(frozen),'image_sha256':item['image_sha256']})
        decisions={i['id']:i for i in check['usable_items']}
        for obj in doc['parsed']['objects']:
            decision=decisions.get(obj['id'],{})
            rows.append({'frame_id':fid,'region_id':obj['id'],'florence_label':obj['label'],'bbox_xyxy':json.dumps(obj['bbox_xyxy']),
                         'input_sha256':digest(frozen),canonical+'_classification':decision.get('classification',''),
                         canonical+'_reason':decision.get('reason',''),canonical+'_whole_valid':str(check['schema_valid_response']),
                         canonical+'_error':'; '.join(check['errors'])})
    if inputs:
        write(bundle/'manifest.json',{'protocol':'full-local-cached-region-adapter-v1','config':{'models':[{'name':canonical}]},'inputs':inputs})
        fields=['frame_id','region_id','florence_label','bbox_xyxy','input_sha256']+[canonical+'_'+key for key in ['classification','reason','whole_valid','error']]
        with (bundle/'region-classes.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        command=docker_command(pilot['sam_image']);command[3:3]=['--gpus','all','--network','none']
        command+=['python','-m','llm_object_classifier.pipeline','--stage','segment','--config',str(ROOT/cfg['pipeline_config']),
                  '--results-from',str(bundle),'--frames',*[i['frame_id'] for i in inputs],'--output',str(cache/'masks')]
        subprocess.run(command,check=True)
        for frame in read(cache/'masks/segmentation.json')['frames']:existing[frame['frame_id']]=(cache/'masks',frame)
    for item in status:
        if item['frame_id'] in existing:
            folder,frame=existing[item['frame_id']];item['mask_report']=str(folder/'segmentation.json')
    write(cache/'mask-status.json',status)
    return status


def filter_scans(cfg,pilot):
    cache=ROOT/cfg['cache'];data=ROOT/cfg['data'];output=ROOT/cfg['output'];replay=output/'changed-replay'
    status=build_masks(cfg,pilot);arrays=dict(np.load(replay/'scans.npz'));times=arrays['times'];ranges=arrays['ranges']
    seq=data/'lt_changes_0';ot,op,_=pose_table(seq/'dingo_velocity_controller/odom.txt')
    records=yaml.safe_load((seq/'transformations.yaml').read_text());lookup=transform_tree(records)
    base_scan=lookup('base_link','scan_front_link');base_camera=lookup('base_link','camera_front_color_optical_frame')
    intrinsics=yaml.safe_load((seq/'camera_front/color/intrinsics.yaml').read_text());laser=read(replay/'input.json')['laser']
    candidates=[item for item in status if item['classification_valid'] and item.get('mask_report')]
    image_times=np.array([item['timestamp'] for item in candidates]);layers={};report_frames={};removed=np.zeros(ranges.shape,bool);valid=np.zeros(len(times),bool)
    associations=np.full(len(times),-1,dtype=int)
    for i,t in enumerate(times):
        if not len(image_times):continue
        nearest=int(np.argmin(abs(image_times-(t+laser['scan_time']/2))))
        if abs(image_times[nearest]-(t+laser['scan_time']/2))>pilot['mask_max_dt']:continue
        item=candidates[nearest];fid=item['frame_id'];report=Path(item['mask_report'])
        if fid not in layers:
            if report not in report_frames:report_frames[report]={f['frame_id']:f for f in read(report)['frames']}
            frame=report_frames[report][fid]
            layers[fid]=mask_layers(report.parent,frame,pilot['classification_model'])
        uv,visible=project_scan(ranges[i],t,item['timestamp'],laser,base_scan,base_camera,intrinsics,ot,op,pilot['pose_max_gap'])
        pixels=np.floor(uv[visible]).astype(int);indices=np.flatnonzero(visible)
        removed[i,indices]=layers[fid][pixels[:,1],pixels[:,0]];valid[i]=True;associations[i]=nearest
    arrays['removed']=removed;np.savez_compressed(replay/'scans.npz',**arrays)
    np.savez_compressed(replay/'mask-coverage.npz',valid=valid,association=associations)
    write(output/'filter-summary.json',{'selected_rgb':len(status),'valid_classified_rgb':len(candidates),
          'reused_rgb':sum(i['origin']=='stored-lilocbench' for i in status),'new_rgb':sum(i['origin']=='new-pipeline' for i in status),
          'mask_valid_scans':int(valid.sum()),'mask_valid_scans_percent':float(valid.mean()*100),
          'scans_with_removed_hits':int(removed.any(axis=1).sum()),'removed_hits':int(removed.sum()),
          'candidate_frames':[i['frame_id'] for i in candidates]})
    print(f'Filter: {valid.sum()}/{len(times)} time-valid masks; {removed.sum()} invalidated hits',flush=True)



def full_delta(c,other):
    return {'position_below_10cm_percentage_points':c['position_below_10cm_percent']-other['position_below_10cm_percent'],
            'yaw_below_4deg_percentage_points':c['yaw_below_4deg_percent']-other['yaw_below_4deg_percent'],
            'joint_success_percentage_points':c['success_percent']-other['success_percent'],
            'position_rmse_cm':c['position_rmse_cm']-other['position_rmse_cm'],
            'yaw_rmse_deg':c['yaw_rmse_deg']-other['yaw_rmse_deg']}


def report(cfg,pilot):
    output=ROOT/cfg['output'];cache=ROOT/cfg['cache'];data=ROOT/cfg['data'];protocol=read(output/'protocol.json')
    branches={};whole={};common={};successes={};errors={}
    for branch in 'ABC':
        replay=output/('reference-replay' if branch=='A' else 'changed-replay')
        arrays=np.load(replay/'scans.npz');evaluation=np.load(replay/'evaluation.npz');times=arrays['times']
        est=load_estimates(output/branch/'poses.json',times)
        gt=evaluation['gt'];has_gt=evaluation['has_gt'];shared=evaluation['common']&has_gt
        whole[branch],pos,yaw,successes[branch]=evaluate(gt,est,has_gt,data/'reference-code/evaluate.py')
        whole[branch]['missing_estimates_total']=int((~np.isfinite(est).all(axis=1)).sum())
        common[branch],_,_,_=evaluate(gt,est,shared,data/'reference-code/evaluate.py')
        common[branch]['times_without_gt']=int((~has_gt).sum())
        common[branch]['outside_common_cells']=int((has_gt&~shared).sum())
        branches[branch]={'times':times,'gt':gt,'has_gt':has_gt,'common':shared,'est':est};errors[branch]=(pos,yaw)
    if not np.array_equal(branches['B']['times'],branches['C']['times']):raise ValueError('B/C evaluation times differ')
    arrays=np.load(output/'changed-replay/scans.npz');drop=arrays['removed'];first=np.flatnonzero(drop.any(axis=1))
    prefix=int(first[0]) if len(first) else len(drop)
    if not np.array_equal(branches['B']['est'][:prefix],branches['C']['est'][:prefix],equal_nan=True):
        raise ValueError('B/C differ before first removed hit; shared initialization/seed check failed')
    laser=read(output/'changed-replay/input.json')['laser'];ranges=arrays['ranges']
    finite=np.isfinite(ranges)&(ranges>laser['range_min'])&(ranges<laser['range_max'])
    coverage=np.load(output/'changed-replay/mask-coverage.npz')['valid'];filter_summary=read(output/'filter-summary.json')
    del filter_summary['candidate_frames']
    filter_summary['new_inference_summary']=read(cache/'inference/summary.json')
    filter_summary['local_model']='qwen3.5:27b-q4_K_M'
    filter_summary['model_digest']=read(cache/'inference/model-metadata.json')['digest']
    filter_summary.update(finite_hits=int(finite.sum()),removed_hits_percent=100*float(drop.sum()/finite.sum()),
                          masked_scan_removed_hits_percent=100*float(drop.sum()/finite[coverage].sum()) if coverage.any() else None)
    paper_comparison={}
    for branch in 'ABC':
        baseline=cfg['paper_table_III']['static_0' if branch=='A' else 'lt_changes_0']
        paper_comparison[branch]={'paper':baseline,
            'position_difference_percentage_points':whole[branch]['position_below_10cm_percent']-baseline['position_below_10cm_percent'],
            'yaw_difference_percentage_points':whole[branch]['yaw_below_4deg_percent']-baseline['yaw_below_4deg_percent']}
    metrics={'method':'ROS2 AMCL 1.3.12, original pilot parameters, seed 42, exactly one full run A/B/C',
             'whole_sequences':whole,'C_minus_B_whole':full_delta(whole['C'],whole['B']),
             'common_visited_cells':{'size_m':cfg['common_cell_size_m'],'count':len(protocol['common_cells']),
                 'A':common['A'],'B':common['B'],'C':common['C'],
                 'C_minus_A_approximate':full_delta(common['C'],common['A'])},
             'filter':filter_summary,'B_C_identical_prefix_scans':prefix,'paper_table_III_comparison':paper_comparison,
             'thresholds':'strict <10 cm; strict absolute yaw <4 deg; joint is logical AND; GT-valid scan times in denominator; missing estimates fail',
             'paper_comparability':cfg['paper_table_III']['note'],
             'spatial_comparability':'C-A only on predeclared intersection of GT-visited 0.5-m cells, with different timestamps and dwell distributions; descriptive, not a paired effect',
             'initialization':protocol['whole_recordings'],
             'download_bytes_total_cache':sum(r['bytes'] for r in read(data/'http-cache/downloads.json'))}
    write(output/'metrics.json',metrics)
    with (output/'poses.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['branch','timestamp','has_gt','in_common_cells','estimated','gt_x','gt_y','gt_yaw','x','y','yaw','position_error_m','yaw_error_deg','joint_success','mask_valid'])
        for branch in 'ABC':
            d=branches[branch]
            for i,t in enumerate(d['times']):
                values=[branch,f'{t:.9f}',bool(d['has_gt'][i]),bool(d['common'][i]),bool(np.isfinite(d['est'][i]).all()),
                        *d['gt'][i].tolist(),*d['est'][i].tolist(),float(errors[branch][0][i]),float(errors[branch][1][i]),
                        bool(successes[branch][i]),bool(coverage[i]) if branch!='A' else False]
                writer.writerow(['' if isinstance(v,float) and not math.isfinite(v) else v for v in values])
    plot_full(output,data,branches,successes)
    paths=[ROOT/'configs/lilocbench-full.json',ROOT/cfg['pilot_config'],ROOT/cfg['pipeline_config'],Path(__file__),
           Path(__file__).with_name('amcl_replay.py'),Path(__file__).with_name('amcl_seed.c'),
           cache/'rgb-plan.json',cache/'mask-status.json',output/'protocol.json',data/'map/map_office.png',data/'map/map_office.yaml']
    for sequence in ['static_0','lt_changes_0']:
        paths += [data/sequence/name for name in ['gt_poses.txt','laser_scan_front/ranges.txt','dingo_velocity_controller/odom.txt','transformations.yaml']]
    for branch in 'ABC':paths += [output/branch/'poses.json',output/branch/'runtime.json']
    write(output/'provenance.json',{'sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths},'seed':42,
                                  'local_inference_run':str((cache/'inference/run.json').relative_to(ROOT))})
    publish_report(cfg,pilot,metrics)
    print(json.dumps(metrics,indent=2))
    return metrics


def plot_full(output,data,branches,successes,*,bounds=None,titles=None,filename="trajectories.png",heading=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    metadata=yaml.safe_load(next((data/'map').glob('*.yaml')).read_text())
    image=np.array(Image.open(data/'map'/Path(metadata['image']).name).convert('L'))
    height,width=image.shape;res=metadata['resolution'];ox,oy,_=metadata['origin'];extent=[ox,ox+width*res,oy,oy+height*res]
    probability=image/255. if metadata['negate'] else 1-image/255.
    known=(probability>metadata['occupied_thresh'])|(probability<metadata['free_thresh'])
    rows,cols=np.where(known)
    low=np.array([ox+cols.min()*res,oy+(height-1-rows.max())*res])-1
    high=np.array([ox+(cols.max()+1)*res,oy+(height-rows.min())*res])+1
    # Include all estimates as well: no failed/localizer-diverged part is clipped away.
    for d in branches.values():
        for poses in [d['gt'],d['est']]:
            valid=np.isfinite(poses).all(axis=1)
            if valid.any():low=np.minimum(low,poses[valid,:2].min(axis=0)-.5);high=np.maximum(high,poses[valid,:2].max(axis=0)+.5)
    fig,axes=plt.subplots(1,3,figsize=(19,8),sharex=True,sharey=True)
    if bounds is not None:low=np.array(bounds[:2]);high=np.array(bounds[2:])
    titles=titles or {'A':'A — static_0, original','B':'B — lt_changes_0, original','C':'C — same lt_changes_0, movable filter'}
    for ax,branch in zip(axes,'ABC'):
        d=branches[branch];gt=d['gt'];valid=d['has_gt'];segments=np.stack([gt[:-1,:2],gt[1:,:2]],axis=1)
        ax.imshow(image,cmap='gray',vmin=0,vmax=255,extent=extent,origin='upper')
        ax.add_collection(LineCollection(segments[valid[:-1]&valid[1:]],colors=np.where(successes[branch][1:][valid[:-1]&valid[1:]],'#159447','#df3025'),linewidths=2.6,zorder=3))
        ax.plot(d['est'][:,0],d['est'][:,1],color='#1858c9',lw=.65,zorder=4)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='x [m]',ylabel='y [m]');ax.set_aspect('equal');ax.set_title(titles[branch],fontsize=10)
        ax.legend(handles=[Line2D([],[],color='#159447',lw=3,label='GT: <10 cm AND <4°'),
                           Line2D([],[],color='#df3025',lw=3,label='GT: failure / missing estimate'),
                           Line2D([],[],color='#1858c9',lw=1,label='AMCL estimate')],fontsize=7,loc='lower right')
    fig.suptitle(heading or 'Full recordings — one AMCL run each, seed 42; no trajectory alignment')
    fig.tight_layout();fig.savefig(output/filename,dpi=150);plt.close(fig)


def publish_report(cfg,pilot,metrics):
    source=ROOT/cfg['output'];public=ROOT/'docs/lilocbench-full';public.mkdir(exist_ok=True)
    shutil.copyfile(source/'trajectories.png',public/'trajectories.png');shutil.copyfile(source/'metrics.json',public/'metrics.json')
    shutil.copyfile(source/'protocol.json',public/'protocol.json')
    whole=metrics['whole_sequences'];shared=metrics['common_visited_cells'];f=metrics['filter']
    lines=['# Celotni vožnji LILocBench: AMCL A / B / C','',
           'En celoten zagon A (`static_0`, original), B (`lt_changes_0`, original) in C (isti skeni kot B, filter `movable`). Iste nastavitve pilota, seme 42; brez glasovanja, tuninga ali poravnave trajektorij.','',
           '![Celotni zemljevid in tri barvne trajektorije](trajectories.png)','',
           'Prikazan je celoten poznani uradni zemljevid z vsemi trajektorijami; odrezan je samo neznani rob slike. Zelena/rdeča GT pot kaže skupni uspeh/neuspeh; modra je ocena AMCL.','',
           '## Celotni vožnji','',
           '| Zagon | Vsi skeni | Ocenljive poze | Položaj <10 cm [%] | Yaw <4° [%] | Oba praga [%] | Uspešne poze | RMSE [cm / °] | Manjkajoče ocene | Brez GT |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in 'ABC':
        s=whole[b];n=metrics['initialization']['static_0' if b=='A' else 'lt_changes_0']['scans']
        lines.append(f'| {b} | {n} | {s["evaluable_poses"]} | {s["position_below_10cm_percent"]:.2f} | {s["yaw_below_4deg_percent"]:.2f} | {s["success_percent"]:.2f} | {s["successful_poses"]} | {s["position_rmse_cm"]:.3f} / {s["yaw_rmse_deg"]:.3f} | {s["missing_estimates"]} | {s["times_without_gt"]} |')
    def delta_row(name,d):return f'| {name} | {d["position_below_10cm_percentage_points"]:+.2f} | {d["yaw_below_4deg_percentage_points"]:+.2f} | {d["joint_success_percentage_points"]:+.2f} |'
    lines += ['','| Primerjava | Položaj <10 cm [odst. točke] | Yaw <4° [odst. točke] | Oba praga [odst. točke] |','|---|---:|---:|---:|',
              delta_row('**C–B: celotna vožnja, učinek filtra**',metrics['C_minus_B_whole']),
              '', '## C–A samo na skupno obiskanih območjih','',
              f'Pred zagoni je iz GT določen presek {shared["count"]} celic velikosti {shared["size_m"]} m. V obeh vožnjah se ocenjujejo samo GT poze v tem preseku. Poti in časovne porazdelitve ostajajo različne, zato je C–A približna opisna referenca.','',
              '| Zagon, skupne celice | Ocenljive | Položaj <10 cm [%] | Yaw <4° [%] | Oba praga [%] | RMSE [cm / °] |',
              '|---|---:|---:|---:|---:|---:|']
    for b in ['A','C']:
        s=shared[b];lines.append(f'| {b} | {s["evaluable_poses"]} | {s["position_below_10cm_percent"]:.2f} | {s["yaw_below_4deg_percent"]:.2f} | {s["success_percent"]:.2f} | {s["position_rmse_cm"]:.3f} / {s["yaw_rmse_deg"]:.3f} |')
    lines += ['','| Primerjava | Položaj <10 cm [odst. točke] | Yaw <4° [odst. točke] | Oba praga [odst. točke] |','|---|---:|---:|---:|',
              delta_row('**C–A: samo skupne celice**',shared['C_minus_A_approximate']),
              '', '## Pokritost in lokalni cache','',
              f'Izbranih je {f["selected_rgb"]} RGB slik pri približno 1 Hz: ponovno uporabljenih {f["reused_rgb"]}, lokalno na novo obdelanih {f["new_rgb"]}. Veljavno klasifikacijo in segmentacijo ima {f["valid_classified_rgb"]} izbranih slik.',
              f'Časovno veljavno masko ima **{f["mask_valid_scans"]}/6540 skenov ({f["mask_valid_scans_percent"]:.2f} %)**. Filter odstrani {f["removed_hits"]} zadetkov ({f["removed_hits_percent"]:.3f} % vseh veljavnih zadetkov; {f["masked_scan_removed_hits_percent"]:.2f} % na skenih z masko).',
              'C uporablja najbližjo posamezno RGB sliko v pilotnem oknu ±50 ms od sredine skena, obstoječo projekcijo s kalibracijo in kompenzacijo z odometrijo. Maske niso razširjene na celoten sekundni interval. Zunaj veljavne povezave, vidnega polja ali pri `uncertain/fixed` ostanejo meritve. Odstranjeni zadetki so NaN. Vsi skeni ostanejo v zaporedju.',
              '', 'Neveljavni ali kontekstno preskočeni Florence/LLM izhodi se ne ponavljajo in ne štejejo kot veljavna maska. Nastavitve modelov/prompti niso spremenjeni. SAM je isti SAM 2.1 Hiera Small. Podatki, uteži, maske, surovi odgovori in veliki izhodi ostajajo lokalno.',
              '', '## Primerjava s tabelo III','',
              '| Zagon | Naš položaj <10 cm [%] | Članek AMCL [%] | Naš yaw <4° [%] | Članek AMCL [%] |','|---|---:|---:|---:|---:|']
    for b in 'ABC':
        s=whole[b];baseline=metrics['paper_table_III_comparison'][b]['paper']
        lines.append(f'| {b} | {s["position_below_10cm_percent"]:.2f} | {baseline["position_below_10cm_percent"]:.2f} | {s["yaw_below_4deg_percent"]:.2f} | {baseline["yaw_below_4deg_percent"]:.2f} |')
    lines += ['','Članek: statična sekvenca 100,00 ± 0 % in 96,78 ± 0,20 %; dolgoročne spremembe 82,89 ± 0,79 % in 84,01 ± 0,85 %. Za C je ista nefiltrirana AMCL vrstica članka samo referenca, članek našega filtra ne testira.',
              'To ni neposredna reprodukcija tabele III: članek poroča povprečje desetih zagonov; tu je en zagon ROS2 AMCL 1.3.12 s pilotnimi nastavitvami. Ocenjujemo na časih vseh skenov, z manjkajočo oceno kot neuspehom. Ločenih deležev iz članka ni mogoče enačiti z njunim skupnim AND deležem.',
              '', '## Ponovitev','', '```bash','python3 -m llm_object_classifier.localization_full --config configs/lilocbench-full.json','```','',
              'Ukaz ponovno uporabi cache in že zaključene zagone. Za sveže enkratne A/B/C uporabi novo izhodno mapo v kopiji konfiguracije; lokalni RGB/inferenčni/segmentacijski cache se ohrani. Posamezni koraki: `--stage prepare`, `rgb`, `infer`, `filter`, `A`, `B`, `C`, `report`.',
              'Preverjeno z 41 regresijskimi testi. Prva časovno ustrezna GT poza inicializira vsako vožnjo; A začne en sken pred prvo interpolabilno GT pozo. B/C imata isto inicializacijo in odometrijo. Poznejši GT se uporablja samo za ocenjevanje, skupno prostorsko območje in prikaz. Na robovih posnetkov je za nekaj skenov uporabljena najbližja odometrija največ 150 ms stran; GT se ne ekstrapolira.',
              '', '[Konfiguracija](../../configs/lilocbench-full.json), [metrike](metrics.json), [vnaprej določen protokol in skupne celice](protocol.json). Lokalno: `results/lilocbench-full/poses.csv`, `protocol.json`, `provenance.json`, dnevniki A/B/C in `data/lilocbench-full/` za cache.',
              '', 'Viri: [LILocBench, podatki in uradni zemljevid](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/), [članek, tabela III](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/trekel2025iros.pdf), [uradni evaluator](https://github.com/PRBonn/LILocBench). Zemljevid: Trekel et al. (2025), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); sestavljena slika pod isto licenco.']
    (public/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/lilocbench-full.json')
    p.add_argument('--stage',choices=['prepare','rgb','infer','A','B','C','filter','report','all'],default='all')
    args=p.parse_args();cfg=read(args.config);pilot=read(ROOT/cfg['pilot_config'])
    if cfg['seed']!=42 or pilot['seed']!=42:raise ValueError('Full run retains pilot seed 42')
    if args.stage=='prepare':prepare(cfg,pilot)
    elif args.stage=='rgb':select_rgb(cfg,pilot)
    elif args.stage=='infer':infer(cfg,pilot)
    elif args.stage in ['A','B','C']:run_amcl(cfg,pilot,args.stage)
    elif args.stage=='filter':filter_scans(cfg,pilot)
    elif args.stage=='report':report(cfg,pilot)
    else:
        prepare(cfg,pilot);select_rgb(cfg,pilot);infer(cfg,pilot);filter_scans(cfg,pilot)
        for branch in 'ABC':run_amcl(cfg,pilot,branch)
        report(cfg,pilot)


if __name__=='__main__':main()
