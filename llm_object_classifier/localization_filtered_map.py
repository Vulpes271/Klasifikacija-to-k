"""Resumable paired AMCL experiment: filter only the mapping sequence."""
import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import yaml

from .box_segmentation import digest
from .lilocbench_data import prepare_data,extract_rgb
from .localization_pilot import (ROOT,write,pose_table,interpolate,transform_tree,
    project_scan,mask_layers,docker_command,evaluate)
from .localization_full import load_full_scans,infer,build_masks,read
from .localization_comparison import load_estimates
from .localization_rgb_rate import added_grid_targets


EXPECTED_QWEN='64ae5b1d651fb2709d5928ccb8dcfb4bb2cc9c2c8001ac34497e36a6cf905b59'


def immutable(path,value):
    if path.exists() and read(path)!=value:raise ValueError(f'Changed immutable protocol: {path}')
    if not path.exists():write(path,value)


def progress(cfg,stage,**values):
    write(ROOT/cfg['output']/'progress.json',{'stage':stage,'unix_time':time.time(),**values})
    print(json.dumps({'stage':stage,**values}),flush=True)


def prepare(cfg,pilot):
    output=ROOT/cfg['output'];cache=ROOT/cfg['cache'];data=ROOT/cfg['data']
    output.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    if len(cfg['seeds'])!=10 or len(set(cfg['seeds']))!=10:raise ValueError('Ten distinct paired seeds required')
    protocol={'configuration':cfg,'amcl':pilot['amcl'],'mask_max_dt':pilot['mask_max_dt'],
        'method':'Table III pose tracking, paired seeds; localization scans NEVER filtered; no alignment',
        'pose_covariance_diagonal':[.2**2,.2**2,float(np.radians(10)**2)],
        'selected_before_results':True,'mapping_only_source':'mapping',
        'trajectory_plot_seed':cfg['seeds'][0], 'std_ddof':0,
        'evaluation':'All scan timestamps; interpolate public GT with max gap 150 ms; missing estimates fail threshold denominator; no extrapolation/alignment',
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'llm_object_classifier/occupancy_rays.c',ROOT/cfg['pipeline_config']]}}
    immutable(output/'protocol.json',protocol)
    for sequence in ['mapping',*cfg['sequences']]:
        seq=prepare_data(data,sequence,ROOT/pilot['stored_results']);target=cache/sequence/'raw.npz'
        if not target.exists():
            d=load_full_scans(seq,pilot);target.parent.mkdir(exist_ok=True)
            np.savez_compressed(target,**{k:v for k,v in d.items() if k not in ['laser','initial','odom_edge_counts']})
            write(target.with_suffix('.json'),{k:d[k] for k in ['laser','initial','initial_gt_index','odom_edge_counts']})
        progress(cfg,'data_ready',sequence=sequence)


def select_mapping_rgb(cfg,pilot):
    cache=ROOT/cfg['cache'];plan=cache/'mapping/rgb-plan-all.json'
    if plan.exists():return read(plan)
    arrays=np.load(cache/'mapping/raw.npz');times=arrays['times']
    stored=ROOT/pilot['stored_results'];pairs=read(stored/'pairs.json')['pairs']
    old=[p['a'] for p in pairs if p['a']['sequence']=='mapping' and times[0]<=p['a']['timestamp']<=times[-1]]
    keep=[{'target':f['timestamp'],'timestamp':f['timestamp'],'frame_id':f['frame_id'],
        'path':str(stored/'images'/(f['frame_id']+'.png')),'image_sha256':f['sha256'],'origin':'stored-lilocbench'} for f in old]
    # Check actual legacy classifier inputs/options before reuse.
    from .pipeline import load_settings
    from .experiment import options
    settings=load_settings(ROOT/cfg['pipeline_config'])['llm'];opts=options(settings,{'temperature':settings['temperature'],'seed':settings['seed']})
    for item in keep:
        response=read(stored/'responses'/pilot['classification_model']/(item['frame_id']+'.json'))
        payload=response['signature']['payload']
        if (payload['options']!=opts or payload['model_digest']!=EXPECTED_QWEN or payload['format']!='json'
            or payload['think'] is not False or payload['messages'][0]['content']!=settings['system_file_text']):
            raise ValueError('Stored Qwen settings differ; cannot reuse mapping classifier')
        if digest(item['path'])!=item['image_sha256']:raise ValueError('Changed stored RGB')
    targets=added_grid_targets(times[0],times[-1],cfg['rgb_hz'],[f['timestamp'] for f in keep])
    # Persist before fetching. Fetch in chunks and checkpoint to avoid redoing work.
    selection={'start':float(times[0]),'end':float(times[-1]),'hz':cfg['rgb_hz'],'retained':keep,'added_targets':targets}
    immutable(cache/'mapping/rgb-selection.json',selection)
    index=read(ROOT/cfg['data']/'archive-index-mapping.json')
    names=[name for name in index if name.startswith('camera_front/color/images/') and name.endswith('.png')]
    stamps=np.array([float(Path(n).stem) for n in names]);rows=list(keep)
    for target in targets:
        name=names[int(np.argmin(abs(stamps-target)))];path=ROOT/cfg['data']/'mapping'/name
        rows.append({'target':target,'timestamp':float(path.stem),'frame_id':'pending-'+path.stem,
            'path':str(path),'origin':'new-pipeline'})
    unique={}
    for row in rows:unique.setdefault(row['timestamp'],row)
    rows=sorted(unique.values(),key=lambda r:r['timestamp']);write(plan,rows)
    return rows


def mapping_masks(cfg,pilot,preflight=False):
    cache=ROOT/cfg['cache'];rows=select_mapping_rgb(cfg,pilot)
    # A first 8-frame room segment is checked locally before any full batch.
    room=[r for r in rows if 1733375693<=r['timestamp']<1733375695][:8]
    chosen={r['frame_id'] for r in room};remaining=[r for r in rows if r['frame_id'] not in chosen]
    batches=[room]+[remaining[i:i+cfg['batch_size']] for i in range(0,len(remaining),cfg['batch_size'])]
    status=[]
    for number,batch in enumerate(batches):
        if preflight and number:break
        folder=cache/'mapping/batches'/f'{number:03d}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'rgb-plan.json').exists():batch=read(folder/'rgb-plan.json')
        else:
            missing=[r for r in batch if not Path(r['path']).exists()]
            if missing:extract_rgb(ROOT/cfg['data'],'mapping',[r['timestamp'] for r in missing])
            batch=[{**r,'image_sha256':digest(r['path']),
                'frame_id':r['frame_id'] if r['origin']=='stored-lilocbench' else 'image-'+digest(r['path'])[:16]} for r in batch]
            immutable(folder/'rgb-plan.json',batch)
        work={**cfg,'cache':str(folder),'output':str(ROOT/cfg['output'])}
        inputs=folder/'rgb-missing';inputs.mkdir(exist_ok=True)
        for item in batch:
            if item['origin']!='new-pipeline':continue
            dest=inputs/(item['frame_id']+'.png')
            if not dest.exists():shutil.copyfile(item['path'],dest)
        if any(i['origin']=='new-pipeline' for i in batch):
            infer(work,pilot)
            if read(folder/'inference/model-metadata.json')['digest']!=EXPECTED_QWEN:raise ValueError('Qwen digest changed')
        report=folder/'masks/segmentation.json'
        if (folder/'masks').exists() and not report.exists():
            # SAM writes its report only after completing a batch; regenerate only an interrupted SAM batch.
            shutil.rmtree(folder/'masks')
        summary=build_masks(work,pilot);status.extend(summary)
        write(cache/('mapping/mask-status-preflight.json' if preflight else 'mapping/mask-status.json'),status)
        progress(cfg,'mapping_models',batch=number+1,batches=len(batches),completed=len(status),total=len(rows),
            valid_classified=sum(i['classification_valid'] for i in status))
    return status


def project_mapping(cfg,pilot,status,preflight=False):
    cache=ROOT/cfg['cache'];dest=cache/'mapping'/('filter-preflight.npz' if preflight else 'filter.npz')
    if dest.exists():return dest
    arrays=np.load(cache/'mapping/raw.npz');seq=ROOT/cfg['data']/'mapping'
    laser=read(cache/'mapping/raw.json')['laser'];ot,op,_=pose_table(seq/'dingo_velocity_controller/odom.txt')
    tree=transform_tree(yaml.safe_load((seq/'transformations.yaml').read_text()))
    base_scan=tree('base_link','scan_front_link');base_camera=tree('base_link','camera_front_color_optical_frame')
    intrinsics=yaml.safe_load((seq/'camera_front/color/intrinsics.yaml').read_text())
    candidates=[s for s in status if s['classification_valid'] and s.get('mask_report')]
    stamps=np.array([s['timestamp'] for s in candidates]);reports={};layers={}
    times=arrays['times'];ranges=arrays['ranges'];removed=np.zeros(ranges.shape,bool);coverage=np.zeros(len(times),bool)
    association=np.full(len(times),-1,np.int32);projection_valid=np.zeros(len(times),np.int32)
    for i,t in enumerate(times):
        if not len(stamps):continue
        nearest=int(np.argmin(abs(stamps-(t+laser['scan_time']/2))))
        if abs(stamps[nearest]-(t+laser['scan_time']/2))>pilot['mask_max_dt']:continue
        item=candidates[nearest];fid=item['frame_id'];report=Path(item['mask_report'])
        if fid not in layers:
            if report not in reports:reports[report]={r['frame_id']:r for r in read(report)['frames']}
            # Bound resident masks to a handful; scans/images are chronological.
            layers={fid:mask_layers(report.parent,reports[report][fid],pilot['classification_model'])}
        uv,visible=project_scan(ranges[i],t,item['timestamp'],laser,base_scan,base_camera,intrinsics,ot,op,pilot['pose_max_gap'])
        pixels=np.floor(uv[visible]).astype(int);indices=np.flatnonzero(visible)
        removed[i,indices]=layers[fid][pixels[:,1],pixels[:,0]];coverage[i]=True;association[i]=nearest
        projection_valid[i]=int(visible.sum())
    np.savez_compressed(dest,removed=removed,coverage=coverage,association=association,projection_valid=projection_valid)
    write(dest.with_suffix('.json'),{'candidates':candidates,'status_count':len(status)})
    if preflight:
        checked=np.flatnonzero(coverage)[:10]
        if len(checked)<5 or not np.all(projection_valid[checked]>0):raise ValueError('Preflight has fewer than 5 valid RGB/scan projections')
        if np.any(removed&~(np.isfinite(ranges)&(ranges>laser['range_min'])&(ranges<laser['range_max']))):raise ValueError('Invalid range removed')
        write(ROOT/cfg['output']/'preflight.json',{'passed':True,'paired_scans':len(checked),
            'timestamps':times[checked].tolist(),'visible_projected_beams':projection_valid[checked].tolist(),
            'removed_hits':removed[checked].sum(axis=1).astype(int).tolist(),
            'max_mask_offset_s':float(max(abs(stamps[association[checked]]-(times[checked]+laser['scan_time']/2)))),
            'same_original_ranges':True,'removed_rays_skip_all_grid_updates':True})
    return dest


def ray_library(cache):
    lib=cache/'occupancy_rays.so';source=ROOT/'llm_object_classifier/occupancy_rays.c'
    subprocess.run(['gcc','-O3','-shared','-fPIC',str(source),'-o',str(lib),'-lm'],check=True)
    fun=ctypes.CDLL(str(lib)).integrate
    fun.argtypes=[np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_float,ctypes.c_float,ctypes.c_float]
    return fun


def beam_world(ranges,t,laser,base_scan,gt_times,gt_poses,max_gap):
    """One common, deskewed GT geometry for both maps (never localization GT)."""
    poses,ok=interpolate(gt_times,gt_poses,t+np.arange(len(ranges))*laser['time_increment'],max_gap)
    angles=laser['angle_min']+np.arange(len(ranges))*laser['angle_increment']
    good=ok&np.isfinite(ranges)&(ranges>laser['range_min'])&(ranges<laser['range_max'])
    safe=np.where(good,ranges,0.)
    local=np.column_stack([safe*np.cos(angles),safe*np.sin(angles),np.zeros(len(ranges)),np.ones(len(ranges))])@base_scan.T
    c=np.cos(poses[:,2]);s=np.sin(poses[:,2]);origin=np.column_stack([poses[:,0]+c*base_scan[0,3]-s*base_scan[1,3],poses[:,1]+s*base_scan[0,3]+c*base_scan[1,3]])
    end=np.column_stack([poses[:,0]+c*local[:,0]-s*local[:,1],poses[:,1]+s*local[:,0]+c*local[:,1]])
    return origin,end,good


def occupancy(log,known,meta):
    p=1/(1+np.exp(-log));out=np.full(log.shape,-1,np.int8)
    out[known.astype(bool)&(p>meta['occupied_thresh'])]=100
    out[known.astype(bool)&(p<meta['free_thresh'])]=0
    return out


def build_maps(cfg,pilot):
    cache=ROOT/cfg['cache'];output=ROOT/cfg['output'];done=output/'map-summary.json'
    if done.exists():return read(done)
    arrays=np.load(cache/'mapping/raw.npz');filtered=np.load(cache/'mapping/filter.npz');meta=cfg['map']
    seq=ROOT/cfg['data']/'mapping';gt_times,gt_poses,_=pose_table(seq/'gt_poses.txt')
    tree=transform_tree(yaml.safe_load((seq/'transformations.yaml').read_text()));base_scan=tree('base_link','scan_front_link')
    laser=read(cache/'mapping/raw.json')['laser'];shape=(meta['height'],meta['width']);fun=ray_library(cache)
    checkpoint=cache/'mapping/grid-checkpoint.npz'
    if checkpoint.exists():
        saved=np.load(checkpoint);logs=saved['logs'];known=saved['known'];start=int(saved['next']);beams=int(saved['valid_beams'])
    else:logs=np.zeros((2,*shape),np.float32);known=np.zeros((2,*shape),np.uint8);start=0;beams=0
    times=arrays['times'];ranges=arrays['ranges'];removed=filtered['removed']
    for i in range(start,len(times)):
        origins,ends,valid=beam_world(ranges[i],times[i],laser,base_scan,gt_times,gt_poses,pilot['pose_max_gap'])
        # Invalid interpolation never reaches C; both maps skip exactly the same missing geometry.
        origins=np.floor((np.nan_to_num(origins)-np.array(meta['origin'][:2]))/meta['resolution']).astype(np.int32)
        ends=np.floor((np.nan_to_num(ends)-np.array(meta['origin'][:2]))/meta['resolution']).astype(np.int32)
        beams+=int(valid.sum())
        for branch in range(2):
            good=np.ascontiguousarray(valid&(~removed[i] if branch else True),dtype=np.uint8)
            fun(logs[branch],known[branch],meta['width'],meta['height'],origins,ends,good,len(good),meta['log_hit'],meta['log_free'],meta['log_limit'])
        if (i+1)%500==0 or i+1==len(times):
            tmp=checkpoint.with_suffix('.part.npz');np.savez_compressed(tmp,logs=logs,known=known,next=i+1,valid_beams=beams);tmp.replace(checkpoint)
            progress(cfg,'map_integration',scans=i+1,total=len(times))
    grids=[occupancy(logs[b],known[b],meta) for b in range(2)]
    for name,grid in zip(['original','filtered'],grids):
        folder=output/'maps'/name;folder.mkdir(parents=True,exist_ok=True)
        image=np.full(shape,205,np.uint8);image[grid==0]=254;image[grid==100]=0
        Image.fromarray(np.flipud(image)).save(folder/'map.png');np.save(folder/'occupancy.npy',grid)
        (folder/'map.yaml').write_text(yaml.safe_dump({k:v for k,v in {**meta,'image':'map.png'}.items() if k in ['image','resolution','origin','occupied_thresh','free_thresh','negate']}))
    finite=np.isfinite(ranges)&(ranges>laser['range_min'])&(ranges<laser['range_max'])
    changed=grids[0]!=grids[1]
    summary={'mapping_scans':len(times),'valid_geometry_beams':beams,'rgb_frames':read(cache/'mapping/rgb-plan-all.json').__len__(),
        'mask_valid_scans':int(filtered['coverage'].sum()),'mask_coverage_percent':100*float(filtered['coverage'].mean()),
        'removed_hits':int(removed.sum()),'finite_hits':int(finite.sum()),'removed_hits_percent':100*float(removed.sum()/finite.sum()),
        'changed_cells':int(changed.sum()),'changed_observed_cells':int((known[0]!=known[1]).sum()),
        'occupied_to_unknown':int(((grids[0]==100)&(grids[1]==-1)).sum()),'occupied_to_free':int(((grids[0]==100)&(grids[1]==0)).sum()),
        'map_sha256':{b:digest(output/'maps'/b/'occupancy.npy') for b in ['original','filtered']}}
    write(done,summary);progress(cfg,'maps_ready',**summary);return summary


def prepare_replays(cfg,pilot):
    cache=ROOT/cfg['cache'];output=ROOT/cfg['output'];data=ROOT/cfg['data']
    for sequence in cfg['sequences']:
        arrays=np.load(cache/sequence/'raw.npz');info=read(cache/sequence/'raw.json')
        transform=next(r for r in yaml.safe_load((data/sequence/'transformations.yaml').read_text()) if r['parent']=='base_link' and r['child']=='scan_front_link')
        for seed in cfg['seeds']:
            for name in ['original','filtered']:
                replay=output/'replays'/sequence/f'seed-{seed}'/name;replay.mkdir(parents=True,exist_ok=True)
                if not (replay/'scans.npz').exists():
                    np.savez_compressed(replay/'scans.npz',**{k:arrays[k] for k in ['times','ranges','odom','odom_end']},
                        removed=np.zeros(arrays['ranges'].shape,bool),occupancy=np.load(output/'maps'/name/'occupancy.npy'))
                immutable(replay/'input.json',{'laser':info['laser'],'map':cfg['map'],'scan_transform':transform,'initial_pose':info['initial'],'seed':seed})
                params=yaml.safe_dump({'amcl':{'ros__parameters':pilot['amcl']}})
                path=replay/'amcl.yaml'
                if path.exists() and path.read_text()!=params:raise ValueError('AMCL parameters changed')
                path.write_text(params)
        progress(cfg,'replays_ready',sequence=sequence,scans=len(arrays['times']))


def run_pairs(cfg,pilot):
    output=ROOT/cfg['output']
    for sequence in cfg['sequences']:
        for seed in cfg['seeds']:
            for name in ['original','filtered']:
                replay=output/'replays'/sequence/f'seed-{seed}'/name;dest=output/'runs'/sequence/f'seed-{seed}'/name
                if (dest/'poses.json').exists():continue
                command=docker_command(pilot['ros_image']);command[3:3]=['--network','none','-e','ROS_DOMAIN_ID=91','-e','ROS_LOCALHOST_ONLY=1']
                command+=['bash','-c','source /opt/ros/jazzy/setup.bash && exec python3 -m llm_object_classifier.amcl_replay "$@"','filtered-map',
                    '--input',str(replay/'input.json'),'--branch','A','--output',str(dest)]
                progress(cfg,'amcl',sequence=sequence,seed=seed,map=name)
                subprocess.run(command,check=True)
    progress(cfg,'amcl_complete',runs=2*len(cfg['seeds'])*len(cfg['sequences']))


def aggregate(rows):
    keys=['position_rmse_cm','position_max_cm','position_below_5cm_percent','position_below_10cm_percent',
          'yaw_rmse_deg','yaw_max_deg','yaw_below_2deg_percent','yaw_below_4deg_percent','success_percent']
    return {k:{'mean':float(np.mean([r[k] for r in rows])),'std':float(np.std([r[k] for r in rows],ddof=0))} for k in keys}


def report(cfg,pilot):
    from .filtered_map_review import make_report
    return make_report(cfg,pilot)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/lilocbench-filtered-map.json')
    parser.add_argument('--stage',choices=['prepare','preflight','models','maps','amcl','report','all'],default='all')
    args=parser.parse_args();cfg=read(args.config);pilot=read(ROOT/cfg['pilot_config']);stage=args.stage
    try:
        if stage in ['prepare','all']:prepare(cfg,pilot)
        if stage in ['preflight','all']:
            rows=mapping_masks(cfg,pilot,True);project_mapping(cfg,pilot,rows,True)
        if stage in ['models','all']:mapping_masks(cfg,pilot)
        if stage in ['maps','all']:
            rows=read(ROOT/cfg['cache']/'mapping/mask-status.json');project_mapping(cfg,pilot,rows);build_maps(cfg,pilot)
        if stage in ['amcl','all']:prepare_replays(cfg,pilot);run_pairs(cfg,pilot)
        if stage in ['report','all']:report(cfg,pilot)
    except Exception as exc:
        write(ROOT/cfg['output']/'error.json',{'error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
