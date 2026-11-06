"""Small stored-mask LILocBench A/B pilot; no Florence/LLM calls or trajectory alignment."""
import argparse
import csv
import ast
import json
import math
import hashlib
import warnings
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import yaml
from scipy.spatial.transform import Rotation, Slerp

from .box_segmentation import digest
from .lilocbench_data import prepare_data

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi


def pose_table(path):
    table=np.loadtxt(path)
    if np.any(np.diff(table[:,0])<=0):raise ValueError(f'Non-monotonic timestamps: {path}')
    # Odom has further covariance/velocity columns, GT is TUM.
    yaw=Rotation.from_quat(table[:,4:8]).as_euler('ZYX')[:,0]
    return table[:,0],np.column_stack([table[:,1:3],np.unwrap(yaw)]),table[:,:8]


def interpolate(times,poses,query,max_gap):
    """No extrapolation and no interpolation through missing-data gaps."""
    query=np.asarray(query);right=np.searchsorted(times,query,side='right');left=np.clip(right-1,0,len(times)-1)
    right=np.clip(right,0,len(times)-1)
    gaps=times[right]-times[left]
    valid=(query>=times[0])&(query<=times[-1])&(gaps<=max_gap)
    output=np.column_stack([np.interp(query,times,poses[:,i]) for i in range(3)])
    output[:,2]=wrap(output[:,2]);output[~valid]=np.nan
    return output,valid


def transform_tree(records):
    edges={}
    for row in records:
        t=np.eye(4);t[:3,:3]=Rotation.from_quat([row['rotation'][k] for k in 'xyzw']).as_matrix()
        t[:3,3]=[row['translation'][k] for k in 'xyz']
        edges.setdefault(row['parent'],[]).append((row['child'],t))
        edges.setdefault(row['child'],[]).append((row['parent'],np.linalg.inv(t)))
    def lookup(parent,child):
        queue=[(parent,np.eye(4))];seen=set()
        while queue:
            frame,t=queue.pop(0)
            if frame==child:return t
            if frame in seen:continue
            seen.add(frame)
            queue.extend((n,t@step) for n,step in edges[frame] if n not in seen)
        raise ValueError(f'Missing official transform {parent} <- {child}')
    return lookup


def scan_points(ranges,laser,base_scan):
    angles=laser['angle_min']+np.arange(len(ranges))*laser['angle_increment']
    points=np.column_stack([ranges*np.cos(angles),ranges*np.sin(angles),np.zeros(len(ranges)),np.ones(len(ranges))])
    return (points@base_scan.T)[:,:3]


def project_scan(ranges,t,image_t,laser,base_scan,base_camera,intrinsics,odom_times,odom_poses,max_gap):
    """Official extrinsics + plumb_bob intrinsics; planar wheel-odom time compensation per beam."""
    good=np.isfinite(ranges)&(ranges>laser['range_min'])&(ranges<laser['range_max'])
    safe=np.where(good,ranges,0.)
    points=scan_points(safe,laser,base_scan)
    beam_times=t+np.arange(len(ranges))*laser['time_increment']
    beam_poses,beam_valid=interpolate(odom_times,odom_poses,beam_times,max_gap)
    camera_pose,camera_valid=interpolate(odom_times,odom_poses,[image_t],max_gap)
    if not camera_valid[0]:return np.zeros((len(ranges),2)),np.zeros(len(ranges),bool)
    c=np.cos(beam_poses[:,2]);s=np.sin(beam_poses[:,2])
    world_x=beam_poses[:,0]+c*points[:,0]-s*points[:,1]
    world_y=beam_poses[:,1]+s*points[:,0]+c*points[:,1]
    dx=world_x-camera_pose[0,0];dy=world_y-camera_pose[0,1];yaw=camera_pose[0,2]
    aligned=np.column_stack([np.cos(yaw)*dx+np.sin(yaw)*dy,-np.sin(yaw)*dx+np.cos(yaw)*dy,points[:,2],np.ones(len(ranges))])
    camera=aligned@np.linalg.inv(base_camera).T
    positive=camera[:,2]>0
    z=np.where(positive,camera[:,2],1)
    x=camera[:,0]/z;y=camera[:,1]/z;r2=x*x+y*y
    if intrinsics['distortion_model']!='plumb_bob':raise ValueError('Unsupported official camera distortion model')
    k1,k2,p1,p2,k3=intrinsics['distortion_coefficients'];radial=1+k1*r2+k2*r2**2+k3*r2**3
    xd=x*radial+2*p1*x*y+p2*(r2+2*x*x);yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    k=intrinsics['K'];uv=np.column_stack([k[0]*xd+k[2],k[4]*yd+k[5]])
    visible=good&beam_valid&positive&np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<intrinsics['width'])&(uv[:,1]>=0)&(uv[:,1]<intrinsics['height'])
    return uv,visible


def mask_layers(folder,frame,model):
    movable=np.zeros((frame['image_size'][1],frame['image_size'][0]),bool);protected=movable.copy()
    for region in frame['regions']:
        path=folder/region['mask_path']
        if digest(path)!=region['mask_sha256']:raise ValueError(f'Changed cached mask: {path}')
        classes=[s['llm']['models'].get(model,{}).get('classification') for s in region['sources']]
        binary=np.asarray(Image.open(path))>0
        if binary.shape!=movable.shape:raise ValueError('Cached mask dimensions differ from original RGB')
        if classes and all(c=='movable' for c in classes):movable|=binary
        else:protected|=binary  # retain fixed/uncertain/missing/conflicting labels at overlaps
    return movable&~protected


def docker_command(image):
    return ['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}',
            '-e','HOME=/tmp','-e','PYTHONDONTWRITEBYTECODE=1','-v',f'{ROOT}:{ROOT}','-w',str(ROOT),image]


def mask_cache(data,stored,selected,cfg):
    reports=[ROOT/cfg['existing_masks']/'segmentation.json',*sorted((data/'mask-cache').rglob('segmentation.json'))]
    found={}
    for report in reports:
        if not report.exists():continue
        content=json.loads(report.read_text())
        for frame in content['frames']:
            if frame['frame_id'] in selected:found.setdefault(frame['frame_id'],(report.parent,frame))
    missing=sorted(set(selected)-set(found))
    if missing:
        import hashlib
        target=data/'mask-cache'/('group-'+hashlib.sha256(','.join(missing).encode()).hexdigest()[:12])
        command=docker_command(cfg['sam_image']);command[3:3]=['--gpus','all','--network','none']
        command+=['python','-m','llm_object_classifier.pipeline','--stage','segment','--config',str(ROOT/'configs/pipeline.json'),
                  '--results-from',str(stored),'--frames',*missing,'--output',str(target)]
        subprocess.run(command,check=True)
        return mask_cache(data,stored,selected,cfg)
    return found


def official_errors(path,gt,estimated):
    """Reuse official evaluator's 2D APE routine on identical, already matched time points."""
    # Execute the three unmodified official numeric functions, without importing
    # its Typer CLI and HTML plotting dependencies. The downloaded file is hashed.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',SyntaxWarning)
        source=ast.parse(Path(path).read_text(),filename=str(path))
    functions=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in {'poses_to_2d','ape','ape2d'}]
    if len(functions)!=3:raise ValueError('Official evaluator numeric API changed')
    namespace={'np':np,'Rotation':Rotation}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    def matrices(poses):
        result=np.tile(np.eye(4),(len(poses),1,1))
        result[:,:3,:3]=Rotation.from_euler('z',poses[:,2]).as_matrix();result[:,:2,3]=poses[:,:2]
        return result
    return namespace['ape2d'](matrices(gt),matrices(estimated))


def evaluate(gt,estimates,has_gt,evaluator):
    exists=np.isfinite(estimates).all(axis=1);valid=has_gt&exists;n=int(has_gt.sum())
    position=np.full(len(gt),np.nan);angle=position.copy()
    if valid.any():position[valid],angle[valid]=official_errors(evaluator,gt[valid],estimates[valid])
    degrees=np.degrees(angle);success=valid&(position<0.1)&(degrees<4)
    def percent(condition):return 100*int(condition.sum())/n if n else None
    summary={'evaluable_poses':n,'estimated_with_gt':int(valid.sum()),'missing_estimates':int((~exists&has_gt).sum()),
             'times_without_gt':int((~has_gt).sum()),'successful_poses':int(success.sum()),'success_percent':percent(success),
             'position_rmse_cm':float(np.sqrt(np.mean(position[valid]**2))*100) if valid.any() else None,
             'position_max_cm':float(np.max(position[valid])*100) if valid.any() else None,
             'yaw_rmse_deg':float(np.sqrt(np.mean(degrees[valid]**2))) if valid.any() else None,
             'yaw_max_deg':float(np.max(degrees[valid])) if valid.any() else None,
             'position_below_5cm_percent':percent(valid&(position<0.05)),
             'position_below_10cm_percent':percent(valid&(position<0.1)),
             'yaw_below_2deg_percent':percent(valid&(degrees<2)),
             'yaw_below_4deg_percent':percent(valid&(degrees<4))}
    return summary,position,degrees,success


def plot_results(output,map_image,meta,gt,estimates,success,valid_gt,diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    res=meta['resolution'];origin=meta['origin'];height,width=map_image.shape
    extent=[origin[0],origin[0]+width*res,origin[1],origin[1]+height*res]
    points=gt[valid_gt,:2];low=points.min(axis=0)-1.5;high=points.max(axis=0)+1.5
    def background(ax):
        ax.imshow(map_image,cmap='gray',vmin=0,vmax=255,extent=extent,origin='upper')
        ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    fig,axes=plt.subplots(1,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,branch in zip(axes,['A','B']):
        background(ax)
        segments=np.stack([gt[:-1,:2],gt[1:,:2]],axis=1)
        good=valid_gt[:-1]&valid_gt[1:]
        ax.add_collection(LineCollection(segments[good],colors=np.where(success[branch][1:][good],'#159447','#dc3030'),linewidths=3,zorder=3))
        ax.plot(estimates[branch][:,0],estimates[branch][:,1],color='#1858c9',lw=1.2,zorder=4,label='AMCL estimate')
        ax.set_title(f'{branch}: '+('original scans' if branch=='A' else 'movable hits invalidated'))
        ax.legend(handles=[Line2D([],[],color='#159447',lw=3,label='GT: position <10 cm AND yaw <4°'),
                           Line2D([],[],color='#dc3030',lw=3,label='GT: failure (incl. missing estimate)'),
                           Line2D([],[],color='#1858c9',label='AMCL estimate')],fontsize=8,loc='upper right')
    fig.tight_layout();fig.savefig(output/'localization_map.png',dpi=170);plt.close(fig)
    # Diagnostic hit placement deliberately uses GT, never the AMCL estimate.
    fig,axes=plt.subplots(1,2,figsize=(13,6));background(axes[0])
    for diag in diagnostics:
        if not diag['gt_valid']:continue
        pose=diag['gt'];p=diag['points'];c=math.cos(pose[2]);s=math.sin(pose[2])
        xy=p[:,:2]@np.array([[c,s],[-s,c]])+pose[:2]
        keep=diag['visible_hits']&~diag['removed'];drop=diag['removed']
        axes[0].scatter(xy[keep,0],xy[keep,1],s=3,c='#1497c4',alpha=.5)
        axes[0].scatter(xy[drop,0],xy[drop,1],s=8,c='#e33320',alpha=.8)
    axes[0].plot(points[:,0],points[:,1],color='#169747',lw=1)
    axes[0].legend(handles=[Line2D([],[],marker='.',ls='',color='#1497c4',label='retained hits (GT placement)'),
                           Line2D([],[],marker='.',ls='',color='#e33320',label='removed movable hits (GT placement)'),
                           Line2D([],[],color='#169747',label='GT trajectory')],fontsize=8)
    axes[0].set_title('Diagnostic only: LiDAR placed using ground truth')
    example=max(diagnostics,key=lambda d:int(d['removed'].sum()))
    image=np.array(Image.open(example['image']).convert('RGB'));mask=example['mask']
    image[mask]=(image[mask]*.5+np.array([255,50,20])*.5).astype(np.uint8)
    axes[1].imshow(image);axes[1].axis('off');axes[1].set_title(example['frame_id']+' — single-image movable mask')
    fig.tight_layout();fig.savefig(output/'lidar_overlay.png',dpi=170);plt.close(fig)


def run_pilot(sequence,start,end,data,output,cfg,preflight_only=False):
    data=Path(data).resolve();output=Path(output).resolve();stored=(ROOT/cfg['stored_results']).resolve()
    if sequence!='lt_changes_0':raise ValueError('Only lt_changes_0 is enabled for this bounded pilot')
    if not start<end or end-start>40:raise ValueError('Pilot interval must be >0 and <=40 seconds')
    seq=prepare_data(data,sequence,stored);output.mkdir(parents=True,exist_ok=True)
    pairs=json.loads((stored/'pairs.json').read_text())['pairs']
    selected={pair['b']['frame_id']:pair['b'] for pair in pairs if pair['b']['sequence']==sequence and start<=pair['b']['timestamp']<=end}
    if len(selected)<5:raise ValueError('Choose an interval with at least 5 already-classified RGB frames')
    selection={'sequence':sequence,'start':start,'end':end,'rgb_frames':list(selected),'config':cfg,
               'selected_before_ab':True,'selection_reason':'Stored RGB descriptions and route: corridor -> changed office -> corridor; no localization results used',
               'mask_policy':'nearest single RGB within mask_max_dt of scan midpoint; odometry time compensation; no voting',
               'gt_policy':'one common initialization only; remaining GT used exclusively for evaluation/diagnostic plots'}
    selection_path=output/'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection:raise ValueError('Changed selection/settings; use a new output directory')
    write(selection_path,selection)  # always persisted before any A/B localization
    caches=mask_cache(data,stored,selected,cfg)
    laser=yaml.safe_load((seq/'laser_scan_front/laser_specs.yaml').read_text())
    records=yaml.safe_load((seq/'transformations.yaml').read_text());lookup=transform_tree(records)
    base_scan=lookup('base_link','scan_front_link');base_camera=lookup('base_link','camera_front_color_optical_frame')
    intrinsics=yaml.safe_load((seq/'camera_front/color/intrinsics.yaml').read_text())
    odom_times,odom_poses,_=pose_table(seq/'dingo_velocity_controller/odom.txt')
    gt_times,gt_poses,gt_raw=pose_table(seq/'gt_poses.txt')
    # GT interpolates full quaternions (Slerp), then projects yaw, matching the official evaluator's 2D convention.
    scans=[];times=[]
    with (seq/'laser_scan_front/ranges.txt').open() as stream:
        for line in stream:
            t=float(line.split(maxsplit=1)[0])
            if start<=t<=end:
                values=np.fromstring(line,sep=' ');times.append(values[0]);scans.append(values[1:])
    times=np.asarray(times);ranges=np.asarray(scans)
    if not len(times) or np.any(np.diff(times)<=0):raise ValueError('Missing/invalid scan interval')
    odom,valid_odom=interpolate(odom_times,odom_poses,times,cfg['pose_max_gap'])
    odom_end,valid_end=interpolate(odom_times,odom_poses,times+laser['scan_time'],cfg['pose_max_gap'])
    if not np.all(valid_odom&valid_end):raise ValueError('Missing wheel odometry inside selected interval')
    gt,valid_gt=interpolate(gt_times,gt_poses,times,cfg['pose_max_gap'])
    if valid_gt.any():
        rotations=Slerp(gt_times,Rotation.from_quat(gt_raw[:,4:8]))(times[valid_gt])
        gt[valid_gt,2]=rotations.as_euler('ZYX')[:,0]
    if not valid_gt[0]:raise ValueError('First scan has no GT initialization; choose a valid start')
    ordered=sorted(selected,key=lambda f:selected[f]['timestamp'])
    image_times=np.array([selected[f]['timestamp'] for f in ordered])
    masks={fid:mask_layers(*caches[fid],cfg['classification_model']) for fid in ordered}
    removed=np.zeros(ranges.shape,bool);mask_valid=np.zeros(len(times),bool);chosen={};projections={}
    for i,t in enumerate(times):
        nearest=int(np.argmin(abs(image_times-(t+laser['scan_time']/2))))
        if abs(image_times[nearest]-(t+laser['scan_time']/2))>cfg['mask_max_dt']:continue
        fid=ordered[nearest];mask_valid[i]=True;chosen[i]=fid
        uv,visible=project_scan(ranges[i],t,image_times[nearest],laser,base_scan,base_camera,intrinsics,odom_times,odom_poses,cfg['pose_max_gap'])
        pixels=np.floor(uv[visible]).astype(int);indices=np.flatnonzero(visible)
        removed[i,indices]=masks[fid][pixels[:,1],pixels[:,0]]
        projections[i]=(uv,visible)
    # Verify 5–10 nearest image/scan pairs before launching either localization branch.
    diagnostics=[]
    for fid in ordered[:10]:
        i=int(np.argmin(abs(times+laser['scan_time']/2-selected[fid]['timestamp'])))
        if i not in projections or chosen[i]!=fid:raise ValueError(f'No time-valid scan for RGB {fid}')
        uv,visible=projections[i];finite=np.isfinite(ranges[i])&(ranges[i]>laser['range_min'])&(ranges[i]<laser['range_max'])
        diagnostics.append({'frame_id':fid,'scan_index':i,'dt_seconds':float(times[i]+laser['scan_time']/2-selected[fid]['timestamp']),
            'image':str(stored/'images'/(fid+'.png')),'mask':masks[fid],'uv':uv,'in_fov':visible,
            'removed':removed[i],'points':scan_points(np.where(finite,ranges[i],0),laser,base_scan),
            'visible_hits':finite,'gt':gt[i],'gt_valid':bool(valid_gt[i])})
    check={'pairs':[{'frame_id':d['frame_id'],'scan_index':d['scan_index'],'dt_seconds':d['dt_seconds'],
                     'projected_hits':int(d['in_fov'].sum()),'removed_hits':int(d['removed'].sum())} for d in diagnostics],
           'passed':len(diagnostics)>=5 and all(d['in_fov'].any() for d in diagnostics) and bool(removed.any()),
           'projection':'official transforms to color optical frame; plumb_bob distortion; per-beam planar wheel-odometry compensation',
           'uncertain_policy':'protected: any overlap with fixed/uncertain/missing/conflicting mask is retained'}
    write(output/'preflight.json',check)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(math.ceil(len(diagnostics)/2),2,figsize=(12,4*math.ceil(len(diagnostics)/2)))
    for ax,diag in zip(np.asarray(axes).flat,diagnostics):
        im=np.array(Image.open(diag['image']).convert('RGB'));mask=diag['mask'];im[mask]=(im[mask]*.5+np.array([255,50,20])*.5).astype(np.uint8)
        ax.imshow(im);uv=diag['uv'];keep=diag['in_fov']&~diag['removed'];drop=diag['removed']
        ax.scatter(uv[keep,0],uv[keep,1],s=4,c='cyan');ax.scatter(uv[drop,0],uv[drop,1],s=9,c='yellow')
        ax.set_title(f'{diag["frame_id"]}: dt={diag["dt_seconds"]*1000:.1f} ms; removed={drop.sum()}');ax.axis('off')
    for ax in list(np.asarray(axes).flat)[len(diagnostics):]:ax.axis('off')
    fig.suptitle('Projection check: red=movable mask, cyan=retained, yellow=removed');fig.tight_layout()
    fig.savefig(output/'projection_check.png',dpi=100);plt.close(fig)
    if not check['passed']:raise RuntimeError('Projection/filter preflight failed; inspect preflight.json and projection_check.png')
    if preflight_only:print(json.dumps(check,indent=2));return
    meta=yaml.safe_load(next((data/'map').glob('*.yaml')).read_text());map_image=np.asarray(Image.open(data/'map'/Path(meta['image']).name).convert('L'))
    probability=map_image/255. if meta['negate'] else 1-map_image/255.
    occupancy=np.full(map_image.shape,-1,dtype=np.int8);occupancy[probability>meta['occupied_thresh']]=100;occupancy[probability<meta['free_thresh']]=0
    replay=output/'replay';replay.mkdir(exist_ok=True)
    np.savez_compressed(replay/'scans.npz',times=times,ranges=ranges,removed=removed,odom=odom,odom_end=odom_end,occupancy=np.flipud(occupancy))
    scan_transform=next(r for r in records if r['parent']=='base_link' and r['child']=='scan_front_link')
    write(replay/'input.json',{'laser':laser,'map':{**meta,'width':map_image.shape[1],'height':map_image.shape[0]},
                            'scan_transform':scan_transform,'initial_pose':gt[0].tolist(),'seed':cfg['seed']})
    (replay/'amcl.yaml').write_text(yaml.safe_dump({'amcl':{'ros__parameters':cfg['amcl']}}))
    signature=hashlib.sha256()
    for array in [times,ranges,removed,odom,odom_end,occupancy]:signature.update(array.tobytes())
    signature.update((replay/'input.json').read_bytes());signature.update((replay/'amcl.yaml').read_bytes())
    for name in ['amcl_replay.py','amcl_seed.c']:signature.update((ROOT/'llm_object_classifier'/name).read_bytes())
    replay_identity={'sha256':signature.hexdigest()}
    signature_path=replay/'signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text())!=replay_identity:
        raise ValueError('Replay inputs/implementation changed; use a new output directory')
    write(signature_path,replay_identity)

    estimates={};summaries={};errors={};success={}
    evaluator=data/'reference-code/evaluate.py'
    if not evaluator.exists():raise FileNotFoundError(f'Add official evaluator src/evaluate.py at {evaluator}')
    for branch in ['A','B']:
        branch_dir=output/branch
        if not (branch_dir/'poses.json').exists():
            command=docker_command(cfg['ros_image']);command[3:3]=['--network','none','-e','ROS_DOMAIN_ID=91','-e','ROS_LOCALHOST_ONLY=1']
            command+=['bash','-c','source /opt/ros/jazzy/setup.bash && exec python3 -m llm_object_classifier.amcl_replay "$@"','pilot',
                      '--input',str(replay/'input.json'),'--branch',branch,'--output',str(branch_dir)]
            subprocess.run(command,check=True)
        rows=json.loads((branch_dir/'poses.json').read_text())
        if len(rows)!=len(times) or any(abs(row['timestamp']-t)>1e-6 for row,t in zip(rows,times)):raise ValueError('A/B replay times differ')
        estimates[branch]=np.array([r['pose'] if r['pose'] is not None else [np.nan]*3 for r in rows])
        summaries[branch],pos,yaw,success[branch]=evaluate(gt,estimates[branch],valid_gt,evaluator);errors[branch]=(pos,yaw)
    finite=np.isfinite(ranges)&(ranges>laser['range_min'])&(ranges<laser['range_max'])
    metrics={'sequence':sequence,'start':start,'end':end,'scans':len(times),'rgb_frames':len(selected),'method':'ROS2 AMCL 1.3.12; OmniMotionModel + likelihood_field',
             'initialization':f'first GT pose, covariance diag=(0.2m,0.2m,10deg)^2; shared seed {cfg["seed"]}; no alignment',
             'mask_valid_scans':int(mask_valid.sum()),'mask_valid_scans_percent':float(mask_valid.mean()*100),
             'finite_hits':int(finite.sum()),'removed_hits':int(removed.sum()),'removed_hits_percent':float(removed.sum()/finite.sum()*100),
             'removed_hits_percent_on_masked_scans':float(removed.sum()/finite[mask_valid].sum()*100),
             'threshold_policy':'strict <; all GT-valid scan times in denominator; missing estimate is failure; RMSE/max on available GT-estimate pairs',
             'downloaded_bytes':sum(r['bytes'] for r in json.loads((data/'http-cache/downloads.json').read_text())),
             'A':summaries['A'],'B':summaries['B'],'success_delta_percentage_points':summaries['B']['success_percent']-summaries['A']['success_percent']}
    write(output/'metrics.json',metrics)
    columns=['timestamp','has_gt','mask_valid','mask_frame','removed_hits','gt_x','gt_y','gt_yaw']+[f'{b}_{field}' for b in ['A','B'] for field in ['x','y','yaw','position_error_m','yaw_error_deg','success']]
    with (output/'poses.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(columns)
        for i,t in enumerate(times):
            row=[f'{t:.9f}',bool(valid_gt[i]),bool(mask_valid[i]),chosen.get(i,''),int(removed[i].sum()),*gt[i].tolist()]
            for b in ['A','B']:row.extend([*estimates[b][i].tolist(),float(errors[b][0][i]),float(errors[b][1][i]),bool(success[b][i])])
            writer.writerow(['' if isinstance(v,float) and not math.isfinite(v) else v for v in row])
    plot_results(output,map_image,meta,gt,estimates,success,valid_gt,diagnostics)
    identities={str(p):digest(p) for p in [seq/'gt_poses.txt',seq/'transformations.yaml',seq/'camera_front/color/intrinsics.yaml',seq/'laser_scan_front/ranges.txt',seq/'laser_scan_front/laser_specs.yaml',seq/'dingo_velocity_controller/odom.txt',evaluator,ROOT/'configs/lilocbench-pilot.json',Path(__file__),ROOT/'llm_object_classifier/amcl_replay.py',ROOT/'llm_object_classifier/amcl_seed.c']}
    for path in sorted((data/'map').glob('*')):
        if path.is_file():identities[str(path)]=digest(path)
    for fid in ordered:
        folder,frame=caches[fid];report=folder/'segmentation.json';identities[str(report)]=digest(report)
        for region in frame['regions']:
            path=folder/region['mask_path'];identities[str(path)]=digest(path)
    for name in ['box_segmentation.py','lilocbench_data.py']:
        path=ROOT/'llm_object_classifier'/name;identities[str(path)]=digest(path)
    write(output/'provenance.json',{'sha256':identities,'seed':cfg['seed'],'replay_signature':replay_identity})
    print(json.dumps(metrics,indent=2))
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/lilocbench-pilot.json')
    p.add_argument('--sequence');p.add_argument('--start',type=float);p.add_argument('--end',type=float)
    p.add_argument('--data',type=Path,default=ROOT/'data/lilocbench-pilot');p.add_argument('--output',type=Path,default=ROOT/'results/lilocbench-pilot')
    p.add_argument('--preflight-only',action='store_true');args=p.parse_args();cfg=json.loads(args.config.read_text())
    run_pilot(args.sequence or cfg['sequence'],args.start if args.start is not None else cfg['start'],
              args.end if args.end is not None else cfg['end'],args.data,args.output,cfg,args.preflight_only)


if __name__=='__main__':main()
