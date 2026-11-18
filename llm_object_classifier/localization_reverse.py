"""Reverse LILocBench experiment; conservative multi-camera fusion and geometry audit.

All localization variants use the existing ROS2 AMCL executable. Public GT is
used for map construction/geometry/evaluation and once for initialization only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import yaml

from .box_segmentation import digest
from .lilocbench_data import extract_rgb
from .localization_pilot import (ROOT, write, pose_table, interpolate, transform_tree,
    project_scan, docker_command, evaluate, wrap)
from .localization_full import read, infer, build_masks
from .localization_filtered_map import immutable, beam_world, ray_library, occupancy, aggregate, EXPECTED_QWEN
from .localization_comparison import load_estimates
from .odometry_bridge import propagate

CAMERAS = ('camera_front', 'camera_left', 'camera_right')
BRANCHES = ('original', 'front', 'three')


def fuse(decisions):
    """Each column is an immutable beam ID: +1 movable, -1 veto, 0 abstain.

    A visible camera with missing/invalid masks is a veto. Outside that camera's
    FOV it abstains. Fixed, uncertain and conflicting regions always veto.
    """
    decisions = np.asarray(decisions)
    return (decisions > 0).any(axis=0) & ~(decisions < 0).any(axis=0)


def decision_layer(folder, frame, model):
    shape = (frame['image_size'][1], frame['image_size'][0])
    movable = np.zeros(shape, bool); protected = movable.copy()
    for region in frame['regions']:
        path = folder / region['mask_path']
        if digest(path) != region['mask_sha256']:raise ValueError(f'Changed mask {path}')
        binary = np.asarray(Image.open(path)) > 0
        if binary.shape != shape:raise ValueError('Mask/image shape mismatch')
        labels = [s['llm']['models'].get(model, {}).get('classification') for s in region['sources']]
        if labels and all(v == 'movable' for v in labels):movable |= binary
        else:protected |= binary
    layer = np.zeros(shape, np.int8)
    layer[movable] = 1; layer[protected] = -1
    return layer


def nearest(stamps, t, tolerance):
    if not len(stamps):return None
    i = int(np.searchsorted(stamps, t)); choices = [j for j in (i-1, i) if 0 <= j < len(stamps)]
    j = min(choices, key=lambda j: (abs(stamps[j]-t), j))
    return j if abs(stamps[j]-t) <= tolerance else None


def gap_plan(times, ranges, laser, cfg):
    usable = (np.isfinite(ranges) & (ranges > laser['range_min']) & (ranges < laser['range_max'])).sum(axis=1)
    natural = usable < cfg['minimum_hits']
    artificial = np.zeros(len(times), bool)
    for start, duration in cfg['artificial_gaps_s']:
        artificial |= (times >= times[0]+start) & (times < times[0]+start+duration)
    return usable, natural, artificial


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def gap_metrics(times, odom, gt, has_gt, estimates, bridge, gap, cfg):
    finite = np.isfinite(estimates).all(axis=1)
    position = np.linalg.norm(estimates[:,:2]-gt[:,:2], axis=1)
    angle = np.abs(np.degrees(wrap(estimates[:,2]-gt[:,2])))
    success = has_gt & finite & (position < .1) & (angle < 4)
    rows = []
    # A duration includes one final scan period, bounded by the next timestamp.
    dt = float(np.median(np.diff(times)))
    for a,b in intervals(gap):
        end = float(times[b]) if b < len(times) else float(times[-1]+dt)
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(odom[a:b,:2],axis=0),axis=1))]
        prefix = np.flatnonzero(~success[a:b]); stop = int(prefix[0]) if len(prefix) else b-a
        valid = has_gt[a:b] & finite[a:b]
        recovery = None
        for j in range(b,len(times)):
            k = int(np.searchsorted(times,times[j]+cfg['recovery_stable_s'],side='right'))
            if k > len(times) or times[-1] < times[j]+cfg['recovery_stable_s']:break
            if gap[j:k].any():break
            if np.all(success[j:k]):recovery=float(times[j]-end);break
        safe_runs=intervals(success[a:b])
        longest_duration=max((float((times[a+v] if a+v<len(times) else end)-times[a+u]) for u,v in safe_runs),default=0.)
        longest_distance=max((float(distance[v-1]-distance[u]) for u,v in safe_runs),default=0.)
        rows.append({'start_index':int(a),'end_index_exclusive':int(b),'start':float(times[a]),'end':end,
            'duration_s':end-float(times[a]),'distance_m':float(distance[-1]),
            'bridged_scans':int(bridge[a:b].sum()),'missing_estimates':int((~finite[a:b]).sum()),
            'times_without_gt':int((~has_gt[a:b]).sum()),
            'joint_success_percent':100*float(success[a:b].sum()/(b-a)),
            'max_position_error_cm':float(np.max(position[a:b][valid])*100) if valid.any() else None,
            'max_yaw_error_deg':float(np.max(angle[a:b][valid])) if valid.any() else None,
            'safe_prefix_duration_s':float((times[a+stop] if a+stop<len(times) else end)-times[a]),
            'safe_prefix_distance_m':float(distance[min(stop,len(distance)-1)]),
            'max_safe_duration_s':longest_duration,'max_safe_distance_m':longest_distance,
            'recovery_s':recovery,'recovered':recovery is not None})
    return rows


def classify_points(points, reference_tree, reference_grid, meta, tolerance, known=None):
    """Disjoint categories: matched, confirmed-free excess, unobserved, other occupied.

    KD-tree uses continuous endpoints, not occupied cell centers. The map only
    confirms free evidence for unmatched points; unknown cells never count as free.
    """
    distance = reference_tree.query(points, workers=1)[0]
    cells = np.floor((points-np.asarray(meta['origin'][:2]))/meta['resolution']).astype(int)
    inside = (cells[:,0]>=0)&(cells[:,0]<meta['width'])&(cells[:,1]>=0)&(cells[:,1]<meta['height'])
    values = np.full(len(points),-1,np.int8)
    values[inside] = reference_grid[cells[inside,1],cells[inside,0]]
    labels = np.full(len(points),3,np.uint8)
    unobserved=values==-1
    if known is not None:
        unobserved=np.ones(len(points),bool)
        unobserved[inside]=~known[cells[inside,1],cells[inside,0]].astype(bool)
    labels[unobserved]=2; labels[values==0]=1; labels[distance<=tolerance]=0
    return labels


class Experiment:
    def __init__(self, config):
        self.cfg=read(config);self.pilot=read(ROOT/self.cfg['pilot_config'])
        self.output=ROOT/self.cfg['output'];self.cache=ROOT/self.cfg['cache']
        self.data=ROOT/self.cfg['data'];self.raw=ROOT/self.cfg['raw_cache']
        self.output.mkdir(parents=True,exist_ok=True);self.cache.mkdir(parents=True,exist_ok=True)

    def progress(self,stage,**info):
        row={'stage':stage,'unix_time':time.time(),**info};write(self.output/'progress.json',row)
        print(json.dumps(row),flush=True)

    def arrays(self,sequence):return dict(np.load(self.raw/sequence/'raw.npz'))
    def laser(self,sequence):return read(self.raw/sequence/'raw.json')['laser']

    def prepare(self):
        cfg=self.cfg
        immutable(self.output/'protocol.json',{'configuration':cfg,'amcl':self.pilot['amcl'],
            'gt_policy':'Common public frame, no alignment. Mapping/geometry/evaluation only; localization initialization once.',
            'sequence_evidence':{'source':cfg['source'],'full':'lt_changes_0: rearranged furniture and added boxes',
                'empty_query':'static_0: static baseline; NOT a certified completely empty room',
                'independent_empty_map':'mapping: independent baseline; ordinary furniture remains'},
            'fusion':'Beam IDs; any movable and no protected/uncertain/conflict/missing-mask veto in any visible camera. Outside coverage retain.',
            'geometry':'Same Euclidean tolerance, continuous deskewed GT endpoints; reference free occupancy confirms excess; unknown separate.',
            'bridge':'Skip scan publication below fixed hit threshold; SE2 odometry from last low-covariance AMCL output. No GT gating.',
            'missing':'Missing estimates fail threshold denominator; GT missing reported separately',
            'seeds':cfg['seeds'],'std_ddof':0})
        summary={}
        for seq in [cfg['full_sequence'],cfg['empty_sequence'],cfg['reference_sequence']]:
            a=self.arrays(seq);gt=a['gt'];v=a['has_gt'];x0,y0,x1,y1=cfg['room_bounds_xyxy']
            room=v&(gt[:,0]>=x0)&(gt[:,0]<=x1)&(gt[:,1]>=y0)&(gt[:,1]<=y1)
            summary[seq]={'scans':len(a['times']),'start':float(a['times'][0]),'end':float(a['times'][-1]),
                'room_scans':int(room.sum()),'room_bounds_xyxy':cfg['room_bounds_xyxy']}
        write(self.output/'sequence-audit.json',summary)
        self.progress('prepared',sequences=summary)

    def model_batches(self,preflight=False):
        cfg=self.cfg;seq=cfg['full_sequence'];a=self.arrays(seq)
        start,end=cfg['preflight_section'];times=a['times']
        targets=np.arange(start,end,1/cfg['rgb_hz']) if preflight else np.arange(times[0],times[-1],1/cfg['rgb_hz'])
        # Include preflight stamps in full plan so the first verified images are reused.
        if not preflight:targets=np.unique(np.r_[targets,np.arange(start,end,1/cfg['rgb_hz'])])
        # Reuse all available full-sequence front responses/masks, including invalid outcomes.
        old={}
        for name in cfg['reuse_status']:
            for row in read(ROOT/name):old.setdefault(row['image_sha256'],row)
        for status in sorted((self.cache/'models').glob('**/work/mask-status.json')):
            for row in read(status):old.setdefault(row['image_sha256'],row)
        statuses={c:[] for c in CAMERAS}
        for camera in CAMERAS:
            for batch_index,offset in enumerate(range(0,len(targets),cfg['batch_size'])):
                folder=self.cache/'models'/camera/('pilot' if preflight else 'full')/f'{batch_index:04d}'
                folder.mkdir(parents=True,exist_ok=True);plan=folder/'rgb-plan.json'
                if not plan.exists():
                    paths=extract_rgb(self.data,seq,targets[offset:offset+cfg['batch_size']],camera)
                    rows=[];seen=set()
                    for target,path in zip(targets[offset:offset+cfg['batch_size']],paths):
                        sha=digest(path)
                        if sha in seen:continue
                        seen.add(sha)
                        row=old.get(sha)
                        if row is None:
                            row={'target':float(target),'timestamp':float(path.stem),'path':str(path),
                                'frame_id':'image-'+sha[:16],'image_sha256':sha,'origin':'new-pipeline'}
                        rows.append(row)
                    write(plan,rows)
                rows=read(plan);new=[r for r in rows if r['image_sha256'] not in old]
                cached=[old[r['image_sha256']] for r in rows if r['image_sha256'] in old]
                if new:
                    # The adapter expects its plan to list only the batch it owns.
                    work=folder/'work';work.mkdir(exist_ok=True)
                    immutable(work/'rgb-plan.json',new);inputs=work/'rgb-missing';inputs.mkdir(exist_ok=True)
                    for row in new:
                        dest=inputs/(row['frame_id']+'.png')
                        if not dest.exists():shutil.copyfile(row['path'],dest)
                    local={**cfg,'cache':str(work)}
                    report=work/'mask-status.json'
                    if not report.exists():
                        infer(local,self.pilot)
                        if read(work/'inference/model-metadata.json')['digest']!=EXPECTED_QWEN:raise ValueError('Qwen weights changed')
                        build_masks(local,self.pilot)
                    new=read(report)
                merged=cached+new
                for row in merged:old.setdefault(row['image_sha256'],row)
                statuses[camera].extend(merged)
                self.progress('models',camera=camera,preflight=preflight,batch=batch_index+1,
                    batches=int(np.ceil(len(targets)/cfg['batch_size'])),frames=len(statuses[camera]))
            unique={r['timestamp']:r for r in statuses[camera]}
            statuses[camera]=sorted(unique.values(),key=lambda r:r['timestamp'])
            write(self.cache/f'{camera}-status-{"pilot" if preflight else "full"}.json',statuses[camera])
        return statuses

    def project(self,statuses,preflight=False):
        cfg=self.cfg;seq=cfg['full_sequence'];a=self.arrays(seq);laser=self.laser(seq)
        cache=self.cache/('projection-pilot.npz' if preflight else 'projection.npz')
        if cache.exists():return dict(np.load(cache))
        times=a['times'];selected=np.arange(len(times))
        if preflight:selected=selected[(times>=cfg['preflight_section'][0])&(times<cfg['preflight_section'][1])]
        tree=transform_tree(yaml.safe_load((self.data/seq/'transformations.yaml').read_text()))
        base_scan=tree('base_link','scan_front_link');ot,op,_=pose_table(self.data/seq/'dingo_velocity_controller/odom.txt')
        geometry={c:(tree('base_link',c+'_color_optical_frame'),yaml.safe_load((self.data/seq/c/'color/intrinsics.yaml').read_text())) for c in CAMERAS}
        valid={c:[r for r in statuses[c] if r['classification_valid'] and r.get('mask_report')] for c in CAMERAS}
        stamps={c:np.array([r['timestamp'] for r in valid[c]]) for c in CAMERAS}
        all_stamps={c:np.array([r['timestamp'] for r in statuses[c]]) for c in CAMERAS}
        shape=(len(selected),a['ranges'].shape[1]);removed=np.zeros((2,*shape),bool)
        visible_out=np.zeros((3,*shape),bool);associated=np.full((len(selected),3),np.nan)
        decisions=np.zeros((3,*shape),np.int8);layers={};reports={};examples=[];example_indices=set()
        if preflight:
            jointly_valid=[k for k,i in enumerate(selected) if all(nearest(stamps[c],times[i]+laser['scan_time']/2,self.pilot['mask_max_dt']) is not None for c in CAMERAS)]
            if not jointly_valid:raise ValueError('Pilot has no simultaneous valid three-camera association')
            example_indices={jointly_valid[0],jointly_valid[len(jointly_valid)//2],jointly_valid[-1]}
        for k,i in enumerate(selected):
            t=times[i];mid=t+laser['scan_time']/2
            for ci,camera in enumerate(CAMERAS):
                j=nearest(stamps[camera],mid,self.pilot['mask_max_dt'])
                candidate=valid[camera][j] if j is not None else None
                # Missing mask still needs actual image pose/FOV to veto overlaps.
                fallback=nearest(all_stamps[camera],mid,self.pilot['mask_max_dt'])
                image_t=candidate['timestamp'] if candidate else (statuses[camera][fallback]['timestamp'] if fallback is not None else mid)
                extrinsic,intrinsic=geometry[camera]
                uv,visible=project_scan(a['ranges'][i],t,image_t,laser,base_scan,extrinsic,intrinsic,ot,op,self.pilot['pose_max_gap'])
                visible_out[ci,k]=visible;decisions[ci,k,visible]=-1
                if candidate:
                    associated[k,ci]=candidate['timestamp'];fid=candidate['frame_id'];report=Path(candidate['mask_report'])
                    if fid not in layers:
                        if report not in reports:reports[report]={r['frame_id']:r for r in read(report)['frames']}
                        layers={key:value for key,value in layers.items() if key in [r['frame_id'] for r in valid[camera][max(0,j-1):j+2]]}
                        layers[fid]=decision_layer(report.parent,reports[report][fid],self.pilot['classification_model'])
                    pixels=np.floor(uv[visible]).astype(int)
                    decisions[ci,k,visible]=layers[fid][pixels[:,1],pixels[:,0]]
                if preflight and k in example_indices:
                    examples.append({'scan_index':int(i),'camera':camera,'timestamp':float(t),'image_timestamp':float(image_t),
                        'image_path':candidate['path'] if candidate else None,'uv':uv[visible].tolist(),
                        'beam_ids':np.flatnonzero(visible).tolist(),'decision':decisions[ci,k,visible].tolist()})
            removed[0,k]=fuse(decisions[:1,k]);removed[1,k]=fuse(decisions[:,k])
            if k%1000==0:self.progress('projection',preflight=preflight,scans=k+1,total=len(selected))
        result={'indices':selected,'removed':removed,'visible':visible_out,'decisions':decisions,'association':associated}
        finite=np.isfinite(a['ranges'][selected])&(a['ranges'][selected]>laser['range_min'])&(a['ranges'][selected]<laser['range_max'])
        front=visible_out[0];side=visible_out[1:].any(axis=0)
        summary={'scans':len(selected),'finite_hits':int(finite.sum()),'front_covered_hits':int(front.sum()),
            'side_additional_covered_hits':int((side&~front).sum()),'three_unique_covered_hits':int(visible_out.any(axis=0).sum()),
            'front_removed_hits':int(removed[0].sum()),'three_removed_hits':int(removed[1].sum()),
            'side_additional_removed_hits':int((removed[1]&~removed[0]).sum()),
            'front_removals_vetoed_by_sides':int((removed[0]&~removed[1]).sum()),
            'valid_associations_by_camera':np.isfinite(associated).sum(axis=0).tolist(),
            'visible_hits_by_camera':visible_out.sum(axis=(1,2)).tolist()}
        if np.any(removed&~finite[None]):raise ValueError('Invalid beam removed')
        if preflight:
            if not len(selected) or np.any(visible_out.sum(axis=(1,2))==0):raise ValueError('Pilot fails camera coverage')
            if np.any(np.isfinite(associated).sum(axis=0)==0):raise ValueError('Pilot has no valid mask for a camera')
            write(self.output/'projection-examples.json',examples)
            self.projection_plot(examples)
        write(self.output/('preflight-projection.json' if preflight else 'projection-summary.json'),summary)
        np.savez_compressed(cache,**result)
        return result

    def projection_plot(self,examples):
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,3,figsize=(16,12))
        for ax,row in zip(axes.flat,examples):
            if row['image_path']:ax.imshow(Image.open(row['image_path']))
            uv=np.asarray(row['uv']);dec=np.asarray(row['decision'])
            if len(uv):ax.scatter(uv[:,0],uv[:,1],c=np.where(dec>0,'red',np.where(dec<0,'lime','cyan')),s=5)
            ax.set(xlim=(0,640),ylim=(480,0),title=f"{row['camera']} scan {row['scan_index']}")
        fig.tight_layout();fig.savefig(self.output/'preflight-projection.png',dpi=140);plt.close(fig)

    def build_map(self,sequence,removals,names):
        meta=self.cfg['map'];a=self.arrays(sequence);laser=self.laser(sequence)
        seq=self.data/sequence;gt_t,gt_p,_=pose_table(seq/'gt_poses.txt')
        base=transform_tree(yaml.safe_load((seq/'transformations.yaml').read_text()))('base_link','scan_front_link')
        fun=ray_library(self.cache);shape=(len(names),meta['height'],meta['width'])
        checkpoint=self.cache/f'grid-{sequence}.npz'
        if checkpoint.exists():
            saved=np.load(checkpoint);logs=saved['logs'];known=saved['known'];start=int(saved['next'])
        else:logs=np.zeros(shape,np.float32);known=np.zeros(shape,np.uint8);start=0
        for i in range(start,len(a['times'])):
            origins,ends,good=beam_world(a['ranges'][i],a['times'][i],laser,base,gt_t,gt_p,self.pilot['pose_max_gap'])
            origins=np.floor((np.nan_to_num(origins)-np.array(meta['origin'][:2]))/meta['resolution']).astype(np.int32)
            ends=np.floor((np.nan_to_num(ends)-np.array(meta['origin'][:2]))/meta['resolution']).astype(np.int32)
            for b in range(len(names)):
                keep=np.ascontiguousarray(good&~removals[b][i],dtype=np.uint8)
                fun(logs[b],known[b],meta['width'],meta['height'],origins,ends,keep,len(keep),meta['log_hit'],meta['log_free'],meta['log_limit'])
            if (i+1)%1000==0 or i+1==len(a['times']):
                temp=checkpoint.with_suffix('.part.npz');np.savez_compressed(temp,logs=logs,known=known,next=i+1);temp.replace(checkpoint)
                self.progress('maps',sequence=sequence,scans=i+1,total=len(a['times']))
        for b,name in enumerate(names):
            grid=occupancy(logs[b],known[b],meta);dest=self.output/'maps'/name;dest.mkdir(parents=True,exist_ok=True)
            np.save(dest/'occupancy.npy',grid);pixels=np.full(grid.shape,205,np.uint8);pixels[grid==0]=254;pixels[grid==100]=0
            Image.fromarray(np.flipud(pixels)).save(dest/'map.png')
            (dest/'map.yaml').write_text(yaml.safe_dump({k:v for k,v in {**meta,'image':'map.png'}.items() if k in ['image','resolution','origin','occupied_thresh','free_thresh','negate']}))

    def maps(self):
        seq=self.cfg['full_sequence'];a=self.arrays(seq);removed=np.load(self.cache/'projection.npz')['removed']
        self.build_map(seq,[np.zeros(a['ranges'].shape,bool),*removed],BRANCHES)
        ref=self.cfg['reference_sequence'];r=self.arrays(ref)
        self.build_map(ref,[np.zeros(r['ranges'].shape,bool)],['empty'])
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        meta=self.cfg['map'];extent=[meta['origin'][0],meta['origin'][0]+meta['width']*meta['resolution'],meta['origin'][1],meta['origin'][1]+meta['height']*meta['resolution']]
        fig,axes=plt.subplots(2,4,figsize=(20,10))
        for column,name in enumerate([*BRANCHES,'empty']):
            grid=np.load(self.output/'maps'/name/'occupancy.npy')
            for row,bounds in enumerate([[-13,-14,18,15],self.cfg['room_bounds_xyxy']]):
                ax=axes[row,column];ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.7)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
                ax.set(title=name,xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal')
        fig.tight_layout();fig.savefig(self.output/'maps-comparison.png',dpi=160);plt.close(fig)

    def cloud(self,sequence):
        dest=self.cache/f'cloud-{sequence}.npz'
        if dest.exists():return dict(np.load(dest))
        a=self.arrays(sequence);seq=self.data/sequence;laser=self.laser(sequence)
        gt_t,gt_p,_=pose_table(seq/'gt_poses.txt');base=transform_tree(yaml.safe_load((seq/'transformations.yaml').read_text()))('base_link','scan_front_link')
        points=[];ids=[]
        for i,t in enumerate(a['times']):
            _,end,valid=beam_world(a['ranges'][i],t,laser,base,gt_t,gt_p,self.pilot['pose_max_gap'])
            points.append(end[valid].astype(np.float32));ids.append(i*a['ranges'].shape[1]+np.flatnonzero(valid))
        result={'points':np.concatenate(points),'ids':np.concatenate(ids)};np.savez_compressed(dest,**result);return result

    def geometry(self,preflight=False):
        cfg=self.cfg;full=self.cloud(cfg['full_sequence']);ref=self.cloud(cfg['empty_sequence'])
        # Empty query builds geometry evidence; independent mapping is reserved for AMCL.
        empty_dest=self.output/'maps'/'geometry-reference'/'occupancy.npy'
        if not empty_dest.exists():
            a=self.arrays(cfg['empty_sequence']);self.build_map(cfg['empty_sequence'],[np.zeros(a['ranges'].shape,bool)],['geometry-reference'])
        grid=np.load(empty_dest);known=np.load(self.cache/f'grid-{cfg["empty_sequence"]}.npz')['known'][0]
        ref_tree=cKDTree(ref['points'])
        projection=dict(np.load(self.cache/('projection-pilot.npz' if preflight else 'projection.npz')))
        points=full['points'];ids=full['ids'];n=self.arrays(cfg['full_sequence'])['ranges'].shape[1]
        if preflight:
            chosen=np.isin(ids//n,projection['indices']);points=points[chosen];ids=ids[chosen]
            rem=np.zeros((2,len(self.arrays(cfg['full_sequence'])['times']),n),bool)
            rem[:,projection['indices']]=projection['removed']
        else:rem=projection['removed']
        masks=[np.ones(len(points),bool),~rem[0].ravel()[ids],~rem[1].ravel()[ids]]
        labels=classify_points(points,ref_tree,grid,cfg['map'],cfg['point_tolerance_m'],known)
        x0,y0,x1,y1=cfg['room_bounds_xyxy'];room=(points[:,0]>=x0)&(points[:,0]<=x1)&(points[:,1]>=y0)&(points[:,1]<=y1)
        ref_room=(ref['points'][:,0]>=x0)&(ref['points'][:,0]<=x1)&(ref['points'][:,1]>=y0)&(ref['points'][:,1]<=y1)
        summary={};plot_data=[]
        for name,keep in zip(BRANCHES,masks):
            summary[name]={}
            missing=cKDTree(points[keep]).query(ref['points'],workers=1)[0]>cfg['point_tolerance_m'] if keep.any() else np.ones(len(ref['points']),bool)
            for region,select,refselect in [('whole',np.ones(len(points),bool),np.ones(len(ref['points']),bool)),('room',room,ref_room)]:
                used=keep&select;counts=np.bincount(labels[used],minlength=4);total=int(used.sum())
                summary[name][region]={'retained_points':total,'matched_points':int(counts[0]),'confirmed_free_excess_points':int(counts[1]),
                    'unobserved_points':int(counts[2]),'other_observed_points':int(counts[3]),
                    **{key:100*int(counts[j])/total if total else None for j,key in enumerate(['matched_percent','confirmed_free_excess_percent','unobserved_percent','other_observed_percent'])},
                    'reference_missing_percent_secondary':100*float(missing[refselect].mean()) if refselect.any() else None}
            plot_data.append((name,keep))
        write(self.output/('preflight-geometry.json' if preflight else 'geometry.json'),summary)
        np.savez_compressed(self.cache/('point-labels-pilot.npz' if preflight else 'point-labels.npz'),ids=ids,labels=labels)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for region,bounds in [('whole',[-13,-14,18,15]),('room',cfg['room_bounds_xyxy'])]:
            fig,axes=plt.subplots(1,3,figsize=(18,6),sharex=True,sharey=True)
            for ax,(name,keep) in zip(axes,plot_data):
                for j,color,label in [(2,'silver','unobserved'),(3,'orange','other observed'),(0,'green','matched'),(1,'red','confirmed free excess')]:
                    selected=np.flatnonzero(keep&(labels==j));stride=max(1,len(selected)//150000)
                    p=points[selected[::stride]];ax.scatter(p[:,0],p[:,1],s=.15,c=color,label=label,rasterized=True)
                ax.set(title=name,xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal');ax.legend(markerscale=8)
            fig.tight_layout();fig.savefig(self.output/f'{"preflight-" if preflight else ""}geometry-{region}.png',dpi=180);plt.close(fig)
        self.progress('geometry_complete',preflight=preflight,metrics=summary)
        return summary

    def replays(self):
        cfg=self.cfg;full=cfg['full_sequence'];empty=cfg['empty_sequence'];removed=np.load(self.cache/'projection.npz')['removed']
        tasks=[]
        cases=[(f'reverse-{b}',empty,b,None) for b in BRANCHES]+[('reverse-empty',empty,'empty',None)]
        cases += [(f'forward-{b}',full,'empty',None if b=='original' else removed[BRANCHES.index(b)-1]) for b in BRANCHES]
        for name,seq,map_name,remove in cases:
            a=self.arrays(seq);info=read(self.raw/seq/'raw.json');ranges=a['ranges'].copy()
            if remove is not None:ranges[remove]=np.nan
            usable,natural,artificial=gap_plan(a['times'],ranges,info['laser'],cfg['bridge'])
            # Same damaged scans for standard AMCL and actual skip-and-odom replay.
            for scenario,gap in [('natural',natural),('artificial',natural|artificial)]:
                damaged=ranges.copy()
                if scenario=='artificial':damaged[artificial]=np.nan
                shared=self.output/'replays'/name/scenario;shared.mkdir(parents=True,exist_ok=True)
                if not (shared/'scans.npz').exists():
                    np.savez_compressed(shared/'scans.npz',times=a['times'],ranges=damaged,odom=a['odom'],odom_end=a['odom_end'],
                        removed=np.zeros(ranges.shape,bool),occupancy=np.load(self.output/'maps'/map_name/'occupancy.npy'),
                        skip_lidar=gap,usable_hits=usable,natural_gap=natural,artificial_gap=artificial if scenario=='artificial' else np.zeros(len(natural),bool))
                transform=next(r for r in yaml.safe_load((self.data/seq/'transformations.yaml').read_text()) if r['parent']=='base_link' and r['child']=='scan_front_link')
                for seed in cfg['seeds']:
                    for mode in ['standard','bridge']:
                        folder=shared/f'seed-{seed}'/mode;folder.mkdir(parents=True,exist_ok=True)
                        dest=folder/'scans.npz'
                        if not dest.exists():dest.symlink_to(os.path.relpath(shared/'scans.npz',folder))
                        immutable(folder/'input.json',{'laser':info['laser'],'map':cfg['map'],'scan_transform':transform,
                            'initial_pose':info['initial'],'seed':seed,'bridge':cfg['bridge'] if mode=='bridge' else None})
                        (folder/'amcl.yaml').write_text(yaml.safe_dump({'amcl':{'ros__parameters':self.pilot['amcl']}}))
                        output=self.output/'runs'/name/scenario/f'seed-{seed}'/mode
                        tasks.append((folder,output,name,scenario,seed,mode,seq))
        return tasks

    def amcl(self):
        tasks=self.replays();pending=[t for t in tasks if not (t[1]/'poses.json').exists()]
        def run(item):
            index,task=item;folder,output,name,scenario,seed,mode,seq=task
            command=docker_command(self.pilot['ros_image']);command[3:3]=['--network','none','-e',f'ROS_DOMAIN_ID={20+index%self.cfg["amcl_workers"]}','-e','ROS_LOCALHOST_ONLY=1']
            command+=['bash','-c','source /opt/ros/jazzy/setup.bash && exec python3 -m llm_object_classifier.amcl_replay "$@"','reverse',
                '--input',str(folder/'input.json'),'--branch','A','--output',str(output)]
            # Thread pool workers have stable domain IDs, independent of job completion order.
            import threading
            domain=20+int(threading.current_thread().name.rsplit('_',1)[-1])
            command[command.index(next(v for v in command if v.startswith('ROS_DOMAIN_ID=')))]=f'ROS_DOMAIN_ID={domain}'
            self.progress('amcl',case=name,scenario=scenario,seed=seed,mode=mode)
            output.mkdir(parents=True,exist_ok=True)
            with (output/'replay.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=self.cfg['amcl_workers'],thread_name_prefix='reverse') as pool:list(pool.map(run,enumerate(pending)))
        self.progress('amcl_complete',runs=len(tasks))

    def report(self):
        tasks=self.replays();metrics={};gaps={};missing=[]
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for folder,dest,name,scenario,seed,mode,seq in tasks:
            if not (dest/'poses.json').exists():missing.append(str(dest));continue
            a=self.arrays(seq);est=load_estimates(dest/'poses.json',a['times'])
            summary,position,angle,success=evaluate(a['gt'],est,a['has_gt'],self.data/'reference-code/evaluate.py')
            key=f'{name}/{scenario}/{mode}';metrics.setdefault(key,[]).append(summary)
            scans=np.load(folder/'scans.npz');rows=read(dest/'poses.json');bridge=np.array([r.get('bridged',False) for r in rows])
            gap= scans['natural_gap'] if scenario=='natural' else scans['natural_gap']|scans['artificial_gap']
            gaprows=gap_metrics(a['times'],a['odom'],a['gt'],a['has_gt'],est,bridge,gap,self.cfg['bridge'])
            for row in gaprows:
                span=slice(row['start_index'],row['end_index_exclusive'])
                row['natural']=bool(scans['natural_gap'][span].any());row['artificial']=bool(scans['artificial_gap'][span].any())
            gaps[f'{key}/seed-{seed}']=gaprows
            if seed==self.cfg['seeds'][0]:
                fig,axes=plt.subplots(1,2,figsize=(15,6))
                x0,y0,x1,y1=self.cfg['room_bounds_xyxy']
                map_name=name.removeprefix('reverse-') if name.startswith('reverse') else 'empty'
                grid=np.load(self.output/'maps'/map_name/'occupancy.npy');meta=self.cfg['map']
                extent=[meta['origin'][0],meta['origin'][0]+meta['width']*meta['resolution'],meta['origin'][1],meta['origin'][1]+meta['height']*meta['resolution']]
                for ax,bounds in zip(axes,[[-13,-14,18,15],[x0,y0,x1,y1]]):
                    ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.7)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
                    ax.plot(a['gt'][:,0],a['gt'][:,1],c='green',lw=.7,label='GT')
                    ax.plot(est[:,0],est[:,1],c='royalblue',lw=.5,label='AMCL')
                    failed=a['has_gt']&~success
                    ax.scatter(a['gt'][failed,0],a['gt'][failed,1],c='red',s=3,label='failure (<10cm AND <4deg)')
                    ax.scatter(est[bridge,0],est[bridge,1],c='darkorange',s=5,label='odometry bridge')
                    ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',title=f'{key}, seed {seed}');ax.legend(fontsize=6)
                fig.tight_layout();fig.savefig(self.output/f'trajectory-{name}-{scenario}-{mode}.png',dpi=150);plt.close(fig)
        write(self.output/'metrics.json',{'complete':not missing,'missing_runs':missing,
            'aggregate':{k:aggregate(v) for k,v in metrics.items()},'runs':metrics})
        write(self.output/'gaps.json',gaps)
        lines=['# Obratni test LILocBench','',f'Končano: {len(tasks)-len(missing)}/{len(tasks)} AMCL zagonov.','',
            '| Primer | Položaj <10 cm [%] | Yaw <4° [%] | Oba [%] |','|---|---:|---:|---:|']
        for key,rows in metrics.items():
            agg=aggregate(rows);values=[agg[k] for k in ['position_below_10cm_percent','yaw_below_4deg_percent','success_percent']]
            lines.append('| '+key+' | '+' | '.join(f'{v["mean"]:.2f} ± {v["std"]:.2f}' for v in values)+' |')
        lines+=['','Semena 42–51; populacijski std. Manjkajoča ocena je neuspeh. Oranžno: odometrija; rdeče: neuspeh.',
            'Osnovno stanje ni potrjeno povsem prazen prostor. Spremembe vključujejo tudi prestavljeno pohištvo.',
            'Referenčni zemljevid mapping je neodvisen od static_0 in lt_changes_0. GT se ne uporablja za proženje premostitve.',
            'Podrobnosti: protocol.json, sequence-audit.json, projection-summary.json, geometry.json, gaps.json.']
        lines+=['','![Zemljevidi](maps-comparison.png)','![Geometrija sobe](geometry-room.png)']
        for name in ['reverse-original','reverse-front','reverse-three','reverse-empty','forward-original','forward-front','forward-three']:
            lines.append(f'\n![{name}](trajectory-{name}-natural-bridge.png)')
        (self.output/'README.md').write_text('\n'.join(lines)+'\n')
        self.progress('complete' if not missing else 'incomplete',completed=len(tasks)-len(missing),total=len(tasks))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/lilocbench-reverse.json')
    p.add_argument('--stage',choices=['prepare','preflight','models','project','maps','geometry','amcl','report','all'],default='all')
    args=p.parse_args();e=Experiment(args.config)
    try:
        if args.stage in ['prepare','all']:e.prepare()
        if args.stage in ['preflight','all']:
            statuses=e.model_batches(True);e.project(statuses,True);e.geometry(True)
            write(e.output/'preflight.json',{'passed':True,'projection':'preflight-projection.json','geometry':'preflight-geometry.json'})
        if args.stage in ['models','all']:
            if not (e.output/'preflight.json').exists():raise RuntimeError('Run preflight before full sequence')
            immutable(e.output/'source-hashes.json',{str(p.relative_to(ROOT)):digest(p) for p in [
                Path(__file__),ROOT/'llm_object_classifier/amcl_replay.py',ROOT/'llm_object_classifier/odometry_bridge.py',
                ROOT/'llm_object_classifier/lilocbench_data.py',ROOT/'llm_object_classifier/occupancy_rays.c',ROOT/e.cfg['pipeline_config']]})
            e.model_batches()
        if args.stage in ['project','all']:
            statuses={c:read(e.cache/f'{c}-status-full.json') for c in CAMERAS};e.project(statuses)
        if args.stage in ['maps','all']:e.maps()
        if args.stage in ['geometry','all']:e.geometry()
        if args.stage in ['amcl','all']:e.amcl()
        if args.stage in ['report','all']:e.report()
    except Exception as exc:
        write(e.output/'error.json',{'error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
