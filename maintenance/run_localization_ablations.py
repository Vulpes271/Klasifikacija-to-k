"""Three independent ablations against the unchanged 695570a confirmed reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from maintenance.run_scan_to_map import CorrelativeMatcher,separated,distinct,csvwrite,dump
from maintenance.run_reverse_short import sha,room,wrap
from maintenance.run_confirmed_relocalization import ConfirmingTracker
import maintenance.run_sequential_scan_to_map as tracking
from llm_object_classifier.localization_reverse import Experiment
from llm_object_classifier.localization_pilot import transform_tree


def ray_penalties(matcher,poses,pts,sensor_origin,cfg):
    """Known occupied cells substantially BEFORE the return add bounded cost.

    Both free and unknown cells give zero cost; no beam gets more than one cost.
    Sample from the calibrated LiDAR origin, not the base_link origin.
    """
    if not len(pts):return np.zeros(len(poses)),[dict(checked=0,conflicts=0) for _ in poses]
    pts=pts[np.unique(np.linspace(0,len(pts)-1,min(len(pts),cfg['beams'])).astype(int))]
    vector=pts-np.asarray(sensor_origin);length=np.linalg.norm(vector,axis=1);unit=vector/np.maximum(length[:,None],1e-12)
    distances=np.arange(cfg['origin_guard_m'],max(cfg['origin_guard_m'],length.max()-cfg['endpoint_guard_m']),cfg['step_m'])
    penalty=[];audit=[];meta=matcher.meta
    for pose in poses:
        if not len(distances):penalty.append(0.);audit.append(dict(checked=len(pts),conflicts=0));continue
        local=np.asarray(sensor_origin)[None,None,:]+unit[:,None,:]*distances[None,:,None]
        c=np.cos(pose[2]);s=np.sin(pose[2]);x=pose[0]+c*local[:,:,0]-s*local[:,:,1];y=pose[1]+s*local[:,:,0]+c*local[:,:,1]
        ix=np.floor((x-meta['origin'][0])/meta['resolution']).astype(int);iy=np.floor((y-meta['origin'][1])/meta['resolution']).astype(int)
        valid=(distances[None,:]<length[:,None]-cfg['endpoint_guard_m'])&(ix>=0)&(ix<matcher.grid.shape[1])&(iy>=0)&(iy<matcher.grid.shape[0])
        occupied=np.zeros(ix.shape,bool);occupied[valid]=matcher.grid[iy[valid],ix[valid]]==100
        blocked=occupied.any(axis=1);first=distances[np.argmax(occupied,axis=1)]
        severity=np.where(blocked,np.clip((length-first-cfg['endpoint_guard_m'])/cfg['severity_saturation_m'],0,1),0.)
        penalty.append(float(cfg['maximum_mean_penalty']*severity.mean()));audit.append(dict(checked=len(pts),conflicts=int(blocked.sum())))
    return np.array(penalty),audit


class RayCheckedMatcher(CorrelativeMatcher):
    def __init__(self,grid,cfg,ray_cfg,sensor_origin):
        super().__init__(grid,cfg);self.ray_cfg=ray_cfg;self.sensor_origin=np.asarray(sensor_origin);self.ray_audit=[];self.current_time=None

    def match(self,pts,candidates):
        if not len(pts):return super().match(pts,candidates)
        def sample(n):return pts[np.unique(np.linspace(0,len(pts)-1,min(len(pts),n)).astype(int))]
        scores=self.score(candidates,sample(self.cfg['coarse']['beams']),'coarse');setting=self.cfg['coarse']
        indices=separated(candidates,scores,setting['keep'],setting['separation_m'],setting['separation_deg'])
        middle,scores=self.local(candidates[indices],sample(self.cfg['middle']['beams']),'middle');setting=self.cfg['middle']
        indices=separated(middle,scores,setting['keep'],setting['separation_m'],setting['separation_deg'])
        fine,endpoints=self.local(middle[indices],pts,'fine')
        # Exactly the same final hypothesis pool as the reference. No truncation
        # confound: at most its 16 highest spatial modes receive the ray check.
        assert len(fine)<=self.ray_cfg['top_candidates']
        costs,details=ray_penalties(self,fine,pts,self.sensor_origin,self.ray_cfg);combined=endpoints-costs
        best=int(np.argmax(combined));pose=fine[best]
        others=[i for i in range(len(fine)) if distinct(pose,fine[i],self.cfg['alternative_separation_m'],self.cfg['alternative_separation_deg'])]
        runner=float(combined[others].max()) if others else None;margin=float(combined[best]-runner) if runner is not None else None
        rank_before=np.empty(len(fine),int);rank_before[np.argsort(-endpoints,kind='stable')]=np.arange(len(fine))+1
        rank_after=np.empty(len(fine),int);rank_after[np.argsort(-combined,kind='stable')]=np.arange(len(fine))+1
        for i,p in enumerate(fine):self.ray_audit.append(dict(timestamp=self.current_time,candidate=i,x=float(p[0]),y=float(p[1]),yaw_rad=float(p[2]),
            endpoint_score=float(endpoints[i]),path_penalty=float(costs[i]),combined_score=float(combined[i]),rank_before=int(rank_before[i]),rank_after=int(rank_after[i]),selected=i==best,**details[i]))
        _,labels=self.score(pose[None],pts,categories=True);matched,inconsistent,unknown=np.bincount(labels[0],minlength=3).tolist()
        a=self.cfg['acceptance'];n=len(pts);reasons=[]
        for failed,reason in [(n<a['minimum_valid_hits'],'few_hits'),(matched/n<a['minimum_match_fraction'],'few_matches'),((n-unknown)/n<a['minimum_known_fraction'],'unknown'),
            (combined[best]<a['minimum_score'],'low_score'),(margin is None or margin<a['minimum_distinct_margin'],'ambiguous')]:
            if failed:reasons.append(reason)
        return dict(pose=pose.tolist(),accepted=not reasons,reason=';'.join(reasons) or 'accepted',score=float(combined[best]),runner_up_score=runner,margin=margin,
            matched=matched,inconsistent=inconsistent,unknown=unknown,valid_hits=n)


class ParallelTracker(ConfirmingTracker):
    def __init__(self,matcher,cfg,candidates,confirmation,parallel):
        super().__init__(matcher,cfg,candidates,confirmation);self.parallel=parallel;self.comparisons=[]

    def step(self,timestamp,pts,motion):
        if self.pending is None or self.pose is None:
            result=super().step(timestamp,pts,motion)
            if self.pending is not None:self.pending.setdefault('advantage_hits',0)
            return result
        p=self.pending;predicted=tracking.move(self.pose,motion);remote=tracking.move(p['odometry_pose'],motion);p['odometry_pose']=remote
        local=self.local.match(predicted,pts);check=self.local.match(remote,pts);p['event']['scans_checked']+=1
        self.pose=np.array(local['pose']) if local['accepted'] else predicted;self.uncertain=not local['accepted']
        if local['accepted']:self.last_lidar=timestamp
        residual=np.inf;angle=np.inf
        if check['pose'] is not None:
            residual=float(np.linalg.norm(check['pose'][:2]-remote[:2]));angle=float(abs(np.rad2deg(wrap(check['pose'][2]-remote[2]))))
        a=self.cfg['global_acceptance'];n=len(pts)
        valid=bool(check['accepted'] and n>=a['minimum_valid_hits'] and check['matched']/n>=a['minimum_match_fraction'] and
            (n-check['unknown'])/n>=a['minimum_known_fraction'] and check['score']>=a['minimum_score'] and
            residual<=self.confirmation['maximum_odometry_residual_m'] and angle<=self.confirmation['maximum_odometry_residual_deg'])
        local_score=float(local['score']) if local['score'] is not None else float(self.matcher.score(predicted[None],pts)[0]) if n else 0.
        advantage=bool(valid and check['score']-local_score>=self.parallel['minimum_score_advantage'])
        p['advantage_hits']=p.get('advantage_hits',0)+1 if advantage else 0
        self.comparisons.append(dict(timestamp=timestamp,proposal_id=p['event']['proposal_id'],local_accepted=local['accepted'],local_score=local_score,
            remote_valid=valid,remote_score=check['score'],advantage=advantage,advantage_hits=p['advantage_hits'],
            local_x=float(self.pose[0]),local_y=float(self.pose[1]),remote_x=float(remote[0]),remote_y=float(remote[1]),
            delta_x=float(motion[0]),delta_y=float(motion[1]),delta_yaw=float(motion[2])))
        state='local_lidar' if local['accepted'] else 'prediction_only'
        cancelled=None
        if not valid:cancelled='rejected'
        elif local['accepted'] and not advantage:cancelled='cancelled_local_recovered'
        elif timestamp-p['start']>self.parallel['maximum_pending_s']:cancelled='timed_out'
        if cancelled:
            p['event'].update(status=cancelled,end_time=timestamp,latency_s=timestamp-p['start']);self.pending=None
            return self.result(timestamp,local,state,False,not local['accepted'],cancelled)
        p['hits']+=1;p['event']['maximum_residual_m']=max(p['event']['maximum_residual_m'],residual)
        if p['hits']>=self.confirmation['consecutive_scans'] and timestamp-p['start']>=self.confirmation['minimum_span_s'] and p['advantage_hits']>=self.parallel['consecutive_advantage_scans']:
            delay=timestamp-p['start'];p['event'].update(status='confirmed',end_time=timestamp,latency_s=delay)
            self.pose=np.array(check['pose']);self.last_lidar=timestamp;self.uncertain=False;self.pending=None
            return self.result(timestamp,check,'confirmed_global',False,False,'confirmed',delay)
        return self.result(timestamp,local,state,False,not local['accepted'],'checking_parallel')


def loss_episodes(rows,cfg):
    """Post-hoc only. Censor at any evaluation-mask/GT boundary."""
    episodes=[];failure_start=None;active=None;success_start=None;last=None
    def close(status,end):
        nonlocal active,failure_start,success_start
        if active is not None:episodes.append(dict(start_time=active,end_time=end,status=status,recovery_s=None))
        active=None;failure_start=None;success_start=None
    for r in rows:
        eligible=r['has_gt'] and r['covered']
        if not eligible:
            close('censored_boundary',last['timestamp'] if last else r['timestamp']);last=r;continue
        t=r['timestamp']
        if r['success']:
            if active is None:failure_start=None
            else:
                if success_start is None:success_start=t
                if t-success_start>=cfg['stable_recovery_s']:
                    episodes.append(dict(start_time=active,end_time=success_start,status='recovered',recovery_s=success_start-active))
                    active=None;failure_start=None;success_start=None
        else:
            success_start=None
            if failure_start is None:failure_start=t
            if active is None and t-failure_start>=cfg['minimum_loss_s']:active=failure_start
        last=r
    close('censored_end',last['timestamp'] if last else None)
    return episodes


def prepare(cfg,base,matcher):
    out=ROOT/cfg['output'];cache=ROOT/cfg['cache'];out.mkdir(exist_ok=True);cache.mkdir(exist_ok=True)
    old_protocol=json.loads((ROOT/'docs/lilocbench-relocalization/protocol.json').read_text())
    assert old_protocol['tracking']==base and old_protocol['matcher']==matcher
    assert old_protocol['source_sha256']==sha(ROOT/'maintenance/run_confirmed_relocalization.py')
    assert old_protocol['tracking_source_sha256']==sha(ROOT/'maintenance/run_sequential_scan_to_map.py')
    frozen=dict(configuration=cfg,tracking=base,matcher=matcher,confirmation=old_protocol['configuration']['confirmation'],
        source_sha256=sha(Path(__file__)),reference_map_sha256=sha(ROOT/matcher['maps']/'three/occupancy.npy'),
        test_sha256=sha(ROOT/matcher['raw']/'raw.npz'),coverage_sha256=old_protocol['coverage_sha256'],
        covered_pose_list_sha256=sha(ROOT/'docs/lilocbench-relocalization/scan-coverage.csv'),
        reference_csv_sha256=sha(ROOT/'data/lilocbench-relocalization/full/three-wheel_odometry.csv'),
        temporal_rule_sha256=sha(ROOT/'docs/lilocbench-temporal-cells/map-protocol.json'),
        lidar_origin_policy='official base_link <- scan_front_link translation; endpoints already include the official extrinsic',
        new_runs=['ray_paths on three','parallel on three','unchanged confirmed localizer on additive map'],no_combined_ablations=True)
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==frozen,'Frozen code/settings/input changed'
    else:dump(path,frozen)
    old=np.load(ROOT/cfg['additive']['old_removals'])['removed'][1];temporal=np.load(ROOT/cfg['additive']['temporal_removals'])['removed']
    assert old.shape==temporal.shape;union=old|temporal;additional=union&~old
    assert not (old&~union).any(),'Old removals reintroduced'
    np.savez_compressed(cache/'union-removals.npz',removed=union,additional=additional)
    e=Experiment(ROOT/'configs/lilocbench-reverse.json');cloud=e.cloud('lt_changes_0');p=cloud['points'];ids=cloud['ids'];meta=e.cfg['map']
    evidence=np.load(ROOT/cfg['additive']['temporal_evidence']);xy=np.floor((p-np.asarray(meta['origin'][:2]))/meta['resolution']).astype(int)
    inside=(xy[:,0]>=0)&(xy[:,0]<meta['width'])&(xy[:,1]>=0)&(xy[:,1]<meta['height']);protected=np.zeros(len(p),bool)
    protected[inside]=evidence['protected'][xy[inside,1]*meta['width']+xy[inside,0]]
    assert not (additional.ravel()[ids]&protected).any()
    e.cache=cache/'ray-build';e.cache.mkdir(exist_ok=True);e.output=cache/'map-build';e.output.mkdir(exist_ok=True)
    e.build_map('lt_changes_0',[old,union],['reference','additive'])
    assert np.array_equal(np.load(e.output/'maps/reference/occupancy.npy'),np.load(ROOT/matcher['maps']/'three/occupancy.npy'))
    native=out/'map-additive';native.mkdir(exist_ok=True)
    for filename in ['map.png','map.yaml']:shutil_copy(e.output/'maps/additive'/filename,native/filename)
    labels=np.load(ROOT/'data/lilocbench-temporal-cells/reference-labels.npy');geometry=[];old_keep=~old.ravel()[ids];extra=additional.ravel()[ids]
    for region,select in [('whole',np.ones(len(p),bool)),('room',room(p,e.cfg['room_bounds_xyxy']))]:
        kept=old_keep&select;remaining=kept&~extra
        geometry.append(dict(region=region,before_retained=int(kept.sum()),after_retained=int(remaining.sum()),additional_removed_hits=int((extra&select).sum()),
            before_excess_free=int((kept&(labels==1)).sum()),after_excess_free=int((remaining&(labels==1)).sum()),additional_excess_removed=int((extra&select&(labels==1)).sum()),
            before_matched=int((kept&(labels==0)).sum()),after_matched=int((remaining&(labels==0)).sum()),matching_hits_lost=int((extra&select&(labels==0)).sum()),
            reintroduced_hits=0))
    csvwrite(out/'additive-geometry.csv',geometry)
    dump(out/'map-audit.json',dict(old_removed_hits=int(old.sum()),union_removed_hits=int(union.sum()),additional_removed_hits=int(additional.sum()),
        no_reintroduced_hits=True,no_additional_hits_in_fixed_protected_cells=True,reference_map_reconstructed_equal=True,
        additional_mask_sha256=hashlib.sha256(additional.tobytes()).hexdigest(),new_map_sha256=sha(e.output/'maps/additive/occupancy.npy')))
    print('PREPARED',json.dumps(geometry),flush=True)
    return frozen


def shutil_copy(source,target):
    import shutil
    shutil.copyfile(source,target)


def run_variant(variant,cfg,base,matcher,confirmation,stage):
    start=time.perf_counter();tracking_cfg=copy.deepcopy(base);tracking_cfg.update(cache=str(Path(cfg['cache'])/variant),smoke_scans=cfg['smoke_scans'])
    (ROOT/tracking_cfg['cache']).mkdir(exist_ok=True)
    name='additive' if variant=='additive' else 'three';matcher_cfg=copy.deepcopy(matcher)
    if variant=='additive':matcher_cfg['maps']=str(Path(cfg['cache'])/'map-build/maps')
    laser_transform=transform_tree(yaml.safe_load((ROOT/matcher['calibration']).read_text()))('base_link','scan_front_link')
    if variant=='ray_paths':tracking.CorrelativeMatcher=lambda grid,setting:RayCheckedMatcher(grid,setting,cfg['rays'],laser_transform[:2,3])
    holder=[]
    def factory(engine,setting,candidates):
        cls=ParallelTracker if variant=='parallel' else ConfirmingTracker
        class Observed(cls):
            def step(self,timestamp,pts,motion):
                self.matcher.current_time=timestamp
                return super().step(timestamp,pts,motion)
        args=[engine,setting,candidates,confirmation]+([cfg['parallel']] if variant=='parallel' else [])
        tracker=Observed(*args);holder.append(tracker);return tracker
    tracking.Tracker=factory
    tracking.run_case(name,'wheel_odometry',tracking_cfg,matcher_cfg,stage)
    t=holder[0]
    for event in t.events:
        if event['status']=='pending':event['status']='incomplete_at_end'
    folder=ROOT/tracking_cfg['cache']/stage
    dump(folder/'events.json',t.events)
    if variant=='ray_paths':csvwrite(folder/'ray-candidates.csv',t.matcher.ray_audit)
    if variant=='parallel':
        if t.comparisons:csvwrite(folder/'parallel-comparisons.csv',t.comparisons)
        else:(folder/'parallel-comparisons.csv').write_text('timestamp,proposal_id\n')
    dump(folder/'runtime.json',dict(wall_s=time.perf_counter()-start,variant=variant))
    return variant


def collect(cfg,stage):
    selection=list(csv.DictReader((ROOT/'docs/lilocbench-relocalization/scan-coverage.csv').open()));n=cfg['smoke_scans'] if stage=='smoke' else len(selection)
    cases=['reference',*cfg['variants']];rows=[];events=[]
    for case in cases:
        if case=='reference':folder=ROOT/'data/lilocbench-relocalization/full';name='three';event=json.loads((folder/'three-events.json').read_text())
        else:
            folder=ROOT/cfg['cache']/case/stage;name='additive' if case=='additive' else 'three';event=json.loads((folder/'events.json').read_text())
        used=tracking.load_rows(folder,dict(variants=[name],modes=['wheel_odometry']))[:n]
        for e in event:
            if e['start_time']<=float(selection[n-1]['timestamp']):events.append(dict(case=case,**e))
        previous=None
        for r in used:
            r['case']=case;r['covered']=selection[r['scan_index']]['covered']=='True'
            for k in ['uncertain','pending']:r[k]=r[k]=='True'
            for k in ['candidate_x','candidate_y','candidate_yaw_rad','confirmation_latency_s']:r[k]=float(r[k]) if r[k] else None
            r['jump_m']=float(np.hypot(r['x']-previous['x'],r['y']-previous['y'])) if previous and previous['has_estimate'] and r['has_estimate'] else None
            r['covered_jump_gt_pair']=bool(previous and previous['covered'] and r['covered']);previous=r
        assert [(r['scan_index'],r['timestamp']) for r in used]==[(int(r['scan_index']),float(r['timestamp'])) for r in selection[:n]]
        rows+=used
    return rows,events


def summarize(rows,cfg):
    stats=[];episodes=[]
    for case in dict.fromkeys(r['case'] for r in rows):
        allrows=[r for r in rows if r['case']==case]
        for region in ['whole','room']:
            selected=[r for r in allrows if r['has_gt'] and r['covered'] and (region=='whole' or r['room'])];est=[r for r in selected if r['has_estimate']];N=len(selected)
            for_episode=[dict(r,covered=r['covered'] and (region=='whole' or r['room'])) for r in allrows]
            lost=loss_episodes(for_episode,cfg['evaluation']);episodes += [dict(case=case,region=region,**e) for e in lost]
            recovered=[e['recovery_s'] for e in lost if e['status']=='recovered'];confirmed=[r for r in selected if r['confirmation_event']=='confirmed']
            near=[r for r in selected if r['confirmation_event']=='near_accepted']
            stats.append(dict(case=case,region=region,N=N,n=sum(r['success'] for r in selected),success_percent=100*sum(r['success'] for r in selected)/N if N else None,
                p95_position_cm=float(np.percentile([r['position_error_cm'] for r in est],95)) if est else None,p95_yaw_deg=float(np.percentile([r['yaw_error_deg'] for r in est],95)) if est else None,
                jumps_over_1m=sum(r['covered_jump_gt_pair'] and r['jump_m'] is not None and r['jump_m']>1 for r in selected),
                global_confirmations=len(confirmed),false_global_confirmations=sum(not r['success'] for r in confirmed),near_global_acceptances=len(near),false_near_acceptances=sum(not r['success'] for r in near),
                recovered_episodes=len(recovered),censored_episodes=sum(e['status']!='recovered' for e in lost),median_relocalization_s=float(np.median(recovered)) if recovered else None,
                p95_relocalization_s=float(np.percentile(recovered,95)) if recovered else None,processing_sum_s=sum(r['processing_s'] for r in allrows),
                median_scan_s=float(np.median([r['processing_s'] for r in allrows])),p95_scan_s=float(np.percentile([r['processing_s'] for r in allrows],95)),
                first_correct_s=next((r['timestamp']-allrows[0]['timestamp'] for r in selected if r['success']),None),excluded_unmapped=sum(r['has_gt'] and not r['covered'] and (region=='whole' or r['room']) for r in allrows)))
    return stats,episodes


def plots(out,rows,stats,cfg,matcher):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    meta=matcher['map'];extent=[meta['origin'][0],meta['origin'][0]+meta['width']*meta['resolution'],meta['origin'][1],meta['origin'][1]+meta['height']*meta['resolution']]
    cmap=LinearSegmentedColormap.from_list('E',['green','yellow','red']);norm=Normalize(0,2,clip=True)
    reference=np.load(ROOT/matcher['maps']/'three/occupancy.npy');additive=np.load(ROOT/cfg['cache']/'map-build/maps/additive/occupancy.npy')
    for region,bounds in [('whole',[-14,-17,19,7]),('room',matcher['room_xyxy'])]:
        fig,axes=plt.subplots(2,2,figsize=(15,12),layout='constrained')
        for ax,case in zip(axes.flat,['reference',*cfg['variants']]):
            grid=additive if case=='additive' else reference;used=[r for r in rows if r['case']==case]
            ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.65)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
            ref=np.array([[r['gt_x'] if r['has_gt'] else np.nan,r['gt_y'] if r['has_gt'] else np.nan] for r in used]);estimate=np.array([[r['x'] if r['has_estimate'] else np.nan,r['y'] if r['has_estimate'] else np.nan] for r in used]);segments=[];colors=[];styles=[]
            for i in range(1,len(used)):
                if not np.isfinite(ref[i-1:i+1]).all():continue
                pair=used[i-1:i+1];unmapped=not all(r['covered'] for r in pair);segments.append(ref[i-1:i+1]);styles.append('dashed' if unmapped else 'solid')
                colors.append('purple' if unmapped else 'gray' if any(r['E'] is None for r in pair) else cmap(norm(max(r['E'] for r in pair))))
            ax.add_collection(LineCollection(segments,colors=colors,linestyles=styles,linewidths=1.5,zorder=3));ax.plot(estimate[:,0],estimate[:,1],c='blue',lw=.25,zorder=4)
            s=next(s for s in stats if s['case']==case and s['region']==region);p='—' if s['success_percent'] is None else f"{s['success_percent']:.2f}"
            ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',xlabel='x [m]',ylabel='y [m]',title=f"{case}, {region}: {s['n']}/{s['N']} ({p}%)\nskoki {s['jumps_over_1m']}; napacne potrditve {s['false_global_confirmations']}/{s['global_confirmations']}")
        bar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=list(axes.flat),orientation='horizontal',fraction=.025,pad=.02,ticks=[0,1,2]);bar.ax.set_xticklabels(['0 zelena','1 rumena','≥2 rdeca']);bar.set_label('E = max(napaka polozaja / 10 cm, napaka zasuka / 4°)')
        fig.legend(handles=[Line2D([0],[0],c='blue',lw=.6,label='ocenjena pot'),Line2D([0],[0],c='purple',ls='--',label='izven iste pokritosti'),Line2D([0],[0],c='gray',label='brez ocene')],loc='outside upper center',ncol=3)
        fig.savefig(out/f'trajectories-{region}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,grid,name in zip(axes,[reference,additive],['reference three','old three OR temporal']):
        ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.65)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1);ax.set(xlim=(-14,19),ylim=(-17,7),aspect='equal',title=name,xlabel='x [m]',ylabel='y [m]')
    fig.tight_layout();fig.savefig(out/'maps-comparison.png',dpi=155);plt.close(fig)


def report(out,stats,cfg):
    f=lambda x:'—' if x is None else f'{x:.2f}'
    lines=['# Tri ločene ablacijske primerjave lokalizacije','',
        'Izhodišče `f4631d0`; referenca je prejšnji filter treh kamer in lokalizator s potrjevanjem iz `695570a`. Ponovno uporabljen je njegov celotni preverjeni zagon. Vsaka nova različica je izvedena enkrat, po tehničnem preizkusu prvih 120 skenov. Brez nove Florence/Qwen/SAM/AMCL inference. Nastavitve in vhodi so vnaprej zamrznjeni v [protocol.json](protocol.json).',
        '', '1. **ray_paths**: ista referenčna mreža in potrjevanje. Hitro grobo/srednje/fino ocenjevanje endpointov ostane enako; vseh največ 16 ohranjenih vrhnjih globalnih hipotez dobi dodatno preverjanje poti 192 enakomerno izbranih žarkov. Izhodišče je uradna LiDAR translacija v base_link, ne izhodišče robota. Korak 2,5 cm; prvih in zadnjih 20 cm se ne kaznuje. Samo znane occupied celice pred izmerjenim koncem štejejo kot ovira; unknown in free sta nevtralna. Kazen žarka raste z manjkajočo razdaljo in saturira po 0,5 m, povprečna kazen kandidata je omejena na 0,20. Imenovalec so vsi izbrani veljavni žarki. Score = endpoint score − path penalty; nato isti sprejem in isto potrjevanje. Množica kandidatov se ne skrajša, zato se učinek ne meša z drugo nastavitvijo iskanja.',
        '', '2. **parallel**: ista referenčna mreža in endpoint matcher. Med oddaljenim predlogom aktivna lokalna poza nadaljuje normalno LiDAR sledenje. Obe hipotezi prejmeta isto relativno odometrijo in isti trenutni sken. Oddaljena mora poleg vseh dosedanjih pogojev potrditve presegati najboljšo lokalno oceno za ≥0,03 vsaj 3 zaporedne skene. Če se lokalni fit obnovi in prednost izgine, se predlog prekliče. Čakanje je omejeno na 2 s; po izteku se ohrani lokalna pot. Pri neznanem začetku brez lokalne hipoteze ostane prejšnje potrjevanje. Noben GT podatek ne odloča o preklopu.',
        '', '3. **additive**: spremeni se samo mreža, lokalizator je nespremenjena referenca s kolesno odometrijo in potrjevanjem. Maska je old_three OR temporal_confirmed_without_fixed. Vse stare odstranitve ostanejo, dodatne v celicah s fixed so prepovedane. Ponovna vključitev je preverjeno nič za vse surove beam ID-je. Vsak odstranjen žarek je v celoti izpuščen iz integracije; ne ustvari prostega prostora za predmetom. Rekonstruirana referenčna mreža je celico za celico enaka izvirniku.',
        '', 'Vsi obdelajo vseh 8982 `static_0` skenov po vrsti, brez resetov na mejah pokritosti. Glavna tabela uporablja samo istih **7550** pokritih GT poz in **478** sobnih; izločeno 1426 nezmapiranih GT poz, v sobi 38; dodatnih 6 skenov nima GT. GT služi samo naknadni evalvaciji in izrisu. Uspeh: položaj <10 cm IN zasuk <4°, tudi za negotovo odometrijsko napoved. Manjkajoča ocena je neuspeh. P95 je nad razpoložljivimi ocenami. Skok >1 m šteje, če sta obe sosednji GT pozi v skupni maski.',
        '', 'Napačna globalna potrditev: `confirmation_event=confirmed` pri GT v maski, kjer oba praga uspeha nista izpolnjena; vključuje tudi začetno potrditev. Bližnji globalni sprejemi so šteti ločeno v CSV. Čas do ponovne lokalizacije: po vsaj 1 s napake iščemo vsaj 1 s neprekinjenega uspeha; zakasnitev teče do začetka tega stabilnega uspeha. Epizode ob izstopu iz pokritosti/GT/sobe so cenzurirane, ne pretvorjene v uspeh. Algoritem se na teh mejah ne resetira. Tabela daje mediano/P95 samo za obnovljene epizode in posebej število cenzuriranih.',
        '', 'Čas izvajanja je vsota merjenih stenskih časov pretvorbe skena in ujemanja (`processing_sum_s`), brez enkratne priprave polj in I/O izvoza. Referenčni čas je shranjen iz prejšnjega zagona. Novi trije procesi tečejo sočasno na CPU, po ena BLAS nit na proces; časi niso ponavljan kontroliran mikrobenchmark. CSV doda mediano/P95 časa na sken ter prvo pravilno pozo. Izvedba je deterministična, seme 42.',
        '', '| Različica / območje | Uspeh n/N (%) | P95 [cm / °] | Skoki >1m | Napačne globalne potrditve | Obnova med/P95 [s] | Obnovljene / cenzurirane | Izračun [s] |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in stats:lines.append(f"| {s['case']} / {s['region']} | {s['n']}/{s['N']} ({f(s['success_percent'])}) | {f(s['p95_position_cm'])} / {f(s['p95_yaw_deg'])} | {s['jumps_over_1m']} | {s['false_global_confirmations']}/{s['global_confirmations']} | {f(s['median_relocalization_s'])}/{f(s['p95_relocalization_s'])} | {s['recovered_episodes']} / {s['censored_episodes']} | {f(s['processing_sum_s'])} |")
    reference=next(s for s in stats if s['case']=='reference' and s['region']=='whole')
    for name in cfg['variants']:
        s=next(s for s in stats if s['case']==name and s['region']=='whole')
        lines+=['',f"Samostatni učinek `{name}` na isti pokriti poti: uspeh {s['success_percent']-reference['success_percent']:+.2f} odstotne točke, skoki {reference['jumps_over_1m']} → {s['jumps_over_1m']}, napačne globalne potrditve {reference['false_global_confirmations']} → {s['false_global_confirmations']}. Ne gre za kombinacijo sprememb ali statistično oceno več ponovitev."]
    geometry=list(csv.DictReader((out/'additive-geometry.csv').open()))
    lines+=['','| Dodatni filter / območje | Dodatno odstranjeni odvečni | Izgubljeni ujemajoči | Odvečni prej → potem | Ujemajoči prej → potem | Ponovno vključeni |','|---|---:|---:|---:|---:|---:|']
    for g in geometry:lines.append(f"| {g['region']} | {g['additional_excess_removed']} | {g['matching_hits_lost']} | {g['before_excess_free']} → {g['after_excess_free']} | {g['before_matched']} → {g['after_matched']} | {g['reintroduced_hits']} |")
    lines+=['','Geometrija uporablja vse gt-kompenzirane `lt_changes_0` konce žarkov in neodvisni `mapping`: ujemanje ≤10 cm; odvečno samo v potrjeno prostem prostoru. Ne gre za GT semantične oznake. Soba je isti endpoint pravokotnik kot prej.',
        '', '![Mreži](maps-comparison.png)','![Celotne poti](trajectories-whole.png)','![Soba, metrike samo sobe](trajectories-room.png)',
        '', 'GT: E=max(napaka položaja/10 cm, napaka zasuka/4°), zelena 0, rumena 1, rdeča ≥2. Vijolično črtkano je izven iste pokritosti, siva brez ocene; tanka modra je celotna ocenjena pot.',
        '', '[Poze](poses.csv) · [Povzetek](summary.csv) · [Globalne potrditve](confirmation-events.csv) · [Epizode izgube in obnove](relocalization-episodes.csv) · [Žarkovni kandidati](ray-candidates.csv) · [Vzporedni oceni](parallel-comparisons.csv) · [Dodatna geometrija](additive-geometry.csv) · [Revizija maske](map-audit.json) · [Preverjanje](validation.json).',
        '', '## Ponovitev','', '```bash','OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_localization_ablations.py --stage prepare',
        'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_localization_ablations.py --stage smoke',
        'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_localization_ablations.py --stage full','```',
        '', 'Potrebuje shranjene skene/odometrijo/kalibracijo static_0, mrežo treh kamer ter referenčne potrjene CSV iz `data/lilocbench-relocalization/full`, projekcijsko masko starega filtra in časovno masko/evidence iz `data/lilocbench-temporal-cells`. Vsi ostanejo lokalno, brez novih uteži. Za novo ponovitev uporabi novo output/cache mapo v `configs/lilocbench-ablations.json`. LILocBench, Trekel et al., IROS 2025, CC BY-NC-SA 4.0.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','smoke','full'],required=True);args=p.parse_args()
    cfg=json.loads((ROOT/'configs/lilocbench-ablations.json').read_text());base=json.loads((ROOT/'configs/lilocbench-tracking.json').read_text());matcher=json.loads((ROOT/base['base_config']).read_text())
    out=ROOT/cfg['output'];cache=ROOT/cfg['cache'];out.mkdir(exist_ok=True);cache.mkdir(exist_ok=True)
    if args.stage=='prepare':prepare(cfg,base,matcher);return
    frozen=json.loads((out/'protocol.json').read_text());assert frozen['configuration']==cfg and frozen['source_sha256']==sha(Path(__file__))
    assert frozen['test_sha256']==sha(ROOT/matcher['raw']/'raw.npz') and frozen['reference_map_sha256']==sha(ROOT/matcher['maps']/'three/occupancy.npy')
    if args.stage=='full':
        smoke=json.loads((cache/'smoke-validation.json').read_text());assert smoke['passed'] and smoke['source_sha256']==frozen['source_sha256']
        assert not (out/'summary.csv').exists(),'Full evaluation already completed'
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('fork')) as pool:
        jobs=[pool.submit(run_variant,v,cfg,base,matcher,frozen['confirmation'],args.stage) for v in cfg['variants']]
        for job in as_completed(jobs):print('completed',job.result(),flush=True)
    rows,events=collect(cfg,args.stage);stats,episodes=summarize(rows,cfg);dest=out if args.stage=='full' else cache/'smoke';dest.mkdir(exist_ok=True)
    csvwrite(dest/'poses.csv',rows);csvwrite(dest/'summary.csv',stats);csvwrite(dest/'confirmation-events.csv',events)
    if episodes:csvwrite(dest/'relocalization-episodes.csv',episodes)
    else:(dest/'relocalization-episodes.csv').write_text('case,region,start_time,end_time,status,recovery_s\n')
    plots(dest,rows,stats,cfg,matcher)
    validation=dict(passed=True,source_sha256=frozen['source_sha256'],same_scans_and_coverage=True,covered_gt_each=7550,room_gt_each=478,
        independent_ablations=True,baseline_reused=True,gt_only_posthoc=True,rows=len(rows))
    dump(cache/f'{args.stage}-validation.json',validation)
    if args.stage=='full':
        for v,filename in [('ray_paths','ray-candidates.csv'),('parallel','parallel-comparisons.csv')]:shutil_copy(cache/v/'full'/filename,out/filename)
        dump(out/'validation.json',validation);report(out,stats,cfg)
    print('COMPLETE',args.stage,flush=True)


if __name__=='__main__':main()
