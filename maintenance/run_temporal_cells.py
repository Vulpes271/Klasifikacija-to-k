"""Offline temporal semantic cells, then one unchanged confirmed localization.

The build stage reads lt_changes_0 and mapping only. No inference is invoked.
"""
import argparse
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm_object_classifier.localization_reverse import Experiment,CAMERAS,nearest,classify_points
from llm_object_classifier.localization_pilot import pose_table,transform_tree,project_scan
from maintenance.run_scan_to_map import csvwrite,dump
from maintenance.run_reverse_short import sha,room
import maintenance.run_sequential_scan_to_map as tracking
import maintenance.run_confirmed_relocalization as relocal


MOVABLE=1
FIXED=2


def semantic_layer(folder,frame,model):
    """Recover actual classes; never reinterpret cached -1 as fixed evidence."""
    shape=(frame['image_size'][1],frame['image_size'][0]);layer=np.zeros(shape,np.uint8)
    for region in frame['regions']:
        labels=[s['llm']['models'].get(model,{}).get('classification') for s in region['sources']]
        bit=MOVABLE if labels and all(v=='movable' for v in labels) else FIXED if labels and all(v=='fixed' for v in labels) else 0
        if not bit:continue
        path=folder/region['mask_path'];assert sha(path)==region['mask_sha256'],f'Changed mask: {path}'
        binary=np.asarray(Image.open(path))>0;assert binary.shape==shape
        layer[binary]|=bit
    return layer


def accumulate_votes(cells,flags,times,cell_count,rule):
    """One independent epoch per class/cell/scan, with an explicit time gate."""
    counts=np.zeros((2,cell_count),np.uint16);camera_bits=np.zeros((2,cell_count),np.uint8)
    last=np.full((2,cell_count),-np.inf);first=np.full((2,cell_count),np.nan)
    per_camera=np.zeros((2,3),np.int64)
    for scan,t in enumerate(times):
        valid=cells[scan]>=0;cell=cells[scan,valid]
        unique,inverse=np.unique(cell,return_inverse=True)
        for label,bit in enumerate([MOVABLE,FIXED]):
            bits=np.zeros(len(cell),np.uint8)
            for ci in range(3):bits|=np.where((flags[ci,scan,valid]&bit)!=0,1<<ci,0).astype(np.uint8)
            merged=np.zeros(len(unique),np.uint8);np.bitwise_or.at(merged,inverse,bits)
            accepted=(merged!=0)&(t-last[label,unique]>=rule['minimum_epoch_gap_s'])
            selected=unique[accepted];sources=merged[accepted]
            counts[label,selected]+=1;camera_bits[label,selected]|=sources;last[label,selected]=t
            missing=np.isnan(first[label,selected]);first[label,selected[missing]]=t
            for ci in range(3):per_camera[label,ci]+=np.count_nonzero(sources&(1<<ci))
    popcount=np.array([i.bit_count() for i in range(8)],np.uint8)
    protected=counts[1]>=rule['minimum_fixed_epochs_to_protect']
    confirmed=(counts[0]>=rule['minimum_movable_epochs'])&(popcount[camera_bits[0]]>=rule['minimum_distinct_cameras'])
    removed=confirmed&~protected
    return dict(counts=counts,camera_bits=camera_bits,first=first,last=last,protected=protected,confirmed=confirmed,removed_cells=removed,accepted_epochs_by_camera=per_camera)


def algorithm_sha():
    return hashlib.sha256((inspect.getsource(semantic_layer)+inspect.getsource(accumulate_votes)).encode()).hexdigest()


def freeze_map_protocol(cfg,e):
    out=ROOT/cfg['output'];out.mkdir(exist_ok=True);cache=ROOT/cfg['cache'];cache.mkdir(exist_ok=True)
    value=dict(configuration=cfg,map=e.cfg['map'],room_bounds=e.cfg['room_bounds_xyxy'],algorithm_sha256=algorithm_sha(),
        projection_sha256=sha(e.cache/'projection.npz'),mapping_scans_sha256=sha(e.raw/'lt_changes_0/raw.npz'),
        gt_poses_sha256=sha(e.data/'lt_changes_0/gt_poses.txt'),extrinsics_sha256=sha(e.data/'lt_changes_0/transformations.yaml'),
        camera_status_sha256={c:sha(e.cache/f'{c}-status-full.json') for c in CAMERAS},
        camera_intrinsics_sha256={c:sha(e.data/'lt_changes_0'/c/'color/intrinsics.yaml') for c in CAMERAS},
        inference_calls=0,static_0_read_in_build=False,reference_geometry='independent mapping; continuous endpoints, known free occupancy confirms excess')
    path=out/'map-protocol.json'
    if path.exists():
        old=json.loads(path.read_text());assert all(old[k]==v for k,v in value.items()),'Frozen map rule/input changed'
    else:dump(path,dict(**value,frozen_utc=datetime.now(timezone.utc).isoformat(),build_source_sha256=sha(Path(__file__))))
    return out,cache


def recover_flags(e,cache):
    dest=cache/'explicit-projection.npz';saved=dict(np.load(e.cache/'projection.npz'))
    if dest.exists():return dict(np.load(dest))['flags']
    a=e.arrays('lt_changes_0');laser=e.laser('lt_changes_0');assert np.array_equal(saved['indices'],np.arange(len(a['times'])))
    tree=transform_tree(yaml.safe_load((e.data/'lt_changes_0/transformations.yaml').read_text()))
    base_scan=tree('base_link','scan_front_link');ot,op,_=pose_table(e.data/'lt_changes_0/dingo_velocity_controller/odom.txt')
    flags=np.zeros(saved['decisions'].shape,np.uint8);valid={};stamps={}
    for c in CAMERAS:
        statuses=json.loads((e.cache/f'{c}-status-full.json').read_text());valid[c]=[r for r in statuses if r['classification_valid'] and r.get('mask_report')];stamps[c]=np.array([r['timestamp'] for r in valid[c]])
    @lru_cache(maxsize=8)
    def report(path):return {r['frame_id']:r for r in json.loads(Path(path).read_text())['frames']}
    @lru_cache(maxsize=8)
    def layer(path,fid):return semantic_layer(Path(path).parent,report(path)[fid],e.pilot['classification_model'])
    verified=0
    for ci,c in enumerate(CAMERAS):
        intrinsic=yaml.safe_load((e.data/'lt_changes_0'/c/'color/intrinsics.yaml').read_text());extrinsic=tree('base_link',c+'_color_optical_frame')
        for i,t in enumerate(a['times']):
            associated=saved['association'][i,ci]
            if not np.isfinite(associated):continue
            j=nearest(stamps[c],t+laser['scan_time']/2,e.pilot['mask_max_dt']);assert j is not None
            row=valid[c][j];assert row['timestamp']==associated
            uv,visible=project_scan(a['ranges'][i],t,associated,laser,base_scan,extrinsic,intrinsic,ot,op,e.pilot['pose_max_gap'])
            assert np.array_equal(visible,saved['visible'][ci,i]),'Projection differs from cached camera geometry'
            pixels=np.floor(uv[visible]).astype(int);flags[ci,i,visible]=layer(row['mask_report'],row['frame_id'])[pixels[:,1],pixels[:,0]]
            assert np.all((flags[ci,i,saved['decisions'][ci,i]>0]&MOVABLE)!=0)
            verified+=1
        print('Recovered explicit labels:',c,'movable',np.count_nonzero(flags[ci]&MOVABLE),'fixed',np.count_nonzero(flags[ci]&FIXED),flush=True)
    np.savez_compressed(dest,flags=flags)
    dump(cache/'projection-audit.json',dict(valid_camera_scan_associations_verified=verified,all_cached_visibility_equal=True,
        cached_movable_votes_preserved=True,movable_hits_by_camera=[int(np.count_nonzero(x&MOVABLE)) for x in flags],
        fixed_hits_by_camera=[int(np.count_nonzero(x&FIXED)) for x in flags],missing_uncertain_conflict_abstain=True))
    return flags


def build(cfg):
    e=Experiment(ROOT/cfg['reverse_config']);out,cache=freeze_map_protocol(cfg,e)
    flags=recover_flags(e,cache);a=e.arrays('lt_changes_0');cloud=e.cloud('lt_changes_0');meta=e.cfg['map'];p=cloud['points'];ids=cloud['ids']
    xy=np.floor((p-np.asarray(meta['origin'][:2]))/meta['resolution']).astype(int)
    inside=(xy[:,0]>=0)&(xy[:,0]<meta['width'])&(xy[:,1]>=0)&(xy[:,1]<meta['height'])
    cell_ids=np.full(a['ranges'].size,-1,np.int32);cell_ids[ids[inside]]=xy[inside,1]*meta['width']+xy[inside,0];cells=cell_ids.reshape(a['ranges'].shape)
    evidence=accumulate_votes(cells,flags,a['times'],meta['width']*meta['height'],cfg['rule'])
    assert np.all(evidence['accepted_epochs_by_camera'].sum(axis=0)>0),'A camera supplied no evidence'
    removed=np.zeros(a['ranges'].shape,bool);good=cells>=0;removed[good]=evidence['removed_cells'][cells[good]]
    np.savez_compressed(cache/'cell-evidence.npz',**evidence);np.savez_compressed(cache/'removed.npz',removed=removed)
    old_removed=np.load(e.cache/'projection.npz')['removed'][1]
    audit=dict(cameras=list(CAMERAS),accepted_movable_epochs_by_camera=evidence['accepted_epochs_by_camera'][0].tolist(),
        accepted_fixed_epochs_by_camera=evidence['accepted_epochs_by_camera'][1].tolist(),
        movable_candidate_cells=int(evidence['confirmed'].sum()),removed_cells=int(evidence['removed_cells'].sum()),
        protected_fixed_cells=int(evidence['protected'].sum()),movable_confirmed_but_fixed_protected_cells=int((evidence['confirmed']&evidence['protected']).sum()),
        old_removed_raw_hits=int(old_removed.sum()),new_removed_raw_hits=int(removed.sum()),all_three_cameras_contributed=True,
        projections=json.loads((cache/'projection-audit.json').read_text()))
    evidence_rows=[]
    for cell in np.flatnonzero(evidence['counts'].any(axis=0)):
        row,col=divmod(int(cell),meta['width'])
        evidence_rows.append(dict(cell_id=int(cell),row=row,column=col,x=meta['origin'][0]+(col+.5)*meta['resolution'],y=meta['origin'][1]+(row+.5)*meta['resolution'],
            movable_epochs=int(evidence['counts'][0,cell]),fixed_epochs=int(evidence['counts'][1,cell]),movable_camera_bits=int(evidence['camera_bits'][0,cell]),
            fixed_camera_bits=int(evidence['camera_bits'][1,cell]),protected=bool(evidence['protected'][cell]),removed=bool(evidence['removed_cells'][cell])))
    csvwrite(out/'cell-evidence.csv',evidence_rows)
    old_grid=np.load(e.output/'maps/three/occupancy.npy');original_grid=np.load(e.output/'maps/original/occupancy.npy')
    original_cache=e.cache;e.cache=cache/'ray-build';e.cache.mkdir(exist_ok=True);e.output=cache/'map-build';e.output.mkdir(exist_ok=True)
    e.build_map('lt_changes_0',[np.zeros(a['ranges'].shape,bool),removed],['reconstructed-original','temporal'])
    assert np.array_equal(np.load(e.output/'maps/reconstructed-original/occupancy.npy'),original_grid)
    audit.update(original_grid_reconstructed_identically=True,removed_rays_omitted_entirely=True,ray_integrator_sha256=sha(ROOT/'llm_object_classifier/occupancy_rays.c'))
    new_grid=np.load(e.output/'maps/temporal/occupancy.npy');audit['new_grid_sha256']=sha(e.output/'maps/temporal/occupancy.npy')
    for name,grid in [('before',old_grid),('after',new_grid)]:
        dest=out/'maps'/name;dest.mkdir(parents=True,exist_ok=True)
        pixels=np.full(grid.shape,205,np.uint8);pixels[grid==0]=254;pixels[grid==100]=0
        Image.fromarray(np.flipud(pixels)).save(dest/'map.png')
        (dest/'map.yaml').write_text(yaml.safe_dump(dict(image='map.png',resolution=meta['resolution'],origin=meta['origin'],negate=0,occupied_thresh=meta['occupied_thresh'],free_thresh=meta['free_thresh'])))
    e.cache=original_cache
    reference=e.cloud('mapping')['points'];ref_grid=np.load(ROOT/'results/lilocbench-reverse/maps/empty/occupancy.npy');known=np.load(e.cache/'grid-mapping.npz')['known'][0]
    labels_path=cache/'reference-labels.npy'
    if labels_path.exists():labels=np.load(labels_path)
    else:
        labels=classify_points(p,cKDTree(reference),ref_grid,meta,cfg['point_tolerance_m'],known);np.save(labels_path,labels)
    old_keep=~old_removed.ravel()[ids];new_keep=~removed.ravel()[ids];regions=room(p,e.cfg['room_bounds_xyxy']);geo=[];effects=[]
    for region,select in [('whole',np.ones(len(p),bool)),('room',regions)]:
        for variant,keep in [('before',old_keep),('after',new_keep)]:
            counts=np.bincount(labels[keep&select],minlength=4);total=int(counts.sum())
            geo.append(dict(variant=variant,region=region,retained_hits=total,matched_hits=int(counts[0]),matched_percent=100*counts[0]/total,
                excess_free_hits=int(counts[1]),excess_free_percent=100*counts[1]/total,unobserved_hits=int(counts[2]),other_observed_hits=int(counts[3])))
        excess=select&(labels==1);matched=select&(labels==0);old_matched=int((matched&old_keep).sum())
        effects.append(dict(region=region,excess_disappeared=int((excess&old_keep&~new_keep).sum()),excess_reintroduced=int((excess&~old_keep&new_keep).sum()),
            net_excess_reduction=int((excess&old_keep).sum()-(excess&new_keep).sum()),old_matching_hits=old_matched,
            old_matching_retained=int((matched&old_keep&new_keep).sum()),old_matching_retained_percent=100*(matched&old_keep&new_keep).sum()/old_matched,
            matching_reintroduced=int((matched&~old_keep&new_keep).sum()),old_matching_removed=int((matched&old_keep&~new_keep).sum())))
    csvwrite(out/'geometry.csv',geo);csvwrite(out/'geometry-effects.csv',effects);dump(out/'map-audit.json',audit)
    map_plots(out,old_grid,new_grid,evidence,meta,e.cfg['room_bounds_xyxy'],geo)
    print('MAP COMPLETE',json.dumps(audit),flush=True)


def map_plots(out,old,new,evidence,meta,room_bounds,geometry):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    extent=[meta['origin'][0],meta['origin'][0]+meta['width']*meta['resolution'],meta['origin'][1],meta['origin'][1]+meta['height']*meta['resolution']]
    def background(ax,grid,bounds):
        ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.65)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
        ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',xlabel='x [m]',ylabel='y [m]')
    for region,bounds in [('whole',[-14,-17,19,7]),('room',room_bounds)]:
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for ax,name,grid in zip(axes,['before','after'],[old,new]):
            background(ax,grid,bounds);g=next(g for g in geometry if g['variant']==name and g['region']==region)
            ax.set_title(f"{name}, {region}\nujemanje {g['matched_hits']}; odvecno v prostem {g['excess_free_hits']}")
        fig.tight_layout();fig.savefig(out/f'maps-{region}.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for ax,key,color,label in zip(axes,['removed_cells','protected'],['red','royalblue'],['odstranjene celice','zascitene z fixed']):
            background(ax,old,bounds);row,col=np.where(evidence[key].reshape(old.shape));x=meta['origin'][0]+(col+.5)*meta['resolution'];y=meta['origin'][1]+(row+.5)*meta['resolution']
            keep=(x>=bounds[0])&(x<=bounds[2])&(y>=bounds[1])&(y<=bounds[3]);ax.scatter(x[keep],y[keep],s=4,c=color,marker='s')
            ax.set_title(f'{region}: {label}, n={keep.sum()}')
        fig.tight_layout();fig.savefig(out/f'cell-decisions-{region}.png',dpi=160);plt.close(fig)


def localization(cfg,stage):
    e=Experiment(ROOT/cfg['reverse_config']);out,cache=freeze_map_protocol(cfg,e)
    assert (out/'map-audit.json').exists(),'Build the map and freeze its rule first'
    base_relocal=json.loads((ROOT/cfg['relocalization_config']).read_text());base=json.loads((ROOT/base_relocal['tracking_config']).read_text());matcher=json.loads((ROOT/base['base_config']).read_text())
    baseline_protocol=json.loads((ROOT/'docs/lilocbench-relocalization/protocol.json').read_text())
    assert baseline_protocol['configuration']==base_relocal and baseline_protocol['tracking']==base and baseline_protocol['matcher']==matcher
    assert baseline_protocol['source_sha256']==sha(ROOT/'maintenance/run_confirmed_relocalization.py')
    baseline_audit=json.loads((ROOT/base_relocal['cache']/'full/three-wheel_odometry-audit.json').read_text())
    assert baseline_audit['scan_sha256']==sha(ROOT/matcher['raw']/'raw.npz') and baseline_audit['map_sha256']==sha(ROOT/matcher['maps']/'three/occupancy.npy')
    coverage=np.flipud(np.asarray(Image.open(ROOT/'docs/lilocbench-relocalization/coverage-mask.png'))>0)
    assert hashlib.sha256(coverage.tobytes()).hexdigest()==baseline_protocol['coverage_sha256']
    selection=list(__import__('csv').DictReader((ROOT/'docs/lilocbench-relocalization/scan-coverage.csv').open()))
    inside=np.array([r['covered']=='True' for r in selection]);assert inside.sum()==7550
    stamp=dict(source_sha256=sha(Path(__file__)),config_sha256=sha(ROOT/'configs/lilocbench-temporal-cells.json'),new_grid_sha256=sha(cache/'map-build/maps/temporal/occupancy.npy'),
        static_0_sha256=baseline_audit['scan_sha256'],coverage_sha256=baseline_protocol['coverage_sha256'],same_localizer_and_settings=True,baseline_reused=True,included_gt=7550)
    if stage=='full':
        smoke=json.loads((cache/'smoke-validation.json').read_text());assert all(smoke[k]==v for k,v in stamp.items()) and smoke['passed']
        assert not (out/'localization.csv').exists(),'Full run already complete; no repeats'
    derived=copy_json(matcher);derived['maps']=str((cache/'map-build/maps').relative_to(ROOT));dump(cache/'matcher.json',derived)
    run_base=copy_json(base);run_base['base_config']=str((cache/'matcher.json').relative_to(ROOT))
    run_cfg=copy_json(base_relocal);run_cfg.update(cache=cfg['cache'],smoke_scans=cfg['smoke_scans'],variants=['temporal'])
    relocal.run_confirmed('temporal',run_cfg,run_base,stage)
    folder=cache/stage
    shutil.copyfile(ROOT/base_relocal['cache']/'full/three-events.json',folder/'three-events.json')
    rows=tracking.load_rows(ROOT/base_relocal['cache']/'full',dict(variants=['three'],modes=['wheel_odometry']))
    n=cfg['smoke_scans'] if stage=='smoke' else len(selection);rows=rows[:n]+tracking.load_rows(folder,dict(variants=['temporal'],modes=['wheel_odometry']))
    for r in rows:
        r['case']=r['map']+'-confirmed'
        for key in ['uncertain','pending']:r[key]=r[key]=='True'
        for key in ['candidate_x','candidate_y','candidate_yaw_rad','confirmation_latency_s']:r[key]=float(r[key]) if r[key] else None
        r['covered']=bool(inside[r['scan_index']]);r['confident_success']=r['success'] and not r['uncertain']
    for case in dict.fromkeys(r['case'] for r in rows):
        previous=None;used=[r for r in rows if r['case']==case]
        assert [r['timestamp'] for r in used]==[float(r['timestamp']) for r in selection[:n]]
        for r in used:
            r['jump_m']=float(np.hypot(r['x']-previous['x'],r['y']-previous['y'])) if previous and previous['has_estimate'] and r['has_estimate'] else None
            r['covered_jump_gt_pair']=bool(previous and previous['covered'] and r['covered']);previous=r
    summary_cfg=dict(cache=cfg['cache'],variants=['three','temporal']);stats,events=relocal.summarize(rows,summary_cfg,stage)
    dest=folder if stage=='smoke' else out;csvwrite(dest/'poses.csv',rows);csvwrite(dest/'localization.csv',stats);csvwrite(dest/'confirmation-events.csv',events)
    trajectory_plots(dest,rows,stats,np.load(ROOT/matcher['maps']/'three/occupancy.npy'),np.load(cache/'map-build/maps/temporal/occupancy.npy'),matcher)
    dump(cache/f'{stage}-validation.json',dict(passed=True,**stamp,scans_each=n,same_scan_indices_timestamps_and_mask=True))
    if stage=='full':
        shutil.copyfile(cache/'full-validation.json',out/'validation.json');write_report(out,cfg,stats)
    print('LOCALIZATION COMPLETE',stage,flush=True)


def copy_json(value):return json.loads(json.dumps(value))


def trajectory_plots(out,rows,stats,old,new,matcher):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    cmap=LinearSegmentedColormap.from_list('error',['green','yellow','red']);norm=Normalize(0,2,clip=True);meta=matcher['map']
    extent=[meta['origin'][0],meta['origin'][0]+meta['width']*meta['resolution'],meta['origin'][1],meta['origin'][1]+meta['height']*meta['resolution']]
    for region,bounds in [('whole',[-14,-17,19,7]),('room',matcher['room_xyxy'])]:
        fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained')
        for ax,case,grid in zip(axes,['three-confirmed','temporal-confirmed'],[old,new]):
            used=[r for r in rows if r['case']==case];ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.65)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
            ref=np.array([[r['gt_x'] if r['has_gt'] else np.nan,r['gt_y'] if r['has_gt'] else np.nan] for r in used]);est=np.array([[r['x'] if r['has_estimate'] else np.nan,r['y'] if r['has_estimate'] else np.nan] for r in used])
            segments=[];colors=[];styles=[]
            for i in range(1,len(used)):
                if not np.isfinite(ref[i-1:i+1]).all():continue
                pair=used[i-1:i+1];unmapped=not all(r['covered'] for r in pair);segments.append(ref[i-1:i+1]);styles.append('dashed' if unmapped else 'solid')
                colors.append('purple' if unmapped else 'gray' if any(r['E'] is None for r in pair) else cmap(norm(max(r['E'] for r in pair))))
            ax.add_collection(LineCollection(segments,colors=colors,linestyles=styles,linewidths=1.5,zorder=3));ax.plot(est[:,0],est[:,1],c='blue',lw=.25,zorder=4)
            s=next(s for s in stats if s['case']==case and s['region']==region);percent='—' if s['success_percent'] is None else f"{s['success_percent']:.2f}"
            ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',xlabel='x [m]',ylabel='y [m]',title=f"{case}, {region}: {s['n']}/{s['N']} ({percent}%)\nskoki >1 m: {s['jumps_over_1m']}; izloceno {s['excluded_unmapped']}")
        bar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=list(axes),orientation='horizontal',fraction=.035,pad=.02,ticks=[0,1,2]);bar.ax.set_xticklabels(['0 zelena','1 rumena','≥2 rdeca']);bar.set_label('E = max(napaka polozaja / 10 cm, napaka zasuka / 4°)')
        fig.legend(handles=[Line2D([0],[0],c='blue',lw=.6,label='ocenjena pot'),Line2D([0],[0],c='purple',ls='--',label='izven zamrznjene pokritosti'),Line2D([0],[0],c='gray',label='brez ocene')],loc='outside upper center',ncol=3)
        fig.savefig(out/f'trajectories-{region}.png',dpi=155);plt.close(fig)


def write_report(out,cfg,stats):
    import csv
    geometry=list(csv.DictReader((out/'geometry.csv').open()));effects=list(csv.DictReader((out/'geometry-effects.csv').open()));audit=json.loads((out/'map-audit.json').read_text())
    fmt=lambda v:'—' if v is None else f'{float(v):.2f}'
    lines=['# Časovno združevanje semantičnih oznak po celicah','',
        'Izhodišče `695570a`. Florence → Qwen → SAM ostaja nespremenjen; uporabljene so shranjene maske, Qwen oznake, časovne asociacije in kalibracije vseh treh kamer. Brez nove inference. Nova mreža iz vseh istih 6540 skenov `lt_changes_0`, 5 cm, isti izvor, GT poze za gradnjo in ista integracija žarkov.',
        '', '## Pravilo, zamrznjeno pred static_0','',
        'Celica potrebuje ≥3 neodvisne movable epohe, med njimi ≥0,5 s, in ≥2 različni kameri med sprejetimi epohami. Vsi žarki in kamere istega skena za isto celico skupaj štejejo največ eno epoho posameznega razreda. Kamere dajejo samo bitno evidenco pogledov, ne dodatnih časovnih glasov. Najmanj en veljaven fixed dokaz celico trajno zaščiti. Pravilo je konservativno in ni bilo izbrano glede na static_0.',
        '', 'Maskirana regija glasuje samo, če so vsi njeni izvorni Qwen razredi movable ali vsi fixed. Uncertain, manjkajoči in konfliktni razredi ne glasujejo. Čiste prekrivajoče regije se združijo; fixed ima prednost. Stari cache −1 ni fixed oznaka: združuje fixed, uncertain in manjkajoče. Zato so dejanski razredi obnovljeni iz shranjenih mask; iste časovne asociacije in UV projekcije so preverjene proti stari vidnosti. Vsaka uporabljena maska ima preverjen SHA256.',
        '', 'Gradnja je offline v dveh prehodih: najprej dokazovanje na celotni mapirni vožnji, nato odstranitev vseh zadetkov v potrjenih celicah brez fixed. Predhodna per-beam odstranitev se ne ohrani samodejno: novi filter jo nadomesti. Tudi zgodnejši zadetki take celice so odstranjeni šele v drugem prehodu. Izpuščen je celoten žarek, brez endpoint posodobitve ali prostega prostora za odstranjenim zadetkom. Drugi ohranjeni žarki lahko isti prostor še vedno opazujejo. Ne gre za sprotni filter.',
        '', f"Odstranjenih celic: **{audit['removed_cells']}**; fixed zaščitenih: **{audit['protected_fixed_cells']}**; movable potrjenih, a zaščitenih s fixed: **{audit['movable_confirmed_but_fixed_protected_cells']}**. Prejšnji filter odstrani {audit['old_removed_raw_hits']} surovih zadetkov, novi {audit['new_removed_raw_hits']}. Vse tri kamere prispevajo veljavne dokaze. [Revizija](map-audit.json), [dokazi po celicah](cell-evidence.csv), [zamrznjeni mapirni protokol](map-protocol.json). Ponovno zgrajena nefiltrirana kontrola je celico za celico enaka izvirniku.",
        '', '## Geometrija proti neodvisnemu mapping','', 'Razdalja ≤10 cm med zveznimi konci LiDAR žarkov. Odvečen zadetek mora biti neujemajoč in ležati v referenčno potrjenem prostem prostoru. Neopaženo je ločena kategorija. Imenovalci so zadetki, ne deduplicirane celice; soba pomeni koordinate endpointov v istem vnaprej določenem pravokotniku.',
        '', '| Zemljevid / območje | Ohranjeni zadetki | Ujemajoči n (%) | Odvečni v prostem n (%) | Neopaženi |','|---|---:|---:|---:|---:|']
    for r in geometry:lines.append(f"| {r['variant']} / {r['region']} | {r['retained_hits']} | {r['matched_hits']} ({fmt(r['matched_percent'])}) | {r['excess_free_hits']} ({fmt(r['excess_free_percent'])}) | {r['unobserved_hits']} |")
    lines+=['','| Območje | Izginili odvečni | Ponovno vključeni odvečni | Neto manj odvečnih | Stari ujemajoči ostali n (%) | Stari ujemajoči odstranjeni |','|---|---:|---:|---:|---:|---:|']
    for r in effects:lines.append(f"| {r['region']} | {r['excess_disappeared']} | {r['excess_reintroduced']} | {r['net_excess_reduction']} | {r['old_matching_retained']} ({fmt(r['old_matching_retained_percent'])}) | {r['old_matching_removed']} |")
    lines+=['','Negativno »neto manj« pomeni povečanje odvečne geometrije. Ločeno štetje ponovnih vključitev prepreči, da bi prikazali samo ugodne odstranitve. Podrobnosti: [geometry.csv](geometry.csv), [parne spremembe](geometry-effects.csv).',
        '', '![Stari in novi zemljevid](maps-whole.png)','![Soba: stari in novi zemljevid](maps-room.png)','![Odstranjene in zaščitene celice](cell-decisions-whole.png)','![Odločitve samo v sobi](cell-decisions-room.png)',
        '', '## Lokalizacija, ista pokritost in nastavitve','',
        'Predvajanje vseh 8982 nefiltriranih `static_0` skenov po vrsti, brez resetov ob vstopu v pokritost. Isti večločljivostni matcher, kolesna odometrija, pet zaporednih podpor za potrditev oddaljenega predloga in vse nastavitve iz `695570a`. GT je samo za evalvacijo in risanje. Stari `three-confirmed` zagon se ponovno uporabi po preverjanju kode, nastavitev, mreže in skenov. Nova različica: tehnični preizkus 120 skenov in en celoten zagon; brez izbora semena ali ponavljanja zaradi rezultata.',
        '', 'Glavna maska je nespremenjeni presek iz `695570a`; **7550** istih referenčnih poz za celotno pot in **478** za sobo. Izločeno je 1426 nezmapiranih GT poz, od teh 38 v sobi; 6 skenov nima GT. Nova mreža maske ne spremeni. Uspeh zahteva položaj <10 cm in zasuk <4°, tudi pri negotovi odometrijski napovedi; `confident_n` je sekundarno število brez negotovih. Manjkajoče ocene so neuspeh. P95 je nad razpoložljivimi ocenami v maski. Skoki >1 m zahtevajo obe sosednji GT pozi v isti zamrznjeni maski.',
        '', '| Zemljevid / območje | Uspeh n/N (%) | P95 položaj [cm] | P95 zasuk [°] | Skoki >1 m |','|---|---:|---:|---:|---:|']
    for s in stats:lines.append(f"| {s['case']} / {s['region']} | {s['n']}/{s['N']} ({fmt(s['success_percent'])}) | {fmt(s['p95_position_cm'])} | {fmt(s['p95_yaw_deg'])} | {s['jumps_over_1m']} |")
    for region in ['whole','room']:
        old=next(s for s in stats if s['case']=='three-confirmed' and s['region']==region);new=next(s for s in stats if s['case']=='temporal-confirmed' and s['region']==region)
        lines+=['',f"Na skupno pokritih pozah ({region}) časovno združevanje spremeni uspeh za **{new['success_percent']-old['success_percent']:+.2f} odstotne točke**, P95 položaja {fmt(old['p95_position_cm'])} → {fmt(new['p95_position_cm'])} cm in skoke {old['jumps_over_1m']} → {new['jumps_over_1m']}. To je en parni preizkus; ne ocenjuje statistične zanesljivosti."]
    lines+=['','![Celotna pot](trajectories-whole.png)','![Soba: samo metrike sobe](trajectories-room.png)',
        '', 'E = max(napaka položaja/10 cm, napaka zasuka/4°): zelena 0, rumena 1, rdeča ≥2; siva brez ocene. Vijolično črtkano: izven iste zamrznjene pokritosti; tanka modra: celotna ocenjena pot. Barva segmenta uporabi največji E koncev.',
        '', '[Meritve po pozah](poses.csv) · [Lokalizacijska tabela](localization.csv) · [Potrditve](confirmation-events.csv) · [Preverjanje](validation.json). Native 5 cm PNG/YAML izvoza sta v `maps/before` in `maps/after`, z ohranjeno razliko occupied/free/unknown.',
        '', '## Ponovitev','', 'Iz korena repozitorija z obstoječimi lokalnimi maskami/cacheom in podatki iz `695570a`:','', '```bash',
        'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_temporal_cells.py --stage build',
        'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_temporal_cells.py --stage smoke',
        'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_temporal_cells.py --stage full','```',
        '', 'Nastavitve: `configs/lilocbench-temporal-cells.json`. Lokalni vmesni podatki: `data/lilocbench-temporal-cells`; velike projekcije in raw skeni niso dodani v Git. Polni zagon se ob obstoječi končni tabeli ne ponovi. Za novo ponovitev uporabi novo izhodno/cache mapo. Seme 42, deterministična izvedba na CPU, brez inference. LILocBench: Trekel et al., IROS 2025, CC BY-NC-SA 4.0.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['build','smoke','full'],required=True);args=p.parse_args()
    cfg=json.loads((ROOT/'configs/lilocbench-temporal-cells.json').read_text())
    if args.stage=='build':build(cfg)
    else:localization(cfg,args.stage)


if __name__=='__main__':main()
