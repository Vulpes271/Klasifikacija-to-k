"""Global correlative initialization, odometry prediction and local LiDAR correction.

Ground truth is available only to the evaluator and renderer, never Tracker.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from maintenance.run_scan_to_map import CorrelativeMatcher, csvwrite, dump, separated, distinct
from maintenance.run_reverse_short import points, room, wrap, sha, deltas
from llm_object_classifier.localization_pilot import transform_tree


def move(pose, delta):
    c=np.cos(pose[2]);s=np.sin(pose[2])
    return np.array([pose[0]+c*delta[0]-s*delta[1],pose[1]+s*delta[0]+c*delta[1],wrap(pose[2]+delta[2])])


def lattice(xy_radius,xy_step,yaw_radius,yaw_step):
    xy=np.arange(-xy_radius,xy_radius+1e-7,xy_step)
    angles=np.deg2rad(np.arange(-yaw_radius,yaw_radius+1e-7,yaw_step))
    return np.array(np.meshgrid(xy,xy,angles,indexing='ij')).reshape(3,-1).T


class LocalMatcher:
    def __init__(self, matcher, config):
        self.matcher=matcher;self.cfg=config
        self.coarse=lattice(config['xy_radius_m'],config['xy_step_m'],config['yaw_radius_deg'],config['yaw_step_deg'])
        self.fine=lattice(config['fine_xy_radius_m'],config['fine_xy_step_m'],config['fine_yaw_radius_deg'],config['fine_yaw_step_deg'])
        self.probes=np.array([[.12,0,0],[-.12,0,0],[0,.12,0],[0,-.12,0],[0,0,np.deg2rad(3)],[0,0,-np.deg2rad(3)]])

    def match(self, predicted, pts):
        if len(pts)<self.cfg['minimum_valid_hits']:
            return dict(pose=None,accepted=False,reason='few_hits',score=None,margin=None,matched=0,inconsistent=0,unknown=len(pts))
        def sample(n):return pts[np.unique(np.linspace(0,len(pts)-1,min(n,len(pts))).astype(int))]
        trial=predicted+self.coarse;trial[:,2]=wrap(trial[:,2]);bounds=self.matcher.cfg['allowed_xyxy']
        trial=trial[room(trial,bounds)]
        if not len(trial):return dict(pose=None,accepted=False,reason='outside_prior',score=None,margin=None,matched=0,inconsistent=0,unknown=len(pts))
        scores=self.matcher.score(trial,sample(self.cfg['coarse_beams']),'middle')
        ids=separated(trial,scores,self.cfg['keep'],self.cfg['alternative_separation_m'],self.cfg['alternative_separation_deg'])
        winners=[]
        for seed in trial[ids]:
            fine=seed+self.fine;fine[:,2]=wrap(fine[:,2]);fine=fine[room(fine,bounds)]
            values=self.matcher.score(fine,sample(self.cfg['fine_beams']))
            winners.append(fine[int(np.argmax(values))])
        winners=np.array(winners);values=self.matcher.score(winners,pts);best=winners[int(np.argmax(values))]
        probes=best+self.probes;probes[:,2]=wrap(probes[:,2])
        alternatives=np.vstack([winners,probes])
        alternatives=np.array([p for p in alternatives if distinct(best,p,self.cfg['alternative_separation_m']-1e-8,self.cfg['alternative_separation_deg']-1e-8)])
        # Explicit nearby probes prevent a single retained mode from implying certainty.
        score=float(values.max());runner=float(self.matcher.score(alternatives,pts).max())
        margin=score-runner
        _,labels=self.matcher.score(best[None],pts,categories=True)
        matched,inconsistent,unknown=np.bincount(labels[0],minlength=3).tolist();n=len(pts);reasons=[]
        for failed,reason in [(matched/n<self.cfg['minimum_match_fraction'],'few_matches'),((n-unknown)/n<self.cfg['minimum_known_fraction'],'unknown'),
            (score<self.cfg['minimum_score'],'low_score'),(margin<self.cfg['minimum_distinct_margin'],'ambiguous')]:
            if failed:reasons.append(reason)
        return dict(pose=best,accepted=not reasons,reason=';'.join(reasons) or 'accepted',score=score,margin=margin,matched=matched,inconsistent=inconsistent,unknown=unknown)


class Tracker:
    def __init__(self, matcher, cfg, candidates):
        self.matcher=matcher;self.local=LocalMatcher(matcher,cfg['local']);self.cfg=cfg;self.candidates=candidates
        self.pose=None;self.last_lidar=None;self.last_global=-np.inf

    def step(self, timestamp, pts, motion):
        result=None;global_attempt=False;prediction_only=False;state='unlocalized'
        predicted=move(self.pose,motion) if self.pose is not None else None
        if predicted is not None:
            result=self.local.match(predicted,pts)
            if result['accepted']:
                self.pose=np.array(result['pose']);self.last_lidar=timestamp;state='local_lidar'
            else:self.pose=predicted;state='prediction_only';prediction_only=True
        lost=self.last_lidar is None or timestamp-self.last_lidar>=self.cfg['lost_after_s']
        if (self.pose is None or (lost and prediction_only)) and timestamp-self.last_global>=self.cfg['global_retry_s']:
            global_attempt=True;self.last_global=timestamp
            result=self.matcher.match(pts,self.candidates)
            if result['accepted']:
                self.pose=np.array(result['pose']);self.last_lidar=timestamp;state='global_lidar';prediction_only=False
        return dict(pose=None if self.pose is None else self.pose.copy(),state=state,prediction_only=prediction_only,
            global_attempt=global_attempt,match_accepted=bool(result and result['accepted']),reason=result['reason'] if result else 'waiting_global_retry',
            score=result['score'] if result else None,margin=result['margin'] if result else None,
            matched=result['matched'] if result else None,inconsistent=result['inconsistent'] if result else None,unknown=result['unknown'] if result else None,
            time_since_lidar_s=None if self.last_lidar is None else timestamp-self.last_lidar)


def run_case(name, mode, cfg, base, stage):
    """Only this evaluator reads GT. Tracker receives points, time and motion only."""
    cache=ROOT/cfg['cache'];case=f'{name}-{mode}';dest=cache/stage;dest.mkdir(exist_ok=True)
    a=dict(np.load(ROOT/base['raw']/'raw.npz'))
    count=min(len(a['times']),cfg['smoke_scans']) if stage=='smoke' else len(a['times'])
    a={k:v[:count] if v.ndim and v.shape[0]==len(a['times']) else v for k,v in a.items()}
    laser=json.loads((ROOT/base['raw']/'raw.json').read_text())['laser']
    transform=transform_tree(yaml.safe_load((ROOT/base['calibration']).read_text()))('base_link','scan_front_link')
    start=time.perf_counter();grid=np.load(ROOT/base['maps']/name/'occupancy.npy')
    matcher_cfg=copy.deepcopy(base);matcher_cfg['acceptance']=cfg['global_acceptance']
    matcher=CorrelativeMatcher(grid,matcher_cfg);prep=time.perf_counter()-start
    bounds=base['allowed_xyxy'];xy_step=base['coarse']['xy_step_m']
    candidates=np.array(np.meshgrid(np.arange(bounds[0],bounds[2]+1e-7,xy_step),np.arange(bounds[1],bounds[3]+1e-7,xy_step),
        np.deg2rad(np.arange(-180,180,base['coarse']['yaw_step_deg'])),indexing='ij')).reshape(3,-1).T
    motion=deltas(a['odom'],mode=='wheel_odometry')
    if mode=='zero_motion':assert np.count_nonzero(motion)==0
    tracker=Tracker(matcher,cfg,candidates);rows=[]
    for i,timestamp in enumerate(a['times']):
        start=time.perf_counter();pts=points(a['ranges'][i],laser,transform,a['ranges'].shape[1])
        result=tracker.step(float(timestamp),pts,motion[i]);duration=time.perf_counter()-start;pose=result.pop('pose')
        valid=bool(a['has_gt'][i] and np.isfinite(a['gt'][i]).all());present=pose is not None
        pos=float(np.linalg.norm(pose[:2]-a['gt'][i,:2])) if valid and present else None
        yaw=float(abs(np.rad2deg(wrap(pose[2]-a['gt'][i,2])))) if valid and present else None
        rows.append(dict(case=case,map=name,motion_mode=mode,scan_index=i,timestamp=float(timestamp),has_gt=valid,has_estimate=present,
            room=bool(room(a['gt'][i:i+1],base['room_xyxy'])[0]) if valid else False,
            x=float(pose[0]) if present else None,y=float(pose[1]) if present else None,yaw_rad=float(pose[2]) if present else None,
            gt_x=float(a['gt'][i,0]) if valid else None,gt_y=float(a['gt'][i,1]) if valid else None,gt_yaw_rad=float(a['gt'][i,2]) if valid else None,
            **result,delta_x_m=float(motion[i,0]),delta_y_m=float(motion[i,1]),delta_yaw_rad=float(motion[i,2]),valid_hits=len(pts),
            position_error_cm=None if pos is None else pos*100,yaw_error_deg=yaw,E=None if pos is None else max(pos/.1,yaw/4),
            success=bool(pos is not None and pos<.1 and yaw<4),processing_s=duration))
        if (i+1)%1000==0: print(case,i+1,'/',len(a['times']),flush=True)
    csvwrite(dest/f'{case}.csv',rows)
    import hashlib
    dump(dest/f'{case}-audit.json',dict(case=case,scans=count,zero_motion_verified=mode=='zero_motion',
        nonzero_motion_components=int(np.count_nonzero(motion)),motion_sha256=hashlib.sha256(motion.tobytes()).hexdigest(),
        scan_sha256=sha(ROOT/base['raw']/'raw.npz'),map_sha256=sha(ROOT/base['maps']/name/'occupancy.npy'),preparation_s=prep))
    return case


def load_rows(folder,cfg):
    rows=[]
    for name in cfg['variants']:
        for mode in cfg['modes']:
            for r in csv.DictReader((folder/f'{name}-{mode}.csv').open()):
                for k in ['has_gt','has_estimate','room','prediction_only','global_attempt','match_accepted','success']:r[k]=r[k]=='True'
                for k in ['scan_index','valid_hits']:r[k]=int(r[k])
                for k in ['timestamp','x','y','yaw_rad','gt_x','gt_y','gt_yaw_rad','score','margin','time_since_lidar_s','delta_x_m','delta_y_m','delta_yaw_rad','position_error_cm','yaw_error_deg','E','processing_s']:
                    r[k]=float(r[k]) if r[k] else None
                rows.append(r)
    return rows


def summarize(rows,cfg):
    result=[]
    for case in dict.fromkeys(r['case'] for r in rows):
        case_rows=[r for r in rows if r['case']==case]
        for region in ['whole','room']:
            used=[r for r in case_rows if r['has_gt'] and (region=='whole' or r['room'])];evaluated=[r for r in used if r['has_estimate']]
            success=[r for r in used if r['success']];first=success[0] if success else None;n=len(used)
            result.append(dict(case=case,region=region,N=n,n=sum(r['success'] for r in used),success_percent=100*len(success)/n if n else None,
                median_position_cm=float(np.median([r['position_error_cm'] for r in evaluated])) if evaluated else None,
                p95_position_cm=float(np.percentile([r['position_error_cm'] for r in evaluated],95)) if evaluated else None,
                median_yaw_deg=float(np.median([r['yaw_error_deg'] for r in evaluated])) if evaluated else None,
                p95_yaw_deg=float(np.percentile([r['yaw_error_deg'] for r in evaluated],95)) if evaluated else None,
                missing_estimates=n-len(evaluated),prediction_only_count=sum(r['prediction_only'] for r in used),
                prediction_only_percent=100*sum(r['prediction_only'] for r in used)/n if n else None,
                first_correct_elapsed_s=first['timestamp']-case_rows[0]['timestamp'] if first else None,
                scans_through_first_correct=first['scan_index']+1 if first else None,
                processing_through_first_correct_s=sum(r['processing_s'] for r in case_rows if first and r['scan_index']<=first['scan_index']) if first else None,
                median_processing_s=float(np.median([r['processing_s'] for r in used])) if n else None,
                p95_processing_s=float(np.percentile([r['processing_s'] for r in used],95)) if n else None,
                global_searches=sum(r['global_attempt'] for r in case_rows)))
    return result


def render(output,rows,stats,base,cfg):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    cmap=LinearSegmentedColormap.from_list('error',['green','yellow','red']);norm=Normalize(0,2,clip=True)
    m=base['map'];extent=[m['origin'][0],m['origin'][0]+m['width']*m['resolution'],m['origin'][1],m['origin'][1]+m['height']*m['resolution']]
    cases=list(dict.fromkeys(r['case'] for r in rows))
    def fmt(v):return '—' if v is None else f'{v:.2f}'
    for region in ['whole','room']:
        fig,axes=plt.subplots(2,2,figsize=(15,13),layout='constrained')
        for ax,case in zip(axes.flat,cases):
            used=[r for r in rows if r['case']==case];grid=np.load(ROOT/base['maps']/used[0]['map']/'occupancy.npy')
            ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.65)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
            ref=np.array([[r['gt_x'] if r['has_gt'] else np.nan,r['gt_y'] if r['has_gt'] else np.nan] for r in used])
            est=np.array([[r['x'] if r['has_estimate'] else np.nan,r['y'] if r['has_estimate'] else np.nan] for r in used])
            colors=[];segments=[]
            for i in range(1,len(used)):
                if np.isfinite(ref[i-1:i+1]).all():
                    segments.append(ref[i-1:i+1]);errors=[used[j]['E'] for j in [i-1,i]]
                    colors.append('gray' if any(e is None for e in errors) else cmap(norm(max(errors))))
            ax.add_collection(LineCollection(segments,colors=colors,linewidths=1.6,zorder=3))
            ax.plot(est[:,0],est[:,1],c='blue',lw=.25,alpha=.75,zorder=4)
            bounds=base['room_xyxy'] if region=='room' else [extent[0],extent[2],extent[1],extent[3]]
            ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',xlabel='x [m]',ylabel='y [m]')
            s=next(s for s in stats if s['case']==case and s['region']==region)
            ax.set_title(f"{case}, {region}: {s['n']}/{s['N']} ({fmt(s['success_percent'])}%)\npoz. med/P95 {fmt(s['median_position_cm'])}/{fmt(s['p95_position_cm'])} cm; kot {fmt(s['median_yaw_deg'])}/{fmt(s['p95_yaw_deg'])}°\nsamo napoved {fmt(s['prediction_only_percent'])}%; prvi uspeh {fmt(s['first_correct_elapsed_s'])} s",fontsize=9)
        bar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=list(axes.flat),orientation='horizontal',fraction=.025,pad=.025,ticks=[0,1,2])
        bar.ax.set_xticklabels(['0: zelena','1: rumena','≥2: rdeca']);bar.set_label('E = max(napaka polozaja / 10 cm, napaka zasuka / 4°)')
        fig.legend(handles=[Line2D([0],[0],color='blue',lw=.6,label='ocenjena pot'),Line2D([0],[0],color='gray',lw=2,label='GT brez ocene')],loc='outside upper center',ncol=2)
        fig.savefig(output/f'trajectories-{region}.png',dpi=155);plt.close(fig)


def report(out,stats,cfg,base):
    def fmt(v):return '—' if v is None else f'{v:.2f}'
    lines=['# Sekvenčno globalno iskanje in sledenje LiDAR','',
        'Nadgradnja matcherja iz `ba62e13`. Vsi skeni `static_0` v časovnem vrstnem redu, isti nefiltrirani LiDAR, zemljevida `three` (tri kamere, `lt_changes_0`) in `empty` (neodvisni osnovni `mapping` z običajnim pohištvom). Na vsakem en zagon z relativnimi premiki kolesne odometrije in en z ničelnimi premiki. Brez gradnje zemljevidov, modelov ali AMCL.',
        '', 'Začetna poza je neznana. Globalno iskanje uporablja isto mrežo položajev in vseh zasukov kot prejšnji matcher, brez GT. Za sprejem zahteva ≥100 zadetkov, ≥45 % ujemanj, ≥70 % znanih zadetkov, score ≥0,35 in razliko ločene hipoteze ≥0,03. Po sprejemu napoved v SE(2) uporablja samo relativni premik med sosednjima odometrijskima pozama. V načinu `zero_motion` funkcija vrne ničle pred branjem izmerjenih razlik; ničelnost je preverjena in premiki so v CSV.',
        '', 'Lokalno: ±24 cm / ±6°, korak 8 cm / 2°, 192 žarkov; tri ločene hipoteze, nato ±4 cm / ±1°, korak 2 cm / 0,5°, 256 žarkov. Končna izbira in preverjanje na vseh veljavnih žarkih. Ambigvnost preverimo tudi s šestimi odmiki ±12 cm oziroma ±3°. Lokalni sprejem: ≥40 % ujemanj, ≥60 % znanih, score ≥0,25, margin ≥0,01 in ≥100 zadetkov. Neznano ne daje podpore; kazen v prostem ostane omejena kot prej.',
        '', 'Ob zavrnitvi se LiDAR popravek preskoči; objavi se samo napoved in stanje `prediction_only`. V ničelnem načinu to pomeni nespremenjeno zadnjo pozo, ne izmerjene odometrije. Po treh sekundah brez sprejetega LiDAR popravka poskusimo ponovno globalno; med globalnimi poskusi je najmanj tri sekunde. Dokler ni prvega sprejema, ocene ni. Globalni neuspeh ob že obstoječi pozi ohrani napoved; lokalna ponovna pridobitev lahko prekine izgubo. GT ne vpliva na sprejem, izgubo ali izbiro kandidata.',
        '', 'GT je samo za vrednotenje in risanje. Uspeh: položaj <10 cm in zasuk <4°, tudi pri napovednih pozah. Manjkajoča ocena ob GT je neuspeh. Mediana/P95 napak veljata za razpoložljive ocene z GT; manjkajoče so posebej preštete. Delež samo napovedi ima imenovalec vse poze z GT v območju, vključno z manjkajočimi ocenami. Čas prve pravilne poze je čas posnetka od prvega skena, za sobo prav tako od začetka posnetka. CSV doda število skenov in seštevek procesnega časa do tega trenutka. Pomanjkanje GT je izločeno in navedeno v reviziji.',
        '',f"Seme 42, brez naključnih korakov. Štirje ločeni CPU procesi ({cfg['workers']} sočasni delavci), ista koda in nastavitve; vsak obdeluje skene zaporedno. Časi vključujejo pretvorbo žarkov in lokalno/globalno ujemanje, brez enkratne priprave polj. Ni kompenzacije skena z odometrijo. Soba: {base['room_xyxy']} m.",
        '', '| Kombinacija / območje | Uspeh n/N (%) | Položaj med/P95 [cm] | Zasuk med/P95 [°] | Prvi pravilen [s] | Samo napoved [%] | Manjkajoče |','|---|---:|---:|---:|---:|---:|---:|']
    for s in stats:
        lines.append(f"| {s['case']} / {s['region']} | {s['n']}/{s['N']} ({fmt(s['success_percent'])}) | {fmt(s['median_position_cm'])}/{fmt(s['p95_position_cm'])} | {fmt(s['median_yaw_deg'])}/{fmt(s['p95_yaw_deg'])} | {fmt(s['first_correct_elapsed_s'])} | {fmt(s['prediction_only_percent'])} | {s['missing_estimates']} |")
    for name in cfg['variants']:
        w=next(s for s in stats if s['case']==name+'-wheel_odometry' and s['region']=='whole');z=next(s for s in stats if s['case']==name+'-zero_motion' and s['region']=='whole')
        lines+=['',f"Na zemljevidu `{name}` izmerjena odometrija spremeni uspešnost za {w['success_percent']-z['success_percent']:+.2f} odstotne točke. To je ena parna izvedba, ne ocena statistične zanesljivosti."]
    lines+=['','![Celotni zemljevidi in poti](trajectories-whole.png)','![Soba: metrike samo sobe](trajectories-room.png)',
        '', 'GT je obarvan po E: zelena pri 0, rumena pri 1, rdeča pri ≥2, siva brez ocene. Barva segmenta je največji E njegovih koncev. Tanka modra je ocenjena pot, tudi med odometrijsko napovedjo. Ob manjkajočem GT so odseki prekinjeni.',
        '', '[Vse poze in prehodi stanj](poses.csv) · [Povzetek](summary.csv) · [Nastavitve](config.json) · [Revizija](validation.csv).',
        '', '## Ponovitev','', '```bash','OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_sequential_scan_to_map.py --stage smoke',
        'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_sequential_scan_to_map.py --stage full','```',
        '', 'Potrebni lokalni vhodi: `data/lilocbench-filtered-map/static_0/raw.npz`, `raw.json`, `data/lilocbench-pilot/static_0/transformations.yaml`, `results/lilocbench-reverse/maps/{three,empty}/occupancy.npy`; izvor in merilo določa osnovna konfiguracija. Potrebni Python paketi: NumPy, SciPy, PyYAML, Matplotlib. Uporabljeni pomožni moduli so v repozitoriju. Podatki LILocBench, Trekel et al., IROS 2025, CC BY-NC-SA 4.0.',
        '', 'Polni zagon se ne ponavlja, če obstajajo končni rezultati; za novo ponovitev uporabi kopijo konfiguracije z drugim `output` in `cache`. Tehnični preizkus uporabi prvih 120 skenov vseh kombinacij in zamrzne hash kode in nastavitev. [Inventar in kandidati za ločen pregled čiščenja](INVENTORY.md). Brisanje ni bilo izvedeno.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['smoke','full','render'],required=True);p.add_argument('--config',default='configs/lilocbench-tracking.json');args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());base=json.loads((ROOT/cfg['base_config']).read_text());cache=ROOT/cfg['cache'];cache.mkdir(exist_ok=True);out=ROOT/cfg['output'];out.mkdir(exist_ok=True)
    stamp=dict(source_sha256=sha(__file__),config_sha256=sha(ROOT/args.config),base_config_sha256=sha(ROOT/cfg['base_config']))
    stage='full' if args.stage=='render' else args.stage;folder=cache/stage;folder.mkdir(exist_ok=True)
    if args.stage=='full':
        smoke=json.loads((cache/'smoke-validation.json').read_text());assert smoke['passed'] and all(smoke[k]==v for k,v in stamp.items())
        assert not (out/'summary.csv').exists(),'Full run already complete; render does not rerun localization.'
    if args.stage!='render':
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('fork')) as pool:
            jobs=[pool.submit(run_case,n,m,cfg,base,stage) for n in cfg['variants'] for m in cfg['modes']]
            for job in as_completed(jobs):print('completed',job.result(),flush=True)
    rows=load_rows(folder,cfg);stats=summarize(rows,cfg);dest=folder if stage=='smoke' else out
    csvwrite(dest/'poses.csv',rows);csvwrite(dest/'summary.csv',stats);render(dest,rows,stats,base,cfg)
    cases=list(dict.fromkeys(r['case'] for r in rows));reference=[r['timestamp'] for r in rows if r['case']==cases[0]]
    for case in cases:
        used=[r for r in rows if r['case']==case];assert [r['timestamp'] for r in used]==reference
        if used[0]['motion_mode']=='zero_motion':assert all(r[k]==0 for r in used for k in ['delta_x_m','delta_y_m','delta_yaw_rad'])
    validation=dict(passed=True,**stamp,scans_each=len(reference),cases=cases,same_scans_and_timestamps=True,zero_motion_recomputed_verified=True,
        missing_gt_each=sum(not r['has_gt'] for r in rows if r['case']==cases[0]),audits=[json.loads((folder/f'{case}-audit.json').read_text()) for case in cases])
    dump(cache/f'{stage}-validation.json',validation)
    if stage=='full':
        dump(out/'config.json',dict(tracking=cfg,matcher=base));csvwrite(out/'validation.csv',[dict(key=k,value=json.dumps(v)) for k,v in validation.items()]);report(out,stats,cfg,base)
    print('COMPLETE',args.stage,flush=True)


if __name__=='__main__':main()
