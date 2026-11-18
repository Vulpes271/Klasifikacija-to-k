"""Confirm distant global proposals using consecutive odometry-linked scans.

Coverage and GT belong exclusively to evaluation; the tracker never sees them.
"""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import maintenance.run_sequential_scan_to_map as tracking
from maintenance.run_scan_to_map import csvwrite,dump
from maintenance.run_reverse_short import room,wrap,sha


class ConfirmingTracker(tracking.Tracker):
    def __init__(self,matcher,cfg,candidates,confirmation):
        super().__init__(matcher,cfg,candidates)
        self.confirmation=confirmation;self.pending=None;self.events=[];self.uncertain=True

    def result(self,timestamp,result,state,global_attempt,prediction_only,event='none',latency=None):
        p=self.pending
        return dict(pose=None if self.pose is None else self.pose.copy(),state=state,prediction_only=prediction_only,
            global_attempt=global_attempt,match_accepted=state in ['local_lidar','global_lidar','confirmed_global'],
            reason=result['reason'] if result else 'waiting_global_retry',score=result['score'] if result else None,margin=result['margin'] if result else None,
            matched=result['matched'] if result else None,inconsistent=result['inconsistent'] if result else None,unknown=result['unknown'] if result else None,
            time_since_lidar_s=None if self.last_lidar is None else timestamp-self.last_lidar,uncertain=self.uncertain,
            pending=p is not None,pending_hits=p['hits'] if p else 0,proposal_id=p['event']['proposal_id'] if p else None,
            candidate_x=float(p['odometry_pose'][0]) if p else None,candidate_y=float(p['odometry_pose'][1]) if p else None,
            candidate_yaw_rad=float(p['odometry_pose'][2]) if p else None,confirmation_event=event,confirmation_latency_s=latency)

    def step(self,timestamp,pts,motion):
        predicted=tracking.move(self.pose,motion) if self.pose is not None else None
        if self.pending is not None:
            p=self.pending;p['odometry_pose']=tracking.move(p['odometry_pose'],motion)
            check=self.local.match(p['odometry_pose'],pts);p['event']['scans_checked']+=1
            residual=np.inf;angle=np.inf
            if check['pose'] is not None:
                residual=float(np.linalg.norm(check['pose'][:2]-p['odometry_pose'][:2]));angle=float(abs(np.rad2deg(wrap(check['pose'][2]-p['odometry_pose'][2]))))
            quality=self.cfg['global_acceptance'];n=len(pts)
            valid=bool(check['accepted'] and n>=quality['minimum_valid_hits'] and check['matched']/n>=quality['minimum_match_fraction'] and
                (n-check['unknown'])/n>=quality['minimum_known_fraction'] and check['score']>=quality['minimum_score'] and
                residual<=self.confirmation['maximum_odometry_residual_m'] and angle<=self.confirmation['maximum_odometry_residual_deg'])
            self.pose=predicted;self.uncertain=True
            if valid:
                p['hits']+=1;p['event']['maximum_residual_m']=max(p['event']['maximum_residual_m'],residual)
                if p['hits']>=self.confirmation['consecutive_scans'] and timestamp-p['start']>=self.confirmation['minimum_span_s']:
                    self.pose=np.array(check['pose']);self.last_lidar=timestamp;self.uncertain=False
                    delay=timestamp-p['start'];p['event'].update(status='confirmed',end_time=timestamp,latency_s=delay)
                    self.pending=None
                    return self.result(timestamp,check,'confirmed_global',False,False,'confirmed',delay)
                return self.result(timestamp,check,'pending_prediction' if self.pose is not None else 'pending_initialization',False,self.pose is not None,'checking')
            p['event'].update(status='rejected',end_time=timestamp,latency_s=timestamp-p['start']);self.pending=None
            return self.result(timestamp,check,'prediction_only' if self.pose is not None else 'unlocalized',False,self.pose is not None,'rejected')
        result=None;state='unlocalized';prediction_only=False;global_attempt=False;event='none'
        if predicted is not None:
            result=self.local.match(predicted,pts)
            if result['accepted']:
                self.pose=np.array(result['pose']);self.last_lidar=timestamp;self.uncertain=False;state='local_lidar'
            else:self.pose=predicted;self.uncertain=True;prediction_only=True;state='prediction_only'
        lost=self.last_lidar is None or timestamp-self.last_lidar>=self.cfg['lost_after_s']
        if (self.pose is None or (lost and prediction_only)) and timestamp-self.last_global>=self.cfg['global_retry_s']:
            global_attempt=True;self.last_global=timestamp;result=self.matcher.match(pts,self.candidates)
            if result['accepted']:
                proposal=np.array(result['pose']);distance=None if self.pose is None else float(np.linalg.norm(proposal[:2]-self.pose[:2]))
                angle=None if self.pose is None else float(abs(np.rad2deg(wrap(proposal[2]-self.pose[2]))))
                distant=self.pose is None or distance>self.confirmation['distant_position_m'] or angle>self.confirmation['distant_yaw_deg']
                if distant:
                    record=dict(proposal_id=len(self.events)+1,start_time=timestamp,end_time=None,kind='initial' if self.pose is None else 'distant',status='pending',
                        innovation_m=distance,innovation_deg=angle,scans_checked=1,latency_s=None,maximum_residual_m=0.)
                    self.events.append(record);self.pending=dict(odometry_pose=proposal,start=timestamp,hits=1,event=record);self.uncertain=True
                    state='pending_prediction' if self.pose is not None else 'pending_initialization';event='proposal'
                else:
                    self.pose=proposal;self.last_lidar=timestamp;self.uncertain=False;state='global_lidar';prediction_only=False;event='near_accepted'
        return self.result(timestamp,result,state,global_attempt,prediction_only,event)


def run_confirmed(name,cfg,base,stage):
    local_cfg=copy.deepcopy(base);local_cfg.update(cache=cfg['cache'],smoke_scans=cfg['smoke_scans'])
    holder=[]
    def factory(matcher,tracking_cfg,candidates):
        instance=ConfirmingTracker(matcher,tracking_cfg,candidates,cfg['confirmation']);holder.append(instance);return instance
    tracking.Tracker=factory
    case=tracking.run_case(name,'wheel_odometry',local_cfg,json.loads((ROOT/base['base_config']).read_text()),stage)
    events=holder[0].events
    for e in events:
        if e['status']=='pending':e['status']='incomplete_at_end'
    dump(ROOT/cfg['cache']/stage/f'{name}-events.json',events)
    return case


def freeze_coverage(cfg,base,matcher):
    out=ROOT/cfg['output'];out.mkdir(exist_ok=True);meta=matcher['map']
    original=np.load(ROOT/matcher['maps']/'original/occupancy.npy');control=np.load(ROOT/matcher['maps']/'empty/occupancy.npy')
    assert original.shape==control.shape==(meta['height'],meta['width'])
    mask=(original!=-1)&(control!=-1)
    path=out/'coverage-mask.png';pixels=np.flipud(mask.astype(np.uint8)*255)
    if path.exists():assert np.array_equal(np.array(Image.open(path)),pixels),'Frozen coverage changed'
    else:Image.fromarray(pixels).save(path)
    a=dict(np.load(ROOT/matcher['raw']/'raw.npz'));valid=a['has_gt']&np.isfinite(a['gt']).all(axis=1);inside=np.zeros(len(valid),bool)
    cells=np.floor((np.nan_to_num(a['gt'][:,:2])-np.array(meta['origin'][:2]))/meta['resolution']).astype(int)
    within=valid&(cells[:,0]>=0)&(cells[:,0]<meta['width'])&(cells[:,1]>=0)&(cells[:,1]<meta['height'])
    inside[within]=mask[cells[within,1],cells[within,0]]
    csvwrite(out/'scan-coverage.csv',[dict(scan_index=i,timestamp=float(t),has_gt=bool(valid[i]),covered=bool(inside[i]),room=bool(room(a['gt'][i:i+1],matcher['room_xyxy'])[0]) if valid[i] else False) for i,t in enumerate(a['times'])])
    protocol=dict(configuration=cfg,tracking=base,matcher=matcher,source_sha256=sha(__file__),tracking_source_sha256=sha(ROOT/'maintenance/run_sequential_scan_to_map.py'),
        config_sha256=sha(ROOT/'configs/lilocbench-relocalization.json'),test_sha256=sha(ROOT/matcher['raw']/'raw.npz'),
        coverage_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),coverage_cells=int(mask.sum()),
        coverage_map_sha256={n:sha(ROOT/matcher['maps']/n/'occupancy.npy') for n in ['original','empty']},
        valid_gt=int(valid.sum()),included_gt=int(inside.sum()),excluded_unmapped_gt=int((valid&~inside).sum()),missing_gt=int((~valid).sum()),
        coverage_policy=cfg['coverage'],coverage_not_passed_to_localizer=True,no_reset_at_coverage_entry=True)
    dest=out/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==protocol,'Frozen protocol changed'
    else:dump(dest,protocol)
    return mask,inside,a,protocol


def collect(cfg,base,stage,inside):
    folder=ROOT/cfg['cache']/stage
    rows=tracking.load_rows(folder,dict(variants=cfg['variants'],modes=['wheel_odometry']))
    count=min(cfg['smoke_scans'],len(inside)) if stage=='smoke' else len(inside)
    baseline=tracking.load_rows(ROOT/cfg['baseline_cache'],dict(variants=['three'],modes=['wheel_odometry']))[:count]
    old=json.loads((ROOT/cfg['baseline_validation']).read_text())
    assert old['source_sha256']==sha(ROOT/'maintenance/run_sequential_scan_to_map.py')
    audit=next(x for x in old['audits'] if x['case']=='three-wheel_odometry')
    matcher=json.loads((ROOT/base['base_config']).read_text())
    assert audit['scan_sha256']==sha(ROOT/matcher['raw']/'raw.npz') and audit['map_sha256']==sha(ROOT/matcher['maps']/'three/occupancy.npy')
    for r in baseline:
        r['case']='three-immediate';r.update(uncertain=r['prediction_only'],pending=False,pending_hits=0,proposal_id=None,candidate_x=None,candidate_y=None,candidate_yaw_rad=None,
            confirmation_event='immediate' if r['state']=='global_lidar' else 'none',confirmation_latency_s=None)
    for r in rows:
        r['case']=r['map']+'-confirmed'
        for key in ['uncertain','pending']:r[key]=r[key]=='True'
        for key in ['candidate_x','candidate_y','candidate_yaw_rad','confirmation_latency_s']:r[key]=float(r[key]) if r[key] else None
    rows=baseline+rows
    for case in dict.fromkeys(r['case'] for r in rows):
        previous=None
        for r in [r for r in rows if r['case']==case]:
            r['covered']=bool(inside[r['scan_index']]);r['jump_m']=float(np.hypot(r['x']-previous['x'],r['y']-previous['y'])) if previous and previous['has_estimate'] and r['has_estimate'] else None
            r['covered_jump_gt_pair']=bool(previous and previous['covered'] and r['covered'])
            r['confident_success']=r['success'] and not r['uncertain'];previous=r
    return rows


def summarize(rows,cfg,stage):
    stats=[];all_events=[]
    for name in cfg['variants']:
        for e in json.loads((ROOT/cfg['cache']/stage/f'{name}-events.json').read_text()):all_events.append(dict(case=name+'-confirmed',**e))
    for case in dict.fromkeys(r['case'] for r in rows):
        r=[r for r in rows if r['case']==case];events=[e for e in all_events if e['case']==case]
        for region in ['whole','room']:
            selected=[x for x in r if x['has_gt'] and (region=='whole' or x['room'])];used=[x for x in selected if x['covered']];estimated=[x for x in used if x['has_estimate']]
            success=[x for x in used if x['success']];N=len(used)
            # Event metrics in the main table also use the SAME coverage at the
            # event's decision time. Unmapped events remain in events.csv only.
            ids={x['timestamp'] for x in used};confirmed=[e for e in events if e['status']=='confirmed' and e['end_time'] in ids]
            delays=[e['latency_s'] for e in confirmed]
            stats.append(dict(case=case,region=region,N=N,n=len(success),success_percent=100*len(success)/N if N else None,
                confident_n=sum(x['confident_success'] for x in used),excluded_unmapped=len(selected)-N,missing_estimates=N-len(estimated),
                p95_position_cm=float(np.percentile([x['position_error_cm'] for x in estimated],95)) if estimated else None,
                p95_yaw_deg=float(np.percentile([x['yaw_error_deg'] for x in estimated],95)) if estimated else None,
                jumps_over_1m=sum(x['covered_jump_gt_pair'] and x['jump_m'] is not None and x['jump_m']>1 for x in used),
                uncertain_percent=100*sum(x['uncertain'] for x in used)/N if N else None,
                first_correct_elapsed_s=success[0]['timestamp']-r[0]['timestamp'] if success else None,
                confirmed_proposals=len(confirmed),median_confirmation_s=float(np.median(delays)) if delays else None,p95_confirmation_s=float(np.percentile(delays,95)) if delays else None,
                median_processing_s=float(np.median([x['processing_s'] for x in used])) if N else None,p95_processing_s=float(np.percentile([x['processing_s'] for x in used],95)) if N else None))
    return stats,all_events


def render(out,rows,stats,mask,a,cfg,matcher):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    meta=matcher['map'];extent=[meta['origin'][0],meta['origin'][0]+meta['width']*meta['resolution'],meta['origin'][1],meta['origin'][1]+meta['height']*meta['resolution']]
    fig,ax=plt.subplots(figsize=(10,8));ax.imshow(mask,origin='lower',extent=extent,cmap='Greens',vmin=0,vmax=1)
    ax.plot(a['gt'][:,0],a['gt'][:,1],c='royalblue',lw=.8);ax.set(xlim=(-14,19),ylim=(-17,7),aspect='equal',title='Skupna pokritost: original lt_changes_0 ∩ mapping; static_0 GT',xlabel='x [m]',ylabel='y [m]')
    fig.tight_layout();fig.savefig(out/'coverage.png',dpi=160);plt.close(fig)
    cmap=LinearSegmentedColormap.from_list('E',['green','yellow','red']);norm=Normalize(0,2,clip=True)
    cases=list(dict.fromkeys(r['case'] for r in rows))
    for region in ['whole','room']:
        fig,axes=plt.subplots(2,3,figsize=(19,12),layout='constrained')
        for ax,case in zip(axes.flat,cases):
            used=[r for r in rows if r['case']==case];grid=np.load(ROOT/matcher['maps']/used[0]['map']/'occupancy.npy')
            ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.65)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
            ref=np.array([[r['gt_x'] if r['has_gt'] else np.nan,r['gt_y'] if r['has_gt'] else np.nan] for r in used]);est=np.array([[r['x'] if r['has_estimate'] else np.nan,r['y'] if r['has_estimate'] else np.nan] for r in used])
            segments=[];colors=[];styles=[]
            for i in range(1,len(used)):
                if not np.isfinite(ref[i-1:i+1]).all():continue
                pair=used[i-1:i+1];segments.append(ref[i-1:i+1]);unmapped=not all(r['covered'] for r in pair)
                colors.append('purple' if unmapped else 'gray' if any(r['E'] is None for r in pair) else cmap(norm(max(r['E'] for r in pair))))
                styles.append('dashed' if unmapped else 'solid')
            ax.add_collection(LineCollection(segments,colors=colors,linestyles=styles,linewidths=1.5,zorder=3));ax.plot(est[:,0],est[:,1],c='blue',lw=.25,alpha=.7,zorder=4)
            bounds=matcher['room_xyxy'] if region=='room' else [-14,-17,19,7]
            ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',xlabel='x [m]',ylabel='y [m]')
            s=next(s for s in stats if s['case']==case and s['region']==region);percent='—' if s['success_percent'] is None else f"{s['success_percent']:.2f}"
            ax.set_title(f"{case}, {region}: {s['n']}/{s['N']} ({percent}%)\nizloceno {s['excluded_unmapped']}; skoki >1 m {s['jumps_over_1m']}",fontsize=9)
        axes.flat[-1].axis('off')
        axes.flat[-1].legend(handles=[Line2D([0],[0],c='blue',lw=.6,label='ocenjena pot'),Line2D([0],[0],c='purple',ls='--',label='zunaj skupne pokritosti; izloceno'),Line2D([0],[0],c='gray',lw=2,label='brez ocene znotraj pokritosti')],loc='center',fontsize=9)
        bar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=list(axes.flat),orientation='horizontal',fraction=.025,pad=.02,ticks=[0,1,2]);bar.ax.set_xticklabels(['0: zelena','1: rumena','≥2: rdeca']);bar.set_label('E = max(napaka polozaja / 10 cm, napaka zasuka / 4°)')
        fig.savefig(out/f'trajectories-{region}.png',dpi=150);plt.close(fig)


def report(out,stats,protocol):
    f=lambda v:'—' if v is None else f'{v:.2f}'
    lines=['# Potrjevanje oddaljenih globalnih poz','',
        f"Pokritost je **pred lokalizacijo** zamrznjena kot presek znanih 5 cm celic (occupancy != −1) iz nefiltriranega `lt_changes_0` in neodvisnega `mapping`, brez dilatacije. Ista maska velja za vseh pet primerjav. Glavna tabela in zaključek uporabljata samo **{protocol['included_gt']}** GT poz v maski; **{protocol['excluded_unmapped_gt']}** nezmapiranih GT poz je izločenih, dodatnih **{protocol['missing_gt']}** skenov nima GT. Vseeno vsak lokalizator neprekinjeno predvaja vseh 8982 skenov; ob vstopu na pokrito območje ni resetiranja. GT in maska nikoli nista vhod algoritma.",
        '', 'Osnova je sekvenčni matcher iz `0510c225`. Oddaljeno pomeni predlog >1 m ali >15° od trenutne odometrijske napovedi; tudi neznana začetna poza zahteva potrditev. Predlog sprejmemo šele po petih zaporednih LiDAR podporah (vsaj 0,20 s). Vmes kandidat sledi izključno relativni kolesni odometriji, lokalni LiDAR fit pa mora ostati znotraj 12 cm in 4° te verige. Lokalni in globalni pragovi kakovosti so enaki predhodnemu preizkusu. Slaba ali dvoumna podpora predlog zavrne. Bližnji globalni predlogi ne potrebujejo dodatne potrditve.',
        '', 'Do potrditve objavljamo prejšnjo odometrijsko napoved kot negotovo, ne oddaljenega predloga; pri neznanem začetku ocene še ni. Pet zaporednih podpor ne zagotavlja pravilnosti v geometrijsko podobnih prostorih. Po potrditvi sledimo z istim lokalnim matcherjem; pri izgubi se ponovno išče globalno. Vsi potrjeni primeri uporabljajo enako konfiguracijo in kolesno odometrijo. Florence, Qwen, Gemma, SAM in AMCL se ne zaganjajo; mreže niso ponovno zgrajene.',
        '', 'Primer `three-immediate` ponovno uporablja preverjen celotni predhodni zagon iste kode, nastavitev, mreže in skenov. Njegovi hashi so preverjeni pred primerjavo. `three-confirmed` je izračunan enkrat ter uporabljen v obeh primerjavah; ostali trije potrjeni zemljevidi se izračunajo enkrat. Ni izbora boljšega semena ali ponavljanja zaradi rezultatov.',
        '', 'Uspeh je geometrijska napaka položaja <10 cm in zasuka <4°, tudi če je odometrijska napoved označena kot negotova. Sekundarni `confident_n` zahteva še nenegotovo oceno. Manjkajoče ocene pri GT so neuspeh; P95 veljajo za razpoložljive ocene. Skok >1 m je razlika dveh sosednjih ocen, pri katerih sta obe GT pozi v isti skupni maski. Dogodki izven pokritosti so v CSV, ne v glavni tabeli. Zakasnitev potrditve je čas posnetka od predloga do sprejema; v tabeli štejejo dogodki, končani v maski in navedenem območju. Čas prve pravilne poze se meri od začetka celotnega posnetka, ne od vstopa v masko.',
        '', '| Primer / območje | Uspeh n/N (%) | P95 položaj [cm] | P95 zasuk [°] | Skoki >1 m | Potrditev med/P95 [s] | Izločeno |','|---|---:|---:|---:|---:|---:|---:|']
    for s in stats:lines.append(f"| {s['case']} / {s['region']} | {s['n']}/{s['N']} ({f(s['success_percent'])}) | {f(s['p95_position_cm'])} | {f(s['p95_yaw_deg'])} | {s['jumps_over_1m']} | {f(s['median_confirmation_s'])}/{f(s['p95_confirmation_s'])} | {s['excluded_unmapped']} |")
    immediate=next(s for s in stats if s['case']=='three-immediate' and s['region']=='whole');confirmed=next(s for s in stats if s['case']=='three-confirmed' and s['region']=='whole')
    lines+=['',f"Na **skupno pokritih pozah** potrjevanje spremeni uspešnost treh kamer za {confirmed['success_percent']-immediate['success_percent']:+.2f} odstotne točke in število skokov >1 m iz {immediate['jumps_over_1m']} na {confirmed['jumps_over_1m']}. To je en parni preizkus; zaključek se ne nanaša na nezmapirane odseke.",
        '', '![Skupna pokritost](coverage.png)','![Celotna neprekinjena pot](trajectories-whole.png)','![Soba, metrike samo v skupni maski](trajectories-room.png)',
        '', 'Zelena E=0, rumena E=1, rdeča E≥2; siva pomeni manjkajočo oceno znotraj pokritosti. Vijolično črtkano je izven skupne pokritosti in izločeno iz glavnih metrik. Tanka modra je celotna ocenjena pot. Barva segmenta uporablja največji E obeh koncev.',
        '', '[Poze](poses.csv) · [Povzetek](summary.csv) · [Predlogi in potrditve](confirmation-events.csv) · [Pokritost po skenih](scan-coverage.csv) · [Natančna maska PNG](coverage-mask.png) · [Vnaprej zamrznjeni protokol](protocol.json) · [Preverjanje](validation.json).',
        '', '## Ponovitev','', '```bash','OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_confirmed_relocalization.py --stage smoke',
        'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_confirmed_relocalization.py --stage full','```',
        '', 'Potrebuje obstoječe `static_0/raw.npz`, `raw.json`, uradno kalibracijo ter štiri `results/lilocbench-reverse/maps/*/occupancy.npy`. Za ponovno uporabo takojšnjega načina sta potrebna predhodni `data/lilocbench-tracking/full/three-wheel_odometry.csv` in revizija `full-validation.json`. Ostala začasna polja/CSV ostanejo v lokalnem `data/lilocbench-relocalization`. Za nov izračun uporabi drugo izhodno mapo v konfiguraciji. Podatki: LILocBench, Trekel et al., IROS 2025, CC BY-NC-SA 4.0.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','smoke','full','render'],required=True);args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/lilocbench-relocalization.json').read_text());base=json.loads((ROOT/cfg['tracking_config']).read_text());matcher=json.loads((ROOT/base['base_config']).read_text())
    cache=ROOT/cfg['cache'];cache.mkdir(exist_ok=True);out=ROOT/cfg['output'];mask,inside,a,protocol=freeze_coverage(cfg,base,matcher)
    if args.stage=='prepare':print(json.dumps({k:protocol[k] for k in ['included_gt','excluded_unmapped_gt','missing_gt','coverage_cells']}));return
    stage='full' if args.stage=='render' else args.stage
    stamp={k:protocol[k] for k in ['source_sha256','tracking_source_sha256','config_sha256','coverage_sha256']}
    if args.stage=='full':
        smoke=json.loads((cache/'smoke-validation.json').read_text());assert all(smoke[k]==v for k,v in stamp.items()) and smoke['passed']
        assert not (out/'summary.csv').exists(),'Full run already complete'
    if args.stage!='render':
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('fork')) as pool:
            futures=[pool.submit(run_confirmed,n,cfg,base,stage) for n in cfg['variants']]
            for future in as_completed(futures):print('completed',future.result(),flush=True)
    rows=collect(cfg,base,stage,inside);stats,events=summarize(rows,cfg,stage);dest=cache/stage if stage=='smoke' else out
    csvwrite(dest/'poses.csv',rows);csvwrite(dest/'summary.csv',stats)
    if events:csvwrite(dest/'confirmation-events.csv',events)
    else:(dest/'confirmation-events.csv').write_text('case,proposal_id,status\n')
    render(dest,rows,stats,mask,a,cfg,matcher)
    n=min(cfg['smoke_scans'],len(a['times'])) if stage=='smoke' else len(a['times'])
    for case in dict.fromkeys(r['case'] for r in rows):
        used=[r for r in rows if r['case']==case];assert [r['scan_index'] for r in used]==list(range(n));np.testing.assert_array_equal([r['timestamp'] for r in used],a['times'][:n])
    validation=dict(passed=True,**stamp,scans_each=n,cases=list(dict.fromkeys(r['case'] for r in rows)),same_scans_and_coverage=True,gt_only_evaluation=True,
        baseline_reused_and_hash_verified=True,missing_gt=protocol['missing_gt'],included_gt=protocol['included_gt'],excluded_unmapped_gt=protocol['excluded_unmapped_gt'])
    dump(cache/f'{stage}-validation.json',validation)
    if stage=='full':dump(out/'validation.json',validation);report(out,stats,protocol)
    print('COMPLETE',args.stage,flush=True)


if __name__=='__main__':main()
