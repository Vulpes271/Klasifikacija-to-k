"""Deterministic independent multiresolution correlative LiDAR localization.

No previous pose, odometry, GT, model inference or AMCL enters the matcher.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import platform

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt, map_coordinates
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maintenance.run_reverse_short import points, room, wrap, sha
from llm_object_classifier.localization_pilot import transform_tree


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csvwrite(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def distinct(p, q, metres, degrees):
    return np.linalg.norm(p[:2]-q[:2]) >= metres or abs(wrap(p[2]-q[2])) >= np.deg2rad(degrees)


def separated(poses, scores, count, metres, degrees):
    selected=[]
    # Stable tie resolution: enumeration order, independent of GT.
    for i in np.argsort(-scores, kind='stable'):
        if all(distinct(poses[i],poses[j],metres,degrees) for j in selected):
            selected.append(int(i))
            if len(selected)==count: break
    return np.array(selected,dtype=int)


class CorrelativeMatcher:
    def __init__(self, grid, cfg):
        self.grid=grid;self.cfg=cfg;self.meta=cfg['map']
        self.distance=distance_transform_edt(grid!=100)*self.meta['resolution']
        # Reused by every scan. Unknown cells contribute zero support at every level.
        self.fields={level:np.where(grid!=-1,np.exp(-.5*(self.distance/cfg[level]['sigma_m'])**2),0).astype(np.float32)
            for level in ['coarse','middle','fine']}

    def transform(self, poses, pts):
        c=np.cos(poses[:,2,None]);s=np.sin(poses[:,2,None])
        x=poses[:,0,None]+c*pts[:,0]-s*pts[:,1]
        y=poses[:,1,None]+s*pts[:,0]+c*pts[:,1]
        return x,y

    def score(self, poses, pts, level='fine', categories=False):
        scores=[];all_labels=[]
        for start in range(0,len(poses),512):
            x,y=self.transform(poses[start:start+512],pts)
            # Occupancy cells use floor; distance fields live at cell centres.
            u=(x-self.meta['origin'][0])/self.meta['resolution']
            v=(y-self.meta['origin'][1])/self.meta['resolution']
            ix=np.floor(u).astype(int);iy=np.floor(v).astype(int)
            inside=(ix>=0)&(ix<self.grid.shape[1])&(iy>=0)&(iy<self.grid.shape[0])
            occ=np.full(ix.shape,-1,np.int8);occ[inside]=self.grid[iy[inside],ix[inside]]
            known=occ!=-1
            dist=map_coordinates(self.distance,[v-.5,u-.5],order=1,mode='constant',cval=10.,prefilter=False)
            support=map_coordinates(self.fields[level],[v-.5,u-.5],order=1,mode='constant',cval=0.,prefilter=False)
            support=np.where(known,support,0.)
            # A bounded outlier penalty only in known free space; denominator ALL hits.
            penalty=np.where(occ==0,self.cfg['free_penalty']*np.minimum(dist/self.cfg['free_penalty_saturation_m'],1.),0.)
            scores.extend(np.mean(support-penalty,axis=1))
            if categories:
                labels=np.where(~known,2,np.where(dist<=self.cfg['match_tolerance_m'],0,1))
                all_labels.extend(labels)
        return (np.array(scores),np.array(all_labels)) if categories else np.array(scores)

    def local(self, seeds, pts, level):
        setting=self.cfg[level]
        dx=np.arange(-setting['xy_radius_m'],setting['xy_radius_m']+1e-7,setting['xy_step_m'])
        da=np.deg2rad(np.arange(-setting['yaw_radius_deg'],setting['yaw_radius_deg']+1e-7,setting['yaw_step_deg']))
        offsets=np.array(np.meshgrid(dx,dx,da,indexing='ij')).reshape(3,-1).T
        winners=[];scores=[]
        bounds=self.cfg['allowed_xyxy']
        for seed in seeds:
            poses=seed+offsets;poses[:,2]=wrap(poses[:,2])
            allowed=room(poses,bounds);poses=poses[allowed]
            values=self.score(poses,pts,level)
            j=int(np.argmax(values));winners.append(poses[j]);scores.append(values[j])
        return np.array(winners),np.array(scores)

    def match(self, all_points, candidates):
        if len(all_points)==0:
            return dict(pose=None,accepted=False,reason='no_valid_hits',score=None,runner_up_score=None,margin=None,matched=0,inconsistent=0,unknown=0,valid_hits=0)
        def subset(count):
            return all_points[np.unique(np.linspace(0,len(all_points)-1,min(count,len(all_points))).astype(int))]
        coarse=self.score(candidates,subset(self.cfg['coarse']['beams']),'coarse')
        setting=self.cfg['coarse'];indices=separated(candidates,coarse,setting['keep'],setting['separation_m'],setting['separation_deg'])
        middle,values=self.local(candidates[indices],subset(self.cfg['middle']['beams']),'middle')
        setting=self.cfg['middle'];indices=separated(middle,values,setting['keep'],setting['separation_m'],setting['separation_deg'])
        fine,values=self.local(middle[indices],all_points,'fine')
        best=int(np.argmax(values));pose=fine[best]
        alternatives=[i for i in range(len(fine)) if distinct(pose,fine[i],self.cfg['alternative_separation_m'],self.cfg['alternative_separation_deg'])]
        runner=float(max(values[alternatives])) if alternatives else None
        margin=float(values[best]-runner) if runner is not None else None
        _,labels=self.score(pose[None],all_points,categories=True)
        matched,inconsistent,unknown=np.bincount(labels[0],minlength=3).tolist()
        acceptance=self.cfg['acceptance'];reasons=[];n=len(all_points)
        if n<acceptance['minimum_valid_hits']: reasons.append('few_hits')
        if matched/n<acceptance['minimum_match_fraction']: reasons.append('few_matches')
        if (n-unknown)/n<acceptance['minimum_known_fraction']: reasons.append('unknown')
        if values[best]<acceptance['minimum_score']: reasons.append('low_score')
        if margin is None or margin<acceptance['minimum_distinct_margin']: reasons.append('ambiguous')
        return dict(pose=pose.tolist(),accepted=not reasons,reason=';'.join(reasons) or 'accepted',score=float(values[best]),runner_up_score=runner,margin=margin,
            matched=matched,inconsistent=inconsistent,unknown=unknown,valid_hits=n)


def select_scans(a,cfg):
    gt=a['gt'];valid=a['has_gt']&np.isfinite(gt).all(axis=1);ids=np.flatnonzero(valid)
    distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(gt[ids,:2],axis=0),axis=1))]
    route=np.unique(ids[np.searchsorted(distances,np.linspace(0,distances[-1],cfg['route_samples']))])
    extra=[]
    if room(gt[route],cfg['room_xyxy']).sum()<cfg['add_room_if_fewer_than']:
        inside=ids[room(gt[ids],cfg['room_xyxy'])]
        # Use cumulative whole-path distances, including intervals outside the room.
        rd=distances[np.searchsorted(ids,inside)]
        for target in np.linspace(rd[0],rd[-1],cfg['additional_room_samples']):
            i=int(inside[np.argmin(abs(rd-target))])
            if i not in route and i not in extra: extra.append(i)
    return [dict(scan_index=int(i),timestamp=float(a['times'][i]),group='route' if i in route else 'room_extra',
        room=bool(room(gt[i:i+1],cfg['room_xyxy'])[0]),path_distance_m=float(distances[np.searchsorted(ids,i)])) for i in sorted(set(route)|set(extra))]


def summary(rows, names, prep):
    result=[]
    for name in names:
        for region in ['route','room']:
            used=[r for r in rows if r['variant']==name and (r['group']=='route' if region=='route' else r['room'])]
            valid=[r for r in used if r['has_gt']]
            result.append(dict(variant=name,region=region,n=sum(r['success'] for r in valid),N=len(valid),
                success_percent=100*sum(r['success'] for r in valid)/len(valid),rejected=sum(not r['accepted'] for r in used),
                wrong_accepted=sum(r['accepted'] and not r['success'] for r in valid),missing=sum(not r['has_estimate'] for r in valid),
                median_processing_s=float(np.median([r['processing_s'] for r in used])),p95_processing_s=float(np.percentile([r['processing_s'] for r in used],95)),
                map_load_s=prep[name]['map_load_s'],distance_and_fields_s=prep[name]['distance_and_fields_s']))
    return result


def plots(output,cfg,grids,matchers,scans,rows,gt,stats):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=cfg['variants'];m=cfg['map'];extent=[m['origin'][0],m['origin'][0]+m['width']*m['resolution'],m['origin'][1],m['origin'][1]+m['height']*m['resolution']]
    def background(ax,name,bounds):
        ax.imshow(np.where(grids[name]==100,0,np.where(grids[name]==0,1,.65)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
        ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',xlabel='x [m]',ylabel='y [m]')
    for region,bounds in [('whole',cfg['allowed_xyxy']),('room',cfg['room_xyxy'])]:
        for trajectory in [False,True]:
            fig,axes=plt.subplots(2,2,figsize=(13,11))
            for ax,name in zip(axes.flat,names):
                background(ax,name,bounds);ax.set_title(name+' (mapping control)' if name=='empty' else name)
                if trajectory:
                    ax.plot(gt[:,0],gt[:,1],c='royalblue',lw=.6,label='static_0 GT')
                    used=[r for r in rows if r['variant']==name and (region!='room' or r['room'])]
                    for color,label,predicate in [('green','uspeh',lambda r:r['success']),('red','napacna sprejeta',lambda r:r['accepted'] and not r['success']),('gray','zavrnitev',lambda r:not r['accepted'])]:
                        rr=[r for r in used if predicate(r)]
                        for group,marker in [('route','o'),('room_extra','^')]:
                            subset=[r for r in rr if r['group']==group]
                            if subset:
                                ax.scatter([r['gt_x'] for r in subset],[r['gt_y'] for r in subset],c=color,s=22,marker=marker,edgecolors='white',linewidths=.3,label=label+(' (dodatni)' if group=='room_extra' else ''),zorder=3)
                    stat=next(r for r in stats if r['variant']==name and r['region']==('route' if region=='whole' else 'room'))
                    ax.set_title(f"{name}: {stat['n']}/{stat['N']} ({stat['success_percent']:.2f}%), zavrnjeno {stat['rejected']}"+ ('; metrika samo route' if region=='whole' else '; samo soba'))
                    ax.legend(fontsize=8,loc='lower left')
                elif region=='room':
                    stat=next(r for r in stats if r['variant']==name and r['region']=='room')
                    ax.set_title(f"{name}, soba: {stat['n']}/{stat['N']} ({stat['success_percent']:.2f}%)")
            fig.tight_layout();fig.savefig(output/f"{'poses' if trajectory else 'maps'}-{region}.png",dpi=150);plt.close(fig)
    lookup={(r['variant'],r['scan_index']):r for r in rows};ids=sorted(scans)
    different=[i for i in ids if lookup[('front',i)]['success']!=lookup[('three',i)]['success'] or lookup[('front',i)]['accepted']!=lookup[('three',i)]['accepted']]
    if not different:
        different=sorted(ids,key=lambda i:abs(lookup[('front',i)]['score']-lookup[('three',i)]['score']),reverse=True)
    room_ids=[i for i in ids if lookup[('front',i)]['room']]
    chosen=list(dict.fromkeys(different[:1]+room_ids[:1]+ids[:1]))[:3]
    for i in chosen:
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,name in zip(axes,['front','three']):
            r=lookup[(name,i)];pose=np.array([r['x'],r['y'],r['yaw_rad']]);p=scans[i]
            x,y=matchers[name].transform(pose[None],p);_,labels=matchers[name].score(pose[None],p,categories=True)
            x=x[0];y=y[0];labels=labels[0]
            bounds=[min(x.min(),r['gt_x'])-.5,min(y.min(),r['gt_y'])-.5,max(x.max(),r['gt_x'])+.5,max(y.max(),r['gt_y'])+.5]
            background(ax,name,bounds)
            for j,c,label in [(0,'limegreen','ujemanje'),(1,'red','neskladno prosto'),(2,'darkorange','neznano')]:
                used=labels==j;ax.scatter(x[used],y[used],s=6,c=c,label=f'{label}: {used.sum()}')
            ax.scatter(*pose[:2],marker='x',s=60,c='blue',label='najboljsa poza');ax.scatter(r['gt_x'],r['gt_y'],marker='+',s=65,c='magenta',label='GT')
            ax.set_title(f"{name}, sken {i}: {r['reason']}\nnapaka {r['position_error_m']:.3f} m / {r['yaw_error_deg']:.2f} deg; score {r['score']:.3f}; margin {r['margin']:.3f}")
            ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(output/f'diagnostic-{i}.png',dpi=150);plt.close(fig)
    return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['smoke','full'],required=True)
    parser.add_argument('--config',default='configs/lilocbench-scan-to-map.json');args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    cache=ROOT/cfg['cache'];cache.mkdir(parents=True,exist_ok=True)
    a=dict(np.load(ROOT/cfg['raw']/'raw.npz'));laser=json.loads((ROOT/cfg['raw']/'raw.json').read_text())['laser']
    transform=transform_tree(yaml.safe_load((ROOT/cfg['calibration']).read_text()))('base_link','scan_front_link')
    selected=select_scans(a,cfg)
    selection_file=out/'selected-scans.csv'
    if selection_file.exists():
        old=list(csv.DictReader(selection_file.open()));assert [int(r['scan_index']) for r in old]==[r['scan_index'] for r in selected]
    else: csvwrite(selection_file,selected)
    dump(out/'config.json',cfg)
    if args.stage=='smoke':
        smoke_ids={selected[0]['scan_index'],next(r['scan_index'] for r in selected if r['room']),selected[-1]['scan_index']}
        selected=[r for r in selected if r['scan_index'] in smoke_ids]
    else:
        technical=json.loads((cache/'smoke-validation.json').read_text())
        assert technical['passed'] and technical['config_sha256']==sha(ROOT/args.config)
        assert technical['source_sha256']==sha(__file__)
        assert not (out/'summary.csv').exists(),'Full test already exists: use a new output for a new experiment.'
    lo=cfg['allowed_xyxy'];step=cfg['coarse']['xy_step_m']
    x=np.arange(lo[0],lo[2]+1e-7,step);y=np.arange(lo[1],lo[3]+1e-7,step);yaw=np.deg2rad(np.arange(-180,180,cfg['coarse']['yaw_step_deg']))
    candidates=np.array(np.meshgrid(x,y,yaw,indexing='ij')).reshape(3,-1).T
    prep={};grids={};matchers={};rows=[];scans={}
    for selected_row in selected:
        i=selected_row['scan_index'];scans[i]=points(a['ranges'][i],laser,transform,a['ranges'].shape[1])
    for name in cfg['variants']:
        begin=time.perf_counter();path=ROOT/cfg['maps']/name/'occupancy.npy'
        grid=np.load(path);load_time=time.perf_counter()-begin
        assert grid.shape==(cfg['map']['height'],cfg['map']['width']) and set(np.unique(grid))=={-1,0,100}
        map_yaml=yaml.safe_load((path.parent/'map.yaml').read_text());assert map_yaml['resolution']==cfg['map']['resolution'] and map_yaml['origin']==cfg['map']['origin']
        begin=time.perf_counter();matcher=CorrelativeMatcher(grid,cfg);field_time=time.perf_counter()-begin
        prep[name]=dict(map_load_s=load_time,distance_and_fields_s=field_time,map_sha256=sha(path))
        grids[name]=grid;matchers[name]=matcher
        np.savez_compressed(cache/f'fields-{name}.npz',distance=matcher.distance,**matcher.fields)
        for s in selected:
            i=s['scan_index'];begin=time.perf_counter();result=matcher.match(scans[i],candidates);elapsed=time.perf_counter()-begin
            pose=result.pop('pose');has=pose is not None;valid=bool(a['has_gt'][i] and np.isfinite(a['gt'][i]).all())
            pos=float(np.linalg.norm(np.array(pose[:2])-a['gt'][i,:2])) if has and valid else None
            angle=float(abs(np.rad2deg(wrap(pose[2]-a['gt'][i,2])))) if has and valid else None
            rows.append(dict(variant=name,**s,has_gt=valid,has_estimate=has,x=pose[0] if has else None,y=pose[1] if has else None,yaw_rad=pose[2] if has else None,
                gt_x=float(a['gt'][i,0]) if valid else None,gt_y=float(a['gt'][i,1]) if valid else None,gt_yaw_rad=float(a['gt'][i,2]) if valid else None,
                **result,position_error_m=pos,yaw_error_deg=angle,success=bool(result['accepted'] and valid and pos<cfg['success']['position_m'] and angle<cfg['success']['yaw_deg']),processing_s=elapsed))
            print(args.stage,name,i,f'{elapsed:.3f}s',result['reason'],flush=True)
        csvwrite(cache/f'{args.stage}-poses-progress.csv',rows)
    stats=summary(rows,cfg['variants'],prep)
    plotout=cache/'smoke' if args.stage=='smoke' else out;plotout.mkdir(exist_ok=True)
    diagnostics=plots(plotout,cfg,grids,matchers,scans,rows,a['gt'],stats)
    csvwrite(plotout/'poses.csv',rows);csvwrite(plotout/'summary.csv',stats)
    expected=[(s['scan_index'],s['timestamp']) for s in selected]
    for name in cfg['variants']:
        assert [(r['scan_index'],r['timestamp']) for r in rows if r['variant']==name]==expected
    for r in rows: assert r['matched']+r['inconsistent']+r['unknown']==r['valid_hits']
    validation=dict(passed=True,config_sha256=sha(ROOT/args.config),source_sha256=sha(__file__),test_sha256=sha(ROOT/cfg['raw']/'raw.npz'),
        same_scans_and_timestamps=True,category_counts_partition_all_hits=True,gt_and_odometry_not_passed_to_matcher=True,
        candidates=len(candidates),selected=len(selected),route_count=sum(s['group']=='route' for s in selected),room_extra_count=sum(s['group']=='room_extra' for s in selected),
        diagnostics=diagnostics,preparation=prep,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,backend=cfg['backend'])
    dump(cache/f'{args.stage}-validation.json',validation)
    if args.stage=='full':
        csvwrite(out/'validation.csv',[dict(key=k,value=json.dumps(v,ensure_ascii=False)) for k,v in validation.items()])
        write_report(out,cfg,stats,diagnostics)
    print('COMPLETE',args.stage,flush=True)


def write_report(out,cfg,stats,diagnostics):
    lines=['# Globalna lokalizacija iz enega LiDAR skena','',
        'Večločljivostni 2D correlative scan matching na štirih obstoječih 5 cm zemljevidih poskusa `86f7c57`. `original`, `front`, `three`: ista vožnja `lt_changes_0`; `empty`: neodvisni kontrolni `mapping`, osnovno stanje z običajnim pohištvom. Testni nefiltrirani skeni so samo iz `static_0`.',
        '', 'Vsak sken je obravnavan neodvisno. GT služi samo izboru po dolžini poti, oceni in risbam. Odometrija, prejšnje poze in GT niso vhod ujemanja; ni kompenzacije gibanja znotraj skena. Florence, Qwen, SAM in AMCL se ne zaganjajo. Obstoječa izvedba uporablja CPU NumPy/SciPy; GPU podpore nima.',
        '', '## Vnaprej določeni protokol', '',
        '80 determinističnih vzorcev enakomerno po dolžini poti. Če je med njimi manj kot 12 poz v sobi, dodamo do 20 sobnih vzorcev. Dodatni vzorci niso v metriki `route`; metrika `room` uporablja samo poze znotraj sobe iz obeh skupin. Izbor in časi so v [selected-scans.csv](selected-scans.csv), vse nastavitve v [config.json](config.json). Na izrisih so redni vzorci krogi, dodatni sobni vzorci trikotniki. To so **deleži uspešnih vzorčenih poz, ne deleži površine prostora**.',
        '',f"Isti dovoljeni pravokotnik {cfg['allowed_xyxy']} m je povzet iz prejšnjega prikaza celotnega zemljevida, brez omejevanja s kontrolnim zemljevidom ali testnimi GT pozami. Globalno: 0,5 m / 6°, vse orientacije; 48 prostorsko ločenih hipotez. Srednja raven: ±0,5 m / ±6°, korak 0,1 m / 1°; 16 hipotez. Fino: ±0,1 m / ±1°, korak 2,5 cm / 0,25°. Grobo 96, srednje 192, fino vsi veljavni zadetki (do 811). Enaka pravila za vse štiri različice. Deterministična enumeracija, stabilno razreševanje izenačenj; seme 42 je zabeleženo, naključnosti ni.",
        '', 'Polje razdalj in tri polja podpore se pripravijo enkrat na zemljevid ter uporabijo pri vseh skenih. Podpora je exp(−d²/(2σ²)) pri σ = 35 / 16 / 10 cm. Neznana celica ali zunaj mreže: nič podpore in nič kazni. Znani prosti prostor: omejena kazen 0,15·min(d/0,30 m, 1). Vse normaliziramo s številom **vseh** veljavnih zadetkov. Ujemanje: znan zadetek pri razdalji ≤10 cm; preostali znani zadetki so neskladni. Števila teh kategorij so v CSV.',
        '', 'Sprejem (zamrznjen pred tehničnim preizkusom in nespremenjen za polni test): ≥100 veljavnih zadetkov, ≥35 % ujemanj, ≥60 % znanih zadetkov, ocena ≥0,20 in razlika do naslednjega ločenega kandidata ≥0,01. Ločen kandidat pomeni ≥0,5 m ali ≥15°. Razlika je med ohranjenimi fino obdelanimi hipotezami, ne dokaz globalne enoličnosti. Najboljša poza se shrani tudi ob zavrnitvi.',
        '', 'Uspeh zahteva sprejem ter položaj <10 cm **in** zasuk <4°. Zavrnitve, napačne sprejete in manjkajoče ocene so neuspeh pri veljavnem GT. Za vse izbrane vzorce je GT na voljo. Čas na sken meri ujemanje po pripravi polj in pretvorbi koordinat žarkov; priprava zemljevida je ločena. Časi vključujejo iskanje vseh kandidatov in vse tri ravni.',
        '', '## Rezultati', '', '| Zemljevid / območje | Uspeh n/N (%) | Zavrnitve | Napačne sprejete | Mediana [s] | P95 [s] | Branje mreže [s] | Polje razdalj in podpora [s] |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        lines.append(f"| {r['variant']} / {r['region']} | {r['n']}/{r['N']} ({r['success_percent']:.2f}) | {r['rejected']} | {r['wrong_accepted']} | {r['median_processing_s']:.3f} | {r['p95_processing_s']:.3f} | {r['map_load_s']:.4f} | {r['distance_and_fields_s']:.3f} |")
    f=next(r for r in stats if r['variant']=='front' and r['region']=='route');t=next(r for r in stats if r['variant']=='three' and r['region']=='route')
    fr=next(r for r in stats if r['variant']=='front' and r['region']=='room');tr=next(r for r in stats if r['variant']=='three' and r['region']=='room')
    delta=t['success_percent']-f['success_percent'];dr=tr['success_percent']-fr['success_percent']
    conclusion='pomagajo' if delta>0 else 'ne izboljšajo uspešnosti' if delta==0 else 'zmanjšajo uspešnost'
    lines+=['',f"Tri kamere glede na sprednjo v tem determinističnem vzorcu **{conclusion}**: razlika na poti {delta:+.2f} odstotne točke ({f['n']} → {t['n']}); v sobi {dr:+.2f} odstotne točke ({fr['n']} → {tr['n']}). Gre za en vnaprej določen nabor in eno izvedbo na različico; iz tega ne ocenjujemo statistične zanesljivosti ali uspešnosti sekvenčnega lokalizatorja.",
        '', '![Štirje zemljevidi](maps-whole.png)','![Zemljevidi sobe](maps-room.png)','![Pot in izbrane poze](poses-whole.png)','![Poze v sobi](poses-room.png)',
        '', 'Diagnostika: isti sken pri najboljši ocenjeni pozi na `front` in `three`; zelena = ujemanje, rdeča = neskladni znani prosti zadetki, oranžna = neznano. Modri križ je ocenjena poza, vijolični GT. Če ni razlike v sprejemu/uspehu, je izbran primer z največjo razliko ocen.']
    lines += [f'![Sken {i}](diagnostic-{i}.png)' for i in diagnostics]
    lines+=['','## Ponovitev','', 'Iz korena repozitorija, ob obstoječem lokalnem cacheu (za ponovitev v drugo mapo spremeni `output` v kopiji konfiguracije):','', '```bash',
        'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_scan_to_map.py --stage smoke',
        'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_scan_to_map.py --stage full','```',
        '', 'Tehnični preizkus uporablja tri vnaprej določene skene (začetek, prva sobna poza, konec), preveri skupni nabor in izriše diagnostiko v lokalni cache. Polni preizkus zahteva uspešen tehnični preizkus iste kode in konfiguracije. Že končanega polnega preizkusa ne ponavlja. Veliki skeni, mreže in polja ostanejo lokalno.',
        '', '[Meritve po pozah](poses.csv) · [Povzetek](summary.csv) · [Preverjanje in hashi](validation.csv). Številke tabele se programsko izpišejo iz istih povzetkov kot CSV. Osnovni podatki: LILocBench, Trekel et al., IROS 2025, CC BY-NC-SA 4.0.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
