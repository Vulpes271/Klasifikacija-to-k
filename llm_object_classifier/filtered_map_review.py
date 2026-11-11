"""Local diagnostics and compact public report for the paired map experiment."""
import csv
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image,ImageDraw
import yaml

from .localization_pilot import ROOT,write,evaluate,pose_table,transform_tree,project_scan,mask_layers
from .localization_full import read
from .localization_comparison import load_estimates
from .localization_filtered_map import aggregate,beam_world

PAPER_KEYS=['position_rmse_cm','position_max_cm','position_below_5cm_percent','position_below_10cm_percent',
            'yaw_rmse_deg','yaw_max_deg','yaw_below_2deg_percent','yaw_below_4deg_percent']
PAPER={
 'static_0':[(3.12,.05),(7.72,.45),(93.87,.73),(100,0),(1.86,.01),(5.39,.3),(70.39,.44),(96.78,.2)],
 'dynamics_0':[(3.21,.2),(8.56,.51),(89.56,2.42),(100,0),(1.87,.02),(4.83,.21),(73.57,.57),(95.28,1.18)],
 'lt_changes_0':[(13.51,.74),(70.3,5.23),(61.41,1.84),(82.89,.79),(4.49,.31),(25.31,2.91),(61.56,.66),(84.01,.85)],
 'lt_changes_dynamics_0':[(11.13,1.68),(72.07,18.53),(64.18,1.33),(87.34,.96),(2.75,.22),(20.34,2.85),(68.54,.62),(90.09,.61)]}


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def map_image(cfg,name):
    return np.asarray(Image.open(ROOT/cfg['output']/'maps'/name/'map.png').convert('L'))


def extent(cfg):
    m=cfg['map'];x,y,_=m['origin'];return [x,x+m['width']*m['resolution'],y,y+m['height']*m['resolution']]


def bounds(cfg):
    grids=[np.load(ROOT/cfg['output']/'maps'/name/'occupancy.npy') for name in ['original','filtered']]
    y,x=np.where((grids[0]>=0)|(grids[1]>=0));m=cfg['map'];o=m['origin'];r=m['resolution']
    return [o[0]+x.min()*r-1,o[0]+(x.max()+1)*r+1,o[1]+y.min()*r-1,o[1]+(y.max()+1)*r+1]


def background(ax,cfg,name,limits=None):
    ax.imshow(map_image(cfg,name),cmap='gray',vmin=0,vmax=255,origin='upper',extent=extent(cfg))
    x0,x1,y0,y1=limits or bounds(cfg);ax.set(xlim=(x0,x1),ylim=(y0,y1),xlabel='x [m]',ylabel='y [m]');ax.set_aspect('equal')


def plot_maps(cfg):
    plt=plotting();output=ROOT/cfg['output'];fig,axes=plt.subplots(1,3,figsize=(19,7),sharex=True,sharey=True)
    for ax,name in zip(axes,['original','filtered']):background(ax,cfg,name);ax.set_title(name+' mapping scans')
    a=np.load(output/'maps/original/occupancy.npy');b=np.load(output/'maps/filtered/occupancy.npy')
    # Free/unknown/occupied transition categories, common axes, no interpolation.
    rgb=np.full((*a.shape,3),245,np.uint8);rgb[a==100]=50;rgb[a==-1]=205
    rgb[(a==100)&(b==-1)]=[230,50,30];rgb[(a==100)&(b==0)]=[255,165,0]
    rgb[(a!=100)&(b==100)]=[135,35,180];rgb[(a!=b)&(a!=100)&(b!=100)]=[30,135,210]
    axes[2].imshow(np.flipud(rgb),extent=extent(cfg),origin='upper');axes[2].set_aspect('equal');axes[2].set_title('Cell-class differences')
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(color=np.array(c)/255,label=t) for c,t in [([230,50,30],'occupied → unknown'),([255,165,0],'occupied → free'),([135,35,180],'new occupied'),([30,135,210],'free ↔ unknown')]],fontsize=7)
    fig.suptitle('Mapping sequence only — identical public GT beam poses and 5-cm grid')
    fig.tight_layout();fig.savefig(output/'maps_comparison.png',dpi=140);plt.close(fig)


def plot_trajectories(cfg,sequence,arrays,estimates,success):
    plt=plotting();from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    gt=arrays['gt'];valid=arrays['has_gt'];fig,axes=plt.subplots(1,2,figsize=(15,8),sharex=True,sharey=True)
    limits=bounds(cfg)
    # Include diverged estimates, never hide failures outside known map.
    for est in estimates.values():
        good=np.isfinite(est).all(axis=1)
        if good.any():
            limits=[min(limits[0],est[good,0].min()-.5),max(limits[1],est[good,0].max()+.5),
                    min(limits[2],est[good,1].min()-.5),max(limits[3],est[good,1].max()+.5)]
    segments=np.stack([gt[:-1,:2],gt[1:,:2]],axis=1);good=valid[:-1]&valid[1:]
    for ax,name in zip(axes,['original','filtered']):
        background(ax,cfg,name,limits)
        ax.add_collection(LineCollection(segments[good],colors=np.where(success[name][1:][good],'#159447','#df3025'),linewidths=2.5))
        est=estimates[name];ax.plot(est[:,0],est[:,1],color='#1858c9',lw=.7)
        ax.set_title(f'{sequence} — {name} map — original localization scans')
        ax.legend(handles=[Line2D([],[],color='#159447',label='GT <10 cm AND <4°'),Line2D([],[],color='#df3025',label='GT failure / missing'),
                           Line2D([],[],color='#1858c9',label='AMCL')],fontsize=7,loc='lower right')
    fig.suptitle(f'Paired representative run, preselected seed {cfg["seeds"][0]}; no alignment')
    fig.tight_layout();fig.savefig(ROOT/cfg['output']/f'trajectories_{sequence}.png',dpi=140);plt.close(fig)


def overlay_image(item,frame,model,target,uv=None,removed=None):
    image=Image.open(item['path']).convert('RGB');arr=np.array(image);colors={'movable':np.array([240,40,30]),'fixed':np.array([30,210,70]),'uncertain':np.array([135,135,135])}
    folder=Path(item['mask_report']).parent
    fixed=np.zeros(arr.shape[:2],bool);uncertain=fixed.copy();regions=[]
    for region in frame['regions']:
        classes=[s['llm']['models'].get(model,{}).get('classification') for s in region['sources']]
        label=classes[0] if classes and all(c==classes[0] for c in classes) and classes[0] in colors else 'uncertain'
        mask=np.asarray(Image.open(folder/region['mask_path']))>0
        if label=='fixed':fixed|=mask
        elif label=='uncertain':uncertain|=mask
        regions.append((region,label))
    effective=mask_layers(folder,frame,model)
    for binary,color in [(uncertain,colors['uncertain']),(fixed,colors['fixed']),(effective,colors['movable'])]:
        arr[binary]=(arr[binary]*.5+color*.5).astype(np.uint8)
    image=Image.fromarray(arr);draw=ImageDraw.Draw(image)
    for region,label in regions:
        box=region['bbox_xyxy'];color=tuple(colors[label].tolist());draw.rectangle(box,outline=color,width=2)
        text=f'{region["source_ids"]} {region["label"]}: {label}'
        # Pillow default bitmap font is ASCII; keep labels readable on every host.
        draw.text((box[0],max(0,box[1]-12)),text.encode('ascii','replace').decode(),fill=color,stroke_width=1,stroke_fill='black')
    if uv is not None and removed is not None:
        for u,v in uv[removed]:
            if np.isfinite(u+v):draw.ellipse((u-2,v-2,u+2,v+2),fill=(255,0,0),outline='white')
    image.save(target,quality=90)


def gallery(cfg,pilot,representative):
    output=ROOT/cfg['output'];cache=ROOT/cfg['cache'];data=ROOT/cfg['data'];folder=output/'gallery';folder.mkdir(exist_ok=True)
    filters=dict(np.load(cache/'mapping/filter.npz'));mapping=dict(np.load(cache/'mapping/raw.npz'));laser=read(cache/'mapping/raw.json')['laser']
    statuses=read(cache/'mapping/filter.json')['candidates'];reports={};cards=[];selections={}
    x0,y0,x1,y1=cfg['room_bounds_xyxy'];limits=[x0-.5,x1+.5,y0-.5,y1+3.0]
    # Mapping contributors selected geometrically, before examining AMCL errors.
    seq=data/'mapping';gt_times,gt_poses,_=pose_table(seq/'gt_poses.txt');tree=transform_tree(yaml.safe_load((seq/'transformations.yaml').read_text()))
    scan_tf=tree('base_link','scan_front_link');scores=[]
    for i in np.flatnonzero(filters['coverage']):
        _,end,good=beam_world(mapping['ranges'][i],mapping['times'][i],laser,scan_tf,gt_times,gt_poses,pilot['pose_max_gap'])
        inside=good&(end[:,0]>=x0)&(end[:,0]<=x1)&(end[:,1]>=y0)&(end[:,1]<=y1)
        if inside.sum()>=10:scores.append((i,int((inside&filters['removed'][i]).sum()),int(inside.sum())))
    selected=[]
    for index,_,_ in sorted(scores,key=lambda x:(-x[1],-x[2],x[0])):
        if all(abs(mapping['times'][index]-mapping['times'][j])>=2 for j in selected):selected.append(index)
        if len(selected)>=10:break
    selections['mapping']=sorted(int(i) for i in selected)
    # Locate first sustained (3-scan) tracking failure in the fixed problematic section.
    lt=representative['lt_changes_0'];raw=lt['arrays'];times=raw['times'];gt=raw['gt'];has=raw['has_gt']
    section=(times>=cfg['gallery_section'][0])&(times<=cfg['gallery_section'][1])&has
    room=section&(gt[:,0]>=x0)&(gt[:,0]<=x1)&(gt[:,1]>=y0)&(gt[:,1]<=y1)
    bad=room&(~lt['success']['original']|~lt['success']['filtered'])
    sustained=np.flatnonzero(bad[:-2]&bad[1:-1]&bad[2:]);onset=int(sustained[0]) if len(sustained) else int(np.flatnonzero(room)[0])
    targets=[times[onset]-3,times[onset]-1,times[onset],times[onset]+1,times[onset]+3,times[onset]+6]
    indices={int(np.argmin(abs(times-t))) for t in targets}
    indices.update(np.flatnonzero(section)[::max(1,int(section.sum())//8)].tolist());selections['lt_changes_0']=sorted(indices)
    write(folder/'selection.json',{'selection':selections,'first_sustained_failure_timestamp':float(times[onset]),
        'mapping_policy':'Rank room-endpoint contributors by removed hits, spatially checked using mapping GT; >=2 sec apart',
        'lt_policy':'Fixed 51-second section, samples before and during first 3-consecutive-scan threshold failure of either map, representative seed 42'})
    lt_status={}
    for path in [ROOT/'data/lilocbench-full/mask-status.json',ROOT/'results/lilocbench-rgb-rate/rgb-plan-5hz.json']:
        if path.exists():
            for item in read(path):lt_status[item['frame_id']]=item
    lt_candidates=sorted(lt_status.values(),key=lambda s:s['timestamp'])
    plt=plotting()
    for sequence in ['mapping','lt_changes_0']:
        source=mapping if sequence=='mapping' else raw
        seq=data/sequence;meta=read(cache/sequence/'raw.json');laser=meta['laser']
        tree=transform_tree(yaml.safe_load((seq/'transformations.yaml').read_text()));scan_tf=tree('base_link','scan_front_link');camera_tf=tree('base_link','camera_front_color_optical_frame')
        gt_times,gt_poses,_=pose_table(seq/'gt_poses.txt');ot,op,_=pose_table(seq/'dingo_velocity_controller/odom.txt')
        intrinsic=yaml.safe_load((seq/'camera_front/color/intrinsics.yaml').read_text())
        for index in selections[sequence]:
            t=float(source['times'][index]);name=f'{sequence}-{index:05d}';drop=filters['removed'][index] if sequence=='mapping' else np.zeros(source['ranges'].shape[1],bool)
            if sequence=='mapping':item=statuses[int(filters['association'][index])]
            else:item=min(lt_candidates,key=lambda i:abs(i['timestamp']-(t+laser['scan_time']/2)))
            dt=abs(item['timestamp']-(t+laser['scan_time']/2));valid=item['classification_valid'] and item.get('mask_report') and dt<=pilot['mask_max_dt']
            if sequence=='mapping' and not valid:raise ValueError('Mapping gallery hit has no valid applied mask')
            stem=folder/name;Image.open(item['path']).convert('RGB').save(str(stem)+'-rgb.jpg',quality=90)
            frame=None;notes=[]
            if item.get('mask_report'):
                path=Path(item['mask_report'])
                if path not in reports:reports[path]={r['frame_id']:r for r in read(path)['frames']}
                frame=reports[path][item['frame_id']]
                uv,visible=project_scan(source['ranges'][index],t,item['timestamp'],laser,scan_tf,camera_tf,intrinsic,ot,op,pilot['pose_max_gap'])
                overlay_image(item,frame,pilot['classification_model'],str(stem)+'-overlay.jpg',uv,drop&visible)
                for r in frame['regions']:
                    for s in r['sources']:
                        decision=s['llm']['models'].get(pilot['classification_model'],{})
                        notes.append(f'{s["object"]["id"]}: {s["object"]["label"]} → {decision.get("classification")} — {decision.get("reason","")}')
            _,end,good=beam_world(source['ranges'][index],t,laser,scan_tf,gt_times,gt_poses,pilot['pose_max_gap'])
            fig,axes=plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=True)
            for ax,map_name in zip(axes,['original','filtered']):
                background(ax,cfg,map_name,limits);ax.scatter(end[good&~drop,0],end[good&~drop,1],s=3,c='#1478b8',label='original retained hits (GT placement)')
                if drop.any():ax.scatter(end[good&drop,0],end[good&drop,1],s=10,c='red',label='actually removed mapping hits (GT)')
                ax.set_title(map_name+' map');ax.legend(fontsize=7,loc='lower right')
            fig.suptitle(f'{sequence} {t:.6f} — same original points / GT diagnostic placement')
            fig.tight_layout();fig.savefig(str(stem)+'-lidar.png',dpi=110);plt.close(fig)
            detail={'sequence':sequence,'scan_timestamp':t,'rgb_timestamp':item['timestamp'],'mask_time_offset_s':dt,
                'mask_valid':bool(valid),'filter_applied':sequence=='mapping','removed_hits':int(drop.sum()),'frame_id':item['frame_id']}
            write(str(stem)+'.json',detail)
            images=f'<img loading="lazy" src="{name}-rgb.jpg"><img loading="lazy" src="{name}-overlay.jpg">' if frame else f'<img loading="lazy" src="{name}-rgb.jpg"><p>Florence/SAM unavailable.</p>'
            cards.append('<article><h2>'+html.escape(f'{sequence} | scan {t:.6f} | RGB {item["timestamp"]:.6f}')+'</h2><p>'+html.escape(json.dumps(detail))+'</p>'+images+
                f'<img class="wide" loading="lazy" src="{name}-lidar.png"><details><summary>Florence region IDs / Qwen reasons</summary><pre>'+html.escape('\n'.join(notes))+'</pre></details></article>')
    content='''<!doctype html><meta charset="utf-8"><title>Filtered mapping review</title><style>body{font:15px sans-serif;margin:2em;background:#eee}article{background:white;padding:1em;margin:1em 0}img{width:49%;height:auto}.wide{width:100%;max-width:1300px}pre{white-space:pre-wrap}p{overflow-wrap:anywhere}</style><h1>Mapping filter / original localization scans</h1><p>Red: effective movable removal mask after overlap protection; green: fixed; gray: uncertain/missing/conflicting. Florence boxes and Qwen labels preserve source IDs. Red dots: actually removed mapping hits only. Localization scans remain ORIGINAL on both maps; their cached RGB mask is diagnostic and is never applied to localization. All point overlays use public GT beam poses to isolate map/scan differences from AMCL error. Mask validity means schema-valid and within ±50 ms of scan midpoint; invalid nearby RGB is labeled as such.</p>'''
    (folder/'index.html').write_text(content+'\n'.join(cards));return str(folder/'index.html')


def make_report(cfg,pilot):
    output=ROOT/cfg['output'];cache=ROOT/cfg['cache'];data=ROOT/cfg['data'];all_metrics={};representative={};csv_rows=[]
    from .box_segmentation import digest
    identity=output/'dataset-identity.json'
    if identity.exists() and any(digest(ROOT/path)!=sha for path,sha in read(identity).items()):
        raise ValueError('Input dataset/cache/config changed during the experiment')
    checkpoint=np.load(cache/'mapping/grid-checkpoint.npz')
    logs=checkpoint['logs'];map_summary=read(output/'map-summary.json')
    map_summary['log_odds_changed_cells']=int((logs[0]!=logs[1]).sum())
    write(output/'map-summary.json',map_summary)
    from .filtered_map_baseline import adopt
    # A CPU-only cache is usable only after byte-equivalent replay validation.
    adopt(cfg)
    pair_checks={}
    for sequence in cfg['sequences']:
        arrays=dict(np.load(cache/sequence/'raw.npz'));times=arrays['times'];runs={'original':[],'filtered':[]};rep={'arrays':arrays,'est':{},'success':{}}
        paired=[]
        for seed in cfg['seeds']:
            root=output/'replays'/sequence/f'seed-{seed}'
            a=np.load(root/'original/scans.npz');b=np.load(root/'filtered/scans.npz')
            for key in ['times','ranges','odom','odom_end','removed']:
                if not np.array_equal(a[key],b[key],equal_nan=True):raise ValueError(f'Paired replay differs: {sequence}/{seed}/{key}')
            if a['removed'].any() or b['removed'].any():raise ValueError('Localization scans must remain original')
            if read(root/'original/input.json')!=read(root/'filtered/input.json') or (root/'original/amcl.yaml').read_bytes()!=(root/'filtered/amcl.yaml').read_bytes():
                raise ValueError('Paired initialization or AMCL settings differ')
            pair_checks[f'{sequence}/{seed}']=True
            row_pair={}
            for name in ['original','filtered']:
                path=output/'runs'/sequence/f'seed-{seed}'/name/'poses.json'
                est=load_estimates(path,times);summary,pos,yaw,ok=evaluate(arrays['gt'],est,arrays['has_gt'],data/'reference-code/evaluate.py')
                summary.update(seed=seed,scans=len(times));runs[name].append(summary);row_pair[name]=summary
                if seed==cfg['seeds'][0]:rep['est'][name]=est;rep['success'][name]=ok
                for i,t in enumerate(times):
                    csv_rows.append([sequence,seed,name,float(t),*est[i].tolist(),bool(arrays['has_gt'][i]),
                        float(pos[i]) if np.isfinite(pos[i]) else '',float(yaw[i]) if np.isfinite(yaw[i]) else '',bool(ok[i])])
                # TUM output for independent official evaluator use.
                tum=path.with_name('poses.tum.txt');good=np.isfinite(est).all(axis=1)
                np.savetxt(tum,np.column_stack([times[good],est[good,:2],np.zeros(good.sum()),np.zeros((good.sum(),2)),np.sin(est[good,2]/2),np.cos(est[good,2]/2)]),fmt='%.9f')
            paired.append({key:row_pair['filtered'][key]-row_pair['original'][key] for key in PAPER_KEYS+['success_percent']})
        all_metrics[sequence]={'original':aggregate(runs['original']),'filtered':aggregate(runs['filtered']),
            'paired_filtered_minus_original':aggregate(paired),'runs':runs,
            'paper_AMCL':{k:{'mean':v[0],'std':v[1]} for k,v in zip(PAPER_KEYS,PAPER[sequence])}}
        representative[sequence]=rep;plot_trajectories(cfg,sequence,arrays,rep['est'],rep['success'])
    with (output/'poses.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sequence','seed','map','timestamp','x','y','yaw','has_gt','position_error_m','yaw_error_deg','success']);writer.writerows(csv_rows)
    plot_maps(cfg);gallery_path=gallery(cfg,pilot,representative)
    metrics={'method':'Paired ROS2 AMCL 1.3.12 tracking, 10 seeds 42–51; original localization scans on both maps',
        'std_ddof':0,'map':read(output/'map-summary.json'),'sequences':all_metrics,'local_gallery':gallery_path,
        'verified_identical_original_localization_inputs':pair_checks,
        'initialization_and_odom_boundaries':{s:read(cache/s/'raw.json') for s in cfg['sequences']},
        'mapping_classification':{'selected_rgb':len(read(cache/'mapping/mask-status.json')),
            'valid_classified_rgb':sum(s['classification_valid'] for s in read(cache/'mapping/mask-status.json')),
            'model':'qwen3.5:27b-q4_K_M','digest':read(cache/'mapping/batches/000/inference/model-metadata.json')['digest']}}
    write(output/'metrics.json',metrics)
    public=ROOT/'docs/lilocbench-filtered-map';public.mkdir(parents=True,exist_ok=True)
    for name in ['metrics.json','protocol.json','preflight.json','maps_comparison.png',*[f'trajectories_{s}.png' for s in cfg['sequences']]]:
        shutil.copyfile(output/name,public/name)
    paths=[ROOT/cfg['data']/'mapping/laser_scan_front/ranges.txt',ROOT/cfg['data']/'mapping/gt_poses.txt',
           cache/'mapping/filter.npz',output/'maps/original/occupancy.npy',output/'maps/filtered/occupancy.npy']
    for sequence in cfg['sequences']:
        paths.extend([data/sequence/'laser_scan_front/ranges.txt',data/sequence/'dingo_velocity_controller/odom.txt',data/sequence/'gt_poses.txt'])
        for seed in cfg['seeds']:
            for name in ['original','filtered']:paths.append(output/'runs'/sequence/f'seed-{seed}'/name/'poses.json')
    write(output/'provenance.json',{'sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths},'qwen_digest':read(cache/'mapping/batches/000/inference/model-metadata.json')['digest']})
    publish(cfg,metrics,public)
    short={s:{'original_joint':r['original']['success_percent'],'filtered_joint':r['filtered']['success_percent'],
        'paired_difference_pp':r['paired_filtered_minus_original']['success_percent']} for s,r in all_metrics.items()}
    write(output/'summary.json',{'map':metrics['map'],'sequences':short,'gallery':gallery_path})
    print(json.dumps(read(output/'summary.json'),indent=2),flush=True);return metrics


def publish(cfg,metrics,public):
    def val(obj,key):return f'{obj[key]["mean"]:.2f} ± {obj[key]["std"]:.2f}'
    lines=['# LILocBench: filtriranje med gradnjo zemljevida','',
        'Dva lastna zemljevida izključno iz `mapping`; na obeh isti originalni lokalizacijski skeni. Deset parov na vsako sekvenco, semena 42–51. Filter ni uporabljen med lokalizacijo.','',
        '![Zemljevida in razlike](maps_comparison.png)','',
        '| Sekvenca | Zemljevid | RMSE položaja [cm] | Max [cm] | <5 cm [%] | <10 cm [%] | RMSE yaw [°] | Max yaw [°] | <2° [%] | <4° [%] | Oba: <10 cm in <4° [%] |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seq,results in metrics['sequences'].items():
        for name in ['original','filtered','paper_AMCL']:
            r=results[name];values=[val(r,k) for k in PAPER_KEYS];values.append(val(r,'success_percent') if name!='paper_AMCL' else 'ni podano')
            lines.append('| '+seq+' | '+name+' | '+' | '.join(values)+' |')
    lines+=['','Povprečje ± populacijski standardni odklon (`ddof=0`), n=10. Članek ne podaja skupnega AND deleža.','',
        '| Sekvenca | Filtriran − original: skupni uspeh [odstotne točke, parno povprečje ± std] |','|---|---:|']
    for seq,r in metrics['sequences'].items():lines.append(f'| {seq} | {val(r["paired_filtered_minus_original"],"success_percent")} |')
    m=metrics['map'];lines+=['',f'Mapping: {m["mapping_scans"]} skenov, {m["rgb_frames"]} RGB pri ~5 Hz; veljavna maska {m["mask_valid_scans"]} skenov ({m["mask_coverage_percent"]:.2f} %). Odstranjenih {m["removed_hits"]} / {m["finite_hits"]} veljavnih zadetkov ({m["removed_hits_percent"]:.3f} %). Spremenjenih {m["changed_cells"]} klasificiranih celic oziroma {m["log_odds_changed_cells"]} celic z drugačnim log-odds.','',
        '## Metoda in razlike glede na tabelo III','',
        'Uporabljena je obstoječa ROS2 Nav2 AMCL 1.3.12 z omnidirekcijskim gibalnim modelom, likelihood field, 100 žarki, 1.000–10.000 delci in pilotnimi parametri. Vsak par ima isto seme, originalne skene, odometrijo in prvo veljavno GT pozo za inicializacijo; začetni std je 0,2 m / 0,2 m / 10°. Poznejši lokalizacijski GT se uporablja samo za ocenjevanje/prikaz. Brez poravnave trajektorij, tuninga ali glasovanja.','',
        'Članek uporablja uradni zemljevid s scan-matching/pose-graph izboljšavo poz mapping. Tu oba lastna zemljevida uporabljata neposredne javne planarno interpolirane GT poze na čas vsakega žarka, brez te izboljšave. Log-odds mreža 5 cm ima enake meje, model zadetka/prostega prostora in pragove. Na celico je dovoljena ena posodobitev na sken; zadetek ima prednost pred prostim prostorom. Odstranjeni in neveljavni žarki ne posodabljajo mreže niti pred niti za zadetkom. Druge meritve lahko celice neodvisno opazujejo. GT vrzeli se na obeh zemljevidih izpustijo enako.','',
        'Florence-2 large FP16 → Qwen3.5 27B Q4_K_M prek lokalne Ollame → SAM 2.1 Hiera Small. Oznaka posamezne slike, RGB ~5 Hz, najbližja shematsko veljavna maska ±50 ms od sredine skena. Obstoječa časovno kompenzirana projekcija s kalibracijo. Odstrani se unija `movable` minus unija `fixed`/`uncertain`/manjkajočih/konfliktnih mask; prekrivanje ima zaščitno pravilo. Kontekstni preskoki in neveljavne sheme ostanejo v cache in se ne popravljajo.','',
        'Uradne numerične funkcije 2D APE so ponovno uporabljene pri časih vseh skenov. GT se interpolira brez ekstrapolacije ali vrzeli nad 150 ms. Manjkajoča ocena z GT šteje kot neuspeh pri deležih; RMSE/max uporabljata razpoložljive pare. Števila manjkajočih ocen in časov brez GT so v metrics.json za vsak zagon. To ponovi AMCL del eksperimentalnega načrta tabele III, vendar zaradi lastne gradnje zemljevida in ROS2/pilotnih parametrov ni bitno enaka reprodukcija avtorjeve implementacije. Učinek filtra ocenjujemo izključno s parno razliko naših dveh zemljevidov.','',
        '## Trajektorije','',
        'Slike kažejo vnaprej izbrano seme 42; tabelo tvori vseh 10 semen. Modro: AMCL; zeleno: GT, kjer sta hkrati dosežena oba stroga praga; rdeče: ostalo. Enako merilo in izrez znotraj vsakega para.']
    for seq in cfg['sequences']:lines+=['',f'![{seq}](trajectories_{seq}.png)']
    lines+=['','## Ponovitev','', '```bash','python3 -m llm_object_classifier.localization_filtered_map --config configs/lilocbench-filtered-map.json','```','',
        'Posamezni koraki: `--stage prepare`, `preflight`, `models`, `maps`, `amcl`, `report`. Zaključene inference, maske, integracija mreže in AMCL zagoni se ponovno uporabijo. Dnevniki in progress.json so v results/lilocbench-filtered-map/. Podatki, modeli, surovi izhodi in lokalna HTML galerija niso v Git.','',
        f'Lokalna galerija: `{metrics["local_gallery"]}`. Galerija prikaže originalni RGB, okvirje/oznake, SAM maske (odstranitev rdeča, fixed zelena, uncertain siva) in dejansko odstranjene mapping zadetke. LiDAR primerjave uporabljajo GT samo za diagnostiko in prikazujejo iste originalne lokalizacijske zadetke na obeh zemljevidih.','',
        '[Metrike vseh ponovitev](metrics.json), [protokol](protocol.json), [kratek preizkus](preflight.json).','',
        'Viri: [članek, tabela III](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/trekel2025iros.pdf), [uradni podatki, GT in navodila za zemljevide](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/), [evaluator](https://github.com/PRBonn/LILocBench). Trekel et al., IROS 2025. Zemljevidi in izpeljane slike: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).']
    (public/'README.md').write_text('\n'.join(lines)+'\n')
