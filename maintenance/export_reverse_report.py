"""Publish a portable, results-only report; no model calls or AMCL execution.

First use --collect with the local experiment cache. Subsequent rendering reads
only the small checked-in data bundle and JSON metrics beside the report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DEST=ROOT/'docs/lilocbench-reverse'
SOURCE=ROOT/'results/lilocbench-reverse'
BRANCHES=['original','front','three','empty']
NAMES={'original':'Brez filtra','front':'Sprednja kamera','three':'Tri kamere','empty':'Osnovno stanje (mapping)'}
COLORS={'original':'#515967','front':'#d98a17','three':'#2268c5','empty':'#14835e'}
CASES=['reverse-original','reverse-front','reverse-three','reverse-empty',
       'forward-original','forward-front','forward-three']


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def fmt(value):return f'{value["mean"]:.2f} ± {value["std"]:.2f}'


def collect():
    metrics=read(SOURCE/'metrics.json')
    if not metrics['complete'] or sum(len(r) for r in metrics['runs'].values())!=280:raise ValueError('Experiment incomplete')
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'data').mkdir(exist_ok=True)
    for name in ['metrics.json','geometry.json','protocol.json','projection-summary.json','semantic-coverage.json',
        'paired-effects.json','gaps-evaluated.json','sequence-audit.json','validation.json','report-provenance.json']:
        shutil.copyfile(SOURCE/name,DEST/'data'/name)
    shutil.copyfile(SOURCE/'room-pair-evidence.png',DEST/'isti-prostor.png')
    cfg=read(ROOT/'configs/lilocbench-reverse.json');bundle={}
    for branch in BRANCHES:bundle['map_'+branch]=np.load(SOURCE/'maps'/branch/'occupancy.npy')
    for seq in ['static_0','lt_changes_0']:
        with np.load(ROOT/cfg['raw_cache']/seq/'raw.npz') as a:
            for key in ['times','gt','has_gt']:bundle[seq+'_'+key]=a[key]
    for case in CASES:
        rows=read(SOURCE/'runs'/case/'natural/seed-42/standard/poses.json')
        bundle[case]=np.array([r['pose'] if r['pose'] is not None else [np.nan]*3 for r in rows])
    from llm_object_classifier.localization_pilot import evaluate
    from llm_object_classifier.localization_filtered_map import aggregate
    regional={}
    for case in CASES:
        seq='static_0' if case.startswith('reverse-') else 'lt_changes_0';gt=bundle[seq+'_gt'];times=bundle[seq+'_times']
        region=bundle[seq+'_has_gt']&(gt[:,0]>=5)&(gt[:,0]<=13.8)&(gt[:,1]>=-10.8)&(gt[:,1]<=-5);rows=[]
        for seed in cfg['seeds']:
            records=read(SOURCE/'runs'/case/'natural'/f'seed-{seed}'/'standard/poses.json')
            if len(records)!=len(times) or any(abs(r['timestamp']-t)>1e-6 for r,t in zip(records,times)):raise ValueError('Regional trajectory timestamps differ')
            estimate=np.array([r['pose'] if r['pose'] is not None else [np.nan]*3 for r in records])
            result,*_=evaluate(gt,estimate,region,ROOT/cfg['data']/'reference-code/evaluate.py')
            result.pop('times_without_gt');result['seed']=seed;rows.append(result)
        regional[case]={'valid_gt_room_scans':int(region.sum()),'aggregate':aggregate(rows),'runs':rows,
            'scope':'Room subset of full trajectory, GT selected for evaluation only; no reinitialization; outside-room samples are not missing GT.'}
    write(DEST/'data/room-localization.json',regional)
    cache=ROOT/cfg['cache']
    with np.load(cache/'cloud-lt_changes_0.npz') as cloud,np.load(cache/'point-labels.npz') as lab,np.load(cache/'projection.npz') as proj:
        if not np.array_equal(cloud['ids'],lab['ids']):raise ValueError('Point ID mismatch')
        p=cloud['points'];room=(p[:,0]>=5)&(p[:,0]<=13.8)&(p[:,1]>=-10.8)&(p[:,1]<=-5)
        whole=np.arange(0,len(p),max(1,len(p)//100000));ri=np.flatnonzero(room)
        select=np.unique(np.r_[whole,ri[::max(1,len(ri)//100000)]])
        bundle['point_xy']=p[select];bundle['point_label']=lab['labels'][select]
        for i,name in enumerate(['front','three']):bundle['point_keep_'+name]=~proj['removed'][i].ravel()[cloud['ids'][select]]
    np.savez_compressed(DEST/'data/plot-data.npz',**bundle)
    write(DEST/'data/publication-provenance.json',{'scope':'Ordinary AMCL with wheel odometry; no odometry bridge, no injected gaps',
        'completed_runs':280,'reported_primary_runs':70,'primary_seeds':cfg['seeds'],'plot_seed':42,
        'source_sha256':{n:sha(SOURCE/n) for n in ['metrics.json','geometry.json','protocol.json','projection-summary.json']},
        'point_plot_sample_count':len(bundle['point_xy']),'point_sampling':'Deterministic stride plus room oversampling; metrics use ALL retained points',
        'plot_bundle_sha256':sha(DEST/'data/plot-data.npz')})


def background(ax,grid,meta,bounds):
    extent=[meta['origin'][0],meta['origin'][0]+meta['width']*meta['resolution'],meta['origin'][1],meta['origin'][1]+meta['height']*meta['resolution']]
    ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.76)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',xlabel='x [m]',ylabel='y [m]')
    ax.tick_params(labelsize=8)


def figures(data,metrics,geometry,meta):
    images=[];room=[5,-10.8,13.8,-5];whole=[-14.5,-12.4,17.5,3.5]
    for name,bounds in [('zemljevidi-celota',whole),('zemljevidi-soba',room)]:
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,branch in zip(axes.flat,BRANCHES):
            background(ax,data['map_'+branch],meta,bounds);ax.set_title(NAMES[branch],fontsize=12)
        fig.suptitle('Zemljevidi v skupnem koordinatnem sistemu — enako merilo\nČrno: zasedeno; belo: prosto; sivo: neznano. Referenca še vsebuje običajno pohištvo.',fontsize=12)
        fig.savefig(DEST/(name+'.png'),dpi=160);plt.close(fig);images.append(name)
    reference=data['map_empty'];far=distance_transform_edt(reference!=100)*meta['resolution']>.1
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    cell_counts={}
    map_changes={}
    for ax,branch in zip(axes,BRANCHES[:3]):
        grid=data['map_'+branch];excess=(grid==100)&(reference==0)&far
        background(ax,grid,meta,room);y,x=np.nonzero(excess)
        px=meta['origin'][0]+(x+.5)*meta['resolution'];py=meta['origin'][1]+(y+.5)*meta['resolution']
        use=(px>=room[0])&(px<=room[2])&(py>=room[1])&(py<=room[3]);cell_counts[branch]=int(use.sum())
        ax.scatter(px[use],py[use],s=8,c='#cf313e',marker='s',label='Zasedeno tukaj, prosto v referenci')
        ax.set_title(NAMES[branch]+f'\n{use.sum()} označenih celic v sobi');ax.legend(fontsize=7,loc='lower left')
        original=data['map_original'];map_changes[branch]={'occupied_cells':int((grid==100).sum()),
            'occupied_removed_vs_original':int(((original==100)&(grid!=100)).sum()),
            'occupied_added_vs_original':int(((original!=100)&(grid==100)).sum()),'changed_classified_cells':int((grid!=original).sum())}
    fig.suptitle('Spremembe v sobi glede na osnovni mapping\nRdeče: razlika geometrije, več kot 10 cm od zasedene reference; to niso ročne oznake premičnih predmetov.',fontsize=11)
    fig.savefig(DEST/'razlike-soba.png',dpi=160);plt.close(fig);images.append('razlike-soba')
    write(DEST/'data/map-cell-diagnostic.json',{'room_cells_occupied_here_free_in_mapping_beyond_10cm':cell_counts,'map_changes':map_changes,
        'secondary_only':True,'note':'Raster cell diagnostic against mapping; distinct from continuous point metric against static_0.'})
    for direction,branches in [('reverse',BRANCHES),('forward',BRANCHES[:3])]:
        seq='static_0' if direction=='reverse' else 'lt_changes_0';gt=data[seq+'_gt'];valid=data[seq+'_has_gt'];cloud=[gt[valid,:2]]
        for branch in branches:
            p=data[direction+'-'+branch];cloud.append(p[np.isfinite(p).all(axis=1),:2])
        points=np.concatenate(cloud);lo=points.min(axis=0)-.5;hi=points.max(axis=0)+.5;bounds=[lo[0],lo[1],hi[0],hi[1]]
        for detail,crop in [('celota',bounds),('soba',room)]:
            fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
            for ax,branch in zip(axes.flat,branches):
                case=direction+'-'+branch;est=data[case];grid=data['map_'+(branch if direction=='reverse' else 'empty')]
                background(ax,grid,meta,crop)
                finite=np.isfinite(est).all(axis=1);position=np.linalg.norm(est[:,:2]-gt[:,:2],axis=1)
                angle=np.abs(np.degrees((est[:,2]-gt[:,2]+np.pi)%(2*np.pi)-np.pi));success=valid&finite&(position<.1)&(angle<4)
                expected=metrics['runs'][case+'/natural/standard'][0]
                if int(success.sum())!=expected['successful_poses']:raise ValueError('Plot/evaluator discrepancy '+case)
                ax.plot(gt[:,0],gt[:,1],c='#26834c',lw=1.1,label='Referenčna pot')
                ax.plot(est[:,0],est[:,1],c='#2369be',lw=.8,label='AMCL')
                failed=valid&~success;ax.scatter(gt[failed,0],gt[failed,1],c='#d13242',s=5,label='Neuspeh ali manjkajoča ocena',zorder=3)
                mean=metrics['aggregate'][case+'/natural/standard']['success_percent']
                ax.set_title(f'{NAMES[branch]}\nOba praga: {expected["success_percent"]:.1f} % (seme 42); {fmt(mean)} % (10 semen)',fontsize=10)
                ax.legend(fontsize=7,loc='upper right')
            for ax in axes.flat[len(branches):]:ax.axis('off')
            title='Isti originalni static_0 skeni na štirih zemljevidih' if direction=='reverse' else 'Različno filtrirani lt_changes_0 skeni na istem osnovnem zemljevidu'
            fig.suptitle(title+'\nObičajni AMCL s kolesno odometrijo; brez dodatne odometrijske premostitve.',fontsize=12)
            filename=f'lokalizacija-{direction}-{detail}';fig.savefig(DEST/(filename+'.png'),dpi=150);plt.close(fig);images.append(filename)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,direction,branches in zip(axes,['reverse','forward'],[BRANCHES,BRANCHES[:3]]):
        for branch in branches:
            values=[r['success_percent'] for r in metrics['runs'][f'{direction}-{branch}/natural/standard']]
            ax.plot(range(42,52),values,'o-',c=COLORS[branch],lw=1.4,label=NAMES[branch])
        ax.set(ylim=(0,100),xticks=range(42,52),xlabel='Seme AMCL',ylabel='Položaj <10 cm IN orientacija <4° [%]',title='Obratni test' if direction=='reverse' else 'Filtriranje med lokalizacijo')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Vseh deset ponovitev: izboljšanje ni dosledno pri vseh semenih',fontsize=12)
    fig.savefig(DEST/'uspesnost-po-semenih.png',dpi=170);plt.close(fig);images.append('uspesnost-po-semenih')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');labels=['Ujemanje','Odvečno v potrjeno prostem','Neopaženo','Drugo opaženo'];colors=['#45a579','#db5663','#babfc7','#e9ad4e']
    for ax,region in zip(axes,['whole','room']):
        left=np.zeros(3)
        for field,label,color in zip(['matched_percent','confirmed_free_excess_percent','unobserved_percent','other_observed_percent'],labels,colors):
            v=np.array([geometry[b][region][field] for b in BRANCHES[:3]]);ax.barh(range(3),v,left=left,color=color,label=label);left+=v
        ax.set(xlim=(0,100),yticks=range(3),yticklabels=[NAMES[b] for b in BRANCHES[:3]],xlabel='Delež ohranjenih točk [%]',title='Celotni prostor' if region=='whole' else 'Problematična soba');ax.invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=9)
    fig.savefig(DEST/'geometrija-delezi.png',dpi=170);plt.close(fig);images.append('geometrija-delezi')
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained');p=data['point_xy'];labels=data['point_label']
    for ax,branch in zip(axes,BRANCHES[:3]):
        keep=np.ones(len(p),bool) if branch=='original' else data['point_keep_'+branch]
        for value,color,title in [(2,'#aeb5c0','Neopaženo'),(3,'#dfaa42','Drugo opaženo'),(0,'#26965d','Ujemanje'),(1,'#cf3544','Odvečno v prostem')]:
            points=p[keep&(labels==value)];ax.scatter(points[:,0],points[:,1],s=.6,c=color,label=title,rasterized=True)
        ax.set(xlim=(5,13.8),ylim=(-10.8,-5),aspect='equal',title=NAMES[branch],xlabel='x [m]',ylabel='y [m]');ax.legend(fontsize=7,loc='lower left')
    fig.suptitle('Točke v sobi proti static_0, ista toleranca 10 cm\nRisba je deterministično razredčena; vse številčne metrike uporabljajo vse točke.',fontsize=11)
    fig.savefig(DEST/'geometrija-soba.png',dpi=170);plt.close(fig);images.append('geometrija-soba')
    return images


def manuscript(metrics,geometry,projection,effects):
    lines=['# Kako dobro deluje semantično filtriranje LiDAR?','',
        '**Zaključen preizkus LILocBench, 18. november 2026.** Ta pregled najprej obravnava običajni AMCL brez dodatne odometrijske premostitve in brez umetnih izpadov. AMCL v teh rezultatih **še vedno uporablja kolesno odometrijo**. Lokalizacija povsem brez odometrije ni bila preizkušena.','',
        '[PDF za branje in deljenje](porocilo.pdf) · [Vseh 280 rezultatov](data/metrics.json) · [Ponovitev in metoda](../lilocbench-reverse.md) · [Odometrijska premostitev, ločen dodatek](odometrija.md)','',
        '## Kratek odgovor','',
        '- **Zanesljivega izboljšanja lokalizacije še ni.** Tri kamere v obratnem testu dosežejo 45,25 % skupnega uspeha, brez filtra 43,71 %. Parna razlika je +1,54 ± 20,75 odstotne točke; med semeni se učinek močno spreminja.',
        '- **Referenčno osnovno stanje je bistveno boljše:** isti static_0 skeni na neodvisnem mapping zemljevidu dosežejo 89,40 % skupnega uspeha. Ta kontrola se razlikuje tudi v prostorski pokritosti in postavitvi pohištva, zato celotne razlike ne pripisujemo samo filtru.',
        '- **Geometrija se delno izboljša:** v problematični sobi delež ujemanj naraste 74,59 → 77,90 %, delež odvečnih točk v potrjeno prostem prostoru pade 19,28 → 16,67 %. To sta metriki točk, ne uspešnost lokalizacijskih poz.','',
        '## Kaj pomeni »z ovirami« in »brez ovir«','',
        '`lt_changes_0` prikazuje dodano škatlo in prestavljeno pohištvo. `static_0` in neodvisni `mapping` prikazujeta osnovno stanje, ki še vsebuje običajno pohištvo. **Popolnoma prazne sobe oziroma nadzorovanega para pred/po odstranitvi vseh premičnih predmetov v teh podatkih nismo potrdili.** Zato spodaj uporabljamo izraz »referenčno osnovno stanje«, ne »zemljevid brez vseh ovir«.','',
        '![Ista kuhinja v treh vožnjah](isti-prostor.png)','',
        'Pogledi med vožnjami niso povsem isti: primer static_0 je bližje vratom. Vse vožnje so v skupnem javnem koordinatnem sistemu; trajektorij dodatno ne poravnavamo.','',
        '## 1. Zemljevidi in vidne spremembe','',
        'Prvi trije zemljevidi nastanejo iz iste celotne lt_changes_0 vožnje, z istimi GT pozami in isto mrežo 5 cm. Razlikuje se samo filter zadetkov. Četrti je neodvisni osnovni mapping. Odstranjeni žarek ne posodobi niti zasedenosti niti prostega prostora.','',
        '![Celotni zemljevidi](zemljevidi-celota.png)','![Povečava problematične sobe](zemljevidi-soba.png)','![Odstopanja od osnovnega mapping](razlike-soba.png)','',
        'Rdeče označene celice so zasedene v spremenjenem zemljevidu, proste v mapping in več kot 10 cm od njegove zasedene geometrije. To je dodatni rastrski prikaz razlik; ni ročna oznaka premičnega predmeta in ni ista metrika kot primerjava koncev žarkov proti static_0.','',
        '**Ovire v zemljevidu večinoma ostanejo.** Brez filtra je zasedenih 10.028 celic, s tremi kamerami 10.000. Tri kamere razvrstijo 262 prvotno zasedenih celic drugače, hkrati pa je 234 drugih celic na novo zasedenih. V sobi se število rdeče označenih odstopanj glede na mapping spremeni 762 → 756. To ni uspešna odstranitev vseh ovir, čeprav se metrike posameznih ohranjenih točk izboljšajo.','',
        '## 2. Lokalizacija brez dodatne odometrijske premostitve','',
        'Vsi deleži spodaj so povprečje ± populacijski standardni odklon desetih semen 42–51. Skupni uspeh zahteva **položaj <10 cm IN orientacijo <4°**. Manjkajoča ocena ob veljavnem GT šteje kot neuspeh.','',
        '### A. Obratni test: spremenjen zemljevid → originalni static_0 skeni','',
        'Skenov za lokalizacijo tukaj ne filtriramo. Vsak zemljevid dobi identične skene, odometrijo, inicializacijo in parna semena.','',
        '| Zemljevid | Položaj <10 cm [%] | Orientacija <4° [%] | Oba [%] | RMSE položaja [cm] |','|---|---:|---:|---:|---:|']
    for b in BRANCHES:
        r=metrics['aggregate'][f'reverse-{b}/natural/standard'];lines.append('| '+NAMES[b]+' | '+' | '.join(fmt(r[k]) for k in ['position_below_10cm_percent','yaw_below_4deg_percent','success_percent','position_rmse_cm'])+' |')
    lines+=['','Velik RMSE in velik raztros v prvih treh vrsticah pomenita, da nekateri zagoni izgubijo lokalizacijo. Prikazano seme 42 je bilo določeno pred rezultati in ni povprečna trajektorija.','',
        '![Obratna lokalizacija, celotna pot](lokalizacija-reverse-celota.png)','![Obratna lokalizacija v sobi](lokalizacija-reverse-soba.png)','',
        '### B. Isti osnovni mapping → različno filtrirani lt_changes_0 skeni','',
        'Zemljevid je v vseh treh vrsticah enak. Filtriranje poteka samo na lokalizacijskih skenih.','',
        '| Filter skenov | Položaj <10 cm [%] | Orientacija <4° [%] | Oba [%] | RMSE položaja [cm] |','|---|---:|---:|---:|---:|']
    for b in BRANCHES[:3]:
        r=metrics['aggregate'][f'forward-{b}/natural/standard'];lines.append('| '+NAMES[b]+' | '+' | '.join(fmt(r[k]) for k in ['position_below_10cm_percent','yaw_below_4deg_percent','success_percent','position_rmse_cm'])+' |')
    lines+=['','Tri kamere izboljšajo ločeni delež pravilne orientacije, skupni uspeh pa ostane približno enak oziroma nekoliko nižji.','',
        '![Lokalizacija spremenjene vožnje, celota](lokalizacija-forward-celota.png)','![Lokalizacija spremenjene vožnje v sobi](lokalizacija-forward-soba.png)','',
        'Na poteh je zeleno referenčna trajektorija, modro AMCL in rdeče neuspešen odsek na referenčni poti, tudi kadar ocena manjka. V celotnem prikazu ni odrezana nobena veljavna referenčna ali ocenjena poza.','',
        '### Parna primerjava in ponovitve','',
        '| Test | Sprednja − brez filtra [odst. točke] | Tri − brez filtra [odst. točke] |','|---|---:|---:|']
    for direction,title in [('reverse','A: obratni test'),('forward','B: filtriranje skenov')]:
        v=[effects[f'{direction}-{b}-minus-{direction}-original/natural/standard']['success_percent'] for b in ['front','three']]
        lines.append('| '+title+' | '+' | '.join(fmt(r) for r in v)+' |')
    lines+=['','Odklon je izračunan na razlikah istega semena, ne iz razlike dveh odklonov. Rezultati ne kažejo dosledne prednosti treh kamer pri lokalizaciji.','',
        '![Uspešnost vseh semen](uspesnost-po-semenih.png)','',
        '### Posebej: lokalizacija v problematični sobi','',
        'To je izrez že ocenjene celotne trajektorije, brez nove inicializacije. A uporablja 516 veljavnih GT vzorcev static_0 v sobi, B pa 1.224 vzorcev lt_changes_0. Zato rezultatov A in B ne primerjamo kot identičnih poti.','',
        '| Test / filter | Položaj <10 cm [%] | Orientacija <4° [%] | Oba [%] |','|---|---:|---:|---:|']
    regional=read(DEST/'data/room-localization.json')
    for case in CASES:
        direction,branch=case.split('-',1);r=regional[case]['aggregate']
        lines.append('| '+('A' if direction=='reverse' else 'B')+' / '+NAMES[branch]+' | '+' | '.join(fmt(r[k]) for k in ['position_below_10cm_percent','yaw_below_4deg_percent','success_percent'])+' |')
    lines += ['', '**V sobi ostaja lokalizacija slaba:** A s tremi kamerami doseže 20,60 % proti 17,31 % brez filtra; neodvisni osnovni zemljevid doseže 73,86 %. Pri B tri kamere dosežejo 29,26 % proti 26,79 % brez filtra. [Metrike sobe za vsa semena](data/room-localization.json).','',
        '## 3. Koliko geometrije ostane pravilne','',
        'Referenca je celotna static_0 vožnja. Za vsak ohranjen konec žarka preverimo najbližjo referenčno točko z isto toleranco 10 cm. Neujemanje je »odvečno v prostem« samo ob prosti celici referenčne mreže. Neopažena območja ostanejo ločena.','',
        '| Območje / filter | Ohranjenih točk | Ujemanje [%] | Odvečno v prostem [%] | Neopaženo [%] | Manjkajoča referenca, sekundarno [%] |','|---|---:|---:|---:|---:|---:|']
    for region,title in [('whole','Celota'),('room','Soba')]:
        for b in BRANCHES[:3]:
            r=geometry[b][region];lines.append(f'| {title} / {NAMES[b]} | {r["retained_points"]:,} | '+' | '.join(f'{r[k]:.2f}' for k in ['matched_percent','confirmed_free_excess_percent','unobserved_percent','reference_missing_percent_secondary'])+' |')
    lines+=['','Preostanek do 100 % sestavljajo opažene, a neujemajoče se/zasedene ali neodločene celice (0,07–0,45 %). Izboljšanje deleža ujemanj ne pomeni, da so bile odstranjene samo napačne točke: filter odstrani tudi del ujemajočih se zadetkov. Sekundarna metrika manjkajoče reference se rahlo poslabša.','',
        '![Deleži geometrije](geometrija-delezi.png)','![Ujemajoče se in odvečne točke v sobi](geometrija-soba.png)','',
        '## 4. Kaj dodatno prispevata stranski kameri','',
        f'- Geometrijsko dodatno pokrijeta **{projection["side_additional_covered_hits"]:,}** zadetkov.',
        f'- Dodatno odstranita **{projection["side_additional_removed_hits"]:,}** zadetkov; pri **{projection["front_removals_vetoed_by_sides"]:,}** preprečita odstranitev sprednje kamere zaradi zaščitnega pravila.',
        f'- Sprednja kamera odstrani **{projection["front_removed_hits"]:,}**, tri kamere skupaj **{projection["three_removed_hits"]:,}** od **{projection["finite_hits"]:,}** veljavnih zadetkov.','',
        'Vsak sken uporablja časovno najbližjo veljavno masko posamezne kamere znotraj ±50 ms ter njeno kalibracijo. Odločitve se združijo po indeksu istega LiDAR zadetka. Konfliktne/fixed/uncertain oznake in manjkajoča maska v vidni kameri zaščitijo zadetek. Časovna pokritost je delna, zato lahko nefiltrirani skeni iste objekte še vedno vpišejo v zemljevid. Projekcija ne preverja zakrivanja z globinsko sliko.','',
        '## Sklep','',
        'Filtriranje s tremi kamerami izboljša ujemanje ohranjene geometrije in zmanjša odvečne točke, posebej v sobi. **Ta izboljšava se v trenutni postavitvi ne pretvori v zanesljivo boljšo lokalizacijo AMCL.** Najboljša kontrola ostaja neodvisni osnovni zemljevid. Potreben je boljši oziroma dejansko prazen referenčni par, preverjanje zakrivanja in stabilnejša pokritost mask; iz teh rezultatov ne sklepamo, da je prostor uspešno očiščen vseh ovir.','',
        '## Podatki in ponovljivost','',
        'Primarna tabela uporablja 70 zagonov običajnega AMCL brez umetnih vrzeli. Celoten arhiv vsebuje 280 zagonov, tudi primerjave premostitve in umetnih vrzeli. Maske in zemljevidi so za vsa semena zamrznjeni; prikazani odklon meri variabilnost AMCL, ne negotovosti klasifikacije.','',
        '- [Vse metrike po zagonih](data/metrics.json), [geometrija](data/geometry.json), [parne razlike](data/paired-effects.json), [protokol](data/protocol.json).',
        '- [Prenosni podatki za risbe](data/plot-data.npz), [sled izvoza](data/publication-provenance.json).',
        '- Ponovni izris iz priloženih podatkov: `python3 maintenance/export_reverse_report.py`. Zbiranje iz lokalnega cache: isti ukaz z `--collect`.',
        '- Nobena inferenca ali AMCL se med izdelavo tega poročila ne zažene.','',
        'Vir: Trekel et al., LILocBench, IROS 2025. [Uradni opis podatkov](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/). Podatki in izpeljani zemljevidi: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).']
    (DEST/'README.md').write_text('\n'.join(lines)+'\n')


def pdf(metrics,geometry):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    with PdfPages(DEST/'porocilo.pdf') as book:
        fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.92,'Kako dobro deluje lokalizacija?',fontsize=23,weight='bold');fig.text(.06,.875,'LILocBench • 18. november 2026 • semena 42–51 • zaključenih 280/280 zagonov',fontsize=11)
        text=('Najprej običajni AMCL brez dodatne odometrijske premostitve.\n'
              'AMCL še vedno uporablja kolesno odometrijo; povsem brez odometrije ni bil testiran.\n\n'
              'Glavna ugotovitev: tri kamere izboljšajo geometrijo, ne pa zanesljivo lokalizacije.\n'
              'Obratni test: 43,71 % → 45,25 % skupnega uspeha; osnovni mapping: 89,40 %.\n'
              'Velika variabilnost med semeni pomeni nestabilno sledenje na spremenjenih zemljevidih.\n\n'
              '»Osnovno stanje« še vsebuje pohištvo. Potrjenega para povsem prazne in polne sobe nimamo.\n'
              'A: spreminjamo zemljevid; lokalizacija uporablja iste originalne static_0 skene.\n'
              'B: isti osnovni mapping; filtriramo lokalizacijske skene lt_changes_0.\n'
              'Skupni uspeh = položaj <10 cm IN orientacija <4°. Manjkajoča ocena šteje kot neuspeh.')
        fig.text(.06,.80,text,va='top',fontsize=11,linespacing=1.6)
        ax=fig.add_axes([.06,.10,.88,.32]);ax.axis('off');rows=[]
        for direction in ['reverse','forward']:
            for branch in (BRANCHES if direction=='reverse' else BRANCHES[:3]):
                r=metrics['aggregate'][f'{direction}-{branch}/natural/standard'];rows.append([('A: obratno' if direction=='reverse' else 'B: skeni'),NAMES[branch],fmt(r['position_below_10cm_percent']),fmt(r['yaw_below_4deg_percent']),fmt(r['success_percent'])])
        table=ax.table(cellText=rows,colLabels=['Test','Filter / zemljevid','<10 cm [%]','<4° [%]','Oba [%]'],colWidths=[.14,.30,.18,.18,.20],cellLoc='left',loc='center');table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
        for (r,c),cell in table.get_celld().items():
            if r==0:cell.set_facecolor('#dfeaf5');cell.set_text_props(weight='bold')
        fig.text(.06,.035,'Povprečje ± populacijski std. Podrobna razlaga, vse metrike in povezave so v README.md.',fontsize=9,color='#555555');book.savefig(fig);plt.close(fig)
        regional=read(DEST/'data/room-localization.json')
        fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.92,'Kako dobro deluje v sobi?',fontsize=23,weight='bold')
        fig.text(.06,.85,'Izrez celotnih trajektorij; brez nove inicializacije.\nA: 516 veljavnih GT vzorcev static_0. B: 1.224 vzorcev lt_changes_0.\nPovprečje ± populacijski std desetih semen; oba praga hkrati.',fontsize=12,va='top',linespacing=1.6)
        ax=fig.add_axes([.06,.38,.88,.32]);ax.axis('off');rows=[]
        for case in CASES:
            direction,branch=case.split('-',1);r=regional[case]['aggregate'];rows.append(['A' if direction=='reverse' else 'B',NAMES[branch]]+[fmt(r[k]) for k in ['position_below_10cm_percent','yaw_below_4deg_percent','success_percent']])
        table=ax.table(cellText=rows,colLabels=['Test','Filter / zemljevid','<10 cm [%]','<4° [%]','Oba [%]'],colWidths=[.08,.36,.18,.18,.20],cellLoc='left',loc='center');table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
        for (r,c),cell in table.get_celld().items():
            if r==0:cell.set_facecolor('#dfeaf5');cell.set_text_props(weight='bold')
        fig.text(.06,.29,'Soba ostaja problematična: A s tremi kamerami 20,60 %, osnovna kontrola 73,86 %.\n\nOvire v zemljevidu večinoma ostanejo. Tri kamere odstranijo 262 zasedenih celic,\n234 drugih pa je na novo zasedenih. Skupno: 10.028 → 10.000 zasedenih celic.\nIzboljšana statistika točk zato še ne pomeni zemljevida brez ovir.',fontsize=12,va='top',linespacing=1.6)
        book.savefig(fig);plt.close(fig)
        pages=[('Zemljevidi celotnega prostora','zemljevidi-celota.png'),('Ista problematična soba','zemljevidi-soba.png'),
            ('Spremembe glede na osnovno stanje','razlike-soba.png'),('A. Originalni static_0 skeni na različnih zemljevidih','lokalizacija-reverse-celota.png'),
            ('A. Lokalizacija v problematični sobi','lokalizacija-reverse-soba.png'),('B. Različno filtrirani skeni na istem osnovnem zemljevidu','lokalizacija-forward-celota.png'),
            ('Rezultati vseh desetih semen','uspesnost-po-semenih.png'),('Ujemanje geometrije in odvečne točke','geometrija-soba.png')]
        for title,name in pages:
            fig=plt.figure(figsize=(11.69,8.27));fig.text(.04,.95,title,fontsize=18,weight='bold')
            ax=fig.add_axes([.025,.09,.95,.81]);ax.imshow(plt.imread(DEST/name));ax.axis('off')
            fig.text(.04,.035,'Običajni AMCL uporablja odometrijo; brez dodatne premostitve. Referenca vsebuje običajno pohištvo. Več v README.md.',fontsize=9,color='#555555');book.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.92,'Sklep in omejitve',fontsize=23,weight='bold')
        text=('• V sobi naraste delež geometrijskih ujemanj: 74,59 → 77,90 %.\n'
              '• Odvečne točke v potrjeno prostem delu sobe: 19,28 → 16,67 %.\n'
              '• To ni 77,90 % uspešnih lokalizacij: točke in poze imajo različna imenovalca.\n\n'
              '• Obratni test tri kamere − brez filtra: +1,54 ± 20,75 odstotne točke.\n'
              '• Filtriranje lokalizacijskih skenov: −0,32 ± 2,74 odstotne točke.\n'
              '• Izboljšanje lokalizacije ni dosledno med semeni.\n\n'
              '• Osnovni stanji nista povsem prazni; postavitev in vidna pokritost se razlikujeta.\n'
              '• Maske pokrijejo le del skenov; isti predmeti lahko ostanejo v zemljevidu.\n'
              '• Projekcija ne preverja zakrivanja z globino.\n'
              '• Standardni odklon opisuje ponovitve AMCL, ne negotovosti modelskih oznak.\n\n'
              'Zaključek: geometrija je čistejša, zanesljivejše lokalizacije pa še nismo dosegli.\n\n'
              'Odometrijska premostitev in umetne vrzeli so v ločenem dodatku odometrija.md.\n'
              'Vse številke: data/metrics.json; metodologija: README.md in data/protocol.json.\n\n'
              'Podatki: Trekel et al., LILocBench, IROS 2025. CC BY-NC-SA 4.0.')
        fig.text(.06,.83,text,va='top',fontsize=12,linespacing=1.65);book.savefig(fig);plt.close(fig)


def appendix(metrics):
    lines=['# Ločen dodatek: odometrijska premostitev','',
        'Glavno [poročilo](README.md) najprej prikazuje običajni AMCL brez dodatne premostitve. Ta dodatek primerja preskok LiDAR popravka in propagacijo iz zadnje zanesljive poze, kadar ostane manj kot 100 veljavnih zadetkov. Obe metodi sicer uporabljata kolesno odometrijo.','',
        'Pri pragu 100 zadetkov ni bilo naravnih vrzeli. Za naravno premostitev v sobi zato nimamo empiričnega rezultata. Umetni izpadi so vnaprej določeni pri 40, 90 in 180 s ter trajajo 0,5, 1 in 2 s. To niso selektivno izbrane samo uspešne vrzeli.','',
        '| Primer / način, umetne vrzeli | Položaj <10 cm [%] | Orientacija <4° [%] | Oba [%] |','|---|---:|---:|---:|']
    for case in CASES:
        for mode in ['standard','bridge']:
            row=metrics['aggregate'][f'{case}/artificial/{mode}'];lines.append('| '+case+' / '+mode+' | '+' | '.join(fmt(row[k]) for k in ['position_below_10cm_percent','yaw_below_4deg_percent','success_percent'])+' |')
    lines+=['','Ta tabela ocenjuje celotno vožnjo, tudi posledice po vrzelih. [gaps-evaluated.json](data/gaps-evaluated.json) ločeno vsebuje samo napake med vrzelmi, največje uspešno trajanje/razdaljo, manjkajoči GT, manjkajoče ocene in čas do vsaj ene sekunde stabilnega skupnega uspeha. Nerazpoložljiv čas ponovne stabilizacije je neuspeh in ni zamenjan z ničlo.','',
        'Premostitev nima enotne prednosti v vseh primerih. Najdaljša preizkušena umetna vrzel je približno 2 s; ne sklepamo o daljših izpadih.']
    (DEST/'odometrija.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collect',action='store_true');args=parser.parse_args()
    if args.collect:collect()
    metrics=read(DEST/'data/metrics.json');geometry=read(DEST/'data/geometry.json');projection=read(DEST/'data/projection-summary.json');effects=read(DEST/'data/paired-effects.json');meta=read(DEST/'data/protocol.json')['configuration']['map']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.facecolor':'white'})
    with np.load(DEST/'data/plot-data.npz') as data:figures(data,metrics,geometry,meta)
    manuscript(metrics,geometry,projection,effects);appendix(metrics);pdf(metrics,geometry)
    write(DEST/'data/report-files.json',{str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='report-files.json'})
    print(DEST/'README.md');print(DEST/'porocilo.pdf')


if __name__=='__main__':main()
