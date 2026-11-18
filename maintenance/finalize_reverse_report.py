"""Add a concise scientific report after every canonical AMCL run is complete.

Raw experiment outputs remain intact. Additional gap distances are evaluated
on public GT, with unobserved GT edges explicitly excluded and counted.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from llm_object_classifier.localization_reverse import intervals
from llm_object_classifier.localization_pilot import wrap, write
from llm_object_classifier.localization_comparison import load_estimates
from llm_object_classifier.box_segmentation import digest


def distances(gt, has_gt, success, a, b):
    edges=np.linalg.norm(np.diff(gt[a:b,:2],axis=0),axis=1)
    observed=has_gt[a:b-1]&has_gt[a+1:b]
    length=float(edges[observed].sum())
    runs=intervals(success[a:b])
    longest=max((float(edges[u:v-1].sum()) for u,v in runs),default=0.)
    failed=np.flatnonzero(~success[a:b]);stop=int(failed[0]) if len(failed) else b-a
    prefix=float(edges[:max(0,stop-1)].sum())
    return {'reference_distance_m':length,'reference_distance_missing_edges':int((~observed).sum()),
        'max_safe_distance_m_gt':longest,'safe_prefix_distance_m_gt':prefix}


def fmt(value):
    if value is None:return '—'
    if isinstance(value,dict):return f'{value["mean"]:.2f} ± {value["std"]:.2f}'
    return f'{value:.2f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();cfg=json.loads((ROOT/'configs/lilocbench-reverse.json').read_text())
    output=ROOT/cfg['output'];cache=ROOT/cfg['cache'];raw=ROOT/cfg['raw_cache']
    if args.wait:
        while True:
            job=json.loads((output/'job.json').read_text())
            if job['status']=='complete':break
            if job['status']=='failed':raise RuntimeError('Canonical experiment failed; see job.json and error.json')
            time.sleep(10)
    metrics=json.loads((output/'metrics.json').read_text())
    expected=7*2*2*len(cfg['seeds'])
    if not metrics['complete'] or sum(map(len,metrics['runs'].values()))!=expected:raise ValueError('Full canonical experiment is incomplete')
    for rows in metrics['runs'].values():
        if len(rows)!=len(cfg['seeds']):raise ValueError('Incomplete paired seed set')
    geometry=json.loads((output/'geometry.json').read_text());projection=json.loads((output/'projection-summary.json').read_text())
    sources={name:digest(output/name) for name in ['metrics.json','gaps.json','geometry.json','projection-summary.json','protocol.json']}
    memo={};gap_rows=json.loads((output/'gaps.json').read_text());gap_summaries={}
    for key,rows in gap_rows.items():
        case,scenario,mode,seed=key.split('/');sequence=cfg['empty_sequence'] if case.startswith('reverse-') else cfg['full_sequence']
        if sequence not in memo:memo[sequence]=dict(np.load(raw/sequence/'raw.npz'))
        a=memo[sequence];est=load_estimates(output/'runs'/case/scenario/seed/mode/'poses.json',a['times'])
        finite=np.isfinite(est).all(axis=1);position=np.linalg.norm(est[:,:2]-a['gt'][:,:2],axis=1)
        yaw=np.abs(np.degrees(wrap(est[:,2]-a['gt'][:,2])))
        success=a['has_gt']&finite&(position<.1)&(yaw<4)
        x0,y0,x1,y1=cfg['room_bounds_xyxy'];room=a['has_gt']&(a['gt'][:,0]>=x0)&(a['gt'][:,0]<=x1)&(a['gt'][:,1]>=y0)&(a['gt'][:,1]<=y1)
        summary=gap_summaries.setdefault('/'.join([case,scenario,mode]),{'runs':0,'gap_scans_with_gt':0,
            'missing_estimates':0,'position_successes':0,'yaw_successes':0,'joint_successes':0,
            'times_without_gt':0,'room_gap_scans':0,'gaps':0,'recovered_gaps':0,'recovery_times_s':[],
            'max_position_error_cm':None,'max_yaw_error_deg':None,'max_safe_duration_s_by_seed':[],
            'max_safe_distance_m_gt_by_seed':[]})
        longest=0.;furthest=0.;summary['runs']+=1
        for row in rows:
            start,end=row['start_index'],row['end_index_exclusive'];select=slice(start,end)
            row.update(distances(a['gt'],a['has_gt'],success,start,end))
            row['room_scans']=int(room[select].sum());summary['room_gap_scans']+=row['room_scans']
            valid=a['has_gt'][select];paired=valid&finite[select]
            summary['gap_scans_with_gt']+=int(valid.sum());summary['times_without_gt']+=int((~valid).sum())
            summary['missing_estimates']+=int((valid&~finite[select]).sum())
            summary['position_successes']+=int((paired&(position[select]<.1)).sum())
            summary['yaw_successes']+=int((paired&(yaw[select]<4)).sum())
            summary['joint_successes']+=int(success[select].sum());summary['gaps']+=1
            if paired.any():
                for field,value in [('max_position_error_cm',float(position[select][paired].max()*100)),('max_yaw_error_deg',float(yaw[select][paired].max()))]:
                    summary[field]=max(summary[field] if summary[field] is not None else value,value)
            if row['recovered']:
                summary['recovered_gaps']+=1;summary['recovery_times_s'].append(row['recovery_s'])
            longest=max(longest,row['max_safe_duration_s']);furthest=max(furthest,row['max_safe_distance_m_gt'])
        summary['max_safe_duration_s_by_seed'].append(longest);summary['max_safe_distance_m_gt_by_seed'].append(furthest)
    for summary in gap_summaries.values():
        n=summary['gap_scans_with_gt']
        for key in ['position','yaw','joint']:summary[key+'_success_percent']=100*summary[key+'_successes']/n if n else None
        recovery=summary['recovery_times_s'];summary['mean_recovery_s']=float(np.mean(recovery)) if recovery else None
        summary['max_recovery_s']=float(max(recovery)) if recovery else None
    write(output/'gaps-evaluated.json',{'method':'Additional distances use GT; missing GT edges excluded and counted. Missing pose with GT fails.',
        'summary':gap_summaries,'intervals':gap_rows})
    paired={}
    comparisons=[('reverse-front','reverse-original'),('reverse-three','reverse-original'),
        ('forward-front','forward-original'),('forward-three','forward-original')]
    for scenario in ['natural','artificial']:
        for mode in ['standard','bridge']:
            for treatment,control in comparisons:
                label=f'{treatment}-minus-{control}/{scenario}/{mode}'
                a=metrics['runs'][f'{treatment}/{scenario}/{mode}'];b=metrics['runs'][f'{control}/{scenario}/{mode}']
                paired[label]={}
                for field in ['position_below_10cm_percent','yaw_below_4deg_percent','success_percent']:
                    delta=np.array([x[field]-y[field] for x,y in zip(a,b)])
                    paired[label][field]={'mean':float(delta.mean()),'std':float(delta.std(ddof=0)),'paired_deltas':delta.tolist()}
    write(output/'paired-effects.json',paired)
    with np.load(cache/'projection.npz') as projected:
        valid=projected['visible']&np.isfinite(projected['association']).T[:,:,None]
        semantic={'front_valid_mask_covered_hits':int(valid[0].sum()),
            'side_additional_valid_mask_covered_hits':int((valid[1:].any(axis=0)&~valid[0]).sum()),
            'three_unique_valid_mask_covered_hits':int(valid.any(axis=0).sum())}
    write(output/'semantic-coverage.json',semantic)
    lines=['# LILocBench: zaključen obratni preizkus','',f'Zaključenih je vseh **{expected}/{expected}** zagonov AMCL. Semena 42–51; povprečje ± populacijski standardni odklon. Manjkajoča ocena ob veljavnem GT šteje kot neuspeh.','',
        '**Podatkovna omejitev:** isti prostor je potrjen s kalibracijo, skupnim GT in slikami. `lt_changes_0` ima dodano škatlo in prestavljeno pohištvo. `static_0` in neodvisni `mapping` sta osnovni stanji z običajnim pohištvom; ne predstavljata potrjeno povsem prazne sobe ali kontroliranega para pred/po odstranitvi vseh predmetov.','',
        '[Preverjanje sobe in pogledov](room-pair-evidence.png) · [Protokol](protocol.json) · [Vse metrike](metrics.json) · [Parne razlike](paired-effects.json)','',
        '## Obratna lokalizacija: originalni static_0 skeni','',
        '| Zemljevid | Položaj <10 cm [%] | Orientacija <4° [%] | Oba [%] |','|---|---:|---:|---:|']
    labels={'original':'Polna vožnja, brez filtra','front':'Polna vožnja, sprednja kamera','three':'Polna vožnja, tri kamere','empty':'Neodvisni osnovni mapping'}
    for branch in ['original','front','three','empty']:
        row=metrics['aggregate'][f'reverse-{branch}/natural/standard']
        lines.append('| '+labels[branch]+' | '+' | '.join(fmt(row[k]) for k in ['position_below_10cm_percent','yaw_below_4deg_percent','success_percent'])+' |')
    lines+=['','## Lokalizacija lt_changes_0 na neodvisnem osnovnem zemljevidu','',
        '| Vhodni skeni | Položaj <10 cm [%] | Orientacija <4° [%] | Oba [%] |','|---|---:|---:|---:|']
    for branch in ['original','front','three']:
        row=metrics['aggregate'][f'forward-{branch}/natural/standard']
        lines.append('| '+labels[branch]+' | '+' | '.join(fmt(row[k]) for k in ['position_below_10cm_percent','yaw_below_4deg_percent','success_percent'])+' |')
    lines+=['','![Zemljevidi](maps-comparison.png)','','## Geometrija proti static_0','',
        'Evklidska toleranca je 10 cm pri vseh variantah. Primarni imenovalec so ohranjeni zadetki. Odvečna točka mora ležati v referenčno potrjenem prostem prostoru. Neopaženo in opaženo, a neujemajoče se/zasedeno, ostajata ločeni kategoriji.','',
        '| Območje / filter | Točke | Ujemanje [%] | Odvečno v prostem [%] | Neopaženo [%] | Drugo opaženo [%] | Manjkajoča referenca, sekundarno [%] |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for region in ['whole','room']:
        for branch in ['original','front','three']:
            row=geometry[branch][region]
            lines.append('| '+region+' / '+branch+' | '+str(row['retained_points'])+' | '+' | '.join(fmt(row[k]) for k in ['matched_percent','confirmed_free_excess_percent','unobserved_percent','other_observed_percent','reference_missing_percent_secondary'])+' |')
    lines+=['',f'Stranski kameri geometrijsko dodatno pokrijeta **{projection["side_additional_covered_hits"]:,}** zadetkov; z veljavnimi maskami dodatno pokrijeta **{semantic["side_additional_valid_mask_covered_hits"]:,}**. Dodatno odstranita **{projection["side_additional_removed_hits"]:,}**, hkrati pa zaščitita **{projection["front_removals_vetoed_by_sides"]:,}** zadetkov, ki bi jih odstranila sprednja kamera. Unija je šteta po ID-ju žarka.',
        '', '![Geometrija sobe](geometry-room.png)','![Geometrija celote](geometry-whole.png)','','## Premostitev z odometrijo','',
        'Prag je 100 veljavnih zadetkov. Sidro določa kovarianca AMCL (≤20 cm, ≤10°), brez GT. Med vrzeljo se LiDAR popravek preskoči. Umetne vrzeli pri 40 s, 90 s in 180 s trajajo 0,5 s, 1 s in 2 s. Običajni AMCL in premostitev dobita enake poškodovane skene. Čas stabilizacije zahteva vsaj sekundo neprekinjenega uspeha po vrzeli.',
        '', '| Primer / način, umetne vrzeli | Položaj <10 cm [%] | Yaw <4° [%] | Oba [%] | Max napaka [cm / °] | Ponovno stabilizirane vrzeli | Povp. čas stabilizacije [s] |',
        '|---|---:|---:|---:|---:|---:|---:|']
    natural=sum(s['gaps'] for k,s in gap_summaries.items() if '/natural/' in k)
    if natural==0:lines+=['| Naravne vrzeli | — | — | — | — | 0 zaznanih | — |']
    for key,row in gap_summaries.items():
        if '/artificial/' not in key:continue
        lines.append('| '+key.replace('/artificial/',' / ')+' | '+' | '.join(fmt(row[k+'_success_percent']) for k in ['position','yaw','joint'])+' | '+fmt(row['max_position_error_cm'])+' / '+fmt(row['max_yaw_error_deg'])+f' | {row["recovered_gaps"]}/{row["gaps"]} | '+fmt(row['mean_recovery_s'])+' |')
    lines+=['','Deleži v tej tabeli veljajo samo za vzorce med vrzelmi z GT; manjkajoče ocene so neuspeh. `gaps-evaluated.json` ločeno vsebuje najdaljše uspešno trajanje in razdaljo za vsako seme, razdalje iz GT, varen začetni odsek, neopaženi GT, vrzeli v sobi in neuspešne ponovne stabilizacije. Najdaljši preizkušeni umetni izpad je 2 s; iz njega ne sklepamo o daljših izpadih.']
    if natural==0:lines+=['','Pri izbranem pragu naravnih vrzeli ni bilo. Preizkus zato potrjuje le odziv na umetne vrzeli; uspešnosti naravne premostitve v sobi ni mogoče oceniti.']
    lines+=['','## Poti','', 'Modra: ocena. Zelena: GT. Rdeča: neuspeh, tudi manjkajoča ocena. Oranžna: odometrijska premostitev. Prikazano je vnaprej določeno seme 42.']
    for name in ['reverse-original','reverse-front','reverse-three','reverse-empty','forward-original','forward-front','forward-three']:
        lines.append(f'\n![{name}, naravni skeni](trajectory-{name}-natural-standard.png)')
        lines.append(f'\n![{name}, umetne vrzeli in premostitev](trajectory-{name}-artificial-bridge.png)')
    lines+=['','Ponovitev: `MPLCONFIGDIR=/tmp/mpl-lilocbench python3 maintenance/run_reverse_experiment.py`.',
        'Podatki in zemljevidi: Trekel et al., LILocBench, IROS 2025, CC BY-NC-SA 4.0. [Uradni opis podatkov](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/).']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    write(output/'report-provenance.json',{'postprocessor_sha256':digest(__file__),'input_sha256':sources,
        'canonical_experiment_complete':True,'completed_amcl_runs':expected,'paired_seeds':cfg['seeds']})


if __name__=='__main__':main()
