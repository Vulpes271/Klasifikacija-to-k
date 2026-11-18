"""Audit independent ablations and export compact decisions/runtime tables."""
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from maintenance.run_reverse_short import sha
from maintenance.run_sequential_scan_to_map import move
from maintenance.run_localization_ablations import collect,summarize
from maintenance.run_scan_to_map import csvwrite


def read(path):return list(csv.DictReader(path.open()))


def main():
    cfg=json.loads((ROOT/'configs/lilocbench-ablations.json').read_text());out=ROOT/cfg['output'];cache=ROOT/cfg['cache'];protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['configuration']==cfg and protocol['source_sha256']==sha(ROOT/'maintenance/run_localization_ablations.py')
    assert protocol['reference_csv_sha256']==sha(ROOT/'data/lilocbench-relocalization/full/three-wheel_odometry.csv')
    rows,events=collect(cfg,'full');stats,episodes=summarize(rows,cfg);saved=read(out/'poses.csv');table=read(out/'summary.csv');selection=read(ROOT/'docs/lilocbench-relocalization/scan-coverage.csv')
    assert len(rows)==len(saved)==4*8982
    for calculated,actual in zip(rows,saved):
        for key in ['case','scan_index','timestamp','covered','success','confirmation_event']:assert str(calculated[key])==actual[key]
    for case in ['reference',*cfg['variants']]:
        used=[r for r in rows if r['case']==case]
        assert [(r['scan_index'],r['timestamp'],r['covered']) for r in used]==[(int(r['scan_index']),float(r['timestamp']),r['covered']=='True') for r in selection]
        for r in used:
            if r['has_gt'] and r['has_estimate']:
                error=np.hypot(r['x']-r['gt_x'],r['y']-r['gt_y'])*100
                angle=abs((r['yaw_rad']-r['gt_yaw_rad']+np.pi)%(2*np.pi)-np.pi)*180/np.pi
                assert np.isclose(error,r['position_error_cm']) and np.isclose(angle,r['yaw_error_deg'])
                assert r['success']==(error<10 and angle<4) and np.isclose(r['E'],max(error/10,angle/4))
        for region in ['whole','room']:
            included=[r for r in used if r['has_gt'] and r['covered'] and (region=='whole' or r['room'])]
            assert len(included)==(7550 if region=='whole' else 478)
    for expected,actual in zip(stats,table):
        for key,value in expected.items():
            if value is None:assert actual[key]==''
            elif isinstance(value,str):assert actual[key]==value
            else:assert np.isclose(float(actual[key]),value),(key,value,actual[key])
    assert len(read(out/'relocalization-episodes.csv'))==len(episodes)
    # Check each recovered episode's observed stable success, not merely an event label.
    for event in episodes:
        if event['status']!='recovered':continue
        used=[r for r in rows if r['case']==event['case'] and event['end_time']<=r['timestamp']<=event['end_time']+1.10]
        stop=next(i for i,r in enumerate(used) if r['timestamp']-event['end_time']>=cfg['evaluation']['stable_recovery_s'])
        assert all(r['has_gt'] and r['covered'] and r['success'] and (event['region']=='whole' or r['room']) for r in used[:stop+1])
        assert np.isclose(event['recovery_s'],event['end_time']-event['start_time'])
    ray=read(out/'ray-candidates.csv')
    for r in ray:
        penalty=float(r['path_penalty']);assert 0<=penalty<=cfg['rays']['maximum_mean_penalty']
        assert np.isclose(float(r['combined_score']),float(r['endpoint_score'])-penalty)
        assert 0<=int(r['conflicts'])<=int(r['checked'])<=cfg['rays']['beams']
        if int(r['conflicts'])==0:assert penalty==0
    groups={}
    for r in ray:groups.setdefault(r['timestamp'],[]).append(r)
    for group in groups.values():
        assert len(group)<=cfg['rays']['top_candidates'] and sum(r['selected']=='True' for r in group)==1
        chosen=next(r for r in group if r['selected']=='True');assert int(chosen['rank_after'])==1
        assert float(chosen['combined_score'])==max(float(r['combined_score']) for r in group)
    parallel=read(out/'parallel-comparisons.csv');pose_lookup={r['timestamp']:r for r in rows if r['case']=='parallel'}
    for r in parallel:
        t=float(r['timestamp']);row=pose_lookup[t];i=row['scan_index'];previous=next(x for x in rows if x['case']=='parallel' and x['scan_index']==i-1)
        motion=np.array([row[k] for k in ['delta_x_m','delta_y_m','delta_yaw_rad']]);np.testing.assert_allclose(motion,[float(r[k]) for k in ['delta_x','delta_y','delta_yaw']])
        prior=np.array([previous[k] for k in ['candidate_x','candidate_y','candidate_yaw_rad']]);np.testing.assert_allclose(move(prior,motion)[:2],[float(r['remote_x']),float(r['remote_y'])],atol=1e-10)
        advantage=r['remote_valid']=='True' and float(r['remote_score'])-float(r['local_score'])>=cfg['parallel']['minimum_score_advantage']
        assert (r['advantage']=='True')==advantage
        if row['state']!='confirmed_global':np.testing.assert_allclose([row['x'],row['y']],[float(r['local_x']),float(r['local_y'])])
    for event in events:
        if event['case']=='parallel' and event['kind']=='distant' and event['status']=='confirmed':
            last=next(r for r in parallel if float(r['timestamp'])==event['end_time']);assert int(last['advantage_hits'])>=cfg['parallel']['consecutive_advantage_scans']
    reference_rows=[r for r in rows if r['case']=='reference'];parallel_rows=[r for r in rows if r['case']=='parallel']
    if not any(r['local_accepted']=='True' for r in parallel):
        for a,b in zip(reference_rows,parallel_rows):
            for key in ['x','y','yaw_rad']:
                assert (a[key] is None and b[key] is None) or (a[key] is not None and b[key] is not None and np.isclose(a[key],b[key],atol=1e-12))
    old=np.load(ROOT/cfg['additive']['old_removals'])['removed'][1];temporal=np.load(ROOT/cfg['additive']['temporal_removals'])['removed'];union=np.load(cache/'union-removals.npz')
    np.testing.assert_array_equal(union['removed'],old|temporal);assert not (old&~union['removed']).any()
    labels=np.load(ROOT/'data/lilocbench-temporal-cells/reference-labels.npy');cloud=np.load(ROOT/'data/lilocbench-reverse/cloud-lt_changes_0.npz');extra=union['additional'].ravel()[cloud['ids']]
    g=read(out/'additive-geometry.csv')[0];assert int(g['additional_excess_removed'])==int((extra&(labels==1)).sum()) and int(g['matching_hits_lost'])==int((extra&(labels==0)).sum())
    baseline_audit=json.loads((ROOT/'data/lilocbench-relocalization/full/three-wheel_odometry-audit.json').read_text());audit=json.loads((out/'map-audit.json').read_text());runtimes=[]
    for case in ['reference',*cfg['variants']]:
        s=next(s for s in stats if s['case']==case and s['region']=='whole')
        if case=='reference':wall=None;prep=baseline_audit['preparation_s']
        else:
            name='additive' if case=='additive' else 'three';current=json.loads((cache/case/'full'/f'{name}-wheel_odometry-audit.json').read_text())
            assert current['scan_sha256']==baseline_audit['scan_sha256'] and current['motion_sha256']==baseline_audit['motion_sha256']
            assert current['map_sha256']==(audit['new_map_sha256'] if case=='additive' else baseline_audit['map_sha256'])
            wall=json.loads((cache/case/'full/runtime.json').read_text())['wall_s'];prep=current['preparation_s']
        runtimes.append(dict(case=case,processing_sum_s=s['processing_sum_s'],wall_s=wall,preparation_s=prep,median_scan_s=s['median_scan_s'],p95_scan_s=s['p95_scan_s']))
    csvwrite(out/'runtime.csv',runtimes)
    decisions=[dict(case=r['case'],scan_index=r['scan_index'],timestamp=r['timestamp'],event=r['confirmation_event'],covered=r['covered'],room=r['room'],position_error_cm=r['position_error_cm'],yaw_error_deg=r['yaw_error_deg'],success=r['success']) for r in rows if r['confirmation_event'] in ['confirmed','near_accepted']]
    csvwrite(out/'global-decisions.csv',decisions)
    doc=(out/'README.md').read_text().replace('Samostatni učinek','Samostojni učinek')
    for s in stats:assert f"| {s['case']} / {s['region']} | {s['n']}/{s['N']} ({s['success_percent']:.2f}) |" in doc
    if 'Dodatna revizija:' not in doc:
        accepted=sum(r['local_accepted']=='True' for r in parallel);changed=sum(r['selected']=='True' and int(r['rank_before'])>1 for r in ray)
        doc+=f"\nDodatna revizija: v {len(parallel)} vzporednih preverjanjih se je lokalni fit obnovil {accepted}-krat. Zato je v tem naboru vzporedna različica dala povsem iste ocenjene poze kot referenca; preklic ob obnovi je preverjen s sintetičnim testom, ne potrjen z naravno epizodo tega posnetka. Žarkovna kazen spremeni prvo uvrščeno hipotezo pri {changed}/{len(groups)} globalnih iskanjih.\n"
        doc+='\nEpizode izgube so določene iz napak posamezne različice, zato niso isti parni izpadi; časi obnove so opisni in vedno opremljeni s številom obnovljenih/cenzuriranih. [Natančne globalne odločitve](global-decisions.csv) in [časi izvajanja](runtime.csv). Referenčni celotni stenski čas ni bil shranjen; vsota processing_s je na voljo.\n'
        doc+='\nPreverjanje izvoza: `OPENBLAS_NUM_THREADS=1 python3 maintenance/validate_localization_ablations.py`.\n'
    (out/'README.md').write_text(doc)
    for link in re.findall(r'\]\(([^)]+)\)',doc):
        if not link.startswith('http'):assert (out/link).exists(),link
    validation=json.loads((out/'validation.json').read_text());validation.update(independent_csv_audit=True,all_35928_pose_rows_checked=True,ray_penalties_and_reranking_verified=True,
        parallel_odometry_and_advantage_verified=True,additive_removals_superset_verified=True,same_7550_and_478_gt_poses=True,runtime_and_recovery_metrics_verified=True,
        validator_sha256=sha(Path(__file__)))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('PASS: independent ablations; 35928 poses; same coverage/scans/odometry; ray penalties; parallel hypotheses; additive superset; errors, jumps, confirmations, recovery and timings')


if __name__=='__main__':main()
