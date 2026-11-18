"""Audit cell evidence, paired geometry and unchanged localization evaluation."""
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from maintenance.run_reverse_short import sha,room


def read(path):return list(csv.DictReader(path.open()))


def main():
    cfg=json.loads((ROOT/'configs/lilocbench-temporal-cells.json').read_text());out=ROOT/cfg['output'];cache=ROOT/cfg['cache']
    protocol=json.loads((out/'map-protocol.json').read_text());rule=cfg['rule'];e=dict(np.load(cache/'cell-evidence.npz'))
    pop=np.array([i.bit_count() for i in range(8)])
    protect=e['counts'][1]>=rule['minimum_fixed_epochs_to_protect']
    confirmed=(e['counts'][0]>=rule['minimum_movable_epochs'])&(pop[e['camera_bits'][0]]>=rule['minimum_distinct_cameras'])
    np.testing.assert_array_equal(protect,e['protected']);np.testing.assert_array_equal(confirmed&~protect,e['removed_cells'])
    for label in range(2):
        used=e['counts'][label]>0
        assert np.all(e['last'][label,used]-e['first'][label,used]+1e-6>=(e['counts'][label,used]-1)*rule['minimum_epoch_gap_s'])
    assert np.all(e['accepted_epochs_by_camera'].sum(axis=0)>0)
    flags=np.load(cache/'explicit-projection.npz')['flags'];association=np.load(ROOT/'data/lilocbench-reverse/projection.npz')['association']
    for ci in range(3):assert not flags[ci,~np.isfinite(association[:,ci])].any()
    old=np.load(ROOT/'results/lilocbench-reverse/maps/three/occupancy.npy');new=np.load(cache/'map-build/maps/temporal/occupancy.npy')
    for variant,grid in [('before',old),('after',new)]:
        pixels=np.flipud(np.array(Image.open(out/'maps'/variant/'map.png')))
        decoded=np.where(pixels==0,100,np.where(pixels==254,0,-1)).astype(np.int8);np.testing.assert_array_equal(decoded,grid)
    cloud=dict(np.load(ROOT/'data/lilocbench-reverse/cloud-lt_changes_0.npz'));ids=cloud['ids'];p=cloud['points'];meta=protocol['map']
    xy=np.floor((p-np.asarray(meta['origin'][:2]))/meta['resolution']).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<meta['width'])&(xy[:,1]>=0)&(xy[:,1]<meta['height'])
    removed=np.load(cache/'removed.npz')['removed'];expected=np.zeros(len(p),bool);expected[inside]=e['removed_cells'][xy[inside,1]*meta['width']+xy[inside,0]]
    np.testing.assert_array_equal(removed.ravel()[ids],expected)
    old_keep=~np.load(ROOT/'data/lilocbench-reverse/projection.npz')['removed'][1].ravel()[ids];new_keep=~expected;labels=np.load(cache/'reference-labels.npy')
    geo=read(out/'geometry.csv');effects=read(out/'geometry-effects.csv');legacy=read(ROOT/'docs/lilocbench-reverse-short/geometry.csv')
    for region,select in [('whole',np.ones(len(p),bool)),('room',room(p,protocol['room_bounds']))]:
        for variant,keep in [('before',old_keep),('after',new_keep)]:
            row=next(r for r in geo if r['variant']==variant and r['region']==region);counts=np.bincount(labels[keep&select],minlength=4)
            assert int(row['retained_hits'])==counts.sum()
            for key,value in zip(['matched_hits','excess_free_hits','unobserved_hits','other_observed_hits'],counts):assert int(row[key])==value
        prior=next(r for r in legacy if r['variant']=='three' and r['region']==region);before=next(r for r in geo if r['variant']=='before' and r['region']==region)
        assert int(before['matched_hits'])==int(prior['matched_points']) and int(before['excess_free_hits'])==int(prior['confirmed_free_excess_points'])
        effect=next(r for r in effects if r['region']==region)
        assert int(effect['net_excess_reduction'])==int(effect['excess_disappeared'])-int(effect['excess_reintroduced'])
        assert int(effect['old_matching_hits'])==int(effect['old_matching_retained'])+int(effect['old_matching_removed'])
    rows=read(out/'poses.csv');selection=read(ROOT/'docs/lilocbench-relocalization/scan-coverage.csv');stats=read(out/'localization.csv');prior=read(ROOT/'docs/lilocbench-relocalization/summary.csv');doc=(out/'README.md').read_text()
    for case in ['three-confirmed','temporal-confirmed']:
        used=[r for r in rows if r['case']==case];assert len(used)==8982
        assert [(r['scan_index'],r['timestamp'],r['covered']) for r in used]==[(r['scan_index'],r['timestamp'],r['covered']) for r in selection]
        for region in ['whole','room']:
            evaluated=[r for r in used if r['has_gt']=='True' and r['covered']=='True' and (region=='whole' or r['room']=='True')];estimated=[r for r in evaluated if r['has_estimate']=='True']
            s=next(s for s in stats if s['case']==case and s['region']==region)
            assert int(s['N'])==len(evaluated)==(7550 if region=='whole' else 478)
            assert int(s['n'])==sum(float(r['position_error_cm'])<10 and float(r['yaw_error_deg'])<4 for r in estimated)
            assert np.isclose(float(s['p95_position_cm']),np.percentile([float(r['position_error_cm']) for r in estimated],95))
            assert np.isclose(float(s['p95_yaw_deg']),np.percentile([float(r['yaw_error_deg']) for r in estimated],95))
            assert int(s['jumps_over_1m'])==sum(r['covered_jump_gt_pair']=='True' and bool(r['jump_m']) and float(r['jump_m'])>1 for r in evaluated)
            assert f"| {case} / {region} | {s['n']}/{s['N']} ({float(s['success_percent']):.2f}) |" in doc
            if case=='three-confirmed':
                previous=next(s for s in prior if s['case']==case and s['region']==region)
                for key in s:
                    assert (s[key]==previous[key]) or (s[key] and previous[key] and np.isclose(float(s[key]),float(previous[key]))),(key,s[key],previous[key])
    before_audit=json.loads((ROOT/'data/lilocbench-relocalization/full/three-wheel_odometry-audit.json').read_text());after_audit=json.loads((cache/'full/temporal-wheel_odometry-audit.json').read_text())
    assert before_audit['motion_sha256']==after_audit['motion_sha256'] and before_audit['scan_sha256']==after_audit['scan_sha256']
    paired=[]
    for region in ['whole','room']:
        a=next(s for s in stats if s['case']=='three-confirmed' and s['region']==region);b=next(s for s in stats if s['case']=='temporal-confirmed' and s['region']==region)
        ga=next(g for g in geo if g['variant']=='before' and g['region']==region);gb=next(g for g in geo if g['variant']=='after' and g['region']==region);effect=next(e for e in effects if e['region']==region)
        row=dict(region=region,N=a['N'])
        for key in ['n','success_percent','p95_position_cm','p95_yaw_deg','jumps_over_1m']:
            row['before_'+key]=a[key];row['after_'+key]=b[key]
        for key in ['matched_hits','excess_free_hits']:
            row['before_'+key]=ga[key];row['after_'+key]=gb[key]
        for key in ['excess_disappeared','excess_reintroduced','old_matching_retained','old_matching_retained_percent']:row[key]=effect[key]
        paired.append(row)
    with (out/'before-after.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]),lineterminator='\n');writer.writeheader();writer.writerows(paired)
    if 'Sklep za dopolnitev PDF:' not in doc:
        overall=paired[0]
        doc+=f"\nSklep za dopolnitev PDF: pri teh vnaprej zamrznjenih pragih časovno združevanje ne izboljša skupne uspešnosti ({float(overall['before_success_percent']):.2f} → {float(overall['after_success_percent']):.2f} %). Ohrani {float(overall['old_matching_retained_percent']):.2f} % prej ujemajočih se zadetkov, toda ponovno vključi več odvečne geometrije, kot je dodatno odstrani. Manj skokov ne odtehta slabšega P95 položaja. [Združena tabela prej–potem za PDF](before-after.csv).\n"
    if 'Neodvisno preverjanje:' not in doc:
        doc+='\nNeodvisno preverjanje: `OPENBLAS_NUM_THREADS=1 python3 maintenance/validate_temporal_cells.py`. Preveri glasove, abstinenco manjkajočih mask, native izvoze, geometrijo, isto pokritost, oba CSV ter nespremenjeno odometrijo in baseline.\n'
    (out/'README.md').write_text(doc)
    for link in re.findall(r'\]\(([^)]+)\)',doc):
        if not link.startswith('http'):assert (out/link).exists(),link
    validation=json.loads((out/'validation.json').read_text());validation.update(independent_audit=True,same_7550_covered_and_478_room_poses=True,baseline_metrics_identical=True,
        same_odometry_sha256=True,cell_vote_limits_and_fixed_protection_verified=True,native_maps_exact=True,geometry_and_localization_csv_verified=True,validator_sha256=sha(Path(__file__)))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('PASS: temporal cell evidence, missing-mask abstention, maps, paired geometry, baseline, 17964 poses, fixed coverage, odometry, summaries and README')


if __name__=='__main__':main()
