"""Independently audit published CSV and append coverage-only interpretation."""
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from maintenance.run_sequential_scan_to_map import move
from maintenance.run_reverse_short import sha


def main():
    out=ROOT/'docs/lilocbench-relocalization';protocol=json.loads((out/'protocol.json').read_text())
    rows=list(csv.DictReader((out/'poses.csv').open()));summary=list(csv.DictReader((out/'summary.csv').open()))
    selected=list(csv.DictReader((out/'scan-coverage.csv').open()));events=list(csv.DictReader((out/'confirmation-events.csv').open()))
    assert len(rows)==5*8982
    coverage=np.flipud(np.array(Image.open(out/'coverage-mask.png'))>0)
    original=np.load(ROOT/'results/lilocbench-reverse/maps/original/occupancy.npy');control=np.load(ROOT/'results/lilocbench-reverse/maps/empty/occupancy.npy')
    np.testing.assert_array_equal(coverage,(original!=-1)&(control!=-1))
    assert protocol['source_sha256']==sha(ROOT/'maintenance/run_confirmed_relocalization.py')
    assert protocol['tracking_source_sha256']==sha(ROOT/'maintenance/run_sequential_scan_to_map.py')
    doc=(out/'README.md').read_text()
    for case in dict.fromkeys(r['case'] for r in rows):
        used=[r for r in rows if r['case']==case]
        assert [(r['scan_index'],r['timestamp']) for r in used]==[(r['scan_index'],r['timestamp']) for r in selected]
        for i,r in enumerate(used):
            assert r['covered']==selected[i]['covered']
            if r['has_estimate']=='True' and r['has_gt']=='True':
                error=np.hypot(float(r['x'])-float(r['gt_x']),float(r['y'])-float(r['gt_y']))*100
                angle=abs((float(r['yaw_rad'])-float(r['gt_yaw_rad'])+np.pi)%(2*np.pi)-np.pi)*180/np.pi
                assert np.isclose(error,float(r['position_error_cm'])) and np.isclose(angle,float(r['yaw_error_deg']))
                assert np.isclose(float(r['E']),max(error/10,angle/4))
                assert (r['success']=='True')==(error<10 and angle<4)
            if i and r['pending']=='True' and used[i-1]['pending']=='True':
                prev=used[i-1];pose=np.array([float(prev[k]) for k in ['candidate_x','candidate_y','candidate_yaw_rad']]);delta=[float(r[k]) for k in ['delta_x_m','delta_y_m','delta_yaw_rad']]
                np.testing.assert_allclose(move(pose,delta),[float(r[k]) for k in ['candidate_x','candidate_y','candidate_yaw_rad']],atol=1e-12)
        for region in ['whole','room']:
            candidates=[r for r in used if r['has_gt']=='True' and (region=='whole' or r['room']=='True')];included=[r for r in candidates if r['covered']=='True'];estimated=[r for r in included if r['has_estimate']=='True']
            s=next(s for s in summary if s['case']==case and s['region']==region)
            assert int(s['N'])==len(included) and int(s['n'])==sum(r['success']=='True' for r in included)
            assert int(s['excluded_unmapped'])==len(candidates)-len(included)
            assert np.isclose(float(s['success_percent']),100*int(s['n'])/int(s['N']))
            for column in ['position_error_cm','yaw_error_deg']:
                key='p95_position_cm' if column=='position_error_cm' else 'p95_yaw_deg'
                assert np.isclose(float(s[key]),np.percentile([float(r[column]) for r in estimated],95))
            jumps=sum(r['covered_jump_gt_pair']=='True' and bool(r['jump_m']) and float(r['jump_m'])>1 for r in included)
            assert jumps==int(s['jumps_over_1m'])
            assert f"| {case} / {region} | {s['n']}/{s['N']} ({float(s['success_percent']):.2f}) |" in doc
    confirmation=protocol['configuration']['confirmation']
    for event in events:
        if event['status']=='confirmed':
            assert int(event['scans_checked'])>=confirmation['consecutive_scans']
            assert float(event['latency_s'])>=confirmation['minimum_span_s']
    marker='Podrobnejši sklep na isti maski:'
    if marker not in doc:
        lookup={(s['case'],s['region']):s for s in summary};a=lookup['three-immediate','room'];b=lookup['three-confirmed','room'];f=lookup['front-confirmed','whole'];t=lookup['three-confirmed','whole']
        doc+=f"\n{marker} v sobi potrjevanje treh kamer spremeni uspeh {a['n']}/{a['N']} → {b['n']}/{b['N']}, P95 položaja {float(a['p95_position_cm']):.2f} → {float(b['p95_position_cm']):.2f} cm in P95 zasuka {float(a['p95_yaw_deg']):.2f} → {float(b['p95_yaw_deg']):.2f}°. Zmanjšanje skokov na celotni pokriti poti torej ne pomeni izboljšanja sobe. Tri potrjene kamere glede na sprednjo dosežejo {float(t['success_percent'])-float(f['success_percent']):+.2f} odstotne točke več uspeha, vendar slabši P95 položaja ({float(t['p95_position_cm']):.2f} proti {float(f['p95_position_cm']):.2f} cm).\n"
        doc=doc.replace('python3 maintenance/run_confirmed_relocalization.py --stage full\n```','python3 maintenance/run_confirmed_relocalization.py --stage full\nOPENBLAS_NUM_THREADS=1 python3 maintenance/validate_relocalization.py\n```')
        (out/'README.md').write_text(doc)
    for link in re.findall(r'\]\(([^)]+)\)',doc):
        if not link.startswith('http'):assert (out/link).exists(),link
    validation=json.loads((out/'validation.json').read_text());validation.update(independent_csv_audit=True,coverage_reconstructed_equal=True,
        all_pose_ids_times_and_masks_equal=True,odometry_linked_proposal_chain_verified=True,summary_and_readme_verified=True,validator_sha256=sha(Path(__file__)))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('PASS: 44910 rows; exact common mask; timestamps; GT errors/E; pending odometry chains; covered summaries, jumps and README; confirmation count/delay')


if __name__=='__main__':main()
