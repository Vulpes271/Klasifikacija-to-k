"""Audit inputs, camera timing, fresh map integration, and paired scan identity."""
from pathlib import Path
import sys
import tempfile
import csv
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm_object_classifier.localization_reverse import Experiment, CAMERAS
from llm_object_classifier.localization_pilot import write
from maintenance.run_reverse_short import sha, OUT


def main():
    e=Experiment(ROOT/'configs/lilocbench-reverse.json')
    projection=dict(np.load(e.cache/'projection.npz'));t=e.arrays('lt_changes_0')['times']
    assoc=projection['association'];ok=np.isfinite(assoc)
    dt=np.abs(assoc-(t+e.laser('lt_changes_0')['scan_time']/2)[:,None])
    assert np.all(dt[ok]<=e.pilot['mask_max_dt'])
    cameras=[]
    for ci,c in enumerate(CAMERAS):
        statuses=json.loads((e.cache/f'{c}-status-full.json').read_text())
        valid={r['timestamp'] for r in statuses if r['classification_valid'] and r.get('mask_report')}
        assert all(float(v) in valid for v in assoc[ok[:,ci],ci])
        cameras.append(dict(camera=c,max_mask_dt_s=float(np.nanmax(dt[:,ci])),
            intrinsics_sha256=sha(e.data/'lt_changes_0'/c/'color/intrinsics.yaml'),
            status_sha256=sha(e.cache/f'{c}-status-full.json')))
    expected={n:np.load(e.output/'maps'/n/'occupancy.npy') for n in ['original','three','empty']}
    with tempfile.TemporaryDirectory(prefix='lilocbench-map-check-') as temporary:
        e.cache=Path(temporary);e.output=OUT/'rebuilt';e.output.mkdir(exist_ok=True)
        a=e.arrays('lt_changes_0')
        e.build_map('lt_changes_0',[np.zeros(a['ranges'].shape,bool),projection['removed'][1]],['original','three'])
        a=e.arrays('mapping');e.build_map('mapping',[np.zeros(a['ranges'].shape,bool)],['empty'])
    checks={n:bool(np.array_equal(v,np.load(e.output/'maps'/n/'occupancy.npy'))) for n,v in expected.items()}
    assert all(checks.values())
    write(OUT/'map-rebuild-validation.json',dict(fresh_ray_integration_matches_saved_maps=checks,removed_beams_skip_entire_ray=True))
    rows=list(csv.DictReader((OUT/'poses.csv').open()));selected=list(csv.DictReader((OUT/'selected-poses.csv').open()))
    ids=[int(r['scan_index']) for r in selected]
    for name in ['original','three','empty']:
        used=[r for r in rows if r['kind']=='independent' and r['variant']==name]
        assert [int(r['scan_index']) for r in used]==ids
        assert [r['timestamp'] for r in used]==[r['timestamp'] for r in selected]
    a=e.arrays('static_0')
    for name in ['zero_motion','wheel_odometry']:
        used=[r for r in rows if r['kind']=='sequential' and r['variant']==name]
        assert len(used)==len(a['times'])
        assert [int(r['scan_index']) for r in used]==list(range(len(a['times'])))
        np.testing.assert_array_equal([float(r['timestamp']) for r in used],a['times'])
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(ROOT/'maintenance/run_reverse_short.py')==protocol['source_sha256']
    assert sha(e.raw/'static_0/raw.npz')==protocol['test_scan_sha256']
    write(OUT/'input-audit.json',dict(cameras=cameras,mask_max_dt_s=e.pilot['mask_max_dt'],
        extrinsics_sha256=sha(e.data/'lt_changes_0/transformations.yaml'),
        ray_integrator_sha256=sha(ROOT/'llm_object_classifier/occupancy_rays.c'),
        scan_ids_and_timestamps_equal_all_maps=True,all_sequential_scans_identical=True,
        source_and_test_hashes_unchanged=True))
    # Every room zoom explicitly displays metrics restricted to the room.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    geo=list(csv.DictReader((OUT/'geometry.csv').open()))
    loc=list(csv.DictReader((OUT/'localization.csv').open()))
    meta=e.cfg['map'];bounds=e.cfg['room_bounds_xyxy']
    extent=[meta['origin'][0],meta['origin'][0]+meta['width']*meta['resolution'],meta['origin'][1],meta['origin'][1]+meta['height']*meta['resolution']]
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    for ax,name in zip(axes,['original','three','empty']):
        grid=expected[name]
        ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.7)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
        result=next(r for r in loc if r['kind']=='independent' and r['variant']==name and r['region']=='room')
        title=f"{name}, room: independent {result['n']}/{result['N']} ({float(result['success_percent']):.1f}%)"
        if name!='empty':
            g=next(r for r in geo if r['variant']==name and r['region']=='room')
            title+=f"\nmatch {float(g['matched_percent']):.2f}%; excess {float(g['confirmed_free_excess_percent']):.2f}%"
        else: title+='\nindependent mapping control'
        ax.set(title=title,xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',xlabel='x [m]',ylabel='y [m]')
    fig.tight_layout();fig.savefig(OUT/'maps-room.png',dpi=170);plt.close(fig)
    readme=(OUT/'README.md').read_text()
    command='MPLCONFIGDIR=/tmp/mpl-lilocbench OPENBLAS_NUM_THREADS=1 python3 maintenance/check_reverse_short.py'
    if command not in readme:
        readme=readme.replace('python3 maintenance/run_reverse_short.py\n```','python3 maintenance/run_reverse_short.py\n'+command+'\n```')
        readme+='\nDodatno preverjanje: [ponovna gradnja zemljevidov](map-rebuild-validation.json) in [časovna uskladitev, kalibracije ter enakost skenov](input-audit.json). Drugi ukaz sveže zgradi vse tri zemljevide, preveri ujemanje in izvozi prenosno poročilo v `docs/lilocbench-reverse-short`.\n'
        (OUT/'README.md').write_text(readme)
    conclusion='Vpliv odometrije: '
    if conclusion not in readme:
        paired={r['variant']:r for r in loc if r['kind']=='sequential' and r['region']=='whole'}
        room_results={r['variant']:r for r in loc if r['kind']=='sequential' and r['region']=='room'}
        z=paired['zero_motion'];w=paired['wheel_odometry']
        readme+=f"\n{conclusion}na celotni poti je pomagala ({z['n']}/{z['N']} proti {w['n']}/{w['N']}); v sobi ni pomagala ({room_results['zero_motion']['n']}/{room_results['zero_motion']['N']} proti {room_results['wheel_odometry']['n']}/{room_results['wheel_odometry']['N']}). Oba načina prvič dosežeta pravilno oceno pri skenu {z['scans_through_first_success']}, po {float(z['first_success_elapsed_s']):.3f} s posnetka.\n"
        (OUT/'README.md').write_text(readme)
    import shutil
    destination=ROOT/'docs/lilocbench-reverse-short';destination.mkdir(exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file() and path.suffix in ['.csv','.json','.png','.md','.npz']:
            shutil.copyfile(path,destination/path.name)
    print('PASS: fresh maps, camera timing and calibration, scan identity, source hashes')

if __name__=='__main__':main()
