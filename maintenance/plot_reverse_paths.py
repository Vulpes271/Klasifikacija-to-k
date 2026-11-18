"""Render complete trajectories with common limits; never crop a failed excursion."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from llm_object_classifier.localization_pilot import evaluate, write
from llm_object_classifier.localization_comparison import load_estimates
from llm_object_classifier.box_segmentation import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/lilocbench-reverse.json').read_text());output=ROOT/cfg['output'];raw=ROOT/cfg['raw_cache']
    if args.wait:
        while not (output/'report-provenance.json').exists():
            job=json.loads((output/'job.json').read_text())
            if job['status']=='failed':raise RuntimeError('Experiment failed; no complete figures to render')
            time.sleep(10)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seed=cfg['seeds'][0];data={};items=[];limits={}
    for sequence in [cfg['empty_sequence'],cfg['full_sequence']]:
        a=dict(np.load(raw/sequence/'raw.npz'));data[sequence]=a
        limits[sequence]=[a['gt'][a['has_gt'],:2]]
    for path in sorted((output/'runs').glob(f'*/*/seed-{seed}/*/poses.json')):
        case,scenario,_,mode=path.relative_to(output/'runs').parts[:4]
        sequence=cfg['empty_sequence'] if case.startswith('reverse-') else cfg['full_sequence'];a=data[sequence]
        estimate=load_estimates(path,a['times']);finite=np.isfinite(estimate).all(axis=1)
        limits[sequence].append(estimate[finite,:2]);items.append((path,case,scenario,mode,sequence,estimate))
    bounds={}
    for sequence,points in limits.items():
        xy=np.concatenate(points);low=xy.min(axis=0)-.5;high=xy.max(axis=0)+.5
        bounds[sequence]=[float(low[0]),float(low[1]),float(high[0]),float(high[1])]
    for path,case,scenario,mode,sequence,estimate in items:
        a=data[sequence];rows=json.loads(path.read_text());bridge=np.array([r.get('bridged',False) for r in rows])
        _,_,_,success=evaluate(a['gt'],estimate,a['has_gt'],ROOT/cfg['data']/'reference-code/evaluate.py')
        name=case.removeprefix('reverse-') if case.startswith('reverse-') else 'empty'
        grid=np.load(output/'maps'/name/'occupancy.npy');meta=cfg['map']
        extent=[meta['origin'][0],meta['origin'][0]+meta['width']*meta['resolution'],meta['origin'][1],meta['origin'][1]+meta['height']*meta['resolution']]
        fig,axes=plt.subplots(1,2,figsize=(16,7),gridspec_kw={'width_ratios':[1.3,1]})
        for ax,crop in zip(axes,[bounds[sequence],cfg['room_bounds_xyxy']]):
            ax.set_facecolor('.7');ax.imshow(np.where(grid==100,0,np.where(grid==0,1,.7)),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
            ax.plot(a['gt'][:,0],a['gt'][:,1],c='green',lw=.8,label='GT')
            ax.plot(estimate[:,0],estimate[:,1],c='royalblue',lw=.65,label='AMCL')
            failed=a['has_gt']&~success;ax.scatter(a['gt'][failed,0],a['gt'][failed,1],c='red',s=4,label='failure (<10cm AND <4deg)')
            ax.scatter(estimate[bridge,0],estimate[bridge,1],c='darkorange',s=12,label='odometry bridge',zorder=5)
            ax.set(xlim=(crop[0],crop[2]),ylim=(crop[1],crop[3]),aspect='equal',xlabel='x [m]',ylabel='y [m]');ax.legend(fontsize=7)
        fig.suptitle(f'{case} / {scenario} / {mode}, seed {seed}\nShared full-trajectory limits for all variants of this query sequence; room detail at right.')
        fig.tight_layout();fig.savefig(output/f'trajectory-{case}-{scenario}-{mode}.png',dpi=160);plt.close(fig)
    write(output/'path-plot-provenance.json',{'source_sha256':digest(__file__),'seed':seed,
        'rendered_plots':len(items),'shared_full_bounds_by_query_sequence':bounds,
        'policy':'Common extent includes every finite estimate and every GT pose for all rendered cases of the same query sequence.'})


if __name__=='__main__':main()
