"""Overlap CPU-only original-map AMCL with missing GPU mapping inference.

The final paired experiment only adopts these runs after exact occupancy equality.
"""
import argparse
from pathlib import Path
import shutil

import numpy as np
import yaml

from .localization_pilot import ROOT,pose_table,transform_tree,write
from .localization_full import read
from .localization_filtered_map import ray_library,beam_world,occupancy,prepare_replays,run_pairs


def run(cfg,pilot):
    cache=ROOT/cfg['cache'];source=ROOT/cfg['data']/'mapping';arrays=dict(np.load(cache/'mapping/raw.npz'));meta=cfg['map']
    folder=cache/'original-baseline';folder.mkdir(exist_ok=True);grid_file=folder/'grid.npy'
    if not grid_file.exists():
        laser=read(cache/'mapping/raw.json')['laser'];gt_times,gt_poses,_=pose_table(source/'gt_poses.txt')
        scan=transform_tree(yaml.safe_load((source/'transformations.yaml').read_text()))('base_link','scan_front_link')
        log=np.zeros((meta['height'],meta['width']),np.float32);known=np.zeros(log.shape,np.uint8);fun=ray_library(cache)
        for i,t in enumerate(arrays['times']):
            origins,ends,valid=beam_world(arrays['ranges'][i],t,laser,scan,gt_times,gt_poses,pilot['pose_max_gap'])
            origins=np.floor((np.nan_to_num(origins)-np.array(meta['origin'][:2]))/meta['resolution']).astype(np.int32)
            ends=np.floor((np.nan_to_num(ends)-np.array(meta['origin'][:2]))/meta['resolution']).astype(np.int32)
            good=np.ascontiguousarray(valid,dtype=np.uint8)
            fun(log,known,meta['width'],meta['height'],origins,ends,good,len(good),meta['log_hit'],meta['log_free'],meta['log_limit'])
        np.save(grid_file,occupancy(log,known,meta))
    # Both temporary slots contain the same original grid. Only the original slot
    # executes; the filtered slot is never evaluated or adopted.
    work={**cfg,'output':str(folder)}
    for name in ['original','filtered']:
        dest=folder/'maps'/name;dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(grid_file,dest/'occupancy.npy')
    prepare_replays(work,pilot)
    # Create a completion marker only for temporary filtered slots so the unchanged
    # serial runner skips them. These markers are never in the main output.
    for seq in cfg['sequences']:
        for seed in cfg['seeds']:
            dest=folder/'runs'/seq/f'seed-{seed}'/'filtered';dest.mkdir(parents=True,exist_ok=True)
            write(dest/'poses.json',{'not_a_run':'temporary filtered slot skipped'})
    run_pairs(work,pilot)
    write(folder/'baseline-complete.json',{'original_only':True,'runs':len(cfg['seeds'])*len(cfg['sequences'])})


def adopt(cfg):
    from .box_segmentation import digest
    cache=ROOT/cfg['cache'];folder=cache/'original-baseline';output=ROOT/cfg['output']
    if not (folder/'baseline-complete.json').exists():return False
    if not np.array_equal(np.load(folder/'grid.npy'),np.load(output/'maps/original/occupancy.npy')):
        raise ValueError('CPU baseline map differs from final original map')
    adopted=0
    for seq in cfg['sequences']:
        for seed in cfg['seeds']:
            src_replay=folder/'replays'/seq/f'seed-{seed}'/'original';dest_replay=output/'replays'/seq/f'seed-{seed}'/'original'
            if digest(src_replay/'input.json')!=digest(dest_replay/'input.json') or digest(src_replay/'amcl.yaml')!=digest(dest_replay/'amcl.yaml'):
                raise ValueError('CPU baseline initialization/parameters differ')
            a=np.load(src_replay/'scans.npz');b=np.load(dest_replay/'scans.npz')
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k],equal_nan=True) for k in a.files):raise ValueError('CPU baseline replay differs')
            src=folder/'runs'/seq/f'seed-{seed}'/'original';dest=output/'runs'/seq/f'seed-{seed}'/'original'
            if not dest.exists():shutil.copytree(src,dest);adopted+=1
    write(output/'baseline-adoption.json',{'adopted_runs':adopted,'exact_grid_scans_odometry_initialization_parameters_verified':True})
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/lilocbench-filtered-map.json');args=p.parse_args()
    cfg=read(args.config);run(cfg,read(ROOT/cfg['pilot_config']))


if __name__=='__main__':main()
