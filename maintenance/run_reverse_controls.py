"""Run the three filter-independent cases while GPU masks are prepared.

These are canonical experiment inputs/outputs, subsequently reused by the full
runner. No provisional camera filter or substitute map is introduced.
"""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from llm_object_classifier.localization_reverse import Experiment, gap_plan
from llm_object_classifier.localization_filtered_map import immutable, aggregate
from llm_object_classifier.localization_full import read
from llm_object_classifier.localization_pilot import write, evaluate
from llm_object_classifier.localization_comparison import load_estimates
from llm_object_classifier.box_segmentation import digest


class Controls(Experiment):
    def progress(self,stage,**info):
        import time
        row={'stage':stage,'unix_time':time.time(),**info}
        write(self.output/'controls-progress.json',row);print(row,flush=True)

    def replays(self):
        if hasattr(self,'control_tasks'):return self.control_tasks
        cfg=self.cfg;tasks=[]
        cases=[('reverse-original',cfg['empty_sequence'],'original'),
            ('reverse-empty',cfg['empty_sequence'],'empty'),
            ('forward-original',cfg['full_sequence'],'empty')]
        for name,seq,map_name in cases:
            arrays=self.arrays(seq);info=read(self.raw/seq/'raw.json');ranges=arrays['ranges']
            usable,natural,artificial=gap_plan(arrays['times'],ranges,info['laser'],cfg['bridge'])
            transform=next(r for r in yaml.safe_load((self.data/seq/'transformations.yaml').read_text())
                if r['parent']=='base_link' and r['child']=='scan_front_link')
            for scenario,gap in [('natural',natural),('artificial',natural|artificial)]:
                damaged=ranges.copy()
                if scenario=='artificial':damaged[artificial]=np.nan
                shared=self.output/'replays'/name/scenario;shared.mkdir(parents=True,exist_ok=True)
                values=dict(times=arrays['times'],ranges=damaged,odom=arrays['odom'],odom_end=arrays['odom_end'],
                    removed=np.zeros(ranges.shape,bool),occupancy=np.load(self.output/'maps'/map_name/'occupancy.npy'),
                    skip_lidar=gap,usable_hits=usable,natural_gap=natural,
                    artificial_gap=artificial if scenario=='artificial' else np.zeros(len(natural),bool))
                if not (shared/'scans.npz').exists():np.savez_compressed(shared/'scans.npz',**values)
                else:
                    with np.load(shared/'scans.npz') as previous:
                        for key,value in values.items():
                            if not np.array_equal(previous[key],value,equal_nan=True):raise ValueError(f'Changed control input: {name}/{scenario}/{key}')
                for seed in cfg['seeds']:
                    for mode in ['standard','bridge']:
                        folder=shared/f'seed-{seed}'/mode;folder.mkdir(parents=True,exist_ok=True)
                        if not (folder/'scans.npz').exists():(folder/'scans.npz').symlink_to(os.path.relpath(shared/'scans.npz',folder))
                        immutable(folder/'input.json',{'laser':info['laser'],'map':cfg['map'],'scan_transform':transform,
                            'initial_pose':info['initial'],'seed':seed,'bridge':cfg['bridge'] if mode=='bridge' else None})
                        (folder/'amcl.yaml').write_text(yaml.safe_dump({'amcl':{'ros__parameters':self.pilot['amcl']}}))
                        dest=self.output/'runs'/name/scenario/f'seed-{seed}'/mode
                        tasks.append((folder,dest,name,scenario,seed,mode,seq))
        immutable(self.output/'controls-protocol.json',{'source_sha256':digest(__file__),
            'parent_protocol_sha256':digest(self.output/'protocol.json'),
            'inputs':{str(folder.relative_to(self.output)):digest(folder/'input.json') for folder,*_ in tasks},
            'cases':[c[0] for c in cases],'expected_runs':len(tasks)})
        self.control_tasks=tasks;return tasks

    def control_report(self):
        metrics={};memo={}
        for folder,dest,name,scenario,seed,mode,seq in self.replays():
            if seq not in memo:memo[seq]=self.arrays(seq)
            a=memo[seq];est=load_estimates(dest/'poses.json',a['times'])
            result,_,_,_=evaluate(a['gt'],est,a['has_gt'],self.data/'reference-code/evaluate.py')
            metrics.setdefault(f'{name}/{scenario}/{mode}',[]).append(result)
        write(self.output/'controls-complete.json',{'complete':True,'scope':'Three filter-independent cases only',
            'runs':metrics,'aggregate':{k:aggregate(v) for k,v in metrics.items()}})


if __name__=='__main__':
    exp=Controls(ROOT/'configs/lilocbench-reverse.json');exp.replays();exp.amcl();exp.control_report()
