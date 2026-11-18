"""Supervise the frozen reverse experiment; recover only native SAM crashes.

No received LLM outcome is retried. Complete masks and AMCL runs are reused by
the experiment. Incomplete SAM outputs are archived before a bounded retry.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/lilocbench-reverse.json')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    output=ROOT/cfg['output'];cache=ROOT/cfg['cache'];output.mkdir(parents=True,exist_ok=True)
    # Prevent two resumptions from publishing to the same cache/run directories.
    import fcntl
    lock=(output/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate()
    lock.write(str(os.getpid()));lock.flush()
    def finish(code):
        lock.close()
        return code
    state={'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'status':'running','attempt':0,'recoveries':[],'configuration':str(args.config.resolve())}
    counters=Counter();ledger=output/'runtime-recoveries';ledger.mkdir(exist_ok=True)
    old_error=output/'error.json'
    if old_error.exists():shutil.copyfile(old_error,ledger/'previous-error.json')
    while True:
        state['attempt']+=1;write(output/'job.json',state)
        command=[sys.executable,'-u','-m','llm_object_classifier.localization_reverse',
            '--config',str(args.config.resolve()),'--stage','all']
        with (output/'run.log').open('a') as stream:
            stream.write(f'\nSupervisor attempt {state["attempt"]}, {datetime.now(timezone.utc).isoformat()}\n');stream.flush()
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'MPLCONFIGDIR':'/tmp/mpl-lilocbench'})
        if result.returncode==0:
            metrics=json.loads((output/'metrics.json').read_text())
            expected=7*2*2*len(cfg['seeds'])
            count=sum(len(v) for v in metrics['runs'].values())
            if not metrics['complete'] or count!=expected:
                state.update(status='failed',reason=f'Incomplete report: {count}/{expected}')
            else:
                state.update(status='complete',completed_amcl_runs=count,
                    finished_utc=datetime.now(timezone.utc).isoformat())
                if old_error.exists():old_error.rename(ledger/f'last-recovered-error-{state["attempt"]}.json')
                validation=output/'validation.json'
                value=json.loads(validation.read_text()) if validation.exists() else {}
                value.update(full_sequence_complete=True,completed_amcl_runs=count)
                write(validation,value)
            write(output/'job.json',state)
            return finish(0 if state['status']=='complete' else 1)
        error=json.loads(old_error.read_text()) if old_error.exists() else {}
        message=error.get('error','')
        # Only native, result-less SAM crashes qualify. Semantic/model/schema
        # failures are terminal outcomes and are never hidden by a retry loop.
        recoverable=('klasifikacija-sam2:local' in message and
            ('exit status 139' in message or 'exit status 134' in message))
        if not recoverable:
            state.update(status='failed',returncode=result.returncode,reason=message)
            write(output/'job.json',state);return finish(result.returncode)
        progress=json.loads((output/'progress.json').read_text())
        key=f'{progress.get("camera")}/{progress.get("preflight")}/{progress.get("batch")}'
        counters[key]+=1
        if counters[key]>3:
            state.update(status='failed',returncode=result.returncode,reason='Three retries exhausted: '+message)
            write(output/'job.json',state);return finish(result.returncode)
        recovery={'attempt':state['attempt'],'key':key,'unix_time':time.time(),'archived':[],'error':message}
        shutil.copyfile(old_error,ledger/f'error-{state["attempt"]}.json')
        for masks in (cache/'models').glob('**/work/masks'):
            if not (masks/'segmentation.json').exists():
                relative=masks.relative_to(cache)
                target=ledger/f'incomplete-sam-{state["attempt"]}'/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(masks),target)
                recovery['archived'].append(str(relative))
        state['recoveries'].append(recovery);write(output/'job.json',state)
        # This pause is short and the orchestration remains observable.
        time.sleep(2)


if __name__=='__main__':sys.exit(main())
