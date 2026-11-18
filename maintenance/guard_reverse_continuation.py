"""Resume remaining work after narrow transport/native-Florence failures.

The primary supervisor handles native SAM crashes. This outer observer never
removes or changes a received model result, and never bypasses signature/model
or protocol failures. It preserves response hashes across every resumption.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/lilocbench-reverse'
CACHE=ROOT/'data/lilocbench-reverse'
GPU_LOAD_FAILURE='Full GPU/digest/context verification failed; CPU fallback forbidden'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def recoverable(message):
    if any(word in message.lower() for word in ['digest changed','signature','immutable protocol','three retries exhausted']):return False
    if 'thud-florence:local' in message and re.search(r'exit status (134|139)',message):return True
    return bool(re.search(r'temporary failure in name resolution|connection reset|connection refused|remote end closed|(?:urlopen|socket|request|read|connect)[^\n]{0,100}timed?\s*out|(?:HTTP Error|HTTPError)[ :]+(?:429|500|502|503|504)',message,re.I))


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def lost_container_gpu(message,container):
    """Only the observed NVML device-loss failure permits a service restart."""
    if GPU_LOAD_FAILURE not in message:return None
    result=subprocess.run(['docker','exec',container,'nvidia-smi'],capture_output=True,text=True,timeout=30)
    diagnostic=result.stdout+result.stderr
    if result.returncode and 'Failed to initialize NVML: Unknown Error' in diagnostic:return diagnostic
    return None


def restore_container_gpu(container):
    """Restore device access, then run the ORIGINAL full-GPU/digest/context gate."""
    sys.path.insert(0,str(ROOT))
    from llm_object_classifier.pipeline import load_settings
    from llm_object_classifier.experiment import options
    from llm_object_classifier.vision_backends import Backend,request
    from llm_object_classifier.ollama_runtime import load_model
    from llm_object_classifier.localization_filtered_map import EXPECTED_QWEN
    cfg=load_settings(ROOT/'configs/pipeline.json')['llm']
    if container!=cfg['container']:raise ValueError('Unexpected GPU service')
    subprocess.run(['docker','restart',container],check=True)
    probe=subprocess.run(['docker','exec',container,'nvidia-smi'],capture_output=True,text=True,timeout=30,check=True)
    ip=subprocess.check_output(['docker','inspect',container,'--format','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'],text=True).strip()
    backend=Backend({'backend':'ollama','model':cfg['alternatives'][cfg['selected']],
        'base_url':f'http://{ip}:11434','precision':'Q4_K_M'},cfg['timeout'])
    deadline=time.monotonic()+30
    while True:
        try:request(backend.url+'/api/version',timeout=2);break
        except Exception:
            if time.monotonic()>=deadline:raise
            time.sleep(.5)
    opts=options(cfg,{'temperature':cfg['temperature'],'seed':cfg['seed']})
    verified=load_model(backend,opts,EXPECTED_QWEN,container)
    return {'nvidia_smi':probe.stdout,'original_load_check':verified,'new_classifications_generated':0}


def metadata_order_equivalent(before,after):
    """Ollama serializes default parameter maps in an unstable textual order.

    All other metadata, including digest, template, model info and server
    version, must match exactly. Repeated parameter keys are not normalized.
    """
    copies=[json.loads(json.dumps(value)) for value in [before,after]]
    for value in copies:
        show=value.get('show',{});text=show.get('parameters')
        if isinstance(text,str):
            pairs=[line.split(maxsplit=1) for line in text.splitlines() if line.strip()]
            if any(len(pair)!=2 for pair in pairs):return False
            if len({pair[0] for pair in pairs})!=len(pairs):return before==after
            show['parameters']=sorted(pairs)
        text=show.get('modelfile')
        if isinstance(text,str):
            lines=text.splitlines();parameters=[line for line in lines if line.startswith('PARAMETER ')]
            show['modelfile']={'other_lines':[line for line in lines if not line.startswith('PARAMETER ')],
                'parameter_lines':sorted(parameters)}
    return copies[0]==copies[1]


def archive_unstarted_metadata(inference,container,recovery):
    """Regenerate diagnostic metadata only when no text result exists yet."""
    if inference is None:return
    path=inference/'model-metadata.json'
    if not path.exists():return
    if list((inference/'responses').glob('*/*.json')):return
    from llm_object_classifier.pipeline import load_settings
    from llm_object_classifier.vision_backends import Backend
    from llm_object_classifier.llm_checks import installed_digest
    cfg=load_settings(ROOT/'configs/pipeline.json')['llm']
    ip=subprocess.check_output(['docker','inspect',container,'--format','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'],text=True).strip()
    backend=Backend({'backend':'ollama','model':cfg['alternatives'][cfg['selected']],
        'base_url':f'http://{ip}:11434','precision':'Q4_K_M'},cfg['timeout'])
    current={'digest':installed_digest(backend),**backend.metadata()};previous=json.loads(path.read_text())
    if not metadata_order_equivalent(previous,current):raise ValueError('Model metadata changed semantically; refused regeneration')
    archive=OUTPUT/'runtime-recoveries'/f'metadata-order-{time.time_ns()}.json'
    path.rename(archive);recovery['diagnostic_metadata_archive']=str(archive)
    recovery['diagnostic_metadata_policy']='Only PARAMETER ordering differs; zero received responses. Original model/response signatures remain enforced.'


def main():
    state={'status':'watching','pid':os.getpid(),'resumptions':[]};counts={};frozen={}
    write(OUTPUT/'continuation-guard.json',state)
    while True:
        job=json.loads((OUTPUT/'job.json').read_text())
        if job['status']=='complete':
            for path,value in frozen.items():
                if sha(Path(path))!=value:raise ValueError('Received result changed: '+path)
            state['status']='complete';write(OUTPUT/'continuation-guard.json',state);return 0
        if job['status']=='running':time.sleep(10);continue
        started=datetime.fromisoformat(job['started_utc']).timestamp()
        error=json.loads((OUTPUT/'error.json').read_text()) if (OUTPUT/'error.json').exists() else {}
        message=error.get('error','')+'\n'+error.get('traceback','');inference=None
        errors=[p for p in (CACHE/'models').glob('**/work/inference/errors/*.json') if p.stat().st_mtime>=started]
        if errors:
            latest=max(errors,key=lambda p:p.stat().st_mtime);extra=json.loads(latest.read_text())
            message+='\n'+extra.get('error','')+'\n'+extra.get('traceback','');inference=latest.parent.parent
        nvml=None;container=None
        if GPU_LOAD_FAILURE in message:
            container=json.loads((ROOT/'configs/pipeline.json').read_text())['llm']['container']
            nvml=lost_container_gpu(message,container)
        if not recoverable(message) and nvml is None:
            state.update(status='failed',reason=job.get('reason',message));write(OUTPUT/'continuation-guard.json',state);return 1
        for path,value in json.loads((OUTPUT/'source-hashes.json').read_text()).items():
            if sha(ROOT/path)!=value:raise ValueError('Frozen experiment changed: '+path)
        progress=json.loads((OUTPUT/'progress.json').read_text());key=f'{progress.get("camera")}/{progress.get("batch")}'
        counts[key]=counts.get(key,0)+1
        if counts[key]>3:
            state.update(status='failed',reason='Transport/Florence recovery limit exceeded at '+key)
            write(OUTPUT/'continuation-guard.json',state);return 1
        if inference:
            for path in (inference/'responses').glob('*/*.json'):
                if '_technical_checks' not in str(path):frozen.setdefault(str(path),sha(path))
        recovery={'previous_job':job,'key':key,'reason':message,'retained_response_sha256':dict(frozen)}
        if nvml is not None:
            recovery['recovery_type']='verified_nvml_container_restart';recovery['nvml_before']=nvml
            try:
                recovery['gpu_after']=restore_container_gpu(container)
                archive_unstarted_metadata(inference,container,recovery)
            except Exception as exc:
                state.update(status='failed',reason='GPU access/model verification still fails: '+str(exc))
                write(OUTPUT/'continuation-guard.json',state);return 1
        state['resumptions'].append(recovery);write(OUTPUT/'continuation-guard.json',state)
        # Let the old process and its status/report observers exit before reuse.
        time.sleep(20)
        with (OUTPUT/'supervisor.log').open('a') as stream:
            subprocess.Popen([sys.executable,'-u',str(ROOT/'maintenance/run_reverse_experiment.py')],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            updated=json.loads((OUTPUT/'job.json').read_text())
            if updated['started_utc']!=job['started_utc'] and updated['status']=='running':break
            time.sleep(.5)
        else:raise RuntimeError('Resumption did not acquire the experiment lock')
        for script,log,wait in [('watch_reverse_status.py','status-watcher.log',False),
            ('finalize_reverse_report.py','finalize.log',True),('plot_reverse_paths.py','path-plots.log',True)]:
            command=[sys.executable,'-u',str(ROOT/'maintenance'/script)]+(['--wait'] if wait else [])
            with (OUTPUT/log).open('a') as stream:
                subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'MPLCONFIGDIR':'/tmp/mpl-lilocbench'})


if __name__=='__main__':sys.exit(main())
