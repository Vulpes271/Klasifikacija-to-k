"""Read-only sampling of shared GPU; no process control."""
import subprocess,time,json
from pathlib import Path
out=Path('results/thud-vmesno/gpu-samples.jsonl')
while True:
 p=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
 with out.open('a') as f:f.write(json.dumps({'sampled_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu':p.stdout.strip(),'exit_code':p.returncode,'error':p.stderr.strip()})+'\n')
 if Path('results/thud-vmesno/experiment-finished.json').exists():break
 time.sleep(15)
