"""Read-only sampling of shared GPU; no process control."""
import argparse,subprocess,time,json
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('results/thud-vmesno/gpu-samples.jsonl'))
parser.add_argument('--finished',type=Path,default=Path('results/thud-vmesno/experiment-finished.json'))
args=parser.parse_args()
out=args.output
out.parent.mkdir(parents=True,exist_ok=True)
while True:
 p=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
 with out.open('a') as f:f.write(json.dumps({'sampled_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu':p.stdout.strip(),'exit_code':p.returncode,'error':p.stderr.strip()})+'\n')
 if args.finished.exists():break
 time.sleep(15)
