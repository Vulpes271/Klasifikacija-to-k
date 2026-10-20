"""Print a compact read-only progress snapshot for a long GPU run."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results/thud-vmesno'
parts=[]
for prefix,label,expected in [('baseline-','osnovni',600),('extra-','dodatni opisi',80),('reason-','reasoning',120)]:
    records=[p for directory in root.glob(prefix+'*') if '-attempt' not in directory.name
             for p in directory.glob('*/t*/frame-*.json')]
    errors=0
    for p in records:
        try:
            errors+=json.loads(p.read_text()).get('status')!='ok'
        except (ValueError,OSError):
            pass
    parts.append(f'{label} {len(records)}/{expected} (napake/preskoki {errors})')
log=root/'all.log'
if log.exists():
    lines=log.read_text().splitlines()
    current=next((line[6:] for line in reversed(lines) if line.startswith('Zagon ')), '--')
else:current='--'
finished=(root/'experiment-finished.json').exists()
print('GPU poskus: '+'; '.join(parts)+'. Trenutna skupina: '+current+('. GPU_DONE' if finished else '.'))
