"""Small read-only heartbeat for a long stage-two experiment; no model calls."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/stage2-presentation'


def status():
    groups = Counter()
    states = Counter()
    calls = 0
    for path in (OUT / 'runs').glob('*/*/*/*/*/frame-*.json'):
        row = json.loads(path.read_text())  # Runner writes JSON by atomic rename.
        groups['/'.join(path.relative_to(OUT / 'runs').parts[:3])] += 1
        states[row['status']] += 1
        calls += 'request' in row
    diagnostics = len(list((OUT / 'diagnostics/context-16384').glob('*/*/*/*/*/frame-*.json')))
    running = subprocess.run(['pgrep', '-f', '^python3 -m llm_object_classifier.thud_stage2_presentation run$'],
                             capture_output=True).returncode == 0
    return {'at_utc': datetime.now(timezone.utc).isoformat(), 'positions': sum(groups.values()),
            'main_calls': calls, 'diagnostic_calls': diagnostics, 'physical_calls': calls + diagnostics,
            'statuses': dict(states), 'groups': dict(groups),
            'completed_groups': sum(n == 180 for n in groups.values()), 'runner_alive': running}


if __name__ == '__main__':
    while True:
        row = status()
        with (OUT / 'progress-counts.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        (OUT / 'live-status.json').write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)
        if not row['runner_alive'] or row['positions'] == 3240:
            break
        time.sleep(45)
