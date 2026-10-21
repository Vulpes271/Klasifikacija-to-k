"""Read-only counts from saved part-two jobs, including schema failures/skips."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_object_classifier.thud_part2_report import collect

for job in collect():
    rows = job['records']
    if not rows:
        continue
    ok = sum(r['status']=='ok' for r in rows)
    failed = sum(r['status']=='error' for r in rows)
    skipped = sum(r['status']=='skipped' for r in rows)
    print(f'{job["name"]}: {len(rows)}/{job["expected"]}, OK {ok}, napake {failed}, preskoki {skipped}'
          + (' [ponovno uporabljeno]' if job['reused'] else '') + (' [končano]' if job['complete'] else ''))
