"""Keep a small, truthful status artifact alongside a long local experiment."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/lilocbench-reverse'
CONFIG=json.loads((ROOT/'configs/lilocbench-reverse.json').read_text())
CACHE=ROOT/CONFIG['cache']
LAST_READ={}


def read(path):
    # progress.json is written by the existing runner without an atomic rename.
    # A transient partial read must not kill the observer during a long run.
    try:value=json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return LAST_READ.get(path,{})
    LAST_READ[path]=value
    return value


def model_progress(progress):
    audit=read(OUTPUT/'sequence-audit.json').get(CONFIG['full_sequence'],{})
    frames=math.ceil((audit.get('end',0)-audit.get('start',0))*CONFIG['rgb_hz'])
    frames+=math.ceil((CONFIG['preflight_section'][1]-CONFIG['preflight_section'][0])*CONFIG['rgb_hz'])
    batches=math.ceil(frames/CONFIG['batch_size'])
    lines=[];finished=0
    for camera in ['camera_front','camera_left','camera_right']:
        saved=read(CACHE/f'{camera}-status-full.json')
        done=batches if isinstance(saved,list) else 0
        if not done and progress.get('stage')=='models' and not progress.get('preflight') and progress.get('camera')==camera:
            done=progress.get('batch',0)
        finished+=done;lines.append(f'- {camera}: {done}/{batches} paketkov.')
    total=3*batches
    heading=f'Priprava mask: **{finished}/{total} paketkov ({100*finished/total:.1f} %)**.' if total else 'Priprava mask: načrt še ni na voljo.'
    previous={};durations=[]
    logfile=OUTPUT/'run.log'
    if logfile.exists():
        for line in logfile.read_text().splitlines():
            try:row=json.loads(line)
            except (ValueError,TypeError):continue
            if not isinstance(row,dict) or row.get('stage')!='models' or row.get('preflight'):continue
            camera=row.get('camera');before=previous.get(camera)
            if before and row.get('batch',0)==before.get('batch',0)+1:
                dt=row['unix_time']-before['unix_time']
                if 5<dt<3600:durations.append(dt)
            previous[camera]=row
    notes=[]
    if durations and finished<total:
        typical=statistics.median(durations[-15:]);hours=(total-finished)*typical/3600
        notes=[f'Zadnji paketki: približno {typical/60:.1f} min/paketek. Linearna ocena preostanka te faze: **{hours:.1f} h**.',
            'To je orientacijska ocena samo za modele; vsebina slik in razpoložljivost predpomnilnika vplivata na trajanje. Nato sledijo zemljevidi, primerjava točk in preostali AMCL.']
    return [heading,'',*lines,'',*notes]


while True:
    job=read(OUTPUT/'job.json');progress=read(OUTPUT/'progress.json');controls=read(OUTPUT/'controls-progress.json')
    completed=len(list((OUTPUT/'runs').glob('*/*/seed-*/*/poses.json')))
    state={'running':'Izračun teče','complete':'Celotni izračun je zaključen','failed':'Izračun je ustavljen zaradi napake'}.get(job.get('status'),'Stanje še ni na voljo')
    lines=['# '+state,'',datetime.now(ZoneInfo('Europe/Ljubljana')).isoformat(timespec='seconds'),'',
        f'AMCL: **{completed}/280** zaključenih zagonov (vključno s preverjenimi rezultati iz predpomnilnika).',
        f'Stopnja: **{progress.get("stage", "—")}**.']
    if progress.get('stage')=='models':
        lines+=['','Števec AMCL med pripravo mask miruje. 120 kontrol je končanih; filtrirani primeri čakajo na maske vseh treh kamer.','',*model_progress(progress)]
        lines+=['',f'Trenutna kamera: {progress.get("camera")}; {progress.get("frames")} obdelanih slik.']
    if job.get('status')=='running':
        try:os.kill(job['pid'],0);alive=True
        except PermissionError:alive=True
        except ProcessLookupError:alive=False
        lines+=['',f'Proces nadzornika: {"aktiven" if alive else "NI AKTIVEN; zapis running je zastarel"}.']
    lines+=['',f'Nadzornik: poskus {job.get("attempt",0)}, okrevanj po sesutju SAM: {len(job.get("recoveries",[]))}.']
    if job.get('status')=='failed':lines+=['',str(job.get('reason',''))]
    lines+=['','[Poročilo](README.md) · [Celotni dnevnik](run.log) · [Strojno berljivo stanje](job.json) · [Stanje kontrol](controls-progress.json)']
    target=OUTPUT/'STATUS.md';temporary=target.with_suffix('.tmp');temporary.write_text('\n'.join(lines)+'\n');temporary.replace(target)
    if job.get('status') in ['complete','failed']:break
    time.sleep(15)
