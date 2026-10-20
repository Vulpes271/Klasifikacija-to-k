"""Audit saved experiment invariants without making any model requests.

Checks the full planned matrix, image identity, annotation isolation, strict JSON,
GPU allocation, and frozen text-only stage2 inputs. Failures are retained, never
removed from denominators. --require-complete rejects unfinished matrices.
"""
import argparse
import json
from pathlib import Path
from .experiment import read_json, sha256, strict_json, write_json
from .thud_experiment import OUT, REPORT, DATASETS, MODELS
from .description_prompts import validate_description


def verify(require_complete=False):
    errors=[]
    missing=[]
    counts={'baseline':0,'extra':0,'reasoning':0,'gpu_verified':0,'schema_ok':0,'schema_error':0,'skipped':0}
    expected=[]
    for ds in DATASETS:
        for spec in MODELS:
            for mode in ('discovery','supplied'):
                expected.append((f'baseline-{spec["name"]}-{mode}-{ds}',50,'baseline',ds,mode))
        for name in ('short-t0','long-t03','long-t07','long-t03-s43'):
            expected.append((f'extra-{name}-{ds}',10,'extra',ds,'discovery'))
        for prompt,t,seed in [('long-examples',.3,42),('long-no-examples',.3,42),('short-no-examples',.3,42),('long-no-examples',0,42),('long-no-examples',.7,42),('long-no-examples',.3,43)]:
            expected.append((f'reason-{prompt}-t{t}-s{seed}-{ds}',10,'reasoning',ds,None))
    hashes={}
    frozen_vectors={}
    for name,expected_count,stage,ds,mode in expected:
        path=OUT/name
        if not (path/'complete.json').exists():
            missing.append(name)
            continue
        if (path/'resumed-attempt.json').exists():
            path=Path(read_json(path/'resumed-attempt.json')['path'])
        records=[read_json(p) for p in sorted(path.glob('*/t*/frame-*.json'))]
        if len(records)!=expected_count:
            errors.append(f'{name}: expected {expected_count}, got {len(records)}')
        manifest=read_json(path/'manifest.json')
        if stage!='reasoning':
            digests=[r['image_sha256'] for r in records]
            hashes.setdefault((ds,mode,stage),[]).append(digests)
        else:
            vector=[entry['sha256'] for entry in manifest['source_captions']]
            frozen_vectors.setdefault(ds,[]).append(vector)
            for entry in manifest['source_captions']:
                if sha256(Path(entry['path']).read_bytes()) != entry['sha256']:
                    errors.append(f'{name}: frozen caption changed')
        for row in records:
            counts[stage]+=1
            status=row.get('status')
            counts['schema_ok' if status=='ok' else ('skipped' if status=='skipped' else 'schema_error')]+=1
            if status=='skipped':continue
            if 'request' in row:
                request=row['request']
                if stage=='reasoning':
                    if any('images' in message or isinstance(message.get('content'),list) for message in request['messages']):
                        errors.append(f'{name}: image in stage2 request')
                elif mode=='discovery':
                    payload=strict_json(request['messages'][-1]['content'].split('\n',1)[1])
                    if payload['supplied_objects'] is not None or set(payload)!={'frame_id','image_size','annotation_kind','supplied_objects'}:
                        errors.append(f'{name}: annotation leakage')
                    if set(row['source_frame'])!={'frame_id','image_path','image_sha256','image_size'}:
                        errors.append(f'{name}: extra discovery source metadata')
            if row.get('gpu_verified'):
                counts['gpu_verified']+=1
                gpu=row['gpu']
                if gpu['size_vram']<=0 or gpu['size_vram']<gpu.get('size',0):
                    errors.append(f'{name}: partial/CPU allocation')
            elif 'response' in row:
                errors.append(f'{name}: received response without GPU verification')
            if status=='ok':
                parsed=strict_json(row['content'])
                if parsed!=row['parsed']:
                    errors.append(f'{name}: parsed output differs from original JSON')
                if stage!='reasoning':
                    validate_description(parsed)
    for key,vectors in hashes.items():
        if any(vector!=vectors[0] for vector in vectors):
            errors.append(f'{key}: image SHA vectors differ across models/variants')
    for ds in DATASETS:
        discovery=hashes.get((ds,'discovery','baseline'))
        supplied=hashes.get((ds,'supplied','baseline'))
        if discovery and supplied and discovery[0]!=supplied[0]:
            errors.append(f'{ds}: discovery/supplied image bytes differ')
        extra=hashes.get((ds,'discovery','extra'))
        if discovery and extra:
            indices=read_json(REPORT/'data-audit.json')[ds]['subset_indices_zero_based']
            if [discovery[0][i] for i in indices]!=extra[0]:
                errors.append(f'{ds}: extra selection differs from frozen subset')
    for ds,vectors in frozen_vectors.items():
        if any(vector!=vectors[0] for vector in vectors):
            errors.append(f'{ds}: stage2 source SHA vectors differ')
    result={'counts':counts,'complete_jobs':len(expected)-len(missing),'expected_jobs':len(expected),
            'invariant_errors':errors,'missing_jobs':missing,'complete':not missing,
            'passed':not errors and (not require_complete or not missing),
            'note':'Schema errors are measured outcomes, not audit failures. No accuracy reference is invented.'}
    write_json(REPORT/'verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    result=verify(args.require_complete)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
