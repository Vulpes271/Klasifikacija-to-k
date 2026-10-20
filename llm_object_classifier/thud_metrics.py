"""Transparent descriptive statistics, retaining failed calls in denominators.

Name overlap is a vocabulary proxy only: set matching ignores instance counts,
boxes, synonyms, groups and unlabeled objects. It is never detection F1/accuracy.
"""
from __future__ import annotations
from collections import Counter
import math
import statistics
from .experiment import strict_json


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*q
    lo,hi=math.floor(index),math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def aggregate(rows):
    times=[r['elapsed_seconds'] for r in rows if isinstance(r.get('elapsed_seconds'),(int,float))]
    successes=[r for r in rows if r.get('status')=='ok']
    tokens=[r['response']['eval_count'] for r in rows if r.get('response',{}).get('eval_count') is not None]
    durations=[r['response']['eval_duration']/1e9 for r in rows if r.get('response',{}).get('eval_duration',0)>0]
    counts=Counter(r.get('status','unknown') for r in rows)
    parse_valid=0
    content_available=0
    for row in rows:
        if isinstance(row.get('content'),str):
            content_available+=1
            try:
                strict_json(row['content'])
                parse_valid+=1
            except (ValueError, TypeError):
                pass
    return {'attempted':len(rows),'ok':counts['ok'],'errors':counts['error'],'skipped':counts['skipped'],
            'schema_valid_fraction':len(successes)/len(rows) if rows else None,
            'json_text_available_calls':content_available,'json_parse_valid_calls':parse_valid,
            'json_parse_valid_fraction_all_attempts':parse_valid/len(rows) if rows else None,
            'timed_calls':len(times),'mean_s':statistics.mean(times) if times else None,
            'median_s':statistics.median(times) if times else None,'p90_s':percentile(times,.9),
            'output_tokens':sum(tokens),'mean_output_tokens':statistics.mean(tokens) if tokens else None,
            'generation_tokens_s':sum(tokens)/sum(durations) if durations and len(tokens)==len(durations) else None,
            'load_seconds':sum(r.get('response',{}).get('load_duration',0)/1e9 for r in rows),
            'gpu_verified_calls':sum(r.get('gpu_verified',False) for r in rows),
            'max_vram_bytes':max((r.get('gpu',{}).get('size_vram',0) for r in rows),default=0),
            'objects':sum(len(r['parsed']['objects']) for r in successes),
            'timeouts':sum('timed out' in r.get('error','').lower() for r in rows),
            'note':'All attempts in denominator. Latency includes every timed response, including invalid JSON; transport failures may lack timing.'}


def name_overlap(predicted, reference):
    normalize=lambda label: ' '.join(label.lower().strip().split())
    p={normalize(x) for x in predicted}
    r={normalize(x) for x in reference}
    return {'predicted_unique_names':len(p),'reference_unique_names':len(r),
            'shared_unique_names':len(p&r),'unmatched_predicted_names':sorted(p-r),
            'unmatched_reference_names':sorted(r-p),
            'note':'Exact lowercase normalized unique-name overlap; not precision/recall or confirmed detections.'}
