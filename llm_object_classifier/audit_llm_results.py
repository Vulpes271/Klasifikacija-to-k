"""Read-only revalidation of published results; writes only a separate revision folder."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path

from .experiment import read_json, sha256, options, combinations
from .llm_checks import validate_response, validity_counts, make_signature, check_signature, context_after

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results models/florence_grounding_llm_comparison10'
PROMPT=ROOT/'results models/prompt_comparison40'
FLORENCE=ROOT/'results models/florence_caption_grounding_test10'


def audit(destination):
    destination=Path(destination).resolve()
    if any(destination.is_relative_to(p) for p in (BASE,PROMPT,FLORENCE,ROOT/'reviews')):
        raise ValueError('Audit destination must be separate from source results and human reviews.')
    sources={}
    def data(path):
        b=path.read_bytes();sources[str(path.relative_to(ROOT))]=sha256(b);return b
    def read(path):return json.loads(data(path))
    base=read(BASE/'experiment200-manifest.json');comparison=read(PROMPT/'manifest.json')
    records=[];summaries={}
    for name,folder,manifest,role in [('baseline200',BASE,base,'base'),
        ('prompt40_new',PROMPT,comparison,'new'),('prompt40_reused_old',PROMPT,comparison,'old')]:
        checks=[];context_counts=Counter();signature_counts=Counter()
        # Offline historical consistency only: no claim about currently installed weights.
        reference_digests={}
        for spec in manifest['config']['models']:
            first=base['inputs'][0]['frame_id']
            reference_digests[spec['model']]=read(BASE/'responses'/spec['name']/(first+'.json'))['gpu']['digest']
        system=manifest['config']['system_file_text'] if role!='new' else data(folder/'prompts/new-system.txt').decode()
        for frame in manifest['inputs']:
            frozen=folder/frame['frozen'];document=read(frozen);digest=sha256(frozen.read_bytes())
            if digest!=frame.get('input_sha256',frame.get('sha256')):raise ValueError('Changed frozen input: '+str(frozen))
            originals=list((FLORENCE/frame['frame_id']).glob('original.*'))
            if len(originals)!=1 or sha256(data(originals[0]))!=document['image_sha256']:
                raise ValueError('Source image SHA mismatch: '+frame['frame_id'])
            user=frame['user_prompt'] if role=='base' else data(folder/frame['user_prompt']).decode()
            for spec in manifest['config']['models']:
                seeds=[42,43,44] if role=='new' and frame['repeatability'] else [42]
                for seed in seeds:
                    if role=='base':path=folder/'responses'/spec['name']/(frame['frame_id']+'.json')
                    elif role=='old':path=folder/'baseline'/spec['name']/(frame['frame_id']+'.json')
                    else:path=folder/'responses'/spec['name']/str(seed)/(frame['frame_id']+'.json')
                    row=read(path);check=validate_response(row,document['parsed']['objects']);checks.append(check)
                    opts=options(manifest['config'],next(combinations(manifest['config'])));opts['seed']=seed
                    signature=make_signature(document,digest,spec['model'],reference_digests[spec['model']],system,user,opts)
                    identity=check_signature(row,signature,document,digest);budget=context_after(row,opts)
                    state='verified_historical' if identity['verified'] else 'mismatch' if identity['mismatches'] else 'unverifiable'
                    signature_counts[state]+=1;context_counts[budget['status']]+=1
                    records.append({'experiment':name,'frame_id':frame['frame_id'],'model':spec['model'],'seed':seed,
                        'source':str(path.relative_to(ROOT)),'source_sha256':sources[str(path.relative_to(ROOT))],
                        'input_sha256':digest,'image_sha256':document['image_sha256'],
                        **{k:v for k,v in check.items() if k not in ('usable_items','errors')},
                        'validation_errors':'; '.join(check['errors']), 'signature_status':state,
                        'signature_missing':'; '.join(identity['missing_metadata']),
                        'signature_mismatches':'; '.join(identity['mismatches']), 'signature_notes':'; '.join(identity['notes']),
                        'context_status':budget['status'],'prompt_tokens':budget['prompt_eval_count'],
                        'output_tokens':budget['eval_count'],'reserved_output_tokens':opts['num_predict'],
                        'num_ctx':budget['num_ctx'],'actual_sequence_fits':budget['actual_sequence_fits']})
        summaries[name]={**validity_counts(checks),'context_budget':dict(context_counts),
            'historical_signatures':dict(signature_counts),'installed_weights_checked':False}
    for path,digest in sources.items():
        if sha256((ROOT/path).read_bytes())!=digest:raise ValueError('Source changed during audit: '+path)
    summary={'audit_version':1,'experiments':summaries,'source_files_unchanged':True,
        'source_file_count':len(sources),'new_inferences':0,'human_reviews_opened':False,
        'validator_sha256':sha256(Path(__file__).with_name('llm_checks.py').read_bytes()),
        'note':'Received API responses include valid and schema/finish-invalid responses. Reused old prompt40 calls are a subset of baseline200, not new generations. Context budget warnings are separate from response-schema validity.'}
    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=list(records[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(records)
    files={'summary.json':json.dumps(summary,ensure_ascii=False,indent=2),
        'responses.csv':output.getvalue(),'source-sha256.json':json.dumps(sources,ensure_ascii=False,indent=2),
        'README.md':'# Technical revision 1\n\nOffline audit only. Historical manifests, raw answers, Florence regions and human ratings are unchanged.\n\n`summary.json` separates received API responses, full schema-valid responses, API errors, schema/finish errors and skips. `responses.csv` retains per-call diagnostics and context-budget checks. `source-sha256.json` records source bytes. Reused prompt40 baseline responses are not additional generations.\n\nLegacy image identity is verified through the frozen input SHA and original image bytes. Model digests are compared with historical records; installed weights were not queried. Historical pre-call token budgets cannot be reconstructed as exact tokenizer counts. Post-call counts are reported without claiming that they prove no server-side truncation. No content accuracy is evaluated.\n'}
    destination.mkdir(parents=True,exist_ok=True)
    for name,text in files.items():
        path=destination/name
        if path.exists() and path.read_text()!=text:raise ValueError('Existing audit differs; choose a new destination: '+str(path))
        if not path.exists():path.write_text(text,encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results models/technical_revision_v1')
    args=parser.parse_args();print(json.dumps(audit(args.output_dir)['experiments'],ensure_ascii=False,indent=2))
