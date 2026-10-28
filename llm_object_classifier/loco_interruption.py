"""Record an evidenced cancelled LOCO call without rerunning or inventing a reply.

Use only with the runner stopped and a saved server log identifying the pending
position. Requests are reconstructed from immutable inputs/config, and labelled.
No network/model call is made; an existing result is never overwritten.
"""
import argparse
from datetime import datetime, timezone

from .experiment import combinations, options, read_json, sha256, write_json
from .loco_experiment import OUT, ROOT, LLM
from .vision_prompts import reason_prompt


def record(model, prompt_name, repeat, index, evidence_path):
    evidence=read_json(evidence_path)
    assert evidence['input_index']==index and evidence['model']==model
    assert evidence['prompt']==prompt_name and evidence['llm_repeat']==repeat
    assert evidence['server_http_status']==500 and evidence['server_cancel_observed']
    run=OUT/'llm/runs'/model/prompt_name/repeat
    manifest=read_json(run/'manifest.json');config=manifest['config']
    variant,=list(combinations(config));spec,=config['models']
    assert spec['name']==model
    destination=run/model/variant['id']/f'frame-{index:06d}.json'
    if destination.exists():raise ValueError('Existing result must not be overwritten')
    source=read_json(LLM/'frozen/index.json')[index-1]
    path=ROOT/source['path'];assert source['input_index']==index
    assert sha256(path.read_bytes())==source['sha256']==manifest['source_captions'][index-1]['sha256']
    saved=read_json(path);assert saved['status']=='ok'
    caption=saved['parsed'];objects=[{'id':o['id'],'label':o['label']} for o in caption['objects']]
    user=reason_prompt(saved['frame_id'],objects,caption,manifest['instructions'])
    override=read_json(LLM/'execution-context-policy.json')['overrides_by_frozen_sha256'].get(source['sha256'],{})
    opts=options(config,variant);opts.update(override)
    body={'model':spec['model'],'messages':[{'role':'system','content':manifest['system_prompt']},
        {'role':'user','content':user}],'stream':False,'think':False,'format':'json','keep_alive':'5m','options':opts}
    row={'stage':2,'frame_id':saved['frame_id'],'caption_source':str(path),'caption_sha256':source['sha256'],
        'model':spec['model'],'model_name':model,'variant':variant,'system_prompt':manifest['system_prompt'],
        'user_prompt':user,'syntax_valid':None,'schema_valid':None,'status':'error',
        'error':'Execution interrupted before the response was persisted; server cancelled this request (HTTP 500). No response reconstructed and no retry.',
        'request':body,'requested_options':opts,'unsupported_options':[],'input_option_overrides':override,
        'gpu_verified':False,'parallel_slots':1,'execution_url':manifest['model_metadata'][model]['execution_url'],
        'http_status':500,'request_reconstructed_after_interruption':True,
        'interruption_evidence_path':str(evidence_path.relative_to(ROOT)),
        'interruption_evidence_sha256':sha256(evidence_path.read_bytes()),
        'reconstructed_at_utc':datetime.now(timezone.utc).isoformat(),
        'client_elapsed_seconds_unrecoverable':True,
        'server_observed_http_seconds':evidence['server_observed_http_seconds'],
        'note':'The deterministic request was reconstructed from immutable files. No assistant content, API response, token counts, GPU allocation, client latency or creation identity is invented.'}
    write_json(destination,row)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--prompt',required=True)
    parser.add_argument('--repeat',required=True);parser.add_argument('--input-index',type=int,required=True)
    parser.add_argument('--evidence',type=lambda s:ROOT/s,required=True)
    args=parser.parse_args()
    print(record(args.model,args.prompt,args.repeat,args.input_index,args.evidence))
