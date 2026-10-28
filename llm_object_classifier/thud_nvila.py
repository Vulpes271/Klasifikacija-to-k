"""NVILA-8B native VILA inference; avoids the obsolete API model allow-list.

Uses the exact part-two rich prompt and original RGB bytes, with no references.
FP16 reflects VILA's actual loader and image tensors (not a BF16 claim).
"""
import argparse
import copy
import time
import traceback
from pathlib import Path
import uuid

from .description_prompts import user_prompt, validate_description
from .experiment import read_json, strict_json, write_json, sha256
from .thud_extended_common import (OUT, ROOT, prepare, snapshot, environment, progress,
    begin_measure, end_measure, saved_path, check_saved, native_selection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--system-file', type=Path, default=ROOT/'configs/prompts/thud-description-rich.txt')
    parser.add_argument('--schema', choices=['legacy','rich-v2','bbox'], default='rich-v2')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat', default='original')
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    parser.add_argument('--max-new-calls', type=int)
    parser.add_argument('--no-warmup', action='store_true')
    args = parser.parse_args()
    import torch
    import llava
    selected = native_selection(args.selection)
    destination = args.output_dir or OUT / 'nvila'
    system = args.system_file.read_text()
    if args.selection is not None: system=system.strip()
    config = {'model': 'Efficient-Large-Model/NVILA-8B',
              'revision': read_json(OUT / 'setup/nvila-api.txt')['sha'],
              'vila_commit': '57ecbfb904b99b563db69ef7916f6390eecea516',
              'system_prompt': system, 'system_sha256': sha256(system.encode()),
              'precision_requested': 'FP16', 'attention': 'sdpa', 'seed': args.seed,
              'do_sample': args.temperature>0, 'temperature':args.temperature,
              'repeat':args.repeat, 'description_schema':args.schema,
              'max_new_tokens': args.max_new_tokens, 'model_max_length': 16384,
              'num_beams': 1, 'top_p': .8, 'top_k': 20, 'repetition_penalty': 1.05,
              'unsupported_comparison_options': ['presence_penalty'],
              'note': 'Sampling only when temperature>0; no JSON grammar constraint.'}
    if args.selection is None and args.repeat=='original' and args.temperature==0 and args.schema=='rich-v2':
        for key in ('temperature','repeat','description_schema'): config.pop(key)
        config['note']='Greedy temperature-zero equivalent; top_p/top_k inactive, no JSON grammar constraint.'
    if args.no_warmup: config['warmup_per_load'] = False
    snapshot(destination, config, selected)
    pending = [f for f in selected[:args.limit] if not check_saved(saved_path(destination, f, 'rich'), f)]
    if args.max_new_calls is not None: pending = pending[:args.max_new_calls]
    if not pending:
        progress('NVILA: zahtevani odgovori že shranjeni; brez ponovitve inferenc.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    # HF expands the tokenizer embeddings at load time. Keep those new weights
    # constant across repeats; only sampling inside infer() uses 42/43/44.
    initialization_seed=42 if args.selection is not None else args.seed
    torch.manual_seed(initialization_seed)
    start = time.perf_counter()
    model = llava.load(str(OUT / 'models/nvila'), device_map='cuda:0',
                       attn_implementation='sdpa', model_max_length=16384)
    torch.cuda.synchronize()
    embedding_tails={}
    if args.selection is not None:
        for name,layer in [('input',model.get_llm().get_input_embeddings()),
                           ('output',model.get_llm().get_output_embeddings())]:
            embedding_tails[name]=sha256(layer.weight[-64:].detach().float().cpu().numpy().tobytes())
    write_json(destination / 'environments' / f'{time.time_ns()}.json', {
        **environment(torch, model), 'load_seconds': time.perf_counter()-start,
        'initialization_seed':initialization_seed,'embedding_tail_sha256':embedding_tails,
        'config': model.config.to_dict(), 'generation_config': model.default_generation_config.to_dict()})
    generation = copy.deepcopy(model.default_generation_config)
    generation.do_sample = args.temperature>0
    generation.max_new_tokens = args.max_new_tokens
    generation.num_beams = 1
    generation.top_p = .8
    generation.top_k = 20
    # This is the official checkpoint's documented value, unlike the wrapper's
    # default 1.0. Initial repetition failure is retained as a separate diagnostic.
    generation.repetition_penalty = 1.05
    generation.temperature = args.temperature or 1.0  # inactive under greedy
    captured = {}
    original_generate = model.generate

    def capture_generate(*args, **kwargs):
        captured['input_tokens_before_image_expansion'] = int(kwargs['input_ids'].shape[-1])
        if args_selection:
            captured['input_token_ids_before_image_expansion']=kwargs['input_ids'][0].tolist()
        captured['image_tiles'] = len(kwargs.get('media', {}).get('image', []))
        try:
            output = original_generate(*args, **kwargs)
        except ValueError as exc:
            # Upstream silently changes sampling to greedy on ValueError. A
            # comparison must save the failure instead of changing its protocol.
            if args_selection: raise RuntimeError('Native generation failed; greedy fallback disabled: '+str(exc)) from exc
            raise
        captured['token_ids'] = output[0].tolist()
        return output

    args_selection=args.selection is not None
    model.generate = capture_generate

    def infer(frame):
        # Reset each image to its declared sampling seed, independent of resume.
        torch.manual_seed(args.seed)
        prompt = user_prompt({'record': {'frame_id': frame['frame_id']},
                              'image_size': frame['image_size'], 'regions': None})
        result = {**frame, 'model': config['model'], 'system_prompt': system,
                  'user_prompt': prompt, 'generation_config': generation.to_dict(),
                  'repeat':args.repeat,'seed':args.seed,'call_id':uuid.uuid4().hex,
                  'syntax_valid':False,'schema_valid':False}
        text=system+'\n\n'+prompt
        if args.schema=='bbox':
            from .bbox_comparison import bbox_prompt
            text=bbox_prompt(system,frame)
            result['user_prompt']=text
        if args.selection is not None:
            result['native_user_prompt']='<image>'+text
            result['native_chat_prompt_text']=model.tokenizer.apply_chat_template(
                [{'role':'user','content':result['native_user_prompt']}],add_generation_prompt=True,tokenize=False)
        captured.clear()
        tick = begin_measure(torch)
        try:
            text = model.generate_content([llava.Image(frame['image_path']), text],
                                          generation_config=generation)
            result.update(content=text, **captured)
            result['output_tokens'] = len(captured['token_ids'])
            candidate = strict_json(text)
            result['candidate_parsed'] = candidate
            result['syntax_valid'] = True
            if result['output_tokens'] >= args.max_new_tokens:
                raise ValueError('Output reached token limit; potential truncation')
            if args.schema=='bbox':
                from .bbox_comparison import validate_bbox
                result['parsed']=validate_bbox(candidate,frame['image_size'])
            else:result['parsed'] = validate_description(candidate, expected_schema=args.schema)
            result['schema_valid'] = True
            result['status'] = 'ok'
        except Exception as exc:
            result.update(status='error', error=str(exc), traceback=traceback.format_exc())
        result.update(end_measure(torch, tick))
        result['finished_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        return result

    if not args.no_warmup:
        warmup = infer(pending[0])
        write_json(destination / 'warmups' / (str(time.time_ns()) + '.json'),
                   {**warmup, 'excluded_from_quality': True})
        if 'content' not in warmup:
            raise RuntimeError('NVILA GPU warmup failed: ' + warmup.get('error', ''))
        progress('NVILA: prva dejanska slika na GPU uspešno obdelana; veljavnost sheme '+warmup['status']+'.')
    for frame in pending:
        row = infer(frame)
        write_json(saved_path(destination, frame, 'rich'), row)
        progress(f"NVILA: {frame['dataset']} slika {frame['index']+1}, {row['status']}, {row['elapsed_seconds']:.2f} s.")


if __name__ == '__main__':
    main()
