"""NVILA-8B native VILA inference; avoids the obsolete API model allow-list.

Uses the exact part-two rich prompt and original RGB bytes, with no references.
FP16 reflects VILA's actual loader and image tensors (not a BF16 claim).
"""
import argparse
import copy
import time
import traceback

from .description_prompts import user_prompt, validate_description
from .experiment import read_json, strict_json, write_json, sha256
from .thud_extended_common import (OUT, ROOT, prepare, snapshot, environment, progress,
    begin_measure, end_measure, saved_path, check_saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    import torch
    import llava
    selected = prepare()
    destination = OUT / 'nvila'
    system = (ROOT / 'configs/prompts/thud-description-rich.txt').read_text()
    config = {'model': 'Efficient-Large-Model/NVILA-8B',
              'revision': read_json(OUT / 'setup/nvila-api.txt')['sha'],
              'vila_commit': '57ecbfb904b99b563db69ef7916f6390eecea516',
              'system_prompt': system, 'system_sha256': sha256(system.encode()),
              'precision_requested': 'FP16', 'attention': 'sdpa', 'seed': 42,
              'do_sample': False, 'max_new_tokens': 8192, 'model_max_length': 16384,
              'num_beams': 1, 'top_p': .8, 'top_k': 20, 'repetition_penalty': 1.05,
              'unsupported_comparison_options': ['presence_penalty'],
              'note': 'Greedy temperature-zero equivalent; top_p/top_k inactive, no JSON grammar constraint.'}
    snapshot(destination, config, selected)
    pending = [f for f in selected[:args.limit] if not check_saved(saved_path(destination, f, 'rich'), f)]
    if not pending:
        progress('NVILA: zahtevani odgovori že shranjeni; brez ponovitve inferenc.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(42)
    start = time.perf_counter()
    model = llava.load(str(OUT / 'models/nvila'), device_map='cuda:0',
                       attn_implementation='sdpa', model_max_length=16384)
    torch.cuda.synchronize()
    write_json(destination / 'environments' / f'{time.time_ns()}.json', {
        **environment(torch, model), 'load_seconds': time.perf_counter()-start,
        'config': model.config.to_dict(), 'generation_config': model.default_generation_config.to_dict()})
    generation = copy.deepcopy(model.default_generation_config)
    generation.do_sample = False
    generation.max_new_tokens = 8192
    generation.num_beams = 1
    generation.top_p = .8
    generation.top_k = 20
    # This is the official checkpoint's documented value, unlike the wrapper's
    # default 1.0. Initial repetition failure is retained as a separate diagnostic.
    generation.repetition_penalty = 1.05
    generation.temperature = 1.0  # inactive under greedy decoding
    captured = {}
    original_generate = model.generate

    def capture_generate(*args, **kwargs):
        captured['input_tokens_before_image_expansion'] = int(kwargs['input_ids'].shape[-1])
        captured['image_tiles'] = len(kwargs.get('media', {}).get('image', []))
        output = original_generate(*args, **kwargs)
        captured['token_ids'] = output[0].tolist()
        return output

    model.generate = capture_generate

    def infer(frame):
        prompt = user_prompt({'record': {'frame_id': frame['frame_id']},
                              'image_size': frame['image_size'], 'regions': None})
        result = {**frame, 'model': config['model'], 'system_prompt': system,
                  'user_prompt': prompt, 'generation_config': generation.to_dict()}
        captured.clear()
        tick = begin_measure(torch)
        try:
            text = model.generate_content([llava.Image(frame['image_path']), system + '\n\n' + prompt],
                                          generation_config=generation)
            result.update(content=text, **captured)
            result['output_tokens'] = len(captured['token_ids'])
            candidate = strict_json(text)
            result['candidate_parsed'] = candidate
            if result['output_tokens'] >= 8192:
                raise ValueError('Output reached token limit; potential truncation')
            result['parsed'] = validate_description(candidate, expected_schema='rich-v2')
            result['status'] = 'ok'
        except Exception as exc:
            result.update(status='error', error=str(exc), traceback=traceback.format_exc())
        result.update(end_measure(torch, tick))
        return result

    warmup = infer(pending[0])
    write_json(destination / 'warmups' / (str(time.time_ns()) + '.json'),
               {**warmup, 'excluded_from_quality': True})
    if 'content' not in warmup:
        raise RuntimeError('NVILA GPU warmup failed: ' + warmup.get('error', ''))
    progress('NVILA: prva dejanska slika na GPU uspešno obdelana; veljavnost sheme '+warmup['status']+'.')
    for frame in pending:
        row = infer(frame)
        write_json(saved_path(destination, frame, 'rich'), row)
        progress(f"NVILA: {frame['dataset']} {frame['index']+1}/50, {row['status']}, {row['elapsed_seconds']:.2f} s.")


if __name__ == '__main__':
    main()
