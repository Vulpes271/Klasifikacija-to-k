"""Missing native Qwen3-VL connection; reference-free RGB, resumable raw records.

Uses the existing THUD selection, measurements, strict parser and legacy schema.
All model parameters must reside on CUDA. Sampling is reset per image, so resume
does not change the seed. A failed record is retained, never silently retried.
"""
import copy
from pathlib import Path
import time
import traceback
import uuid

from .description_prompts import user_prompt, validate_description
from .experiment import read_json, write_json, strict_json, sha256
from .thud_extended_common import (ROOT, native_selection, snapshot, environment,
                                   begin_measure, end_measure, saved_path, check_saved, progress)

OUT = ROOT / 'results/stage1-presentation'
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
SEEDS = {'r1': 42, 'r2': 43, 'r3': 44}


def configuration(repeat, instruction):
    return {'model': MODEL, 'revision': REVISION, 'repeat': repeat,
            'seed': SEEDS[repeat], 'initialization_seed': 42,
            'instruction': instruction, 'instruction_sha256': sha256(instruction.encode()),
            'native_system_prompt': 'You are a helpful assistant',
            'precision_requested': 'BF16', 'attention': 'sdpa',
            'do_sample': True, 'temperature': .3, 'top_p': .8, 'top_k': 20,
            'repetition_penalty': 1.05, 'num_beams': 1,
            'max_new_tokens': 4096, 'input_plus_output_limit': 16384,
            'unsupported_comparison_options': ['presence_penalty'],
            'processor_resize': 'checkpoint defaults, recorded image_grid_thw',
            'description_schema': 'legacy', 'json_grammar': False}


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    selected = native_selection(ROOT / 'results/ponovitve/frames.json')
    instruction = (ROOT / 'configs/ponovitve/nvila-description-v2.txt').read_text().strip()
    configs = {r: configuration(r, instruction) for r in SEEDS}
    for repeat, config in configs.items():
        snapshot(OUT / 'qwen3-vl' / repeat, config, selected)
    pending = [(r, f) for r in SEEDS for f in selected
               if not check_saved(saved_path(OUT / 'qwen3-vl' / r, f, 'description'), f)]
    if not pending:
        progress('Qwen3-VL: vseh 60 odgovorov že shranjenih; brez novih generacij.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback forbidden')
    torch.manual_seed(42)
    tick = time.perf_counter()
    weights = OUT / 'models/qwen3-vl'
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        weights, torch_dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True).eval()
    torch.cuda.synchronize()
    env = {**environment(torch, model), 'load_seconds': time.perf_counter()-tick,
           'revision': REVISION, 'config': model.config.to_dict(),
           'processor': processor.to_dict() if hasattr(processor, 'to_dict') else str(processor),
           'generation_defaults': model.generation_config.to_dict()}
    write_json(OUT / 'environment.json', env)

    def infer(repeat, frame):
        config = configs[repeat]
        torch.manual_seed(config['seed'])
        prompt = user_prompt({'record': {'frame_id': frame['frame_id']},
                              'image_size': frame['image_size'], 'regions': None})
        text = instruction + '\n\n' + prompt
        record = {**frame, 'model': MODEL, 'revision': REVISION, 'repeat': repeat,
                  'seed': config['seed'], 'call_id': uuid.uuid4().hex,
                  'system_prompt': config['native_system_prompt'], 'user_prompt': text,
                  'instruction': instruction, 'generic_input_prompt': prompt,
                  'syntax_valid': False, 'schema_valid': False}
        start = begin_measure(torch)
        try:
            # Neither annotations nor reference paths are read by this runner.
            image = Image.open(frame['image_path']).convert('RGB')
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': config['native_system_prompt']}]},
                        {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                    {'type': 'text', 'text': text}]}]
            record['native_chat_prompt_text'] = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda:0')
            count = int(inputs['input_ids'].shape[-1])
            record['input_token_ids'] = inputs['input_ids'][0].tolist()
            record['input_tokens'] = count
            record['image_grid_thw'] = inputs['image_grid_thw'].tolist()
            if count + config['max_new_tokens'] > config['input_plus_output_limit']:
                raise ValueError('Input plus output exceeds declared context limit')
            generation = copy.deepcopy(model.generation_config)
            for key in ('do_sample', 'temperature', 'top_p', 'top_k',
                        'repetition_penalty', 'num_beams', 'max_new_tokens'):
                setattr(generation, key, config[key])
            record['generation_config'] = generation.to_dict()
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=generation)
            tokens = output[0, count:].tolist()
            record.update(token_ids=tokens, output_tokens=len(tokens),
                content=processor.decode(tokens, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False),
                raw_decoded_with_special_tokens=processor.decode(tokens, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False))
            candidate = strict_json(record['content'])
            record.update(candidate_parsed=candidate, syntax_valid=True)
            if len(tokens) >= config['max_new_tokens']:
                raise ValueError('Output reached token limit; potential truncation')
            record['parsed'] = validate_description(candidate, expected_schema='legacy')
            record.update(schema_valid=True, status='ok')
        except Exception as exc:
            record.update(status='error', error=str(exc), traceback=traceback.format_exc())
        record.update(end_measure(torch, start))
        record['finished_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        return record

    repeat, frame = pending[0]
    warmup = infer(repeat, frame)
    write_json(OUT / 'warmups' / (str(time.time_ns()) + '.json'),
               {**warmup, 'excluded_from_quality': True})
    if 'content' not in warmup:
        raise RuntimeError('Qwen3-VL GPU warmup failed: ' + warmup.get('error', ''))
    progress('Qwen3-VL: prva slika na GPU uspešna; ogrevanje izključeno iz primerjave.')
    for position, (repeat, frame) in enumerate(pending, 1):
        record = infer(repeat, frame)
        write_json(saved_path(OUT / 'qwen3-vl' / repeat, frame, 'description'), record)
        progress(f"Qwen3-VL {repeat} {frame['frame_id']}: {record['status']}, {record['elapsed_seconds']:.2f} s ({position}/{len(pending)}).")
        if position <= 2 and 'content' not in record:
            raise RuntimeError('Initial GPU pilot failed; see retained raw error')


if __name__ == '__main__':
    main()
