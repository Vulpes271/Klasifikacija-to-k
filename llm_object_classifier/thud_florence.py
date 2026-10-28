"""Native Florence-2 tasks, no Qwen JSON prompt; one persistent CUDA model.

Raw decoded tokens and IDs plus Microsoft's parsed predictions are both saved.
Each task and the three-task per-image procedure have separate synchronized
wall-clock measurements. Loading and physical warmups are excluded.
"""
import argparse
from pathlib import Path
import time
import traceback
import uuid

from .experiment import read_json, write_json
from .thud_extended_common import (OUT, ROOT, prepare, snapshot, environment, progress,
    begin_measure, end_measure, saved_path, check_saved, validate_boxes, draw_boxes, native_selection)

TASKS = {'caption': '<MORE_DETAILED_CAPTION>', 'od': '<OD>', 'dense': '<DENSE_REGION_CAPTION>'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat', default='original')
    parser.add_argument('--max-new-calls', type=int, help='Maximum new images (three native tasks per image)')
    parser.add_argument('--no-warmup', action='store_true')
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    selected = native_selection(args.selection)
    destination = args.output_dir or OUT / 'florence'
    model_path = OUT / 'models/florence'
    config = {'model': 'microsoft/Florence-2-large',
              'revision': read_json(OUT / 'setup/florence-api.txt')['sha'],
              'tasks': TASKS, 'precision_requested': 'FP16', 'num_beams': 3,
              'do_sample': False, 'seed': args.seed, 'repeat':args.repeat,'max_new_tokens': 1024,
              'attention': 'eager', 'bbox_coordinates': 'original image pixels, xyxy',
              'unsupported_comparison_options': ['rich JSON prompt', '16384 context', '8192 output',
                                                  'top_k', 'top_p', 'presence_penalty']}
    if args.selection is None and args.repeat=='original': config.pop('repeat')
    if args.no_warmup: config['warmup_per_load'] = False
    snapshot(destination, config, selected)
    pending = [f for f in selected[:args.limit] if any(not check_saved(saved_path(destination, f, t), f) for t in TASKS)]
    if args.max_new_calls is not None: pending = pending[:args.max_new_calls]
    if not pending:
        progress('Florence: zahtevani odgovori že shranjeni; brez ponovitve inferenc.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    initialization_seed = 42 if args.selection is not None else args.seed
    torch.manual_seed(initialization_seed)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True,
        torch_dtype=torch.float16, attn_implementation='eager', local_files_only=True).cuda().eval()
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    torch.cuda.synchronize()
    write_json(destination / 'environments' / f'{time.time_ns()}.json',
               {**environment(torch, model), 'load_seconds': time.perf_counter()-start,
                'initialization_seed': initialization_seed,
                'generation_config': model.generation_config.to_dict()})

    def infer(frame, task):
        torch.manual_seed(args.seed)
        result = {**frame, 'model': config['model'], 'task': task, 'prompt': task,
                  'system_prompt':'','user_prompt':task,'repeat':args.repeat,'seed':args.seed,
                  'call_id':uuid.uuid4().hex,
                  'generation': {k: config[k] for k in ('num_beams', 'do_sample', 'max_new_tokens', 'seed')}}
        tick = begin_measure(torch)
        try:
            image = Image.open(frame['image_path']).convert('RGB')
            inputs = processor(text=task, images=image, return_tensors='pt').to('cuda', torch.float16)
            with torch.inference_mode():
                ids = model.generate(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'],
                    max_new_tokens=1024, num_beams=3, do_sample=False)
            text = processor.batch_decode(ids, skip_special_tokens=False)[0]
            result.update({'content': text, 'token_ids': ids[0].tolist(),
                           'output_tokens': int(ids.shape[-1]-1),
                           'input_tensor_shapes': {k: list(v.shape) for k,v in inputs.items()}})
            parsed = processor.post_process_generation(text, task=task, image_size=tuple(frame['image_size']))
            result['parsed'] = parsed
            validate_boxes(parsed, task, frame['image_size'])
            if ids.shape[-1]-1 >= 1024:
                raise ValueError('Output reached token limit; potential truncation')
            result['status'] = 'ok'
        except Exception as exc:
            result.update(status='error', error=str(exc), traceback=traceback.format_exc())
        result.update(end_measure(torch, tick))
        result['finished_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        return result

    # Physical warmup for every task after this model load, even on resume.
    warmdir = destination / 'warmups' / str(time.time_ns())
    for name, task in ({} if args.no_warmup else TASKS).items():
        row = infer(pending[0], task)
        write_json(warmdir / (name + '.json'), {**row, 'excluded_from_quality': True})
        if 'content' not in row:
            raise RuntimeError('Florence GPU warmup failed: ' + row.get('error', ''))
    progress('Florence: začetek merjenih slik; ' + ('brez dodatnih ogrevalnih generacij.' if args.no_warmup else 'fizično ogrevanje treh nalog opravljeno.'))
    for frame in pending:
        combined_tick = begin_measure(torch)
        performed = []
        measured_rows = []
        for name, task in TASKS.items():
            path = saved_path(destination, frame, name)
            if check_saved(path, frame):
                continue
            row = infer(frame, task)
            write_json(path, row)
            performed.append(name)
            measured_rows.append(row)
        timing = {**frame, **end_measure(torch, combined_tick), 'tasks_executed': performed,
                  'complete_three_task_measurement': len(performed) == 3,
                  'scope': 'image load, 3 preprocessing/generation/postprocess calls, JSON writes; excludes overlays'}
        # Each task resets CUDA peak counters. The procedure peak must therefore
        # be the maximum of its task peaks, not only the final task's counter.
        timing['peak_allocated_bytes'] = max(r['peak_allocated_bytes'] for r in measured_rows)
        timing['peak_reserved_bytes'] = max(r['peak_reserved_bytes'] for r in measured_rows)
        write_json(saved_path(destination, frame, 'combined'), timing)
        # Drawing is offline output preparation, outside combined inference time.
        for name in ('od', 'dense'):
            row = read_json(saved_path(destination, frame, name))
            if row['status'] == 'ok':
                draw_boxes(frame['image_path'], row['parsed'], TASKS[name],
                    destination / 'overlays' / frame['dataset'] / name / f"frame-{frame['index']+1:06d}.jpg")
        progress(f"Florence: {frame['dataset']} slika {frame['index']+1}, tri naloge shranjene.")


if __name__ == '__main__':
    main()
