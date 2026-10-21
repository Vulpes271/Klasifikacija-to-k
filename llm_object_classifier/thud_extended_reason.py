"""Freeze native observations and reuse the existing text-only Qwen BF16 stage.

NVILA valid rich JSON is passed intact. Florence's caption, OD labels and dense
phrases remain literal; box coordinates are rendered as text. No merging across
tasks, synonym invention, inferred contacts or attachment evidence. Adapter-only
flags explicitly mean identity/mounting was not independently verified.
"""
import os
import json
from types import SimpleNamespace

from .description_prompts import validate_description
from .experiment import load_config, read_json, write_json, sha256, strict_json
from .stage2_reason import run
from .thud_experiment import immutable_json, MODELS
from .thud_extended_common import OUT, ROOT, DATASETS, prepare, saved_path, progress, snapshot
from .thud_florence import TASKS
from .vision_backends import Backend


def florence_caption(rows):
    """A transparent transport conversion, not a new visual prediction."""
    for name, task in TASKS.items():
        if rows[name].get('status') != 'ok':
            raise ValueError('Invalid native Florence task: ' + name)
    objects = []
    for name in ('od', 'dense'):
        value = rows[name]['parsed'][TASKS[name]]
        for label, box in zip(value['labels'], value['bboxes']):
            objects.append({'id': len(objects)+1, 'label': label,
                'observation': f"Florence task {TASKS[name]} literally returned label/phrase {label!r}; bbox_xyxy in original image pixels: {box}. No attachment/contact/occlusion fields were produced. Identity across tasks is not established.",
                'attachment_evidence': 'not_visible', 'identifiable': True})
    # True here records that the native model named this tentative region;
    # it is never manual confirmation. Do not mark all observations unusable.
    return validate_description({'scene_description': rows['caption']['parsed'][TASKS['caption']],
                                 'objects': objects})


def nvila_caption(row):
    """Transport literal native claims without making rejected rich JSON valid.

    Some native outputs contain a JSON object followed by echoed instructions,
    or omit rich fields. Keep that original failure and the ENTIRE raw text.
    A separately audited transport view uses only explicitly returned labels;
    the complete native object is the observation. Nothing is filled in as a
    visual fact, and no trailing content is discarded. Duplicate JSON keys or
    missing labels remain fatal. This is not rich-v2 validation or repair.
    """
    if row['status'] == 'ok':
        return row['parsed'], 'unchanged-valid-rich-v2'
    content=row.get('content','')
    decoder=json.JSONDecoder()
    _, end=decoder.raw_decode(content.lstrip())
    # Validate duplicate keys in the consumed prefix as well as its syntax.
    native=strict_json(content.lstrip()[:end])
    if not isinstance(native,dict) or not isinstance(native.get('objects'),list) or not native['objects']:
        raise ValueError('No literal NVILA object list available for transport')
    objects=[]
    for obj in native['objects']:
        if not isinstance(obj,dict) or not isinstance(obj.get('label'),str) or not obj['label'].strip():
            raise ValueError('NVILA object has no literal label')
        evidence=obj.get('attachment_evidence')
        if evidence not in {'visible_installation','appears_freestanding','not_visible','mixed'}:
            evidence='not_visible'  # Transport missing/invalid enum, never assert mounting.
        objects.append({'id':len(objects)+1,'label':obj['label'],
                        'observation':json.dumps(obj,ensure_ascii=False),
                        'attachment_evidence':evidence,
                        'identifiable':obj.get('identifiable') if type(obj.get('identifiable')) is bool else True})
    return validate_description({'scene_description':content,'objects':objects}), 'literal-native-text-v1; original rich rejection retained; full raw text included'


def freeze(model, dataset, selected):
    destination = OUT / 'frozen' / model / dataset
    for frame in selected:
        if frame['dataset'] != dataset or not frame['subset20']:
            continue
        task_names = ['rich'] if model == 'nvila' else list(TASKS)
        sources = {name: saved_path(OUT/model, frame, name) for name in task_names}
        row = {'stage': 1, 'frame_id': frame['frame_id'], 'model_name': model,
               'model': ('Efficient-Large-Model/NVILA-8B' if model == 'nvila' else 'microsoft/Florence-2-large'),
               'variant': {'id': 'native-t0-s42'},
               'image_sha256': frame['image_sha256'],
               'adapter_policy': 'Literal observations only. Florence task regions kept distinct; not_visible means mounting not provided, identifiable means native model named a tentative region, not human confirmation.',
               'source_records': {name: {'path': str(p), 'sha256': sha256(p.read_bytes()) if p.exists() else None}
                                  for name,p in sources.items()}}
        try:
            native = {name: read_json(p) for name,p in sources.items()}
            if model == 'nvila':
                row['native_status']=native['rich']['status']
                row['native_error']=native['rich'].get('error')
                row['parsed'],row['conversion']=nvila_caption(native['rich'])
            else:
                row['parsed'] = florence_caption(native)
            row['status'] = 'ok'
        except (ValueError, FileNotFoundError) as exc:
            row.update(status='error', error=str(exc))
        immutable_json(destination / f"frame-{frame['index']+1:06d}.json", row)
    return destination


def main():
    selected = prepare()
    url = os.environ['THUD_OLLAMA_URL']
    raw = read_json(ROOT / 'configs/stage1/thud-part2.json')
    raw.pop('description_schema')
    raw['models'] = [{**MODELS[0], 'base_url': url, 'require_gpu': True}]
    raw['temperatures'] = [.3]
    raw['system_file'] = str(ROOT / 'configs/prompts/thud-short-no-examples.txt')
    raw['instruction_file'] = str(ROOT / 'configs/prompts/reasoning.txt')
    path = OUT / 'reason-config.json'
    immutable_json(path, raw)
    config = load_config(path, 2)
    snapshot(OUT/'reason', {'config':config,'conversion_policy':'literal-native-text-v1; florence literal regions'},selected)
    for model in ('nvila', 'florence'):
        for dataset in DATASETS:
            source = freeze(model, dataset, selected)
            destination = OUT / 'reason' / model / dataset
            progress(f'Razvrščanje novih opisov: {model}, {dataset}, isti Qwen BF16, T0.3/seed42.')
            code = run(SimpleNamespace(captions=source, output_dir=destination, limit=10,
                                       resume=True, warmup=True), config)
            Backend(config['models'][0], config['timeout']).unload()
            if code == 1:
                raise RuntimeError('Qwen GPU preparation failed; partial outputs retained')
            write_json(destination / 'complete.json', {'exit_code': code, 'positions': 10})


if __name__ == '__main__':
    main()
