"""Shared, offline checks for the Florence text experiments. Never repair answers."""
import json
from pathlib import Path

from .experiment import read_json, strict_json, sha256, write_json

def typed_id(value):
    # Keep the offline viewer independent of vision_pipeline's Pillow dependency.
    if type(value) not in (int, str) or value == '':
        raise ValueError('Object IDs must be integers or nonempty strings, never booleans.')
    return type(value).__name__, value


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                             separators=(',', ':')).encode())


def validate_response(row, objects):
    """Full-call validity and separately usable classes; input/output IDs are typed."""
    response = row.get('response') or {}
    if not isinstance(response, dict):response = {}
    received = row.get('status') == 'response_received' or 'message' in response
    skipped = row.get('status') == 'skipped'
    errors = []
    usable = []
    if received:
        if response.get('done') is not True or row.get('finish_reason') != 'stop':
            errors.append('Generation did not finish with done=true and finish_reason=stop.')
        if response.get('done_reason', row.get('finish_reason')) != row.get('finish_reason'):
            errors.append('Conflicting generation finish metadata.')
        expected = {typed_id(o['id']) for o in objects}
        seen = set(); duplicates = set(); candidates = {}
        try:
            value = strict_json(row.get('content', ''))
            if not isinstance(value, dict) or not isinstance(value.get('objects'), list):
                raise ValueError('Missing JSON objects list.')
            for item in value['objects']:
                if not isinstance(item, dict):
                    errors.append('Prediction must be an object.'); continue
                try:
                    key = typed_id(item.get('id'))
                except ValueError:
                    errors.append('Invalid ID type; only integer/string IDs are allowed.'); continue
                if key in seen:
                    duplicates.add(key); errors.append('Duplicate ID: ' + repr(item['id']))
                seen.add(key)
                if key not in expected:
                    errors.append('Unexpected ID or ID type: ' + repr(item['id'])); continue
                classification = item.get('classification')
                if classification not in ('fixed', 'movable', 'uncertain'):
                    errors.append('Invalid classification for ID ' + repr(item['id']))
                else:
                    candidates[key] = item
                if not isinstance(item.get('reason'), str) or not item['reason'].strip():
                    errors.append('Missing/nonempty reason required for ID ' + repr(item['id']))
            if expected - seen:
                errors.append('Missing IDs: ' + repr(sorted(expected - seen)))
            usable = [v for k, v in candidates.items() if k not in duplicates]
        except (ValueError, TypeError, KeyError) as exc:
            errors.append('Invalid JSON/schema: ' + str(exc))
    return {'api_response_received': received, 'schema_valid_response': received and not errors,
            'api_error': row.get('status') == 'error' and not received,
            'schema_or_finish_error': received and bool(errors), 'skipped': skipped,
            'errors': errors, 'usable_items': usable}


def validity_counts(checks):
    fields = ('api_response_received', 'schema_valid_response', 'api_error',
              'schema_or_finish_error', 'skipped')
    return {'positions': len(checks), **{k: sum(bool(c[k]) for c in checks) for k in fields},
            'quality_evaluated': False, 'accuracy': None}


def make_signature(document, input_sha, model, digest, system, prompt, options):
    image_sha = document.get('image_sha256')
    if not image_sha or not digest:
        raise ValueError('Signature requires image SHA256 and model weight digest.')
    payload = {'input_sha256': input_sha, 'image_sha256': image_sha, 'model': model,
               'model_digest': digest, 'messages': [{'role': 'system', 'content': system},
                   {'role': 'user', 'content': prompt}], 'options': options,
               'format': 'json', 'think': False}
    return {'version': 1, 'sha256': fingerprint(payload), 'payload': payload}


def check_signature(row, expected, document, document_sha):
    """Legacy image identity can be verified through the exact frozen document SHA."""
    target = expected['payload']; actual = {}; missing = []; mismatches = []; notes = []
    signed = (row.get('signature') or {}).get('payload', {})
    request = row.get('request') or {}
    for key in ('input_sha256', 'image_sha256', 'model'):
        actual[key] = row.get(key, signed.get(key))
    if actual['image_sha256'] is None and actual['input_sha256'] == document_sha:
        actual['image_sha256'] = document.get('image_sha256')
        notes.append('Legacy image SHA verified transitively through the frozen input; no direct image SHA in response.')
    actual['model_digest'] = (row.get('gpu') or {}).get('digest', signed.get('model_digest'))
    for key in ('messages', 'options', 'format', 'think'):
        actual[key] = request.get(key, signed.get(key))
    for key, value in target.items():
        if actual.get(key) is None:missing.append(key)
        elif fingerprint(actual[key]) != fingerprint(value):mismatches.append(key)
    if document_sha != target['input_sha256']:mismatches.append('frozen_document_bytes')
    if document.get('image_sha256') != target['image_sha256']:mismatches.append('frozen_image_sha256')
    if request.get('model', target['model']) != target['model']:mismatches.append('request.model')
    if signed and (fingerprint(signed) != row['signature'].get('sha256') or signed != target):
        mismatches.append('stored_signature')
    return {'verified': not missing and not mismatches, 'missing_metadata': missing,
            'mismatches': mismatches, 'notes': notes}


def context_before(system, prompt, options, template=None):
    """One estimated token per UTF-8 byte plus template bytes and 512 overhead tokens.

    This is deliberately conservative, NOT a tokenizer or a guaranteed upper bound.
    No exact count is claimed. Unknown templates and failed estimates block new calls.
    """
    context = options.get('num_ctx'); output = options.get('num_predict')
    valid = type(context) is int and context > 0 and type(output) is int and output > 0
    estimate = None if not isinstance(template, str) else sum(len(s.encode('utf-8')) for s in (system, prompt, template)) + 512
    status = 'unverifiable' if not valid or estimate is None else (
        'estimated_sufficient' if estimate + output <= context else 'estimated_insufficient')
    return {'status': status, 'method': 'conservative_utf8_bytes_plus_template_plus_512',
            'exact': False, 'estimated_prompt_tokens': estimate, 'reserved_output_tokens': output,
            'num_ctx': context, 'note': 'Heuristic, not an exact token count; false rejections are possible.'}


def context_after(row, options):
    raw = row.get('response') or {}; prompt = raw.get('prompt_eval_count'); output = raw.get('eval_count')
    context = (row.get('gpu') or {}).get('context_length', options.get('num_ctx'))
    budget = options.get('num_predict')
    valid = all(type(n) is int and n >= 0 for n in (prompt, output, context, budget)) and context > 0
    return {'status': 'unverifiable' if not valid else ('sufficient' if prompt + budget <= context else 'reserved_budget_exceeded'),
            'prompt_eval_count': prompt, 'eval_count': output, 'num_ctx': context,
            'reserved_output_tokens': budget, 'actual_sequence_fits': None if not valid else prompt + output <= context,
            'context_matches_request': None if context is None else context == options.get('num_ctx'),
            'note': 'Reported token counts do not by themselves rule out server-side input truncation.'}


def installed_digest(backend):
    # Metadata only; never loads a model or performs a generation.
    from .vision_backends import request
    tags = request(backend.url + '/api/tags', timeout=backend.timeout, headers=backend.headers)
    entry = next((m for m in tags['models'] if m['name'] == backend.spec['model']), None)
    if not entry or not entry.get('digest'):raise ValueError('Installed model digest is unavailable.')
    return entry['digest']


def prepare_position(target, frozen, input_sha, spec, digest, system, prompt, options, template):
    """Check every saved position before reuse. Never replace an existing result."""
    content = Path(frozen).read_bytes(); document = strict_json(content.decode()); actual_sha = sha256(content)
    if actual_sha != input_sha:raise ValueError('Frozen input SHA256 mismatch: ' + str(frozen))
    signature = make_signature(document, input_sha, spec['model'], digest, system, prompt, options)
    if target.exists():
        saved=read_json(target)
        report = check_signature(saved, signature, document, actual_sha)
        report['context_budget_after']=context_after(saved,options)
        write_json(target.parent/'_technical_checks'/(target.stem+'.json'), report)
        if not report['verified']:
            raise ValueError('Result cannot be reused; original preserved: '+str(target)+' '+json.dumps(report))
        return None
    before = context_before(system, prompt, options, template)
    return {'signature': signature, 'image_sha256': document['image_sha256'],
            'input_sha256': input_sha, 'context_budget_before': before}


def budget_skip(row):
    if row['context_budget_before']['status'] == 'estimated_sufficient':return False
    row.update(status='skipped', generation_attempted=False,
               error='Context budget is insufficient or unverifiable; no generation attempted.')
    return True


def finish_position(row, document, options):
    row['context_budget_after']=context_after(row,options)
    row['validation']={k:v for k,v in validate_response(row,document['parsed']['objects']).items() if k!='usable_items'}
    row['signature_check_after']=check_signature(row,row['signature'],document,row['input_sha256'])
