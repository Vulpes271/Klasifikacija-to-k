"""Technical regressions only; no model/server calls and no user's review database."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from llm_object_classifier.llm_checks import (validate_response, validity_counts,
    make_signature, check_signature, context_before, context_after, prepare_position)
from llm_object_classifier.experiment import sha256


class LLMChecksTests(unittest.TestCase):
    def setUp(self):
        self.objects=[{'id':1,'label':'shelf'},{'id':'2','label':'box'}]
        self.items=[{'id':1,'classification':'uncertain','reason':'Installation not established.'},
                    {'id':'2','classification':'movable','reason':'A transportable box.'}]
        self.row={'status':'response_received','finish_reason':'stop',
            'response':{'done':True,'done_reason':'stop'},'content':json.dumps({'objects':self.items})}

    def test_shared_validator_requires_reason_and_preserves_partial_classes(self):
        self.items[0]['description']=self.items[0].pop('reason')
        row={**self.row,'content':json.dumps({'objects':self.items})};before=copy.deepcopy(row)
        checked=validate_response(row,self.objects)
        self.assertTrue(checked['api_response_received']);self.assertFalse(checked['schema_valid_response'])
        self.assertTrue(checked['schema_or_finish_error']);self.assertEqual(len(checked['usable_items']),2)
        self.assertTrue(any('reason' in error for error in checked['errors']));self.assertEqual(row,before)
        for reason in ('', '   ', None, 7):
            items=copy.deepcopy(self.items);items[0]['reason']=reason
            self.assertFalse(validate_response({**row,'content':json.dumps({'objects':items})},self.objects)['schema_valid_response'])

    def test_ids_json_and_generation_finish(self):
        for bad_id in ('1', True, 1.0, 9, None):
            items=copy.deepcopy(self.items);items[0]['id']=bad_id
            with self.subTest(bad_id=bad_id):
                self.assertFalse(validate_response({**self.row,'content':json.dumps({'objects':items})},self.objects)['schema_valid_response'])
        duplicate=validate_response({**self.row,'content':json.dumps({'objects':self.items+[self.items[0]]})},self.objects)
        self.assertFalse(duplicate['schema_valid_response']);self.assertEqual([x['id'] for x in duplicate['usable_items']],['2'])
        for content in ('not JSON','{"objects":[],"objects":[]}',json.dumps({'objects':self.items[:1]})):
            self.assertFalse(validate_response({**self.row,'content':content},self.objects)['schema_valid_response'])
        for change in ({'finish_reason':'length'},{'response':{'done':False}}, {'response':{}},
                       {'response':{'done':True,'done_reason':'length'}}):
            self.assertTrue(validate_response({**self.row,**change},self.objects)['schema_or_finish_error'])
        checks=[validate_response(r,self.objects) for r in [self.row,{'status':'error'},
            {'status':'skipped'},{**self.row,'finish_reason':'length'}]]
        self.assertEqual({k:validity_counts(checks)[k] for k in ('api_response_received','schema_valid_response','api_error','schema_or_finish_error','skipped')},
            {'api_response_received':2,'schema_valid_response':1,'api_error':1,'schema_or_finish_error':1,'skipped':1})

    def test_signature_rejects_changed_fields_and_reports_legacy_unknowns(self):
        doc={'image_sha256':'image'};opts={'num_ctx':8192,'num_predict':2048,'seed':42}
        signed=make_signature(doc,'input','model','weights','system','prompt',opts)
        row={'model':'model','input_sha256':'input','gpu':{'digest':'weights'},
            'request':{'model':'model','messages':signed['payload']['messages'],'options':opts,'format':'json','think':False}}
        check=check_signature(row,signed,doc,'input')
        self.assertTrue(check['verified']);self.assertIn('transitively',check['notes'][0])
        for field,value in [('input_sha256','different'),('image_sha256','changed'),('model','other')]:
            altered={**row,field:value};self.assertFalse(check_signature(altered,signed,doc,'input')['verified'])
        for field,value in [('messages',[{'role':'user','content':'changed'}]),('options',{**opts,'seed':43})]:
            altered={**row,'request':{**row['request'],field:value}}
            self.assertFalse(check_signature(altered,signed,doc,'input')['verified'])
        altered={**row,'gpu':{'digest':'other-weights'}}
        self.assertIn('model_digest',check_signature(altered,signed,doc,'input')['mismatches'])
        unknown={**row};unknown.pop('gpu')
        self.assertIn('model_digest',check_signature(unknown,signed,doc,'input')['missing_metadata'])

    def test_conservative_budget_and_actual_counts_are_distinct(self):
        opts={'num_ctx':8192,'num_predict':2048}
        before=context_before('system','🧪',opts,'template')
        self.assertFalse(before['exact']);self.assertEqual(before['estimated_prompt_tokens'],len('system🧪template'.encode())+512)
        self.assertEqual(before['status'],'estimated_sufficient')
        self.assertEqual(context_before('','a'*8192,opts,'')['status'],'estimated_insufficient')
        self.assertEqual(context_before('','text',opts,None)['status'],'unverifiable')
        after=context_after({'response':{'prompt_eval_count':7000,'eval_count':200}},opts)
        self.assertEqual(after['status'],'reserved_budget_exceeded');self.assertTrue(after['actual_sequence_fits'])
        self.assertEqual(context_after({},opts)['status'],'unverifiable')



    def test_reuse_checks_signature_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frozen=root/'frozen.json'
            frozen.write_text(json.dumps({'image_sha256':'image','status':'ok','parsed':{'objects':self.objects}}))
            digest=sha256(frozen.read_bytes());spec={'name':'test','model':'model'}
            opts={'num_ctx':8192,'num_predict':2048,'seed':42}
            signature=make_signature(json.loads(frozen.read_text()),digest,'model','weights','system','prompt',opts)
            row={**self.row,'signature':signature,'input_sha256':digest,'model':'model','gpu':{'digest':'weights'},
                 'request':{'model':'model','messages':signature['payload']['messages'],'options':opts,'format':'json','think':False}}
            target=root/'response.json';target.write_text(json.dumps(row));original=target.read_bytes()
            self.assertIsNone(prepare_position(target,frozen,digest,spec,'weights','system','prompt',opts,'template'))
            for changed_digest,changed_prompt in [('changed','prompt'),('weights','changed')]:
                with self.assertRaises(ValueError):prepare_position(target,frozen,digest,spec,changed_digest,'system',changed_prompt,opts,'template')
                self.assertEqual(original,target.read_bytes())
