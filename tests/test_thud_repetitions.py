"""Repeat analysis must preserve failures, literal text and immutable identity."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from copy import deepcopy
from llm_object_classifier.thud_extended_report import (
    ROOT,repetition_view,repetition_compact,repetition_consistency,repetition_reason_consistency)


class RepetitionTests(unittest.TestCase):
    def test_reason_orchestrator_does_not_reopen_photographs(self):
        from scripts import thud_ponovitve as runner
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'frames.json').write_text(json.dumps([{'frame_id':'saved metadata'}]))
            with patch.object(runner,'OUT',root),patch.object(runner,'prepare',side_effect=AssertionError('RGB reopened')), \
                 patch.object(runner,'freeze') as freeze,patch.object(runner,'ollama') as calls, \
                 patch('sys.argv',['runner','--phase','reason']):
                runner.main()
            freeze.assert_called_once()
            self.assertEqual(len(calls.call_args_list),13)  # 12 groups + reusable two-image pilot.

    def test_native_json_stream_is_display_only_and_never_repaired(self):
        text='{"scene_description":"room"}\n{"objects":[{"id":2,"label":"chair"}]}\nextra claims'
        row={'model_name':'nvila','task':'description','status':'error','content':text}
        before=deepcopy(row);view=repetition_view(row)
        self.assertEqual(view['objects'],[{'id':2,'label':'chair'}])
        self.assertEqual(row,before)
        self.assertIn('display only',view['extraction'])
        bad={**row,'content':'{"objects":[{"label":"a","label":"b"}]}'}
        self.assertEqual(repetition_view(bad)['objects'],[])

    def test_compact_keeps_entire_response_and_prompt(self):
        text='unchanged '+str(ROOT)+' content\nall words'
        row={'content':text,'system_prompt':text,'request':{'messages':[{'content':text,'images':['base64']} ]},
             'image_path':str(ROOT/'data/a.png'),'traceback':'local debug',
             'parsed':{'objects':[{'images':['literal model claim']} ]}}
        result=repetition_compact(row)
        self.assertEqual(result['content'],text)
        self.assertEqual(result['request']['messages'][0]['content'],text)
        self.assertNotIn('images',result['request']['messages'][0])
        self.assertNotIn('traceback',result)
        self.assertEqual(result['image_path'],'data/a.png')
        self.assertEqual(result['parsed'],row['parsed'])

    def test_duplicate_box_predictions_match_one_to_one(self):
        rows=[]
        for repeat in ('r1','r2','r3'):
            boxes=[[0,0,5,5]]*(2 if repeat=='r1' else 1)
            rows.append({'frame_id':'x','model_name':'florence','task':'od','repeat':repeat,'status':'ok','content':'same',
                         'parsed':{'<OD>':{'labels':['chair']*len(boxes),'bboxes':boxes}}})
        pairs=repetition_consistency(rows)
        self.assertEqual(len(pairs),3)
        self.assertEqual(pairs[0]['box_matching']['matched'],1)
        self.assertEqual(pairs[0]['name_jaccard'],1)
        self.assertEqual(pairs[0]['left_objects'],2)

    def test_reasoning_requires_same_frozen_sha_and_does_not_hide_failures(self):
        base={'frame_id':'x','model_name':'qwen35-bf16','prompt_name':'short','caption_sha256':'same','status':'ok',
              'parsed':{'objects':[{'id':1,'classification':'fixed','reason':'first'}]}}
        rows=[{**base,'repeat':'r1'},{**base,'repeat':'r2','caption_sha256':'changed'},
              {**base,'repeat':'r3','status':'error'}]
        pairs=repetition_reason_consistency(rows)
        self.assertEqual(len(pairs),3)
        self.assertTrue(all(not p['comparable'] for p in pairs))
        self.assertFalse(pairs[0]['same_frozen_input'])


if __name__=='__main__': unittest.main()
