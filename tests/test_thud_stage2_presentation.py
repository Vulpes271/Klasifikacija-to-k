"""No invented objects, no lost malformed text, no cross-source ID matching."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from llm_object_classifier.thud_stage2_presentation import convert
from llm_object_classifier.thud_stage2_comparison import direct_pair, name_pair
from llm_object_classifier.experiment import write_json,sha256
from llm_object_classifier import stage2_reason
from llm_object_classifier import thud_stage2_comparison as report


class StageTwoComparisonTests(unittest.TestCase):
    def test_public_deduplication_reconstructs_full_request_and_invalid_response(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);system=root/'system.txt';user=root/'user.txt'
            system.write_text('actual system\n');user.write_text('actual user\nwith all observations')
            text='invalid JSON "quoted"\nfull trailing text <tag>'
            response={'message':{'role':'assistant','content':text},'created_at':'a unique timestamp'}
            request={'model':'test','messages':[{'role':'system','content':'actual system'},
                {'role':'user','content':user.read_text()}],'options':{'seed':42},'format':'json'}
            row={'status':'error','llm':'test','content':text,'response':response,'request':request,
                 'system_prompt':'actual system','user_prompt':user.read_text(),'error':'bad JSON'}
            with patch.object(report,'ROOT',root),patch.object(report,'OUT',root/'raw'),patch.object(report,'TARGET',root/'public'):
                published=report.public_record(row,{'path':'frozen.json','sha256':'frozen SHA'},user,system)
                restored=report.restore_public_record(published)
                self.assertEqual(restored['response'],response)
                self.assertEqual(restored['request'],request)
                self.assertEqual(restored['content'],text)
                self.assertEqual(restored['status'],'error')
                self.assertNotIn('content',published)
                user.write_text('changed observations')
                with self.assertRaises(AssertionError):report.restore_public_record(published)

    def test_context_override_preserves_frozen_description_and_sampling(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);captions=root/'frozen';output=root/'out';output.mkdir()
            source=captions/'frame-000001.json'
            write_json(source,{'stage':1,'frame_id':'f','model_name':'vlm','variant':{'id':'r1'},'status':'ok',
                'parsed':{'scene_description':'literal scene','objects':[{'id':1,'label':'chair',
                 'observation':'literal location','attachment_evidence':'not_visible','identifiable':True}]}})
            before=source.read_bytes();digest=sha256(before)
            config={'models':[{'name':'test','model':'test','backend':'ollama','base_url':'http://test'}],
                'timeout':10,'temperatures':[.3],'seeds':[43],'max_tokens':4096,'num_ctx':16384,
                'top_p':.8,'top_k':20,'presence_penalty':1.5}
            args=SimpleNamespace(captions=captions,output_dir=output,limit=None,resume=True,warmup=False,
                                 context_overrides={digest:{'num_ctx':32768}})
            with patch.object(stage2_reason,'Backend') as mock:
                mock.return_value.metadata.return_value={}
                mock.return_value.generate.return_value={'finish_reason':'stop','elapsed_seconds':.1,'content':json.dumps({'objects':[
                    {'id':1,'classification':'movable','reason':'can be repositioned'}]})}
                args.defer_caption_sha256={digest}
                self.assertEqual(stage2_reason.run(args,config),0)
                mock.return_value.generate.assert_not_called()
                self.assertFalse(list(output.glob('*/*/frame-*.json')))
                args.defer_caption_sha256=set()
                self.assertEqual(stage2_reason.run(args,config),0)
                opts=mock.return_value.generate.call_args.args[2]
                self.assertEqual(opts['num_ctx'],32768)
                self.assertEqual(opts['seed'],43)
                self.assertEqual(opts['temperature'],.3)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(config['num_ctx'],16384)

    def test_nvila_sequence_preserves_full_invalid_text_and_native_id(self):
        text='{"scene_description":"room"}\n{"objects":[{"id":9,"label":"chair","observation":"left"}]}\nextra unparsed words'
        parsed,partial=convert({'description':{'status':'error','content':text}},'nvila')
        self.assertTrue(partial)
        self.assertEqual(parsed['native_raw_text'],text)
        self.assertEqual(parsed['objects'][0]['id'],1)
        self.assertEqual(parsed['objects'][0]['source_object_id'],9)
        self.assertIn('left',parsed['objects'][0]['observation'])
        with self.assertRaises(ValueError):
            convert({'description':{'status':'error','content':'{"objects":[{"label":"bad "quote""}]}' }},'nvila')

    def test_florence_overlapping_regions_keep_separate_task_provenance(self):
        tasks={'caption':'<MORE_DETAILED_CAPTION>','od':'<OD>','dense':'<DENSE_REGION_CAPTION>'}
        rows={t:{'status':'ok','prompt':p,'content':'literal native output',
                 'parsed':{p:'full scene' if t=='caption' else {'labels':['chair'],'bboxes':[[0,0,10,10]]}}}
              for t,p in tasks.items()}
        parsed,partial=convert(rows,'florence')
        self.assertFalse(partial)
        self.assertEqual([o['id'] for o in parsed['objects']],[1,2])
        self.assertEqual([o['source_task'] for o in parsed['objects']],['<OD>','<DENSE_REGION_CAPTION>'])
        self.assertEqual([o['source_region_id'] for o in parsed['objects']],[1,1])
        self.assertIn('not confirmed distinct physical objects',parsed['adapter_notes'])

    def test_cross_sources_same_id_never_means_same_object(self):
        base={'frame_id':'same image','llm':'reasoner','prompt_name':'short','llm_repeat':'r1',
              'vlm_repeat':'r1','status':'ok','parsed':{'objects':[{'id':1,'classification':'movable','reason':'reason'}]}}
        a={**base,'input_index':1,'vlm':'a'};b={**base,'input_index':2,'vlm':'b'}
        frozen={1:{'document':{'parsed':{'objects':[{'id':1,'label':'chair'}]}}},
                2:{'document':{'parsed':{'objects':[{'id':1,'label':'wall'}]}}}}
        result=name_pair(a,b,frozen,'vlms')
        self.assertEqual(result['candidate_name_pairs'],0)
        self.assertIsNone(result['candidate_agreement'])
        self.assertIn('UNCONFIRMED',result['matching'])
        frozen[2]['document']['parsed']['objects'][0]['label']='Chair'
        self.assertEqual(name_pair(a,b,frozen,'vlms')['candidate_name_pairs'],1)
        frozen[2]['document']['parsed']['objects'].append({'id':2,'label':'chair'})
        b={**b,'parsed':{'objects':b['parsed']['objects']+[{'id':2,'classification':'movable','reason':'another region'}]}}
        self.assertEqual(name_pair(a,b,frozen,'vlms')['candidate_name_pairs'],0)

    def test_same_input_comparison_refuses_different_frozen_sha(self):
        base={'frame_id':'f','vlm':'a','vlm_repeat':'r1','input_index':1,'llm':'llm',
              'prompt_name':'short','llm_repeat':'r1','status':'ok','caption_sha256':'x',
              'parsed':{'objects':[{'id':1,'classification':'fixed','reason':'structure'}]}}
        self.assertFalse(direct_pair(base,{**base,'caption_sha256':'changed'},'llm-repeats')['comparable'])


if __name__=='__main__':unittest.main()
