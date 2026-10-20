"""Regressions for immutable THUD inputs, resume, and honest statistics."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from llm_object_classifier import stage1_describe
from llm_object_classifier.experiment import load_config, sha256
from llm_object_classifier.thud_metrics import aggregate, name_overlap, percentile
from llm_object_classifier.description_prompts import user_prompt
from llm_object_classifier.image_dataset import load_images

class ThudTests(unittest.TestCase):
    def test_failed_calls_are_not_removed_and_p90_is_interpolated(self):
        result=aggregate([{'status':'ok','elapsed_seconds':1,'parsed':{'objects':[]}},
                          {'status':'error','elapsed_seconds':3}, {'status':'error','error':'timed out'}])
        self.assertEqual(result['attempted'],3)
        self.assertEqual(result['json_valid_fraction'],1/3)
        self.assertEqual(result['timed_calls'],2)
        self.assertAlmostEqual(result['p90_s'],2.8)
        self.assertEqual(result['timeouts'],1)
        self.assertIsNone(percentile([], .9))

    def test_group_names_are_not_counted_as_instance_truth(self):
        result=name_overlap(['Chair','chair','products','unlabeled cup'],['chair','Products'])
        self.assertEqual(result['shared_unique_names'],2)
        self.assertNotIn('precision',result)
        self.assertEqual(result['unmatched_predicted_names'],['unlabeled cup'])

    def test_discovery_manifest_contains_no_annotation_metadata(self):
        from llm_object_classifier.thud_experiment import prepare, OUT, DATASETS
        # Exercise repository manifests, including SHA checks and real image bytes.
        for ds in DATASETS:
            path=OUT/'inputs'/f'{ds}-discovery-all.jsonl'
            if not path.exists():
                self.skipTest('Run --phase prepare first')
            frames=load_images(path,50)
            for frame in frames:
                self.assertEqual(set(frame['record']),{'frame_id','image_path','image_sha256','image_size'})
                self.assertIsNone(frame['regions'])
                payload=json.loads(user_prompt(frame).split('\n',1)[1])
                self.assertIsNone(payload['supplied_objects'])
                self.assertNotIn('annotation_path',json.dumps(payload))
            annotated=load_images(Path('data')/ds/'objects.jsonl',50)
            self.assertEqual([f['image_sha256'] for f in frames],[f['image_sha256'] for f in annotated])

    def test_resume_keeps_errors_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.new('RGB',(5,5)).save(root/'one.png')
            path=root/'input.jsonl'
            path.write_text(json.dumps({'frame_id':'one','image_path':'one.png'})+'\n')
            config_path=root/'config.json'
            config_path.write_text(json.dumps({'models':[{'name':'test','model':'fake'}]}))
            config=load_config(config_path,1)
            args=argparse.Namespace(input=path,limit=1,frame_id=None,output_dir=root/'run',resume=True,warmup=False)
            with patch('llm_object_classifier.stage1_describe.Backend') as backend,contextlib.redirect_stdout(io.StringIO()):
                backend.return_value.metadata.return_value={}
                backend.return_value.generate.return_value={'content':'{"x":1,"x":2}', 'elapsed_seconds':2,'finish_reason':'stop'}
                self.assertEqual(stage1_describe.run(args,config),2)
                raw=(root/'run/test/t0-s42/frame-000001.json').read_bytes()
                self.assertEqual(stage1_describe.run(args,config),2)
                self.assertEqual(backend.return_value.generate.call_count,1)
                self.assertEqual(raw,(root/'run/test/t0-s42/frame-000001.json').read_bytes())
                path.write_text(path.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'Resume input changed'):
                    stage1_describe.run(args,config)

    def test_partial_cpu_offload_is_rejected_and_raw_retained(self):
        from llm_object_classifier.vision_backends import Backend
        from llm_object_classifier.vision_pipeline import GPURequiredError
        spec={'backend':'ollama','base_url':'http://test','model':'fake','require_gpu':True}
        replies=[{'message':{'content':'{}'},'done_reason':'stop'},
                 {'models':[{'name':'fake','size':1000,'size_vram':500}]}]
        with patch('llm_object_classifier.vision_backends.request',side_effect=replies):
            with self.assertRaises(GPURequiredError) as caught:
                Backend(spec).generate('system','prompt',{})
        self.assertEqual(caught.exception.result['content'],'{}')
        self.assertIn('Partial CPU offload',str(caught.exception))

    def test_long_reasoning_prompts_only_remove_example_blocks(self):
        original=Path('configs/prompts/thud-long-examples.txt').read_text().strip()
        no_examples=Path('configs/prompts/thud-long-no-examples.txt').read_text().strip()
        before,rest=original.split('\nExamples:\n',1)
        _,rest=rest.split('\nVisual observations',1)
        tail,_=rest.split('\nExample input:',1)
        self.assertEqual(no_examples,before+'\nVisual observations'+tail)

    def test_http_failure_retains_full_body_and_logged_request(self):
        import urllib.error
        from llm_object_classifier.vision_backends import Backend
        error=urllib.error.HTTPError('http://test/api/chat',500,'failed',{},io.BytesIO(b'x'*5000))
        spec={'backend':'ollama','base_url':'http://test','model':'fake','require_gpu':True}
        with patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                Backend(spec).generate('system','prompt',{},b'private-image-bytes')
        result=caught.exception.result
        self.assertEqual(result['response_text'],'x'*5000)
        self.assertEqual(result['http_status'],500)
        self.assertIn('omitted',result['request']['messages'][1]['images'][0])
        self.assertGreaterEqual(result['elapsed_seconds'],0)
