"""Začasni podatki in ponarejeni API odgovori; nobene modelne inference ali prave baze ocen."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from PIL import Image
from llm_object_classifier import pipeline as p
from llm_object_classifier.florence_runtime import run_florence,task_signature
from llm_object_classifier.grounding_common import CAPTION,GROUND
from llm_object_classifier.experiment import write_json,read_json,sha256
from llm_object_classifier.review_viewer import Results,Store,App


class PipelineTests(unittest.TestCase):
    def test_florence_eos_uses_tokenizer_when_outer_generation_config_is_empty(self):
        from llm_object_classifier.gpu_runtime import validate_eos
        self.assertEqual(validate_eos(2,None,2),[2])
        self.assertEqual(validate_eos(2,[2,3],9),[2,3])
        with self.assertRaises(ValueError):validate_eos(7,None,2)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.images=self.root/'input';self.images.mkdir();Image.new('RGB',(40,30),'white').save(self.images/'a.png')
        self.cfg=p.load_settings(p.ROOT/'configs/pipeline.json');self.cfg['llm']['base_url']='http://test.invalid'
        weights=self.root/'weights';weights.mkdir();(weights/'model.safetensors').write_bytes(b'not real weights')
        self.cfg['florence']['model_dir']=str(weights);self.output=self.root/'output'

    def saved_vision(self):
        run=p.prepare_run(copy.deepcopy(self.cfg),self.output,self.images);frame=run['frames'][0]
        cp=self.output/'native'/frame['frame_id']/'caption.json';gp=cp.with_name('grounding.json')
        text='The complete scene description. A chair next to a wall.'
        write_json(cp,{'status':'ok','call_id':'caption','parsed':{CAPTION:text},'signature':task_signature(run,frame,CAPTION)})
        write_json(gp,{'status':'ok','call_id':'ground','user_prompt':GROUND+text,
            'parsed':{GROUND:{'labels':['chair'],'bboxes':[[1,2,20,25]]}},'signature':task_signature(run,frame,GROUND,cp)})
        write_json(self.output/'florence-environment.json',{'test_only':True})
        return run,p.prepare_documents(self.output,run)

    def fake_backend(self):
        backend=Mock();backend.metadata.return_value={'show':{'template':'chat'},'server_version':{'version':'test'}}
        def generate(system,user,opts):
            text=json.dumps({'objects':[{'id':1,'classification':'movable','reason':'A transportable chair.'}]})
            return {'content':text,'finish_reason':'stop','gpu_verified':True,'elapsed_seconds':.1,
                'gpu':{'digest':'digest','size':10,'size_vram':10,'context_length':opts['num_ctx']},
                'response':{'done':True,'done_reason':'stop','message':{'content':text},'prompt_eval_count':400,'eval_count':20},
                'request':{'model':'qwen3.5:27b-q4_K_M','messages':[{'role':'system','content':system},{'role':'user','content':user}],
                    'options':opts,'format':'json','think':False}}
        backend.generate.side_effect=generate;return backend

    def test_default_and_exact_evidence_prompt(self):
        self.assertEqual(self.cfg['llm']['selected'],'qwen')
        self.assertEqual(self.cfg['llm']['alternatives']['qwen'],'qwen3.5:27b-q4_K_M')
        self.assertEqual(self.cfg['llm']['system_file_text'],(p.ROOT/'results models/prompt_comparison40/prompts/new-system.txt').read_text())

    def test_full_caption_regions_export_and_separate_review(self):
        run,manifest=self.saved_vision();summary=p.export_results(self.output)
        self.assertEqual(summary['missing_positions'],1)
        doc=read_json(self.output/manifest['inputs'][0]['frozen'])
        self.assertIn('The complete scene description.',doc['parsed']['scene_description'])
        self.assertEqual(doc['parsed']['objects'][0]['bbox_xyxy'],[1,2,20,25])
        results=Results(self.output);self.assertEqual(results.kind,'pipeline')
        store=Store(self.root/'ratings.sqlite3',results);app=App(results,store)
        store.rate(0,0,{'reference':'movable'})
        self.assertEqual(app.completion('reference')[1]['complete_regions'],1)
        again=Store(self.root/'ratings.sqlite3',Results(self.output))
        self.assertEqual(App(Results(self.output),again).completion('reference')[1]['complete_regions'],1)

    def test_complete_native_resume_never_imports_torch(self):
        self.saved_vision()
        with patch.dict('sys.modules',{'torch':None}):run_florence(self.output)

    def test_llm_resume_checks_signature_without_generation(self):
        run,manifest=self.saved_vision();backend=self.fake_backend()
        with patch.object(p,'Backend',return_value=backend),patch.object(p,'installed_digest',return_value='digest'),\
             patch.object(p,'load_model',return_value={'tokenizer_port':1234}) as load,\
             patch.object(p,'runner_http',return_value={'tokens':[1]*400}):
            p.run_text(self.output,run,manifest);p.run_text(self.output,run,manifest)
            self.assertEqual(backend.generate.call_count,1);self.assertEqual(load.call_count,1)
        summary=p.export_results(self.output);self.assertEqual(summary['schema_valid_response'],1)
        path=next((self.output/'responses/qwen').glob('*.json'));before=path.read_bytes()
        with patch.object(p,'Backend',return_value=backend),patch.object(p,'installed_digest',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'Spremenjen|Result cannot be reused'):p.run_text(self.output,run,manifest)
        self.assertEqual(before,path.read_bytes())

    def test_context_skip_and_configuration_mismatch_preserve_original(self):
        run,manifest=self.saved_vision();backend=self.fake_backend()
        with patch.object(p,'Backend',return_value=backend),patch.object(p,'installed_digest',return_value='digest'),\
             patch.object(p,'load_model',return_value={'tokenizer_port':1234}),patch.object(p,'runner_http',return_value={'tokens':[1]*9000}):
            p.run_text(self.output,run,manifest)
        backend.generate.assert_not_called();self.assertEqual(p.export_results(self.output)['skipped'],1)
        changed=copy.deepcopy(self.cfg);changed['llm']['temperature']=.8
        with self.assertRaisesRegex(ValueError,'Spremenjena konfiguracija'):p.prepare_run(changed,self.output)

    def test_changed_image_and_native_signature_are_rejected(self):
        run,manifest=self.saved_vision();frame=run['frames'][0];cp=self.output/'native'/frame['frame_id']/'caption.json'
        row=read_json(cp);row['signature']='changed';write_json(cp,row)
        with self.assertRaisesRegex(ValueError,'podpisa'):p.prepare_documents(self.output,run)
        (self.output/frame['image_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Spremenjena shranjena slika'):p.prepare_run(copy.deepcopy(self.cfg),self.output)

    def test_alternative_reuses_native_without_calling_florence(self):
        self.saved_vision();cfg=copy.deepcopy(self.cfg);cfg['llm']['selected']='gemma';other=self.root/'gemma'
        run=p.prepare_run(cfg,other,florence_from=self.output);manifest=p.prepare_documents(other,run)
        self.assertEqual(manifest['config']['models'][0]['model'],'gemma3:27b-it-q4_K_M')
        self.assertEqual(len(list((other/'native').glob('*/*.json'))),2)


if __name__=='__main__':unittest.main()
