"""LOCO links: conservative labels, context freezing and arbitrary source count."""
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from llm_object_classifier import loco_report as report
from llm_object_classifier import thud_stage2_presentation as stage
from llm_object_classifier.thud_box_metrics import match_regions
from llm_object_classifier.experiment import write_json, sha256
from llm_object_classifier import loco_interruption as interruption


class LocoLinksTests(TestCase):
    def test_cancelled_call_has_no_fabricated_reply_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);public=root/'public';out=root/'work'
            source=root/'source.json';write_json(source,{'status':'ok','frame_id':'image',
                'parsed':{'scene_description':'A literal crate.','objects':[{'id':7,'label':'crate','observation':'on floor','attachment_evidence':'not_visible','identifiable':True}]}})
            digest=sha256(source.read_bytes())
            write_json(public/'frozen/index.json',[{'input_index':1,'path':'source.json','sha256':digest}])
            write_json(public/'execution-context-policy.json',{'overrides_by_frozen_sha256':{}})
            write_json(out/'llm/runs/local/long/r2/manifest.json',{'config':{
                'models':[{'name':'local','model':'local:4b'}],'temperatures':[.3],'seeds':[43],
                'max_tokens':4096,'num_ctx':16384,'top_p':.8,'top_k':20,'presence_penalty':1.5},
                'source_captions':[{'sha256':digest}],'system_prompt':'fixed/movable/uncertain',
                'instructions':'Use supplied IDs.','model_metadata':{'local':{'execution_url':'http://local'}}})
            evidence=root/'evidence.json';write_json(evidence,{'input_index':1,'model':'local','prompt':'long','llm_repeat':'r2',
                'server_http_status':500,'server_cancel_observed':True,'server_observed_http_seconds':2.0})
            with patch.object(interruption,'ROOT',root),patch.object(interruption,'OUT',out),patch.object(interruption,'LLM',public):
                path=interruption.record('local','long','r2',1,evidence)
                row=json.loads(path.read_text());self.assertNotIn('response',row);self.assertNotIn('elapsed_seconds',row)
                self.assertEqual(row['status'],'error');self.assertTrue(row['request_reconstructed_after_interruption'])
                self.assertIn('literal crate',row['request']['messages'][1]['content'])
                with self.assertRaisesRegex(ValueError,'overwritten'):interruption.record('local','long','r2',1,evidence)

    def test_runtime_contexts_distinguish_qwen_weight_blobs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);log=[]
            for spec,letter,context in zip(stage.MODELS,'abc',(32768,16384,65536)):
                blob='sha256-'+letter*64
                write_json(root/'llm/runs'/spec['name']/'short/r1/manifest.json',{
                    'model_metadata':{spec['name']:{'show':{'modelfile':'FROM /models/'+blob+'\n'}}}})
                log.extend(['starting llama-server --model /models/'+blob,
                    'general.architecture str = qwen35','n_seq_max = 1',
                    'new slot, n_ctx = '+str(context)])
            write_json(root/'llm/setup/server-start.json',{'at_utc':'2026-10-28T15:00:00Z'})
            with patch.object(report,'OUT',root),patch.object(report.subprocess,'check_output',side_effect=['\n'.join(log),'sha256:image']):
                report.capture_runtime()
            runtime=json.loads((root/'llm/setup/runtime-parallel.json').read_text())
            self.assertEqual(runtime['actual_model_slots_by_context'],{
                'qwen35-bf16':{'32768':1},'qwen35-q4':{'16384':1},'gemma3-q4':{'65536':1}})

    def test_explicit_cli_paths_are_absolute_and_do_not_change_legacy_defaults(self):
        with patch.object(stage,'SOURCE'),patch.object(stage,'TARGET'),patch.object(stage,'OUT'):
            stage.configure_paths(source='results/loco/source',target='results/loco/public',out='results/loco/work')
            self.assertEqual(stage.SOURCE,Path('results/loco/source').resolve())
            self.assertTrue(stage.TARGET.is_absolute() and stage.OUT.is_absolute())

    def test_mentions_do_not_count_pallet_truck_as_two_classes(self):
        aliases={'pallet':'pallet','pallet truck':'pallet_truck','forklift':'forklift'}
        self.assertEqual(report.literal_mentions('A pallet truck by a forklift and boxes',aliases),['forklift','pallet_truck'])
        self.assertEqual(report.literal_mentions('boxes and a shelf',aliases),[])

    def test_matching_is_one_to_one_and_unmapped_labels_are_separate(self):
        policy={'aliases':{'pallet':'pallet'},'iou_threshold':.5}
        refs=[{'id':77,'label':'pallet','bbox_xyxy':[0,0,10,10]},
              {'id':88,'label':'wall','bbox_xyxy':[0,0,10,10]}]
        preds=[{'label':'pallet','bbox_xyxy':[0,0,10,10]}]*2+[{'label':'wall','bbox_xyxy':[0,0,10,10]}]
        result=match_regions(preds,refs,policy)
        self.assertEqual((result['matched'],result['mapped_prediction_regions'],result['excluded_reference_regions']),(1,2,1))

    def test_group_uses_all_270_sources_and_separate_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_json(root/'public/frozen/index.json',[{}]*270)
            config={'models':[]}
            with patch.object(stage,'TARGET',root/'public'),patch.object(stage,'OUT',root/'work'),\
                 patch.object(stage,'configuration',return_value=config),patch.object(stage,'progress'),\
                 patch.object(stage,'reason',return_value=0) as run:
                stage.group({'name':'local'},'short','r2',pilot=True)
            args=run.call_args.args[0]
            self.assertEqual(args.limit,270);self.assertEqual(args.captions,root/'public/frozen')
            self.assertTrue(args.resume);self.assertFalse(args.warmup);self.assertEqual(args.max_new_calls,1)

    def test_context_policy_never_changes_frozen_observations(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);public=root/'llm';out=root/'work';(public/'prompts').mkdir(parents=True)
            (public/'prompts/reasoning.txt').write_text('Classify every supplied ID.')
            document={'scene_description':'literal '*5000,'objects':[{'id':'id-1','label':'pallet','observation':'on floor','attachment_evidence':'not_visible','identifiable':True}]}
            path=root/'source.json';write_json(path,{'parsed':document});digest=sha256(path.read_bytes())
            write_json(public/'frozen/index.json',[{'status':'ok','input_index':1,'path':'source.json','sha256':digest,'frame_id':'loco-image','vlm':'qwen3-vl','partial_input':False}])
            write_json(out/'setup/prior-results-sha.json',{})
            with patch.object(report,'ROOT',root),patch.object(report,'LLM',public),patch.object(report,'OUT',out),patch.object(report,'PUBLIC',root/'package'):report.context_policy()
            policy=json.loads((public/'execution-context-policy.json').read_text())
            self.assertEqual(policy['overrides_by_frozen_sha256'][digest],{'num_ctx':32768})
            self.assertEqual(policy['archived_failed_attempts'],[])
            self.assertEqual(sha256(path.read_bytes()),digest)
