"""Rich graph invariants and real HTTP compatibility through both old stages."""
import contextlib
import copy
import io
import json
import unittest
import tempfile
from pathlib import Path
from argparse import Namespace
from tests import test_separate_stages as legacy_tests
from llm_object_classifier import stage2_reason
from llm_object_classifier.description_prompts import validate_description


def rich_description():
    def obj(oid, label):
        return {'id': oid, 'label': label, 'observation': 'Visible in the photograph.',
                'attachment_evidence': 'not_visible', 'identifiable': True,
                'appearance': 'wooden' if oid == 5 else 'grey surface',
                'image_location': 'lower left', 'occlusion': 'partial',
                'support_contacts': 'legs contact floor' if oid == 5 else 'unknown',
                'installation_evidence': 'unknown'}
    return {'schema_version': 'rich-description-v2', 'scene_description': 'A chair on a floor.',
            'objects': [obj(5, 'chair'), obj(6, 'floor')],
            'relations': [{'subject_id': 5, 'predicate': 'on', 'object_id': 6, 'evidence': 'Visible leg contact.'}]}


class RichDescriptionTests(unittest.TestCase):
    def test_graph_rejects_dangling_typed_self_and_duplicate_edges(self):
        valid = rich_description()
        self.assertIs(validate_description(valid, expected_schema='rich-v2'), valid)
        mutations = [lambda p: p['relations'][0].update(object_id=99),
                     lambda p: p['relations'][0].update(object_id='6'),
                     lambda p: p['relations'][0].update(object_id=5),
                     lambda p: p['relations'].append(copy.deepcopy(p['relations'][0])),
                     lambda p: p['relations'][0].update(evidence=''),
                     lambda p: p['objects'][0].update(bbox_xyxy=[0, 0, 1, 1]),
                     lambda p: p['objects'][0].update(classification='fixed'),
                     lambda p: p['objects'][0].pop('installation_evidence'),
                     lambda p: p.update(schema_version='other'),
                     lambda p: p.pop('schema_version')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                parsed = copy.deepcopy(valid)
                mutate(parsed)
                with self.assertRaises(ValueError):
                    validate_description(parsed, expected_schema='rich-v2')

    def fixture(self):
        fixture = legacy_tests.SeparateStageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_rich_evidence_survives_http_stage2_without_photos_and_resume(self):
        fixture = self.fixture()
        fixture.bad['model-a'] = json.dumps(rich_description())
        config = fixture.config(both=False)
        config['description_schema'] = 'rich-v2'
        rc, output = fixture.describe(config)
        self.assertEqual(rc, 0)
        self.assertIn('installation_evidence', (output/'comparison.html').read_text())
        for path in fixture.images.glob('*.jpg'):
            path.unlink()
        fixture.calls.clear()
        fixture.bad['model-a'] = json.dumps({'objects': [
            {'id': 5, 'classification': 'movable', 'reason': 'Repositionable.'},
            {'id': 6, 'classification': 'fixed', 'reason': 'Structural floor.'}]})
        args = Namespace(captions=output/'a/t0-s42', output_dir=fixture.root/'reason-rich', limit=None, resume=True)
        config = fixture.config(stage=2, both=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage2_reason.run(args, config), 0)
        calls = [body for path, body in fixture.calls if path == '/api/chat']
        self.assertEqual(len(calls), 2)
        for body in calls:
            self.assertTrue(all('images' not in m for m in body['messages']))
            payload = json.loads(body['messages'][-1]['content'].split('\n', 1)[1])
            self.assertEqual(payload['visual_observations'], rich_description())
        before = {p: p.read_bytes() for p in args.output_dir.glob('a/t*/frame-*.json')}
        fixture.calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage2_reason.run(args, config), 0)
        self.assertFalse(any(path == '/api/chat' for path, _ in fixture.calls))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(len((args.output_dir/'classifications.csv').read_text().splitlines()), 5)
        changed = json.loads((args.captions/'frame-000001.json').read_text())
        changed['parsed']['scene_description'] = 'Changed input.'
        (args.captions/'frame-000001.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'Resume input changed'):
            stage2_reason.run(args, config)

    def test_resume_retains_rejected_reasoning_response(self):
        fixture = self.fixture()
        _, output = fixture.describe(fixture.config(both=False))
        fixture.bad['model-a'] = '{"objects":[],"objects":[]}'
        args = Namespace(captions=output/'a/t0-s42', output_dir=fixture.root/'rejected', limit=None, resume=True)
        config = fixture.config(stage=2, both=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage2_reason.run(args, config), 2)
        fixture.calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage2_reason.run(args, config), 2)
        self.assertFalse(any(path == '/api/chat' for path, _ in fixture.calls))
        self.assertEqual(json.loads((args.output_dir/'summary.json').read_text())['models']['a']['error'], 2)

    def test_structure_counts_unknown_without_claiming_visual_correctness(self):
        from llm_object_classifier.thud_part2_report import structure
        result = structure([{'status':'ok','parsed':rich_description()}, {'status':'error'}])
        self.assertEqual(result['valid_descriptions'], 1)
        self.assertEqual(result['objects'], 2)
        self.assertEqual(result['relations'], 1)
        self.assertEqual(result['unknown_installation'], 2)
        self.assertEqual(result['unknown_support'], 1)

    def test_decision_name_pairing_excludes_duplicates_and_ignores_local_ids(self):
        from llm_object_classifier.thud_part2_report import reason_labels
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'caption.json'
            path.write_text(json.dumps({'parsed':{'objects':[{'id':1,'label':' Chair '},{'id':2,'label':'chair'},{'id':3,'label':' FLOOR '}]}}))
            row = {'status':'ok','caption_source':str(path),'parsed':{'objects':[
                {'id':1,'classification':'movable'}, {'id':2,'classification':'uncertain'}, {'id':3,'classification':'fixed'}]}}
            self.assertEqual(reason_labels(row), {'floor':{'id':3,'classification':'fixed'}})

    def test_pilot_reuse_preserves_errors_but_does_not_reuse_physical_warmup(self):
        from llm_object_classifier.thud_part2 import reuse_pilot
        from llm_object_classifier.experiment import write_json
        with tempfile.TemporaryDirectory() as directory:
            pilot = Path(directory)/'pilot-model-dataset'
            dest = Path(directory)/'rich-model-dataset'
            config = {'system_file_text':'frozen prompt'}
            write_json(pilot/'manifest.json', {'config':config})
            write_json(pilot/'complete.json', {'exit_code':2})
            write_json(pilot/'model/t0-s42/frame-000001.json', {'status':'error','content':'not JSON'})
            write_json(pilot/'model/warmup.json', {'excluded_from_quality':True})
            reuse_pilot(dest, config)
            self.assertEqual((dest/'model/t0-s42/frame-000001.json').read_bytes(), (pilot/'model/t0-s42/frame-000001.json').read_bytes())
            self.assertFalse((dest/'model/warmup.json').exists())
            self.assertTrue((pilot/'model/warmup.json').exists())
            reuse_pilot(dest, config)
            with self.assertRaisesRegex(ValueError, 'changed settings'):
                reuse_pilot(dest, {'system_file_text':'different prompt'})

    def test_new_group_warms_gpu_even_when_all_pilot_measurements_are_reused(self):
        from llm_object_classifier import stage1_describe
        from llm_object_classifier.thud_part2 import reuse_pilot
        from llm_object_classifier.experiment import write_json
        fixture = self.fixture()
        config = fixture.config(both=False)
        _, pilot = fixture.describe(config, name='pilot-model-data')
        write_json(pilot/'complete.json', {'exit_code':0})
        write_json(pilot/'a/warmup.json', {'excluded_from_quality':True})
        dest = fixture.root/'rich-model-data'
        reuse_pilot(dest, config)
        fixture.calls.clear()
        args = Namespace(input=fixture.input, output_dir=dest, limit=2, frame_id=None, resume=True, warmup=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage1_describe.run(args, config), 0)
        self.assertEqual(len([body for path,body in fixture.calls if path=='/api/chat']), 1)
        self.assertTrue((dest/'a/warmup.json').exists())
        self.assertEqual((dest/'a/t0-s42/frame-000001.json').read_bytes(), (pilot/'a/t0-s42/frame-000001.json').read_bytes())
