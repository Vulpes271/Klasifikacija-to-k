"""The publication must retain malformed answers and reject protocol drift."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_object_classifier import thud_stage1_presentation as report
from llm_object_classifier.thud_qwen3_vl import configuration


class StageOnePublicationTests(unittest.TestCase):
    def test_failed_rebuild_cannot_leave_previous_completeness_claim(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp)
            report.write_json(target / 'verification.json', {'complete': True})
            with patch.object(report, 'TARGET', target), \
                 patch.object(report, 'native_selection', side_effect=ValueError('Changed input')):
                with self.assertRaises(ValueError): report.main()
            self.assertFalse(json.loads((target / 'verification.json').read_text())['complete'])

    def test_reuse_audit_rejects_sampling_drift_and_changed_rgb_selection(self):
        selected = [{'frame_id': 'same RGB'}]
        manifests = {}
        for model in report.MODELS:
            for repeat, seed in report.SEEDS.items():
                if model == 'florence':
                    config = {'model': report.MODELS[model], 'revision': report.REVISIONS[model],
                              'repeat': repeat, 'seed': seed, 'tasks': report.TASKS,
                              'do_sample': False, 'num_beams': 3, 'max_new_tokens': 1024}
                else:
                    config = configuration(repeat, 'same instruction')
                    if model == 'nvila':
                        config.update(model=report.MODELS[model], revision=report.REVISIONS[model],
                                      system_prompt=config.pop('instruction'))
                manifests[(model, repeat)] = {'config': config, 'frames': selected}

        def read(path):
            return deepcopy(manifests[(path.parent.parent.name, path.parent.name)])

        with patch.object(report, 'read_json', side_effect=read):
            report.audit_protocol(selected)
            manifests[('nvila', 'r2')]['config']['temperature'] = .7
            with self.assertRaises(AssertionError): report.audit_protocol(selected)
            manifests[('nvila', 'r2')]['config']['temperature'] = .3
            manifests[('florence', 'r3')]['frames'] = [{'frame_id': 'other RGB'}]
            with self.assertRaises(ValueError): report.audit_protocol(selected)

    def test_collect_preserves_full_invalid_answer_and_has_no_borrowed_boxes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; target = root / 'published'
            image = root / 'image.png'; image.write_bytes(b'exact original RGB file bytes')
            frame = {'dataset': 'sample', 'index': 0, 'frame_id': 'sample/image-000001',
                     'source_frame_id': 'image-000001', 'image_path': str(image),
                     'image_sha256': report.sha256(image.read_bytes()), 'image_size': [960, 540]}
            config = configuration('r1', 'instruction')
            config['system_prompt'] = 'instruction'
            text = '{"scene_description":"full description"}\n{"objects":[{"id":1,"label":"chair"}]}\nUNPARSED TAIL'
            row = {**frame, 'model': report.MODELS['nvila'], 'repeat': 'r1', 'seed': 42,
                   'system_prompt': 'instruction', 'user_prompt': 'generic no annotations',
                   'generation_config': config, 'gpu_verified': True, 'call_id': 'original-call',
                   'finished_at_utc': '2026-10-26T12:00:00Z', 'status': 'error',
                   'content': text, 'error': 'Extra data', 'traceback': 'entire original failure'}
            path = report.saved_path(source / 'nvila/r1', frame, 'rich')
            report.write_json(path, row)
            with patch.object(report, 'ROOT', root), patch.object(report, 'OLD', source), \
                 patch.object(report, 'TARGET', target), patch.object(report, 'MODELS', {'nvila': row['model']}), \
                 patch.object(report, 'SEEDS', {'r1': 42}):
                records, sources, combined = report.collect([frame], {'nvila': {'r1': config}})
            published = target / sources[0]['published_path']
            self.assertEqual(published.read_bytes(), path.read_bytes())
            self.assertEqual(json.loads(published.read_text())['content'], text)
            self.assertEqual(json.loads(published.with_name('description.result.json').read_text())['view'], records[0]['view'])
            self.assertEqual(records[0]['status'], 'error')
            self.assertEqual(records[0]['view']['objects'], [{'id': 1, 'label': 'chair'}])
            self.assertTrue(sources[0]['reused_existing_generation'])
            self.assertFalse(list(target.rglob('*.jpg')))
            self.assertEqual(combined, [])


if __name__ == '__main__':
    unittest.main()
