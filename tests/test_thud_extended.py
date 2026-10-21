"""Native-task transport, references, matching and restart invariants."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm_object_classifier.thud_extended_common import frames, validate_boxes, check_saved
from llm_object_classifier.thud_extended_reason import florence_caption, nvila_caption
from llm_object_classifier.thud_box_metrics import match_regions, iou
from llm_object_classifier.experiment import write_json
from llm_object_classifier.thud_extended_report import stats


class ExtendedTests(unittest.TestCase):
    def test_latency_includes_failed_calls_and_distinguishes_memory_sources(self):
        values=stats([{'status':'ok','elapsed_seconds':1,'gpu_verified':True,
                       'peak_allocated_bytes':123,'peak_reserved_bytes':456},
                      {'status':'error','elapsed_seconds':9,'gpu_verified':True,
                       'gpu':{'size_vram':999}}])
        self.assertEqual(values['positions'],2)
        self.assertEqual(values['mean_s'],5)
        self.assertEqual(values['errors'],1)
        self.assertEqual(values['max_allocated_bytes'],123)
        self.assertEqual(values['max_reserved_bytes'],456)
        self.assertEqual(values['max_ollama_vram_bytes'],999)

    def test_same100_without_annotations(self):
        datasets = ('thud_dense_test50','thud_agtea_test50')
        audit = {ds: {'frames':[{'image_sha256':f'{ds}-{i}'} for i in range(50)],
                      'subset_indices_zero_based':list(range(0,50,5))} for ds in datasets}
        def fake_load(path, limit):
            ds = next(ds for ds in datasets if ds in str(path))
            return [{'record':{'frame_id':str(i),'image_path':f'{i}.png',
                               'image_sha256':f'{ds}-{i}','image_size':[10,10]},
                     'regions':None,'path':f'{i}.png','image_size':[10,10],
                     'image_sha256':f'{ds}-{i}'} for i in range(50)]
        with patch('llm_object_classifier.thud_extended_common.read_json', return_value=audit), \
             patch('llm_object_classifier.thud_extended_common.load_images', side_effect=fake_load):
            selected = frames()
        self.assertEqual(len(selected), 100)
        self.assertEqual(sum(f['subset20'] for f in selected), 20)
        self.assertEqual(len({f['image_sha256'] for f in selected}), 100)
        self.assertTrue(all('objects' not in f and 'annotation_path' not in f for f in selected))
        def leaking_load(path, limit):
            items=fake_load(path,limit)
            items[0]['record']['objects']=[{'label':'reference chair'}]
            return items
        with patch('llm_object_classifier.thud_extended_common.read_json', return_value=audit), \
             patch('llm_object_classifier.thud_extended_common.load_images', side_effect=leaking_load):
            with self.assertRaisesRegex(ValueError,'Annotation leak'):
                frames()

    def test_literal_florence_adapter_preserves_duplicate_regions(self):
        rows = {'caption': {'status':'ok','parsed':{'<MORE_DETAILED_CAPTION>':'Original caption.'}},
                'od': {'status':'ok','parsed':{'<OD>':{'labels':['chair'],'bboxes':[[1,2,3,4]]}}},
                'dense': {'status':'ok','parsed':{'<DENSE_REGION_CAPTION>':{'labels':['blue chair'],'bboxes':[[1,2,3,4]]}}}}
        caption = florence_caption(rows)
        self.assertEqual(caption['scene_description'], 'Original caption.')
        self.assertEqual([o['label'] for o in caption['objects']], ['chair','blue chair'])
        self.assertTrue(all(o['attachment_evidence']=='not_visible' for o in caption['objects']))
        self.assertNotIn('relations', caption)
        self.assertIn('[1, 2, 3, 4]', caption['objects'][1]['observation'])
        rows['od']['status']='error'
        with self.assertRaises(ValueError):
            florence_caption(rows)

    def test_boxes_not_clipped_or_silently_repaired(self):
        good={'<OD>':{'labels':['chair'],'bboxes':[[0,0,10,10]]}}
        validate_boxes(good,'<OD>',[10,10])
        for box in [[-1,0,10,10],[0,0,11,10],[3,2,1,4],[0,0,float('nan'),2]]:
            with self.assertRaises(ValueError):
                validate_boxes({'<OD>':{'labels':['chair'],'bboxes':[box]}},'<OD>',[10,10])

    def test_nvila_literal_transport_keeps_error_and_entire_raw_text(self):
        raw='{"objects":[{"id":9,"label":"shelf","appearance":"red"}]}\nEchoed instructions.'
        row={'status':'error','error':'Extra data','content':raw}
        caption,conversion=nvila_caption(row)
        self.assertEqual(row['status'],'error')
        self.assertEqual(caption['scene_description'],raw)
        self.assertEqual(caption['objects'][0]['label'],'shelf')
        self.assertIn('"appearance": "red"',caption['objects'][0]['observation'])
        self.assertEqual(caption['objects'][0]['attachment_evidence'],'not_visible')
        self.assertNotIn('relations',caption)
        self.assertIn('rejection retained',conversion)
        with self.assertRaises(ValueError):
            nvila_caption({'status':'error','content':'{"objects":[{"label":"a","label":"b"}]}'})

    def test_saved_errors_resume_and_changed_images_reject(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'row.json'
            write_json(path,{'status':'error','image_sha256':'abc'})
            self.assertTrue(check_saved(path,{'image_sha256':'abc'}))
            with self.assertRaises(ValueError):
                check_saved(path,{'image_sha256':'changed'})

    def test_one_to_one_semantics_and_group_exclusion(self):
        policy={'aliases':{'chair':'Chair'},'iou_threshold':.5}
        refs=[{'id':1,'label':'Chair','bbox_xyxy':[0,0,10,10]},
              {'id':2,'label':'Products','bbox_xyxy':[0,0,10,10]}]
        predictions=[{'label':'chair','bbox_xyxy':[0,0,10,10]}]*2
        result=match_regions(predictions,refs,policy)
        self.assertEqual(result['matched'],1)
        self.assertEqual(result['eligible_reference_regions'],1)
        self.assertEqual(result['excluded_reference_regions'],1)
        self.assertEqual(iou([0,0,10,10],[5,0,15,10]),1/3)
        self.assertEqual(match_regions([{'label':'chair near table','bbox_xyxy':[0,0,10,10]}],refs,policy)['matched'],0)

    def test_maximum_cardinality_recovers_greedy_collision(self):
        # P0 overlaps both refs, P1 only R0; greedy highest-IoU would lose a pair.
        policy={'aliases':{'chair':'Chair'},'iou_threshold':.3}
        refs=[{'id':1,'label':'Chair','bbox_xyxy':[0,0,10,10]},
              {'id':2,'label':'Chair','bbox_xyxy':[8,0,18,10]}]
        preds=[{'label':'chair','bbox_xyxy':[0,0,17,10]},
               {'label':'chair','bbox_xyxy':[0,0,8,10]}]
        self.assertEqual(match_regions(preds,refs,policy)['matched'],2)


if __name__ == '__main__':
    unittest.main()
