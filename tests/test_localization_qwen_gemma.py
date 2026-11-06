"""Only class metadata may differ when comparing classifiers on shared SAM geometry."""
import copy
import unittest
from llm_object_classifier.localization_qwen_gemma import reclassify


class ClassifierSwapTests(unittest.TestCase):
    def test_classifier_swap_preserves_original_masks_groups_and_classes(self):
        frame={'image_size':[640,480],'regions':[{'region_id':1,'source_ids':[1,4],
              'bbox_xyxy':[10,20,100,200],'mask_path':'masks/frame/region-0001.png','mask_sha256':'unchanged',
              'sources':[{'object':{'id':1},'llm':{'models':{'qwen':{'classification':'movable'}}}},
                         {'object':{'id':4},'llm':{'models':{'qwen':{'classification':'movable'}}}}]}]}
        before=copy.deepcopy(frame)
        gemma=reclassify(frame,{('int',1):{'classification':'fixed'}},'gemma')
        self.assertEqual(frame,before)
        for field in ['region_id','source_ids','bbox_xyxy','mask_path','mask_sha256']:
            self.assertEqual(gemma['regions'][0][field],frame['regions'][0][field])
        sources=gemma['regions'][0]['sources']
        self.assertEqual(sources[0]['llm']['models']['qwen']['classification'],'movable')
        self.assertEqual(sources[0]['llm']['models']['gemma']['classification'],'fixed')
        self.assertIsNone(sources[1]['llm']['models']['gemma']['classification'])


if __name__=='__main__':unittest.main()
