from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from maintenance.run_temporal_cells import MOVABLE,FIXED,semantic_layer,accumulate_votes
from maintenance.run_reverse_short import sha


class TemporalCellTests(unittest.TestCase):
    def setUp(self):
        self.rule=dict(minimum_movable_epochs=3,minimum_epoch_gap_s=.5,minimum_distinct_cameras=2,minimum_fixed_epochs_to_protect=1)

    def test_beams_and_simultaneous_cameras_count_as_one_epoch(self):
        cells=np.zeros((1,4),np.int32);flags=np.ones((3,1,4),np.uint8)
        e=accumulate_votes(cells,flags,[0.],1,self.rule)
        self.assertEqual(e['counts'][0,0],1);self.assertFalse(e['removed_cells'][0])
        self.assertEqual(e['camera_bits'][0,0],7)

    def test_time_separated_epochs_and_two_views_are_required(self):
        cells=np.zeros((4,2),np.int32);flags=np.zeros((3,4,2),np.uint8)
        flags[0,:3]=MOVABLE;flags[1,3]=MOVABLE
        e=accumulate_votes(cells,flags,[0.,.1,.5,1.],1,self.rule)
        self.assertEqual(e['counts'][0,0],3);self.assertTrue(e['removed_cells'][0])
        self.assertEqual(e['camera_bits'][0,0],3)

    def test_one_camera_even_over_many_scans_is_insufficient(self):
        cells=np.zeros((5,1),np.int32);flags=np.zeros((3,5,1),np.uint8);flags[0]=MOVABLE
        e=accumulate_votes(cells,flags,np.arange(5),1,self.rule)
        self.assertEqual(e['counts'][0,0],5);self.assertFalse(e['removed_cells'][0])

    def test_one_late_fixed_observation_protects_confirmed_cell(self):
        cells=np.zeros((4,1),np.int32);flags=np.zeros((3,4,1),np.uint8)
        flags[0,:3]=MOVABLE;flags[1,1:3]=MOVABLE;flags[2,3]=FIXED
        e=accumulate_votes(cells,flags,[0.,.5,1.,1.1],1,self.rule)
        self.assertTrue(e['confirmed'][0]);self.assertTrue(e['protected'][0]);self.assertFalse(e['removed_cells'][0])

    def test_missing_unknown_and_invalid_world_endpoints_do_not_vote(self):
        cells=np.array([[-1,0],[-1,0]],np.int32);flags=np.zeros((3,2,2),np.uint8);flags[:,:,0]=3
        e=accumulate_votes(cells,flags,[0.,1.],1,self.rule)
        np.testing.assert_array_equal(e['counts'],np.zeros((2,1)));self.assertFalse(e['protected'][0])

    def test_explicit_fixed_is_distinct_from_uncertain_and_overlaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);regions=[]
            for j,(pixels,labels) in enumerate([([1,1,0,0],['movable']),([0,1,0,0],['fixed']),([1,0,1,0],['uncertain']),([0,0,0,1],['fixed','uncertain'])]):
                path=root/f'{j}.png';Image.fromarray(np.array([pixels],np.uint8)*255).save(path)
                regions.append(dict(mask_path=path.name,mask_sha256=sha(path),sources=[dict(llm=dict(models=dict(qwen=dict(classification=label)))) for label in labels]))
            layer=semantic_layer(root,dict(image_size=[4,1],regions=regions),'qwen')
            self.assertEqual(layer.tolist(),[[MOVABLE,MOVABLE|FIXED,0,0]])


if __name__=='__main__':unittest.main()
