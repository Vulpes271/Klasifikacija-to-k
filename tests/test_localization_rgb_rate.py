"""Temporal densification preserves originals; room/corridor partition is disjoint."""
import unittest
import numpy as np
from llm_object_classifier.localization_rgb_rate import added_grid_targets,partition


class SamplingRateTests(unittest.TestCase):
    def test_off_grid_originals_replace_slots_without_dropping_original_times(self):
        retained=[.19,.81]
        added=added_grid_targets(0.,1.,5.,retained)
        np.testing.assert_allclose(added,[0.,.4,.6])
        combined=sorted(retained+added)
        self.assertEqual(len(combined),5)
        self.assertIn(.19,combined);self.assertIn(.81,combined)

    def test_room_and_corridor_have_no_double_count_at_door_boundary(self):
        poses=np.array([[10,-5.01,0],[10,-5.,0],[10,-4.,0],[np.nan,np.nan,np.nan]])
        valid=np.array([True,True,True,False]);regions=partition(poses,valid,-5.)
        self.assertEqual(regions['room'].tolist(),[True,False,False,False])
        self.assertEqual(regions['corridor'].tolist(),[False,True,True,False])
        np.testing.assert_array_equal(regions['room']|regions['corridor'],valid)
        self.assertFalse((regions['room']&regions['corridor']).any())


if __name__=='__main__':unittest.main()
