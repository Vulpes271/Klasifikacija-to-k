import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from llm_object_classifier.localization_reverse import (fuse, nearest, propagate,
    classify_points, gap_plan, gap_metrics, decision_layer)


class ReverseTests(unittest.TestCase):
    def test_overlap_unique_and_missing_or_conflicting_camera_vetoes(self):
        decisions=np.array([[1,1,1,0,0],[1,-1,0,1,0],[0,0,-1,0,0]])
        self.assertEqual(fuse(decisions).tolist(),[True,False,False,True,False])
        self.assertEqual(int(fuse(decisions).sum()),2)

    def test_nearest_valid_requires_time_window_and_stable_tie(self):
        self.assertEqual(nearest(np.array([0.,2.]),1.,1.),0)
        self.assertIsNone(nearest(np.array([0.,2.]),1.,.05))
        self.assertIsNone(nearest(np.array([]),1.,1.))

    def test_odometry_is_composed_in_anchor_frame(self):
        pose=propagate([10.,20.,np.pi/2],[2.,3.,0.],[3.,3.,np.pi/4])
        np.testing.assert_allclose(pose,[10.,21.,3*np.pi/4])

    def test_geometry_free_unknown_occupied_and_match_are_disjoint(self):
        meta={'origin':[0,0,0],'resolution':1.,'width':4,'height':1}
        grid=np.array([[100,0,-1,100]],np.int8)
        points=np.array([[.11,.1],[1.1,.1],[2.1,.1],[3.1,.1],[5.,.1]])
        labels=classify_points(points,cKDTree([[.1,.1]]),grid,meta,.02)
        self.assertEqual(labels.tolist(),[0,1,2,3,2])

    def test_observed_ambiguous_cell_is_not_unobserved(self):
        meta={'origin':[0,0,0],'resolution':1.,'width':2,'height':1}
        labels=classify_points(np.array([[.5,.5],[1.5,.5]]),cKDTree([[10.,10.]]),
            np.array([[-1,-1]],np.int8),meta,.1,np.array([[1,0]],np.uint8))
        self.assertEqual(labels.tolist(),[3,2])

    def test_protected_region_vetoes_movable_mask_overlap(self):
        from PIL import Image
        from llm_object_classifier.box_segmentation import digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);regions=[]
            for j,(pixels,labels) in enumerate([([1,1,1,0],['movable']),([0,1,0,0],['fixed']),([0,0,1,0],['movable','uncertain'])]):
                path=root/f'{j}.png';Image.fromarray(np.array([pixels],np.uint8)*255).save(path)
                regions.append({'mask_path':path.name,'mask_sha256':digest(path),
                    'sources':[{'llm':{'models':{'qwen':{'classification':label}}}} for label in labels]})
            result=decision_layer(root,{'image_size':[4,1],'regions':regions},'qwen')
            self.assertEqual(result.tolist(),[[1,-1,-1,0]])

    def test_missing_estimate_fails_gap_and_prevents_recovery(self):
        times=np.arange(6,dtype=float);gt=np.zeros((6,3));est=gt.copy();est[2]=np.nan
        rows=gap_metrics(times,gt,gt,np.ones(6,bool),est,np.array([0,1,1,0,0,0],bool),
            np.array([0,1,1,0,0,0],bool),{'recovery_stable_s':1.})
        self.assertEqual(rows[0]['joint_success_percent'],50.)
        self.assertEqual(rows[0]['missing_estimates'],1)
        self.assertEqual(rows[0]['safe_prefix_duration_s'],1.)
        self.assertEqual(rows[0]['recovery_s'],0.)

    def test_natural_and_artificial_gaps_are_independent(self):
        ranges=np.ones((5,3));ranges[1,:]=np.nan
        count,natural,artificial=gap_plan(np.arange(5),ranges,{'range_min':.1,'range_max':10.},
            {'minimum_hits':2,'artificial_gaps_s':[[3.,1.]]})
        self.assertEqual(natural.tolist(),[False,True,False,False,False])
        self.assertEqual(artificial.tolist(),[False,False,False,True,False])

    def test_gap_distances_exclude_missing_gt_edges_and_failing_endpoints(self):
        from maintenance.finalize_reverse_report import distances
        gt=np.zeros((5,3));gt[:,0]=np.arange(5)*.1;gt[2]=np.nan
        observed=np.array([True,True,False,True,True])
        result=distances(gt,observed,observed,0,5)
        self.assertAlmostEqual(result['reference_distance_m'],.2)
        self.assertEqual(result['reference_distance_missing_edges'],2)
        self.assertAlmostEqual(result['max_safe_distance_m_gt'],.1)
        self.assertAlmostEqual(result['safe_prefix_distance_m_gt'],.1)


if __name__=='__main__':unittest.main()
