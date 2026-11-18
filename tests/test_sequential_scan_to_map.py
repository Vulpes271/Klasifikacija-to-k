import json
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np
from maintenance.run_sequential_scan_to_map import Tracker, move, summarize
from maintenance.run_reverse_short import deltas


def outcome(accepted,pose=None):
    return dict(accepted=accepted,pose=pose,reason='accepted' if accepted else 'ambiguous',score=.8,margin=.1 if accepted else 0.,matched=700,inconsistent=100,unknown=11)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('configs/lilocbench-tracking.json').read_text())

    def test_robot_frame_odometry_composition(self):
        np.testing.assert_allclose(move(np.array([2.,3.,np.pi/2]),[1.,0.,np.pi/2]),[2.,4.,-np.pi],atol=1e-12)

    def test_ambiguity_predicts_then_long_loss_relocalizes(self):
        matcher=Mock();matcher.match.side_effect=[outcome(True,[1.,2.,0.]),outcome(True,[10.,20.,0.])]
        tracker=Tracker(matcher,self.cfg,np.zeros((1,3)))
        first=tracker.step(0.,np.zeros((811,2)),np.zeros(3))
        self.assertEqual(first['state'],'global_lidar')
        tracker.local.match=Mock(return_value=outcome(False))
        second=tracker.step(1.,np.zeros((811,2)),[.2,0,0])
        self.assertEqual(second['state'],'prediction_only');np.testing.assert_allclose(second['pose'],[1.2,2.,0.])
        self.assertEqual(matcher.match.call_count,1)
        third=tracker.step(3.1,np.zeros((811,2)),[.2,0,0])
        self.assertTrue(third['global_attempt']);self.assertFalse(third['prediction_only'])
        np.testing.assert_allclose(third['pose'],[10,20,0])

    def test_rejected_initialization_has_no_estimate_and_retry_is_throttled(self):
        matcher=Mock();matcher.match.return_value=outcome(False)
        tracker=Tracker(matcher,self.cfg,np.zeros((1,3)))
        self.assertIsNone(tracker.step(0.,np.zeros((811,2)),np.zeros(3))['pose'])
        self.assertIsNone(tracker.step(.1,np.zeros((811,2)),np.zeros(3))['pose'])
        self.assertEqual(matcher.match.call_count,1)
        tracker.step(3.,np.zeros((811,2)),np.zeros(3));self.assertEqual(matcher.match.call_count,2)

    def test_zero_prediction_does_not_read_measured_values(self):
        class Unreadable:
            def __len__(self):return 3
            def __getitem__(self,key):raise AssertionError('measured odometry read')
        np.testing.assert_array_equal(deltas(Unreadable(),False),np.zeros((3,3)))

    def test_missing_estimate_fails_but_is_excluded_from_error_percentiles(self):
        rows=[]
        for i,error in enumerate([5.,20.,None]):
            rows.append(dict(case='x',timestamp=float(i),scan_index=i,room=True,has_gt=True,has_estimate=error is not None,
                position_error_cm=error,yaw_error_deg=1. if error is not None else None,success=i==0,prediction_only=i==1,processing_s=.01,global_attempt=i==0))
        r=summarize(rows,self.cfg)[0]
        self.assertEqual((r['n'],r['N'],r['missing_estimates']),(1,3,1))
        self.assertEqual(r['median_position_cm'],12.5)
        self.assertAlmostEqual(r['prediction_only_percent'],100/3)


if __name__=='__main__':unittest.main()
