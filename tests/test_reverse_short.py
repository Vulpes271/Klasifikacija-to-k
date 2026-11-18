import unittest
import numpy as np
from maintenance.run_reverse_short import deltas, metrics_rows, Matcher


class ShortReverseTests(unittest.TestCase):
    def test_zero_mode_does_not_read_measurements(self):
        class Unreadable:
            def __len__(self): return 4
            def __getitem__(self, key): raise AssertionError('measured odometry accessed')
        np.testing.assert_array_equal(deltas(Unreadable(), False), np.zeros((4,3)))

    def test_relative_wheel_motion_in_previous_robot_frame(self):
        odom = np.array([[0,0,np.pi/2],[0,1,np.pi]])
        np.testing.assert_allclose(deltas(odom,True)[1],[1,0,np.pi/2],atol=1e-12)

    def test_missing_estimate_and_strict_thresholds(self):
        estimates=np.array([[np.nan]*3,[.1,0,0],[0,0,np.deg2rad(4)],[.09,0,np.deg2rad(3.9)]])
        rows=metrics_rows('test','x',np.arange(4),np.arange(4),np.zeros((4,3)),np.ones(4,bool),estimates,np.zeros(4),[-1,-1,1,1])
        self.assertEqual([r['success'] for r in rows],[False,False,False,True])
        self.assertFalse(rows[0]['has_estimate'])

    def test_likelihood_uses_map_frame_and_extrinsic_points(self):
        grid=np.zeros((30,30),np.int8);grid[10,20]=100
        matcher=Matcher(grid,dict(resolution=.1,origin=[-1,-1,0]))
        poses=np.array([[0,0,0],[1,0,0]])
        score=matcher.scores(poses,np.array([[.05,.05]]))
        self.assertLess(score[1],score[0])


if __name__=='__main__': unittest.main()
