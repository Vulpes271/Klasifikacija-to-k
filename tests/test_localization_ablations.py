import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from maintenance.run_localization_ablations import ray_penalties,ParallelTracker,loss_episodes


def outcome(accepted,pose=None,score=.8):
    return dict(accepted=accepted,pose=pose,reason='accepted' if accepted else 'ambiguous',score=score,margin=.1 if accepted else 0.,matched=700,inconsistent=100,unknown=11)


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('configs/lilocbench-ablations.json').read_text())
        self.base=json.loads(Path('configs/lilocbench-tracking.json').read_text())
        self.confirm=json.loads(Path('configs/lilocbench-relocalization.json').read_text())['confirmation']

    def engine(self,grid):return SimpleNamespace(grid=grid,meta=dict(resolution=.1,origin=[0.,0.,0.]))

    def test_early_occupied_cell_is_penalized_with_a_bound(self):
        grid=np.zeros((10,40),np.int8);grid[0,10]=100
        values,audit=ray_penalties(self.engine(grid),np.zeros((1,3)),np.array([[2.05,.05]]),[.35,.05],self.cfg['rays'])
        self.assertGreater(values[0],0);self.assertLessEqual(values[0],.2);self.assertEqual(audit[0]['conflicts'],1)

    def test_unknown_neutral_endpoint_tolerance_and_real_sensor_origin(self):
        for x in [None,19,1]:
            grid=np.full((10,40),-1,np.int8)
            if x is not None:grid[0,x]=100
            values,_=ray_penalties(self.engine(grid),np.zeros((1,3)),np.array([[2.05,.05]]),[.35,.05],self.cfg['rays'])
            self.assertEqual(values[0],0.,x)

    def make_tracker(self):
        matcher=Mock();matcher.match.return_value=outcome(True,[10.,0.,0.])
        tracker=ParallelTracker(matcher,self.base,np.zeros((1,3)),self.confirm,self.cfg['parallel'])
        tracker.pose=np.zeros(3);tracker.last_lidar=0.;tracker.local.match=Mock(return_value=outcome(False))
        first=tracker.step(3.1,np.zeros((811,2)),np.zeros(3));self.assertTrue(first['pending'])
        return tracker

    def test_recovered_local_tracking_cancels_remote_without_an_advantage(self):
        tracker=self.make_tracker()
        tracker.local.match=Mock(side_effect=lambda p,pts:outcome(True,p.copy(),.8 if p[0]<1 else .79))
        result=tracker.step(3.167,np.zeros((811,2)),np.array([.02,0,0]))
        self.assertIsNone(tracker.pending);self.assertEqual(result['confirmation_event'],'cancelled_local_recovered')
        np.testing.assert_allclose(result['pose'],[.02,0,0]);self.assertFalse(result['prediction_only'])
        self.assertFalse(result['uncertain']);self.assertEqual(tracker.events[0]['status'],'cancelled_local_recovered')

    def test_remote_needs_old_confirmation_and_consecutive_advantage(self):
        tracker=self.make_tracker()
        tracker.local.match=Mock(side_effect=lambda p,pts:outcome(True,p.copy(),.4 if p[0]<1 else .8))
        for i in range(1,5):
            result=tracker.step(3.1+i*.067,np.zeros((811,2)),np.array([.01,0,0]))
            if i<4:self.assertLess(result['pose'][0],1)
        self.assertEqual(result['state'],'confirmed_global');np.testing.assert_allclose(result['pose'],[10.04,0,0])
        self.assertGreaterEqual(tracker.comparisons[-1]['advantage_hits'],3)
        self.assertEqual(tracker.events[0]['scans_checked'],5)

    def test_same_odometry_moves_both_hypotheses(self):
        tracker=self.make_tracker()
        tracker.local.match=Mock(side_effect=lambda p,pts:outcome(p[0]>=1,p.copy(),.7 if p[0]<1 else .71))
        tracker.step(3.167,np.zeros((811,2)),np.array([.03,.02,.01]))
        record=tracker.comparisons[-1]
        self.assertAlmostEqual(record['local_x'],.03);self.assertAlmostEqual(record['remote_x'],10.03)
        self.assertAlmostEqual(record['local_y'],.02);self.assertAlmostEqual(record['remote_y'],.02)
        self.assertEqual(record['advantage_hits'],0);self.assertIsNotNone(tracker.pending)

    def test_relocalization_episodes_require_stability_and_are_censored(self):
        times=np.arange(0,6,.5)
        rows=[dict(timestamp=float(t),has_gt=True,covered=True,success=t<.5 or t>=3.) for t in times]
        episodes=loss_episodes(rows,self.cfg['evaluation'])
        self.assertEqual(len(episodes),1);self.assertEqual(episodes[0]['status'],'recovered');self.assertEqual(episodes[0]['recovery_s'],2.5)
        for r in rows:
            if r['timestamp']>=2.:r['covered']=False
        episodes=loss_episodes(rows,self.cfg['evaluation']);self.assertEqual(episodes[0]['status'],'censored_boundary');self.assertIsNone(episodes[0]['recovery_s'])


if __name__=='__main__':unittest.main()
