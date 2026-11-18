import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from maintenance.run_confirmed_relocalization import ConfirmingTracker,summarize


def outcome(accepted,pose=None):
    return dict(accepted=accepted,pose=pose,reason='accepted' if accepted else 'ambiguous',score=.8,margin=.1 if accepted else 0.,matched=700,inconsistent=100,unknown=11)


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('configs/lilocbench-tracking.json').read_text())
        self.confirm=json.loads(Path('configs/lilocbench-relocalization.json').read_text())['confirmation']
        self.matcher=Mock();self.matcher.match.return_value=outcome(True,[10.,0.,0.])
        self.tracker=ConfirmingTracker(self.matcher,self.cfg,np.zeros((1,3)),self.confirm)
        self.pts=np.zeros((811,2))

    def test_initialization_requires_five_consecutive_odometry_linked_scans(self):
        first=self.tracker.step(0.,self.pts,np.zeros(3));self.assertIsNone(first['pose']);self.assertTrue(first['pending'])
        self.tracker.local.match=Mock(side_effect=lambda p,pts:outcome(True,p.copy()))
        for i in range(1,5):
            result=self.tracker.step(i*.067,self.pts,np.array([.02,0,0]))
            if i<4:self.assertIsNone(result['pose'])
        np.testing.assert_allclose(result['pose'],[10.08,0,0])
        self.assertEqual(result['state'],'confirmed_global');self.assertFalse(result['uncertain'])
        self.assertEqual(self.tracker.events[0]['scans_checked'],5)
        self.assertAlmostEqual(result['confirmation_latency_s'],.268)

    def test_distant_proposal_keeps_uncertain_old_prediction_until_confirmation(self):
        self.tracker.pose=np.array([0.,0.,0.]);self.tracker.last_lidar=0.
        self.tracker.local.match=Mock(return_value=outcome(False))
        result=self.tracker.step(3.1,self.pts,np.array([.1,0,0]))
        np.testing.assert_allclose(result['pose'],[.1,0,0]);self.assertTrue(result['uncertain']);self.assertTrue(result['prediction_only'])
        self.tracker.local.match=Mock(side_effect=lambda p,pts:outcome(True,p+[.5,0,0]))
        rejected=self.tracker.step(3.167,self.pts,np.array([.1,0,0]))
        self.assertEqual(rejected['confirmation_event'],'rejected');np.testing.assert_allclose(rejected['pose'],[.2,0,0])
        self.assertIsNone(self.tracker.pending);self.assertEqual(self.tracker.events[0]['status'],'rejected')

    def test_ambiguous_scan_breaks_consecutive_confirmation(self):
        self.tracker.step(0.,self.pts,np.zeros(3));self.tracker.local.match=Mock(return_value=outcome(False))
        result=self.tracker.step(.067,self.pts,np.zeros(3))
        self.assertIsNone(result['pose']);self.assertFalse(result['pending']);self.assertEqual(result['confirmation_event'],'rejected')

    def test_unmapped_outlier_and_boundary_jump_do_not_enter_primary_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'full').mkdir();(Path(tmp)/'full/three-events.json').write_text('[]')
            rows=[]
            for i,(covered,error) in enumerate([(True,5.),(False,10000.),(True,20.)]):
                rows.append(dict(case='three-immediate',has_gt=True,has_estimate=True,room=True,covered=covered,success=error<10,
                    confident_success=error<10,position_error_cm=error,yaw_error_deg=1.,timestamp=float(i),covered_jump_gt_pair=False,jump_m=100.,uncertain=False,processing_s=.01))
            stats,_=summarize(rows,dict(variants=['three'],cache=tmp),'full')
            self.assertEqual((stats[0]['n'],stats[0]['N'],stats[0]['excluded_unmapped']),(1,2,1))
            self.assertLess(stats[0]['p95_position_cm'],20.);self.assertEqual(stats[0]['jumps_over_1m'],0)


if __name__=='__main__':unittest.main()
