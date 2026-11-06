"""Numerical safeguards: timestamp gaps, calibrated projection and missing-pose scoring."""
import unittest
from pathlib import Path
import numpy as np
from llm_object_classifier.localization_pilot import interpolate,project_scan,evaluate


class LocalizationPilotTests(unittest.TestCase):
    def test_interpolation_does_not_extrapolate_or_bridge_gt_gaps(self):
        times=np.array([0.,.05,1.]);poses=np.array([[0,0,0],[.05,0,0],[1,0,0]])
        result,valid=interpolate(times,poses,[-.01,.025,.5,1.01],.15)
        self.assertEqual(valid.tolist(),[False,True,False,False])
        self.assertAlmostEqual(result[1,0],.025)
        self.assertTrue(np.isnan(result[[0,2,3]]).all())

    def test_official_optical_axes_and_fov_preserve_invalid_rays(self):
        # Camera optical z is robot x; optical x is -robot y; optical y is -robot z.
        camera=np.eye(4);camera[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
        laser={'range_min':.05,'range_max':100.,'angle_min':0.,'angle_increment':np.pi,'time_increment':0.}
        intrinsics={'distortion_model':'plumb_bob','distortion_coefficients':[0]*5,
                    'K':[100,0,320,0,100,240,0,0,1],'width':640,'height':480}
        uv,visible=project_scan(np.array([2.,2.,np.nan]),.05,.05,laser,np.eye(4),camera,intrinsics,
                                np.array([0.,.1]),np.zeros((2,3)),.15)
        self.assertEqual(visible.tolist(),[True,False,False])
        np.testing.assert_allclose(uv[0],[320,240])

    def test_wheel_odometry_compensates_lateral_camera_motion(self):
        camera=np.eye(4);camera[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
        laser={'range_min':.05,'range_max':100.,'angle_min':0.,'angle_increment':0.,'time_increment':0.}
        intrinsics={'distortion_model':'plumb_bob','distortion_coefficients':[0]*5,
                    'K':[100,0,320,0,100,240,0,0,1],'width':640,'height':480}
        uv,visible=project_scan(np.array([2.]),0.,.1,laser,np.eye(4),camera,intrinsics,
                                np.array([0.,.1]),np.array([[0,0,0],[0,1,0]]),.15)
        self.assertTrue(visible[0]);np.testing.assert_allclose(uv[0],[370,240])

    def test_missing_pose_is_failure_not_removed_from_denominator(self):
        from unittest.mock import patch
        gt=np.zeros((4,3));est=gt.copy();est[1]=np.nan;est[2,0]=.2
        # Fourth timestamp has no GT and must be counted separately.
        with patch('llm_object_classifier.localization_pilot.official_errors',
                   return_value=(np.array([0.,.2]),np.array([0.,0.]))):
            summary,_,_,success=evaluate(gt,est,np.array([True,True,True,False]),Path('unused'))
        self.assertEqual(summary['evaluable_poses'],3)
        self.assertEqual(summary['missing_estimates'],1)
        self.assertEqual(summary['times_without_gt'],1)
        self.assertEqual(summary['successful_poses'],1)
        self.assertAlmostEqual(summary['success_percent'],100/3)
        self.assertEqual(success.tolist(),[True,False,False,False])


if __name__=='__main__':unittest.main()
