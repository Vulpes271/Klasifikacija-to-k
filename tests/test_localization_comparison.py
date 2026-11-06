"""Shared spatial selection and cache guarantees for the three-way comparison."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from llm_object_classifier.localization_comparison import spatial_mask,differences
from llm_object_classifier.lilocbench_data import extract_rgb


class ComparisonTests(unittest.TestCase):
    def test_common_border_is_assigned_once_and_invalid_gt_is_excluded(self):
        points=np.array([[0,-4.4,0],[0,-5,0],[0,-4,0],[np.nan,-4,0],[2,-4,0]])
        corridor=spatial_mask(points,[-.8,-4.4,1.1,-3.4])
        office=spatial_mask(points,[-.8,-7.8,1.1,-4.4])
        self.assertEqual(corridor.tolist(),[True,False,True,False,False])
        self.assertEqual(office.tolist(),[False,True,False,False,False])
        self.assertFalse((corridor&office).any())

    def test_delta_signs_distinguish_success_increase_from_error_increase(self):
        c={'success_percent':97.,'position_rmse_cm':4.,'yaw_rmse_deg':2.2}
        b={'success_percent':95.,'position_rmse_cm':4.1,'yaw_rmse_deg':2.}
        delta=differences(c,b)
        self.assertEqual(delta['success_percentage_points'],2.)
        self.assertAlmostEqual(delta['position_rmse_cm'],-.1)
        self.assertAlmostEqual(delta['yaw_rmse_deg'],.2)

    def test_exact_cached_rgb_reuse_requires_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp);image=data/'static_0/camera_front/color/images/123.456000000.png'
            image.parent.mkdir(parents=True);image.write_bytes(b'already-cached')
            with patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected network request')):
                self.assertEqual(extract_rgb(data,'static_0',[123.456]),[image])


if __name__=='__main__':unittest.main()
