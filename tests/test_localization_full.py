"""Full-sequence spatial intersection and correctly separated threshold comparisons."""
import unittest
import numpy as np
from llm_object_classifier.localization_full import visited_cells,cell_membership,full_delta


class FullLocalizationTests(unittest.TestCase):
    def test_shared_cells_use_valid_gt_not_estimates_or_nonvisited_areas(self):
        origin=np.array([-25.,-25.]);a=np.array([[0.1,0.1,0],[.6,.1,0],[10,10,0]])
        b=np.array([[.2,.2,0],[2,2,0],[10,10,0]])
        valid=np.array([True,True,False])
        common=visited_cells(a,valid,.5,origin)&visited_cells(b,valid,.5,origin)
        self.assertEqual(common,{(50,50)})
        self.assertEqual(cell_membership(a,valid,common,.5,origin).tolist(),[True,False,False])
        self.assertEqual(cell_membership(b,valid,common,.5,origin).tolist(),[True,False,False])

    def test_separate_threshold_deltas_are_not_joint_success(self):
        b={'position_below_10cm_percent':90.,'yaw_below_4deg_percent':85.,'success_percent':78.,'position_rmse_cm':4.,'yaw_rmse_deg':2.}
        c={**b,'position_below_10cm_percent':91.,'yaw_below_4deg_percent':83.,'success_percent':77.}
        d=full_delta(c,b)
        self.assertEqual(d['position_below_10cm_percentage_points'],1.)
        self.assertEqual(d['yaw_below_4deg_percentage_points'],-2.)
        self.assertEqual(d['joint_success_percentage_points'],-1.)


if __name__=='__main__':unittest.main()
