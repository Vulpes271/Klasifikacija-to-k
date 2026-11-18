import json
from pathlib import Path
import unittest
import numpy as np
from maintenance.run_scan_to_map import CorrelativeMatcher, distinct, separated, summary


class ScanToMapTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('configs/lilocbench-scan-to-map.json').read_text())
        self.cfg['map']=dict(resolution=.1,origin=[-1.,-1.,0.],width=30,height=30)
        self.grid=np.full((30,30),-1,np.int8)
        self.grid[10,20]=100

    def test_se2_coordinates_and_yaw(self):
        matcher=CorrelativeMatcher(self.grid,self.cfg)
        x,y=matcher.transform(np.array([[1.,2.,np.pi/2]]),np.array([[2.,0.]]))
        np.testing.assert_allclose([x[0,0],y[0,0]],[1.,4.],atol=1e-12)

    def test_unknown_gets_no_support_even_beside_occupied(self):
        matcher=CorrelativeMatcher(self.grid,self.cfg)
        scores,labels=matcher.score(np.zeros((1,3)),np.array([[1.15,.05]]),categories=True)
        self.assertEqual(scores[0],0.)
        self.assertEqual(labels[0,0],2)

    def test_all_hits_normalize_and_free_penalty_is_bounded(self):
        self.grid[10,10]=0
        matcher=CorrelativeMatcher(self.grid,self.cfg)
        hit=np.array([[1.05,.05]])
        one=matcher.score(np.zeros((1,3)),hit)[0]
        both=matcher.score(np.zeros((1,3)),np.r_[hit,[[10.,10.]]])[0]
        self.assertAlmostEqual(one/2,both)
        free,labels=matcher.score(np.zeros((1,3)),np.array([[.05,.05]]),categories=True)
        self.assertGreaterEqual(free[0],-self.cfg['free_penalty'])
        self.assertEqual(labels[0,0],1)

    def test_spatial_modes_and_stable_ties(self):
        poses=np.array([[0.,0.,0.],[.01,0.,0.],[2.,0.,0.]])
        self.assertEqual(separated(poses,np.ones(3),2,.5,15).tolist(),[0,2])
        self.assertFalse(distinct(np.array([0,0,np.pi-.01]),np.array([0,0,-np.pi+.01]),.5,15))

    def test_route_denominator_excludes_room_extras(self):
        rows=[dict(variant='x',group=g,room=True,has_gt=True,has_estimate=True,success=s,accepted=True,processing_s=.1) for g,s in [('route',False),('room_extra',True)]]
        result=summary(rows,['x'],{'x':dict(map_load_s=0,distance_and_fields_s=0)})
        self.assertEqual((result[0]['n'],result[0]['N']),(0,1))
        self.assertEqual((result[1]['n'],result[1]['N']),(1,2))


if __name__=='__main__': unittest.main()
