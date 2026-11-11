"""Evidence omitted by the mapping filter must not carve phantom free space."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from llm_object_classifier.localization_filtered_map import ray_library,occupancy,aggregate,beam_world


class FilteredMapTests(unittest.TestCase):
    def test_removed_ray_never_updates_endpoint_or_space_behind(self):
        with tempfile.TemporaryDirectory() as folder:
            fun=ray_library(Path(folder));log=np.zeros((3,10),np.float32);known=np.zeros(log.shape,np.uint8)
            origins=np.array([[1,1]],np.int32);ends=np.array([[5,1]],np.int32)
            fun(log,known,10,3,origins,ends,np.array([0],np.uint8),1,.85,-.4,5.)
            self.assertFalse(known.any());self.assertFalse(log.any())
            fun(log,known,10,3,origins,ends,np.array([1],np.uint8),1,.85,-.4,5.)
            self.assertTrue(np.all(log[1,1:5]<0));self.assertGreater(log[1,5],0)
            self.assertFalse(known[1,6:].any())

    def test_one_update_per_cell_per_scan_and_endpoint_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            fun=ray_library(Path(folder));log=np.zeros((3,10),np.float32);known=np.zeros(log.shape,np.uint8)
            origins=np.array([[1,1],[1,1],[1,1]],np.int32);ends=np.array([[5,1],[5,1],[8,1]],np.int32)
            fun(log,known,10,3,origins,ends,np.ones(3,np.uint8),3,.85,-.4,5.)
            self.assertAlmostEqual(float(log[1,2]),-.4,places=6)
            self.assertAlmostEqual(float(log[1,5]),.85,places=6)

    def test_missing_mapping_gt_is_identical_invalid_geometry(self):
        laser={'angle_min':0,'angle_increment':.1,'time_increment':.01,'range_min':.05,'range_max':100}
        _,_,valid=beam_world(np.array([2.,3.]),0,laser,np.eye(4),np.array([1.,2.]),np.zeros((2,3)),.15)
        self.assertFalse(valid.any())

    def test_unknown_evidence_does_not_become_free(self):
        grid=occupancy(np.array([[-5.,5.,0.]],np.float32),np.array([[0,1,1]],np.uint8),{'free_thresh':.196,'occupied_thresh':.65})
        self.assertEqual(grid.tolist(),[[-1,100,-1]])

    def test_paired_std_uses_differences_not_difference_of_stds(self):
        keys=['position_rmse_cm','position_max_cm','position_below_5cm_percent','position_below_10cm_percent','yaw_rmse_deg','yaw_max_deg','yaw_below_2deg_percent','yaw_below_4deg_percent','success_percent']
        rows=[dict.fromkeys(keys,x) for x in [-2.,2.]];out=aggregate(rows)
        self.assertEqual(out['success_percent'],{'mean':0.,'std':2.})


class GalleryColorsTests(unittest.TestCase):
    def test_protected_masks_stay_green_or_gray_inside_movable_overlay(self):
        from PIL import Image
        from llm_object_classifier.box_segmentation import digest
        from llm_object_classifier.filtered_map_review import overlay_image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);Image.new('RGB',(90,60),(255,255,255)).save(root/'rgb.png');regions=[]
            for i,(label,left,right) in enumerate([('movable',0,90),('fixed',0,30),('uncertain',30,60)]):
                binary=np.zeros((60,90),np.uint8);binary[:,left:right]=255;path=root/f'{i}.png';Image.fromarray(binary).save(path)
                regions.append({'source_ids':[i],'label':'test','bbox_xyxy':[left,0,right-1,59],
                    'mask_path':path.name,'mask_sha256':digest(path),
                    'sources':[{'llm':{'models':{'qwen':{'classification':label}}}}]})
            item={'path':str(root/'rgb.png'),'mask_report':str(root/'segmentation.json')}
            overlay_image(item,{'regions':regions,'image_size':[90,60]},'qwen',root/'overlay.png')
            result=np.asarray(Image.open(root/'overlay.png'))
            green=result[40,15];gray=result[40,45];red=result[40,75]
            self.assertGreater(green[1],green[0]);self.assertEqual(int(gray[0]),int(gray[1]));self.assertGreater(red[0],red[1])


if __name__=='__main__':unittest.main()
