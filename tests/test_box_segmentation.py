"""Identity, overlap and provenance guarantees for optional region filtering."""
import unittest
from llm_object_classifier.box_segmentation import filter_regions, near_equal


def obj(i,label,box):
    return {'id':i,'label':label,'bbox_xyxy':box,'source_region_id':i}


class BoxFilteringTests(unittest.TestCase):
    def run_filter(self,objects,scene):
        decisions={o['id']:{'classification':'fixed' if o['id']==1 else 'movable'} for o in objects}
        return filter_regions(objects,scene,decisions)

    def test_duplicate_box_preserves_disagreeing_classes_and_ids(self):
        originals=[obj(1,'box',[10,10,110,110]),obj(2,'The box',[10.5,10,110,110])]
        regions=self.run_filter(originals,'A cardboard box is in front of the wall.')
        self.assertEqual(len(regions),1)
        self.assertEqual(regions[0]['source_ids'],[1,2])
        self.assertEqual([s['llm']['classification'] for s in regions[0]['sources']],['fixed','movable'])
        self.assertEqual([s['object'] for s in regions[0]['sources']],originals)
        self.assertEqual(regions[0]['bbox_xyxy'],originals[0]['bbox_xyxy'])

    def test_wall_and_box_remain_separate_even_with_identical_boxes(self):
        regions=self.run_filter([obj(1,'box',[0,0,100,100]),obj(2,'wall',[0,0,100,100])],
                                'A box stands in front of a wall.')
        self.assertEqual(len(regions),2)

    def test_disagreeing_boxes_and_containment_are_not_merged(self):
        regions=self.run_filter([obj(1,'box',[0,0,100,100]),obj(2,'box',[0,0,90,100]),
                                 obj(3,'box',[30,30,70,70])],'A box is visible.')
        self.assertEqual(len(regions),3)

    def test_no_drifting_chain(self):
        regions=self.run_filter([obj(1,'box',[0,0,100,100]),obj(2,'box',[1.5,0,101.5,100]),
                                 obj(3,'box',[3,0,103,100])],'A box is visible.')
        self.assertEqual([r['source_ids'] for r in regions],[[1,2],[3]])

    def test_box_anaphora_requires_explicit_unique_caption_reference(self):
        objects=[obj(1,'a cardboard box',[0,0,100,100]),obj(2,'The box',[0,0,100,100])]
        self.assertEqual(len(self.run_filter(objects,'A cardboard box is by the wall. The box is empty.')),1)
        self.assertEqual(len(self.run_filter(objects,'A cardboard box is by the wall.')),2)
        self.assertEqual(len(self.run_filter(objects,'A cardboard box and a white box. The box is empty.')),2)

    def test_caption_evidence_required(self):
        self.assertEqual(len(self.run_filter([obj(1,'box',[0,0,100,100]),obj(2,'box',[0,0,100,100])],
                                              'A room is visible.')),2)

    def test_logo_relationship_needs_description_and_unique_parent(self):
        objects=[obj(1,'box',[0,0,100,100]),obj(2,'logo',[20,20,40,40])]
        self.assertNotIn('part_of',self.run_filter(objects,'A box and a logo are visible.')[1])
        regions=self.run_filter(objects,'A cardboard box with the logo of "Brand.de" on it.')
        self.assertEqual(len(regions),2)
        self.assertEqual(regions[1]['part_of']['source_ids'],[1])
        objects.append(obj(3,'cardboard box',[0,0,100,100]))
        self.assertNotIn('part_of',self.run_filter(objects,'A box with the logo on it.')[1])


if __name__=='__main__':unittest.main()
