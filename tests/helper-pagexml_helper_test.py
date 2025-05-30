import unittest
from typing import List, Tuple

import pagexml.helper.pagexml_helper as helper
import pagexml.model.pagexml_document_model as page_dm
import pagexml.model.physical_document_model as pdm
from pagexml.parser import parse_pagexml_file


def make_region(points: List[Tuple[int, int]], doc_id: str = 'doc') -> page_dm.PageXMLTextRegion:
    coords = pdm.Coords(points)
    return page_dm.PageXMLTextRegion(doc_id=doc_id, coords=coords)


class TestRegionType(unittest.TestCase):

    def test_point(self):
        tr = make_region([(1, 1)])
        self.assertEqual(helper.RegionType.POINT, helper.get_region_type(tr))

    def test_hline(self):
        tr = make_region([(1, 1), (2, 1)])
        self.assertEqual(helper.RegionType.HLINE, helper.get_region_type(tr))

    def test_vline(self):
        tr = make_region([(1, 1), (1, 2)])
        self.assertEqual(helper.RegionType.VLINE, helper.get_region_type(tr))

    def test_box(self):
        tr = make_region([(1, 1), (2, 2)])
        self.assertEqual(helper.RegionType.BOX, helper.get_region_type(tr))


class TestPageXMLHelper(unittest.TestCase):

    def setUp(self) -> None:
        no_coords = None
        point_coords1 = pdm.Coords([(1, 1)])
        point_coords2 = pdm.Coords([(1, 1)])
        point_coords3 = pdm.Coords([(2, 2)])
        hline_coords1 = pdm.Coords([(0, 0), (10, 0)])
        hline_coords2 = pdm.Coords([(5, 0), (15, 0)])
        hline_coords3 = pdm.Coords([(0, 5), (10, 5)])
        vline_coords1 = pdm.Coords([(0, 0), (0, 10)])
        vline_coords2 = pdm.Coords([(0, 5), (0, 15)])
        vline_coords3 = pdm.Coords([(5, 0), (5, 10)])
        self.no_coords_region = page_dm.PageXMLTextRegion(doc_id='no_coords')
        self.point_coords_region1 = page_dm.PageXMLTextRegion(doc_id='point_coords1', coords=point_coords1)
        self.page_file = 'data/example.xml'
        self.page_doc = parse_pagexml_file(self.page_file)

    def test_element_overlap_no_coords(self):
        tr1 = make_region([(1, 1)])
        tr2 = page_dm.PageXMLTextRegion(doc_id='no_coords')
        self.assertEqual(False, helper.regions_overlap(tr1, tr2))
        self.assertEqual(False, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_same_points(self):
        tr1 = make_region([(1, 1)])
        tr2 = make_region([(1, 1)])
        self.assertEqual(True, helper.regions_overlap(tr1, tr2))
        self.assertEqual(True, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_different_points(self):
        tr1 = make_region([(1, 1)])
        tr2 = make_region([(1, 2)])
        self.assertEqual(False, helper.regions_overlap(tr1, tr2))
        self.assertEqual(False, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_point_on_horizontal_line(self):
        tr1 = make_region([(5, 1)])
        tr2 = make_region([(1, 1), (10, 1)])
        self.assertEqual(True, helper.regions_overlap(tr1, tr2))
        self.assertEqual(True, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_point_on_vertical_line(self):
        tr1 = make_region([(1, 5)])
        tr2 = make_region([(1, 1), (1, 10)])
        self.assertEqual(True, helper.regions_overlap(tr1, tr2))
        self.assertEqual(True, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_point_not_on_horizontal_line(self):
        tr1 = make_region([(5, 2)])
        tr2 = make_region([(1, 1), (10, 1)])
        self.assertEqual(False, helper.regions_overlap(tr1, tr2))
        self.assertEqual(False, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_point_not_on_vertical_line(self):
        tr1 = make_region([(2, 5)])
        tr2 = make_region([(1, 1), (1, 10)])
        self.assertEqual(False, helper.regions_overlap(tr1, tr2))
        self.assertEqual(False, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_point_inside_box(self):
        tr1 = make_region([(5, 5)])
        tr2 = make_region([(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertEqual(True, helper.regions_overlap(tr1, tr2))
        self.assertEqual(True, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_point_outside_box(self):
        tr1 = make_region([(5, 15)])
        tr2 = make_region([(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertEqual(False, helper.regions_overlap(tr1, tr2))
        self.assertEqual(False, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_horizontal_line_through_box(self):
        tr1 = make_region([(5, 5), (5, 15)])
        tr2 = make_region([(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertEqual(True, helper.regions_overlap(tr1, tr2))
        self.assertEqual(True, helper.regions_overlap(tr2, tr1))

    def test_element_overlap_horizontal_line_outside_box(self):
        tr1 = make_region([(5, 15), (5, 20)])
        tr2 = make_region([(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertEqual(False, helper.regions_overlap(tr1, tr2))
        self.assertEqual(False, helper.regions_overlap(tr2, tr1))

    def test_translate_point_moves_point(self):
        new_point = helper.translate_point((5, 15), (10, 20))
        self.assertEqual((15, 35), new_point)

    def test_rescale_point_moves_point(self):
        new_point = helper.rescale_point((5, 15), 2)
        self.assertEqual((10, 30), new_point)

    def test_translate_doc_coords_moves_doc_points(self):
        translate_by = (37, 53)
        new_page = helper.transform_doc_coords(self.page_doc, translate_by=translate_by)
        old_point = self.page_doc.coords.points[0]
        new_point = new_page.coords.points[0]
        self.assertEqual(translate_by, (new_point[0] - old_point[0], new_point[1] - old_point[1]))

    def test_translate_doc_coords_moves_child_points(self):
        translate_by = (37, 53)
        new_page = helper.transform_doc_coords(self.page_doc, translate_by=translate_by)
        old_point = self.page_doc.text_regions[0].coords.points[0]
        new_point = new_page.text_regions[0].coords.points[0]
        self.assertEqual(translate_by, (new_point[0] - old_point[0], new_point[1] - old_point[1]))

    def test_rescale_doc_coords_moves_doc_points(self):
        rescale_by = 4
        new_page = helper.transform_doc_coords(self.page_doc, rescale_by=rescale_by)
        old_point = self.page_doc.coords.points[2]
        new_point = new_page.coords.points[2]
        rescale_x = int(new_point[0] / old_point[0])
        rescale_y = int(new_point[1] / old_point[1])
        self.assertEqual((rescale_by, rescale_by), (rescale_x, rescale_y))

    def test_rescale_doc_coords_rescales_child_points(self):
        rescale_by = 4
        new_page = helper.transform_doc_coords(self.page_doc, rescale_by=rescale_by)
        old_point = self.page_doc.text_regions[0].coords.points[0]
        new_point = new_page.text_regions[0].coords.points[0]
        rescale_x = int(new_point[0] / old_point[0])
        rescale_y = int(new_point[1] / old_point[1])
        self.assertEqual((rescale_by, rescale_by), (rescale_x, rescale_y))


if __name__ == '__main__':
    unittest.main()
