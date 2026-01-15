from itertools import combinations
from unittest import TestCase

import pagexml.helper.spatial_helper as spatial_helper
import pagexml.model.physical_document_model as pdm


class TestSpatialHelper(TestCase):

    def setUp(self) -> None:
        self.coords_main = pdm.Coords([(0, 0), (500, 0), (500, 500), (0, 500)])
        self.region_main = pdm.PageXMLTextRegion(doc_id='r1', coords=self.coords_main)
        self.coords1 = pdm.Coords([(100, 100), (200, 100), (200, 200), (100, 200)])
        self.region1 = pdm.PageXMLTextRegion(doc_id='r1', coords=self.coords1)
        self.coords2 = pdm.Coords([(150, 150), (250, 150), (250, 250), (150, 250)])
        self.region2 = pdm.PageXMLTextRegion(doc_id='r2', coords=self.coords2)
        self.coords3 = pdm.Coords([(300, 100), (400, 100), (400, 200), (300, 200)])
        self.region3 = pdm.PageXMLTextRegion(doc_id='r3', coords=self.coords3)
        self.coords4 = pdm.Coords([(200, 100), (300, 100), (300, 200), (200, 200)])
        self.region4 = pdm.PageXMLTextRegion(doc_id='r4', coords=self.coords4)
        self.coords5 = pdm.Coords([(100, 200), (200, 200), (200, 300), (100, 300)])
        self.region5 = pdm.PageXMLTextRegion(doc_id='r5', coords=self.coords5)
        self.line = pdm.PageXMLTextLine(coords=pdm.Coords([(100, 100), (200, 100)]))

    def test_region_1_and_2_overlap(self):
        self.assertEqual(True, spatial_helper.regions_poly_overlap(self.region1, self.region2))

    def test_region_1_and_2_relative_hloc(self):
        self.assertEqual('over', spatial_helper.get_relative_hloc(self.region1, self.region2))

    def test_region_1_and_2_relative_vloc(self):
        self.assertEqual('over', spatial_helper.get_relative_vloc(self.region1, self.region2))

    def test_region_1_and_3_not_overlap(self):
        self.assertEqual(False, spatial_helper.regions_poly_overlap(self.region1, self.region3))

    def test_region_1_and_3_relative_hloc(self):
        self.assertEqual('right', spatial_helper.get_relative_hloc(self.region1, self.region3))

    def test_region_1_and_3_relative_vloc(self):
        self.assertEqual('over', spatial_helper.get_relative_vloc(self.region1, self.region3))

    def test_region_1_and_3_not_touch(self):
        self.assertEqual(False, spatial_helper.region_touches_right(self.region1, self.region3))

    def test_region_1_and_4_touch(self):
        self.assertEqual(True, spatial_helper.region_touches_right(self.region1, self.region4))

    def test_region_1_and_4_relative_hloc(self):
        self.assertEqual('right', spatial_helper.get_relative_hloc(self.region1, self.region4))

    def test_region_1_and_4_relative_vloc(self):
        self.assertEqual('over', spatial_helper.get_relative_vloc(self.region1, self.region4))

    def test_region_3_and_4_touch(self):
        self.assertEqual(True, spatial_helper.region_touches_right(self.region1, self.region4))

    def test_region_1_and_5_touch(self):
        self.assertEqual(True, spatial_helper.region_touches_below(self.region1, self.region5))

    def test_region_1_and_5_relative_hloc(self):
        self.assertEqual('over', spatial_helper.get_relative_hloc(self.region1, self.region5))

    def test_region_1_and_5_relative_vloc(self):
        self.assertEqual('below', spatial_helper.get_relative_vloc(self.region1, self.region5))

    def test_make_neighour_regions_region_1(self):
        neighbours = spatial_helper.make_region_neighbours(self.region1, self.region_main)
        self.assertEqual(4, len(neighbours))

    def test_make_empty_regions_region_1(self):
        empty_regions = spatial_helper.make_empty_regions(self.region1, debug=1)
        self.assertEqual(1, len(empty_regions))

    def test_make_empty_regions_region_main_with_region_1(self):
        self.region_main.text_regions = [self.region1]
        empty_regions = spatial_helper.make_empty_regions(self.region_main, debug=1)
        self.assertEqual(4, len(empty_regions))

    def test_make_empty_regions_region_main_with_regions_1_3_(self):
        self.region_main.text_regions = [self.region1, self.region3]
        empty_regions = spatial_helper.make_empty_regions(self.region_main, debug=1)
        self.assertEqual(5, len(empty_regions))

    def test_make_empty_regions_region_main_with_regions_1_4_(self):
        self.region_main.text_regions = [self.region1, self.region4]
        empty_regions = spatial_helper.make_empty_regions(self.region_main, debug=1)
        self.assertEqual(4, len(empty_regions))

    def test_make_empty_regions_regions_do_not_overlap(self):
        self.region_main.text_regions = [self.region1, self.region4]
        empty_regions = spatial_helper.make_empty_regions(self.region_main, debug=1)
        for r1, r2 in combinations(empty_regions, 2):
            with self.subTest(f"{r1.id}-{r2.id}"):
                overlap = spatial_helper.regions_box_overlap(r1, r2)
                if overlap:
                    print(f"OVERLAP: {r1.coords.box_string} - {r2.coords.box_string}")
                self.assertEqual(False, overlap)

    def test_make_empty_regions_region_main_with_line(self):
        self.region_main.text_regions = [self.line]
        empty_regions = spatial_helper.make_empty_regions(self.region_main, debug=1)
        self.assertEqual(1, len(empty_regions))
