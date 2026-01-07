import unittest
from typing import List

import pagexml.helper.pagexml_helper as pagexml_helper
import pagexml.model.physical_document_model as pdm
import pagexml.parsers.column_parser as col_parser
from pagexml.parser import parse_pagexml_file


def generate_page_doc():
    page_file = 'data/example.xml'
    page_doc = parse_pagexml_file(page_file)
    for region in page_doc.text_regions:
        region.set_derived_id(page_doc.id)
    return page_doc


class TestPixelGap(unittest.TestCase):

    def setUp(self) -> None:
        self.lines = [
            pdm.PageXMLTextLine(coords=pdm.Coords([(100, 100), (200, 100)])),
            pdm.PageXMLTextLine(coords=pdm.Coords([(300, 100), (400, 100)])),
            pdm.PageXMLTextLine(coords=pdm.Coords([(100, 200), (200, 200)])),
            pdm.PageXMLTextLine(coords=pdm.Coords([(300, 200), (400, 200)])),
        ]

    def test_compute_pixel_dist_ignores_lines_without_points(self):
        line = pdm.PageXMLTextLine(text='bla')
        pixel_dist = col_parser.compute_text_pixel_dist([line])
        self.assertEqual(0, len(pixel_dist))

    def test_compute_pixel_dist(self):
        pixel_dist = col_parser.compute_text_pixel_dist(self.lines)
        self.assertEqual(2, pixel_dist[100])

    def test_find_column_ranges_returns_text_column_ranges(self):
        column_ranges = col_parser.find_column_ranges(self.lines, min_column_lines=1,
                                                      min_gap_width=50, min_column_width=50)
        self.assertEqual(2, len(column_ranges))

    def test_find_column_ranges_ignores_small_gaps(self):
        column_ranges = col_parser.find_column_ranges(self.lines, min_column_lines=1,
                                                      min_gap_width=150, min_column_width=50)
        self.assertEqual(1, len(column_ranges))

    def test_find_column_ranges_ignores_small_columns(self):
        column_ranges = col_parser.find_column_ranges(self.lines, min_column_lines=1,
                                                      min_gap_width=50, min_column_width=150)
        self.assertEqual(0, len(column_ranges))


def split_region_lines(text_region: pdm.PageXMLTextRegion) -> List[pdm.PageXMLColumn]:
    lines_left, lines_right = [], []
    print(f"text_region.id: #{text_region.id}#")
    for line in text_region.lines:
        if line.coords.left < 3500:
            lines_left.append(line)
        else:
            lines_right.append(line)
    tr_left = pagexml_helper.derive_text_region_from_lines(lines_left, parent=text_region.parent)
    tr_right = pagexml_helper.derive_text_region_from_lines(lines_right, parent=text_region.parent)
    col_left = col_parser.derive_column_from_regions(tr_left)
    col_right = col_parser.derive_column_from_regions(tr_right)
    columns = [col_left, col_right]
    return columns


class TestColumnGeneration(unittest.TestCase):

    def setUp(self) -> None:
        self.page_doc = generate_page_doc()

    def test_generate_column_from_regions_copies_all_text_regions(self):
        column = col_parser.derive_column_from_regions(self.page_doc.text_regions)
        self.assertEqual(len(self.page_doc.text_regions), len(column.text_regions))

    def test_generate_column_from_region_sets_scan_as_parent(self):
        region = self.page_doc.text_regions[0]
        column = col_parser.derive_column_from_regions(region)
        self.assertEqual(self.page_doc, column.parent)

    def test_generate_column_from_region_sets_columns_as_parent(self):
        region = self.page_doc.text_regions[0]
        column = col_parser.derive_column_from_regions(region)
        self.assertEqual(column, column.text_regions[0].parent)

    def test_generate_column_from_region_leaves_original_region_parent_intact(self):
        region = self.page_doc.text_regions[0]
        col_parser.derive_column_from_regions(region)
        self.assertEqual(self.page_doc, region.parent)


class TestColumnNormalisation(unittest.TestCase):

    def setUp(self) -> None:
        self.page_doc = generate_page_doc()
        self.columns = [col_parser.derive_column_from_regions(r) for r in self.page_doc.text_regions]

    def test_normalise_columns_retains_number_of_text_regions(self):
        num_page_regions = len(self.page_doc.text_regions)
        norm_columns = col_parser.normalise_columns(self.columns)
        num_norm_regions = sum(len(col.text_regions) for col in norm_columns)
        self.assertEqual(num_page_regions, num_norm_regions)

    def test_normalise_columns_copies(self):
        columns = split_region_lines(self.page_doc.text_regions[1])
        norm_columns = col_parser.normalise_columns(columns)
        self.assertEqual(1, len(norm_columns))
