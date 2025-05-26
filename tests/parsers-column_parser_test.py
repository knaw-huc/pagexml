import unittest

import pagexml.model.physical_document_model as pdm
import pagexml.parsers.column_parser as col_parser


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

    def test_find_column_gaps_returns_text_column_gap(self):
        column_gaps = col_parser.find_column_gaps(self.lines, min_column_lines=1,
                                                  min_gap_width=50, min_column_width=50)
        self.assertEqual(1, len(column_gaps))
