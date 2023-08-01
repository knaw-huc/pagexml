import unittest

import pagexml.analysis.layout_stats as layout_stats
import pagexml.model.physical_document_model as pdm
import pagexml.parser as parser


class TestLayoutStats(unittest.TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example.xml'
        self.page_doc = parser.parse_pagexml_file(self.page_file)
        self.tr = self.page_doc.text_regions[1]

    def test_sort_above_below(self):
        line = self.tr.lines[0]
        above, below = layout_stats.sort_coords_above_below_baseline(line)
        self.assertEqual(len(line.coords.points), len(above) + len(below))

    def test_sort_above_below_has_no_above(self):
        line1 = self.tr.lines[0]
        line2 = self.tr.lines[1]
        line1.coords.points = line2.coords.points
        above, below = layout_stats.sort_coords_above_below_baseline(line1)
        self.assertEqual(0, len(above))

    def test_sort_above_below_has_no_below(self):
        line1 = self.tr.lines[0]
        line2 = self.tr.lines[1]
        line2.coords.points = line1.coords.points
        above, below = layout_stats.sort_coords_above_below_baseline(line2)
        self.assertEqual(0, len(below))

    def test_sort_above_below_returns_empty_when_coords_left_of_baseline(self):
        line1 = self.tr.lines[0]
        line1.coords = pdm.Coords([(100, 100), (150, 100), (150, 200), (100, 200)])
        line1.baseline = pdm.Coords([(300, 150), (400, 150)])
        above, below = layout_stats.sort_coords_above_below_baseline(line1)
        self.assertEqual(0, len(below) + len(above))

    def test_sort_above_below_returns_empty_when_coords_right_of_baseline(self):
        line1 = self.tr.lines[0]
        line1.coords = pdm.Coords([(1000, 100), (1500, 100), (1500, 200), (1000, 200)])
        line1.baseline = pdm.Coords([(300, 150), (400, 150)])
        above, below = layout_stats.sort_coords_above_below_baseline(line1)
        self.assertEqual(0, len(below) + len(above))


if __name__ == '__main__':
    unittest.main()
