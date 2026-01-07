import unittest

import pagexml.model.pagexml_document_model as pdm
import pagexml.parsers.scan_parser as scan_parser
from pagexml.parser import parse_pagexml_file


class TestScanSplit(unittest.TestCase):

    def setUp(self) -> None:
        self.scan_file = 'data/example.xml'
        self.scan_doc = parse_pagexml_file(self.scan_file)

    def test_set_avg_scan_width(self):
        scan_parser.set_average_scan_width([self.scan_doc])
        self.assertEqual(self.scan_doc.coords.width, self.scan_doc.metadata['avg_scan_width'])

    def test_get_page_split_width_splits_scan_halfway(self):
        even_start, even_end, odd_start, odd_end = scan_parser.get_page_split_widths(self.scan_doc, page_overlap=0)
        self.assertEqual(even_end, odd_start)

    def test_get_page_split_width_can_split_with_page_overlap(self):
        page_overlap = 100
        even_start, even_end, odd_start, odd_end = scan_parser.get_page_split_widths(self.scan_doc,
                                                                                     page_overlap=page_overlap)
        self.assertEqual(even_end - page_overlap, odd_start + page_overlap)

    def test_initiliaze_page(self):
        page_overlap = 100
        even_start, even_end, odd_start, odd_end = scan_parser.get_page_split_widths(self.scan_doc,
                                                                                     page_overlap=page_overlap)
        page_even = scan_parser.initialize_pagexml_page(self.scan_doc, 'even', even_start, even_end)
        self.assertEqual(int(self.scan_doc.coords.width / 2), page_even.coords.width - page_overlap)

    def test_initiliaze_scan_pages(self):
        page_overlap = 100
        pages = scan_parser.initialize_scan_pages(self.scan_doc, page_overlap=page_overlap)
        self.assertEqual(2, len(pages))

    def test_split_scan_pages(self):
        page_overlap = 100
        pages = scan_parser.split_scan_pages(self.scan_doc, page_overlap=page_overlap)
        self.assertEqual(2, len(pages))


class TestScanParser(unittest.TestCase):

    def setUp(self) -> None:
        self.scan_file = 'data/example-2.xml'
        self.scan_doc = parse_pagexml_file(self.scan_file)
        self.table_file = 'data/example_table.xml'
        self.table_doc = parse_pagexml_file(self.table_file)
        """
        print(f"scan.id: {self.scan_doc.id}")
        for tr in self.scan_doc.text_regions:
            print(f"  tr.id: {tr.id}")
        print(f"table.id: {self.table_doc.id}")
        for tr in self.table_doc.table_regions:
            print(f"  tr.id: {tr.id}")
            for row in tr.rows:
                print(row.cells[1].lines[0].parent)
                print(f"    row {row.id}: {[line.text for cell in row.cells for line in cell.lines]}")
        """

    def test_separation_split_returns_non_overlapping_pages(self):
        page_verso, page_recto = scan_parser.split_scan_pages_with_separation_point(self.scan_doc, 2700)
        hor_overlap = pdm.get_horizontal_overlap(page_verso, page_recto)
        self.assertEqual(0, hor_overlap)

    def test_separation_split_adjusts_page_left_to_left_most_region(self):
        page_verso, page_recto = scan_parser.split_scan_pages_with_separation_point(self.scan_doc, 3100)
        tr_left = min(tr.coords.left for tr in page_recto.get_textual_regions())
        self.assertEqual(tr_left, page_recto.coords.left)

    def test_get_parent_region_of_text_line(self):
        tr = self.scan_doc.text_regions[1]
        line = tr.lines[0]
        parent = scan_parser.get_parent_region(line)
        self.assertEqual(tr, parent)

    def test_get_parent_region_of_table_cell_line(self):
        tr = self.table_doc.table_regions[0]
        line = tr.rows[0].cells[1].lines[0]
        parent = scan_parser.get_parent_region(line)
        self.assertEqual(tr, parent)

    def test_group_lines_by_parent_regions_retains_table_structure(self):
        table = self.table_doc.table_regions[0]
        lines = [line for row in table.rows for cell in row.cells for line in cell.lines]
        new_tables = scan_parser.group_lines_by_parent_regions(lines)
        self.assertEqual(table.stats['lines'], new_tables[0].stats['lines'])
