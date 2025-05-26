import unittest

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
