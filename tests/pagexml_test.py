import unittest

import pagexml.model.physical_document_model as pdm
from pagexml.helper.pagexml_helper import pretty_print_textregion
from pagexml.parser import parse_pagexml_file
from pagexml.parsers.scan_parser import split_scan_pages_with_separation_point


class PageXMLTestCase(unittest.TestCase):

    def test_parse_pagexml_file(self):
        file = 'data/example.xml'
        scan = parse_pagexml_file(file)
        # pretty_print_textregion(scan, print_stats=True)
        self.assertEqual(isinstance(scan, pdm.PageXMLScan), True)

    def test_parser_gets_correct_stats(self):
        # example has 2 text regions, 39 text lines and 155 words
        file = 'data/example.xml'
        scan = parse_pagexml_file(file)
        self.assertEqual(scan.stats['pages'], 0)
        self.assertEqual(scan.stats['text_regions'], 2)
        self.assertEqual(scan.stats['columns'], 0)
        self.assertEqual(scan.stats['lines'], 39)
        self.assertEqual(scan.stats['words'], 155)
        self.assertEqual(scan.stats['extra'], 0)

    def test_parser_gets_correct_page_stats(self):
        # example has 2 text regions, 39 text lines and 155 words
        file = 'data/example.xml'
        scan = parse_pagexml_file(file)
        print(f"scan box_string: {scan.coords.box_string}")
        pages = split_scan_pages_with_separation_point(scan, 4000)
        for page in pages:
            print(f"page.stats: {page.stats}")
            self.assertEqual(page.stats['text_regions'], 2)


if __name__ == '__main__':
    unittest.main()
