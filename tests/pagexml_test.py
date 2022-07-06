import unittest

from pagexml.helper.pagexml_helper import pretty_print_textregion
from pagexml.parser import parse_pagexml_file
import pagexml.model.physical_document_model as pdm


class PageXMLTestCase(unittest.TestCase):

    def test_parse_pagexml_file(self):
        file = 'data/example.xml'
        scan = parse_pagexml_file(file)
        pretty_print_textregion(scan, print_stats=True)
        self.assertEqual(isinstance(scan, pdm.PageXMLScan), True)

    def test_parser_gets_correct_stats(self):
        # example has 2 text regions, 39 text lines and 155 words
        file = 'data/example.xml'
        scan = parse_pagexml_file(file)
        self.assertEqual(scan.stats['words'], 155)


if __name__ == '__main__':
    unittest.main()
