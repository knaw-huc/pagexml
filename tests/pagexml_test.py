import unittest

from pagexml.model.physical_document_model import pretty_print_textregion
from pagexml.parser import parse_pagexml_file


class PageXMLTestCase(unittest.TestCase):
    def test_parse_pagexml_file(self):
        file = 'example.xml'
        scan = parse_pagexml_file(file)
        pretty_print_textregion(scan, print_stats=True)


if __name__ == '__main__':
    unittest.main()
