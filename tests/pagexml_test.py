import glob
import traceback
import unittest

from icecream import ic

from pagexml.model.physical_document_model import pretty_print_textregion
from pagexml.parser import parse_pagexml_file


class PageXMLTestCase(unittest.TestCase):
    def test_parse_pagexml_file(self):
        file = 'example.xml'
        scan = parse_pagexml_file(file)
        pretty_print_textregion(scan, print_stats=True)

    def test_all_ga_pagexml_files(self):
        pagexml_basedir = "../golden-agents/pagexml/"
        testdirs = glob.glob(pagexml_basedir + '[0-9AN]*')
        # testdirs = glob.glob(pagexml_basedir + '10025*')
        assert len(testdirs) >0
        for d in testdirs:
            files = glob.glob(d + '/*.xml')
            assert len(files) > 0
            for fname in files:
                ic(fname)
                print(fname)
                try:
                    scan_doc = parse_pagexml_file(fname)
                    pretty_print_textregion(scan_doc, print_stats=True)
                except Exception:
                    print(traceback.format_exc())


if __name__ == '__main__':
    unittest.main()
