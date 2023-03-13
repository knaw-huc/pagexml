import os
import tempfile
import unittest

import pagexml.helper.text_helper as text_helper
from pagexml.parser import parse_pagexml_file


class TestTextHelper(unittest.TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example.xml'
        self.page_doc = parse_pagexml_file(self.page_file)
        self.box_fields = ['doc_box', 'textregion_box', 'line_box']

    def test_box_to_coords(self):
        bbox = text_helper.get_bbox(self.page_doc)
        self.assertEqual(True, all([num.isdigit() for num in bbox.split(',')]))

    def test_get_json_format_generates_dictionaries(self):
        line_format_reader = text_helper.get_line_format_json(self.page_doc)
        line = next(line_format_reader)
        self.assertEqual(True, 'doc_id' in line)
        self.assertEqual(self.page_doc.id, line['doc_id'])

    def test_get_json_format_can_add_bounding_box(self):
        line_format_reader = text_helper.get_line_format_json(self.page_doc, add_bounding_box=True)
        line = next(line_format_reader)
        self.assertEqual(True, all([field in line for field in self.box_fields]))

    def test_line_reader_default_has_no_bounding_box(self):
        line_format_reader = text_helper.LineReader(pagexml_files=self.page_file)
        for line in line_format_reader:
            self.assertEqual(False, all([field in line for field in self.box_fields]))

    def test_line_reader_can_add_bounding_box(self):
        line_format_reader = text_helper.LineReader(pagexml_files=self.page_file, add_bounding_box=True)
        for line in line_format_reader:
            self.assertEqual(True, all([field in line for field in self.box_fields]))


class TestTextHelperWriter(unittest.TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example.xml'
        self.page_doc = parse_pagexml_file(self.page_file)
        self.box_fields = ['doc_box', 'textregion_box', 'line_box']
        self.tmp = tempfile.NamedTemporaryFile(delete=False)
        self.tmp.close()

    def tearDown(self) -> None:
        os.unlink(self.tmp.name)

    def test_read_from_line_file_returns_correct_number_of_scans(self):
        text_helper.make_line_format_file([self.page_doc], self.tmp.name, add_bounding_box=True)
        page_docs = text_helper.read_pagexml_docs_from_line_file(self.tmp.name, add_bounding_box=True)
        page_docs = [page_doc for page_doc in page_docs]
        self.assertEqual(1, len(page_docs))

    def test_read_from_line_file_returns_complete_scan(self):
        stats = self.page_doc.stats
        text_helper.make_line_format_file([self.page_doc], self.tmp.name, add_bounding_box=True)
        page_docs = text_helper.read_pagexml_docs_from_line_file(self.tmp.name, add_bounding_box=True)
        page_doc = [page_doc for page_doc in page_docs][0]
        for field in page_doc.stats:
            self.assertEqual(True, page_doc.stats[field] == stats[field])


if __name__ == '__main__':
    unittest.main()
