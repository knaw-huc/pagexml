import unittest

import pagexml.helper.text_helper as text_helper
from pagexml.parser import parse_pagexml_file


class TestTextHelper(unittest.TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example.xml'
        self.page_doc = parse_pagexml_file(self.page_file)

    def test_box_to_coords(self):
        bbox = text_helper.get_bbox_coords(self.page_doc)
        self.assertEqual(True, all([num.isdigit() for num in bbox.split(',')]))

    def test_get_json_format_generates_dictionaries(self):
        line_format_reader = text_helper.get_line_format_json(self.page_doc)
        line = next(line_format_reader)
        self.assertEqual(True, 'doc_id' in line)
        self.assertEqual(self.page_doc.id, line['doc_id'])

    def test_get_json_format_can_add_bounding_box(self):
        line_format_reader = text_helper.get_line_format_json(self.page_doc, add_bounding_box=True)
        line = next(line_format_reader)
        coords_fields = ['doc_coords', 'textregion_coords', 'line_coords']
        self.assertEqual(True, all([field in line for field in coords_fields]))

    def test_line_reader_default_has_no_bounding_box(self):
        line_format_reader = text_helper.LineReader(pagexml_files=self.page_file)
        for line in line_format_reader:
            coords_fields = ['doc_coords', 'textregion_coords', 'line_coords']
            self.assertEqual(False, all([field in line for field in coords_fields]))

    def test_line_reader_can_add_bounding_box(self):
        line_format_reader = text_helper.LineReader(pagexml_files=self.page_file, add_bounding_box=True)
        for line in line_format_reader:
            coords_fields = ['doc_coords', 'textregion_coords', 'line_coords']
            self.assertEqual(True, all([field in line for field in coords_fields]))


if __name__ == '__main__':
    unittest.main()
