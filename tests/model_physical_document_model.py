import unittest

import pagexml.model.physical_document_model as pdm
from pagexml.parser import parse_pagexml_file


class TestParentage(unittest.TestCase):

    def setUp(self) -> None:
        self.scan_file = 'data/example-2.xml'
        self.scan_doc = parse_pagexml_file(self.scan_file)
        self.table_file = 'data/example_table.xml'
        self.table_doc = parse_pagexml_file(self.table_file)

    def test_get_parent_region_of_text_line(self):
        tr = self.scan_doc.text_regions[1]
        line = tr.lines[0]
        parent = line.parent
        self.assertEqual(tr, parent)

    def test_get_parent_region_of_table_cell_line(self):
        tr = self.table_doc.table_regions[0]
        line = tr.rows[0].cells[1].lines[0]
        parent = line.parent.parent.parent
        self.assertEqual(tr, parent)
