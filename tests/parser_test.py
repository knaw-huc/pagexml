import unittest

import pagexml.parser as parser


class TestParser(unittest.TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example.xml'
        self.page_doc = parser.parse_pagexml_file(self.page_file, custom_tags=['date', 'place'])

    def test_parentage_is_set(self):
        for tr in self.page_doc.text_regions:
            self.assertEqual(True, tr.parent == self.page_doc)
            for line in tr.lines:
                self.assertEqual(True, line.parent == tr)

    def test_parsing_from_json_retains_stats(self):
        page_json = self.page_doc.json
        new_page = parser.parse_pagexml_from_json(page_json)
        for field in new_page.stats:
            self.assertEqual(True, new_page.stats[field] == self.page_doc.stats[field])

    def test_parsing_from_json_retains_type(self):
        page_json = self.page_doc.json
        new_page = parser.parse_pagexml_from_json(page_json)
        self.assertEqual(True, self.page_doc.main_type == new_page.main_type)

    def test_parsing_from_json_sets_parents(self):
        page_json = self.page_doc.json
        new_page = parser.parse_pagexml_from_json(page_json)
        for tr in new_page.text_regions:
            self.assertEqual(True, tr.parent == new_page)
            for line in tr.lines:
                self.assertEqual(True, line.parent == tr)


if __name__ == '__main__':
    unittest.main()
