import unittest
from pagexml import parser


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.page_file = 'tests/issue_10.xml'
        self.page_doc = parser.parse_pagexml_file(self.page_file, custom_tags=['date', 'place'])

    def test_multiple_text_styles_in_r2l1(self):
        text_line = self.page_doc.text_regions[0].lines[0]
        self.assertEqual(text_line.id, 'r2l1')  # add assertion here
        text_styles = text_line.metadata['text_style']
        expected = [{'type': 'textStyle', 'offset': 13, 'length': 1, 'superscript': 'true'},
                    {'type': 'textStyle', 'offset': 20, 'length': 1, 'superscript': 'true'}]
        self.assertEqual(text_styles, expected)

    def test_place_in_r2l13(self):
        text_line = self.page_doc.text_regions[0].lines[12]
        self.assertEqual(text_line.id, 'r2l13')  # add assertion here
        custom_tags = text_line.metadata['custom_tags']
        expected = [{'type': 'place', 'offset': 5, 'length': 6, }]
        self.assertEqual(custom_tags, expected)

    def test_date_in_r2l18(self):
        text_line = self.page_doc.text_regions[0].lines[17]
        self.assertEqual(text_line.id, 'r2l18')  # add assertion here
        custom_tags = text_line.metadata['custom_tags']
        expected = [{'type': 'date', 'offset': 17, 'length': 12, }]
        self.assertEqual(custom_tags, expected)


if __name__ == '__main__':
    unittest.main()
