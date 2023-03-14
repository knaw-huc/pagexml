import unittest

from pagexml.parser import parse_pagexml_file


class TestPageXMLHelper(unittest.TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example.xml'
        self.page_doc = parse_pagexml_file(self.page_file)

    def test_something(self):
        self.assertEqual(True, 1 == 1)


if __name__ == '__main__':
    unittest.main()
