from unittest import TestCase

import pagexml.helper.text_helper as text_helper


class TestGetLineWords(TestCase):

    def setUp(self) -> None:
        self.word_break_chars = '-–'

    def test_get_line_words(self):
        line = 'has no line break char'
        words = text_helper.get_line_words(line, word_break_chars=self.word_break_chars)
        self.assertEqual(' '.join(words), line)
