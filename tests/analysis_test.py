from unittest import TestCase

import pagexml.analysis.text_stats as text_stats


class TestGetLineWords(TestCase):

    def test_get_line_words(self):
        line = {'text': 'has no line break char'}
        line_text = text_stats.get_line_text(line)
        self.assertEqual(line_text, line['text'])
