from unittest import TestCase

import pagexml.analysis.stats as stats
import pagexml.parser as parser


class TestInitStats(TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example.xml'
        self.page_doc = parser.parse_pagexml_file(self.page_file)
        self.tr = self.page_doc.text_regions[1]

    def test_init_doc_stats_has_regions(self):
        line_width_boundary_points = [100, 200]
        doc_stats = stats._init_doc_stats(line_width_boundary_points=line_width_boundary_points)
        print(doc_stats)
        self.assertEqual(True, 'text_regions' in doc_stats)


class TestDocStats(TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example.xml'
        self.page_doc = parser.parse_pagexml_file(self.page_file)
        self.tr = self.page_doc.text_regions[1]

    def test_tr_metadata_has_coords_box_string(self):
        metadata = stats.get_doc_metadata(self.tr)
        self.assertEqual(self.tr.coords.box_string, metadata['doc_coords'])

    def test_tr_line_stats_are_integers(self):
        line_stats = stats.get_doc_line_stats(self.tr)
        self.assertEqual(True, all(isinstance(line_stats[field], int) for field in line_stats))

    def test_tr_word_stats_are_integers(self):
        word_stats = stats.get_doc_word_stats(self.tr)
        self.assertEqual(True, all(isinstance(word_stats[field], int) for field in word_stats if field != 'num_stop_words'))

    def test_page_doc_metadata_has_coords_box_spage_docing(self):
        metadata = stats.get_doc_metadata(self.page_doc)
        self.assertEqual(self.page_doc.coords.box_string, metadata['doc_coords'])

    def test_page_doc_line_stats_are_integers(self):
        line_stats = stats.get_doc_line_stats(self.page_doc)
        values = [line_stats[field] for field in line_stats]
        self.assertEqual(True, all(isinstance(value, int) for value in values))

    def test_page_doc_word_stats_are_integers(self):
        word_stats = stats.get_doc_word_stats(self.page_doc)
        values = [word_stats[field] for field in word_stats if field != 'num_stop_words']
        self.assertEqual(True, all(isinstance(value, int) for value in values))

    def test_page_doc_doc_stats_has_regions(self):
        doc_stats = stats.get_doc_stats(self.page_doc)
        self.assertEqual(True, 'text_regions' in doc_stats)

    def test_region_level_same_as_region_list(self):
        trs = [tr for tr in self.page_doc.get_inner_text_regions()]
        trs_stats = stats.get_doc_stats(trs)
        docs_stats = stats.get_doc_stats(self.page_doc, use_region_level=True)
        self.assertEqual(len(trs_stats), len(docs_stats))
        doc_line_stats = stats.get_doc_line_stats(self.page_doc)
        doc_word_stats = stats.get_doc_word_stats(self.page_doc)
        shared_fields = list(doc_line_stats.keys()) + list(doc_word_stats.keys())
        for i, field in enumerate(shared_fields):
            with self.subTest(i):
                self.assertEqual(trs_stats[field], docs_stats[field])

    def test_page_doc_old_doc_stats_same_as_new_doc_stats(self):
        new_doc_stats = stats.get_doc_stats(self.page_doc)
        old_doc_stats = stats.get_doc_stats_old(self.page_doc)
        for field in new_doc_stats:
            if field not in old_doc_stats:
                print(f"missing in old: {field}")
            elif old_doc_stats[field] != new_doc_stats[field]:
                print(f"different values in old ({old_doc_stats[field]}) and new ({new_doc_stats[field]})")
        for field in old_doc_stats:
            if field not in new_doc_stats:
                print(f"missing in new: {field}")
        del new_doc_stats['chars']
        del new_doc_stats['doc_coords']
        self.assertEqual(new_doc_stats, old_doc_stats)
