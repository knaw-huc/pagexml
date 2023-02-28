from typing import Dict, List, Union

import numpy as np

import pagexml.model.physical_document_model as pdm
import pagexml.analysis.text_stats as text_stats
import pagexml.analysis.layout_stats as layout_stats


def derive_boundary_points(pagexml_doc: pdm.PageXMLTextRegion) -> List[int]:
    bin_width = pagexml_doc.coords.width / 5
    return [point for point in np.arange(bin_width, pagexml_doc.coords.width, bin_width)]


def _init_doc_stats(line_width_boundary_points: List[int],
                    word_length_bin_size: int = 5, max_word_length: int = 30) -> Dict[str, List[any]]:
    fields = ['doc_id', 'doc_num', 'lines', 'words', 'text_regions', 'columns', 'extra', 'pages',
              'num_words', 'num_number_words', 'num_title_words', 'num_non_title_words',
              'num_stop_words', 'num_punctuation_words', 'num_oversized_words']
    doc_stats = {field: [] for field in fields}
    for length_bin in range(word_length_bin_size, max_word_length+1, word_length_bin_size):
        doc_stats[f"num_words_length_{length_bin}"] = []
    for width_range in layout_stats.get_boundary_width_ranges(line_width_boundary_points):
        doc_stats[f"line_width_range_{width_range}"] = []
    return doc_stats


def get_doc_stats(pagexml_docs: Union[pdm.PageXMLTextRegion, List[pdm.PageXMLTextRegion]],
                  line_width_boundary_points: List[int] = None,
                  stop_words: List[str] = None,
                  max_word_length: int = 30, doc_num: int = None,
                  use_re_word_boundaries: bool = False,
                  line_bin_width: int = 300, max_bin: int = 3000) -> Dict[str, List[any]]:
    """Generate basic statistics for a PageXML scan object (number of text regions, lines,
    words, etc.).

    Line widths are categorised based on a list of boundary points that determine the width of
    each bin. If no boundary points are passed, a set of boundary points is generated based on
    the width of the pagexml_doc.

    :param pagexml_docs: a PageXML document object or a list of PageXML document objects
    :type pagexml_docs: PageXMLTextRegion
    :param line_width_boundary_points: a list of points indicating boundaries between categories of
    line widths
    :type line_width_boundary_points: List[int]
    :param stop_words: a list of stopwords to include in number of stopwords the scan statistics
    :type stop_words: List[str],
    :param max_word_length: max word length above which words are considered over sized
    :type max_word_length: int
    :param doc_num: the number of a doc in a sequence of docs
    :type doc_num: int
    :param use_re_word_boundaries: flag whether to use RegEx word boundaries for word count
    :type use_re_word_boundaries: bool
    :param line_bin_width: width of line bins, to aggregate lines of different lengths
    :type line_bin_width: int
    :param max_bin: max line width bin
    :type max_bin: int
    :return: a dictionary with scan statistics
    :rtype: Dict[str, int]
    """
    if line_width_boundary_points is None:
        line_width_boundary_points = [point for point in range(line_bin_width, max_bin, line_bin_width)]
    pagexml_doc_stats = _init_doc_stats(line_width_boundary_points, max_word_length=max_word_length)
    if isinstance(pagexml_docs, pdm.PageXMLTextRegion):
        pagexml_docs = [pagexml_docs]
    for pi, pagexml_doc in enumerate(pagexml_docs):
        pagexml_doc_stats['doc_id'].append(pagexml_doc.id)
        pagexml_doc_stats['doc_num'].append(pi+1)
        lines = [line for line in pagexml_doc.get_lines() if line.text is not None]
        words = text_stats.get_doc_words(pagexml_doc, use_re_word_boundaries=use_re_word_boundaries)
        word_stats = text_stats.get_word_cat_stats(words, stop_words=stop_words,
                                                   max_word_length=max_word_length)
        for field in pagexml_doc.stats:
            pagexml_doc_stats[field].append(pagexml_doc.stats[field])
        for word_cat in word_stats:
            pagexml_doc_stats[word_cat].append((word_stats[word_cat]))
        if line_width_boundary_points is None:
            bin_width = pagexml_doc.coords.width / 5
            line_width_boundary_points = [point for point in np.arange(bin_width, pagexml_doc.coords.width, bin_width)]
        line_width_stats = layout_stats.get_line_width_stats(lines, line_width_boundary_points)
        for line_width_range in line_width_stats:
            pagexml_doc_stats[f'line_width_range_{line_width_range}'].append(line_width_stats[line_width_range])
    return pagexml_doc_stats
