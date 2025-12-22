from collections import defaultdict
from typing import Dict, List, Union

import numpy as np

import pagexml.analysis.layout_stats as layout_stats
import pagexml.analysis.text_stats as text_stats
import pagexml.model.physical_document_model as pdm


DEFAULT_ELEMENTS = ['lines', 'words', 'text_regions', 'columns', 'extra', 'pages']


def get_doc_metadata(doc: pdm.PageXMLRegion):
    return {
        'doc_id': doc.id,
        'doc_coords': doc.coords.box_string,
        'doc_width': doc.coords.width,
        'doc_height': doc.coords.height,
    }


def get_doc_line_stats(doc: pdm.PageXMLRegion, line_width_boundary_points: List[int] = None,
                       line_bin_width: int = 300, max_bin: int = 3000):
    """

    :param doc: a PageXML document
    :param line_width_boundary_points: a list of points indicating boundaries between categories of
        line widths
    :type line_width_boundary_points: List[int]
    :param line_bin_width: width of line bins, to aggregate lines of different lengths
    :type line_bin_width: int
    :param max_bin: max line width bin
    :type max_bin: int
    """
    if line_width_boundary_points is None:
        line_width_boundary_points = [point for point in range(line_bin_width, max_bin, line_bin_width)]
    lines = [line for line in doc.get_lines() if line.text is not None]
    wpl_stats = text_stats.get_words_per_line(lines, alpha_words_only=False)
    awpl_stats = text_stats.get_words_per_line(lines, alpha_words_only=True)
    line_stats = {}
    for wpl_cat in text_stats.wpl_cat_range.values():
        line_stats[f'words_per_line_{wpl_cat}'] = wpl_stats[wpl_cat]
    for wpl_cat in text_stats.wpl_cat_range.values():
        line_stats[f'alpha_words_per_line_{wpl_cat}'] = awpl_stats[wpl_cat]
    line_width_stats = layout_stats.get_line_width_stats(lines, line_width_boundary_points)
    for line_width_range in line_width_stats:
        line_stats[f'line_width_range_{line_width_range}'] = line_width_stats[line_width_range]
    return line_stats


def get_doc_word_stats(doc: pdm.PageXMLRegion, stop_words: List[str] = None,
                       max_word_length: int = 30, use_re_word_boundaries: bool = False):
    """Return a dictionary with statistics on the words in a PageXML document.

    :param doc: a PageXML document
    :param stop_words: a list of stopwords to include in number of stopwords the scan statistics
    :type stop_words: List[str],
    :param max_word_length: max word length above which words are considered oversized
    :type max_word_length: int
    :param use_re_word_boundaries: flag whether to use RegEx word boundaries for word count
    :type use_re_word_boundaries: bool
    """
    words = text_stats.get_doc_words(doc, use_re_word_boundaries=use_re_word_boundaries)
    return text_stats.get_word_cat_stats(words, stop_words=stop_words,
                                         max_word_length=max_word_length)


def derive_boundary_points(pagexml_doc: pdm.PageXMLRegion) -> List[int]:
    bin_width = pagexml_doc.coords.width / 5
    return [point for point in np.arange(bin_width, pagexml_doc.coords.width, bin_width)]


def _init_doc_stats(line_width_boundary_points: List[int],
                    word_length_bin_size: int = 5, max_word_length: int = 30) -> Dict[str, List[any]]:
    fields = ['doc_id', 'doc_num', 'doc_width', 'doc_height',
              'lines', 'words', 'text_regions',
              'columns', 'extra', 'pages',
              'num_words', 'num_alpha_words', 'num_number_words',
              'num_title_words', 'num_non_title_words',
              'num_stop_words', 'num_punctuation_words', 'num_oversized_words']
    doc_stats = {field: [] for field in fields}
    for cat_wpl in text_stats.wpl_cat_range:
        doc_stats[f"words_per_line_{text_stats.wpl_cat_range[cat_wpl]}"] = []
    for cat_wpl in text_stats.wpl_cat_range:
        doc_stats[f"alpha_words_per_line_{text_stats.wpl_cat_range[cat_wpl]}"] = []
    for length_bin in range(word_length_bin_size, max_word_length + 1, word_length_bin_size):
        doc_stats[f"num_words_length_{length_bin}"] = []
    for width_range in layout_stats.get_boundary_width_ranges(line_width_boundary_points):
        doc_stats[f"line_width_range_{width_range}"] = []
    return doc_stats


def get_scan_stats(pagexml_docs: Union[pdm.PageXMLRegion, List[pdm.PageXMLRegion]],
                   line_width_boundary_points: List[int] = None,
                   stop_words: List[str] = None,
                   max_word_length: int = 30, doc_num: int = None,
                   use_re_word_boundaries: bool = False,
                   use_region_level: bool = False,
                   line_bin_width: int = 300, max_bin: int = 3000):
    """Generate basic statistics for a PageXML scan object (number of text regions, lines,
    words, etc.).

    Line widths are categorised based on a list of boundary points that determine the width of
    each bin. If no boundary points are passed, a set of boundary points is generated based on
    the width of the pagexml_doc.

    :param pagexml_docs: a PageXML document object or a list of PageXML document objects
    :type pagexml_docs: PageXMLRegion
    :param line_width_boundary_points: a list of points indicating boundaries between categories of
        line widths
    :type line_width_boundary_points: List[int]
    :param stop_words: a list of stopwords to include in number of stopwords the scan statistics
    :type stop_words: List[str],
    :param max_word_length: max word length above which words are considered oversized
    :type max_word_length: int
    :param use_re_word_boundaries: flag whether to use RegEx word boundaries for word count
    :type use_re_word_boundaries: bool
    :param use_region_level: flag whether to generates stats per text region instead of per document
    :type use_region_level: bool
    :param line_bin_width: width of line bins, to aggregate lines of different lengths
    :type line_bin_width: int
    :param max_bin: max line width bin
    :type max_bin: int
    :return: a dictionary with scan statistics
    :rtype: Dict[str, int]
    """
    docs_stats = defaultdict(list)

    for pi, pagexml_doc in enumerate(pagexml_docs):
        continue


def get_doc_region_stats(doc: pdm.PageXMLRegion, doc_stats: Dict[str, any],
                         line_width_boundary_points: List[int] = None,
                         stop_words: List[str] = None,
                         max_word_length: int = 30,
                         use_re_word_boundaries: bool = False,
                         line_bin_width: int = 300, max_bin: int = 3000):
    regions_stats = defaultdict(list)
    for tr in doc.get_inner_text_regions():
        for field in doc_stats:
            regions_stats[field].append(doc_stats[field])
        tr_metadata = get_doc_metadata(tr)
        for field in tr_metadata:
            regions_stats[field.replace('doc_', 'text_region_')].append(tr_metadata[field])
        tr_line_stats = get_doc_line_stats(tr, line_width_boundary_points=line_width_boundary_points,
                                           line_bin_width=line_bin_width, max_bin=max_bin)
        tr_word_stats = get_doc_word_stats(tr, stop_words=stop_words, max_word_length=max_word_length,
                                           use_re_word_boundaries=use_re_word_boundaries)
        for field in tr_word_stats:
            regions_stats[field].append(tr_word_stats[field])
        for field in tr_line_stats:
            regions_stats[field].append(tr_line_stats[field])
    return regions_stats


def get_doc_stats(docs: Union[pdm.PageXMLRegion, List[pdm.PageXMLRegion]],
                  line_width_boundary_points: List[int] = None,
                  use_region_level: bool = False,
                  stop_words: List[str] = None,
                  max_word_length: int = 30,
                  use_re_word_boundaries: bool = False,
                  line_bin_width: int = 300, max_bin: int = 3000
                  ):
    docs_stats = defaultdict(list)
    if isinstance(docs, pdm.PageXMLRegion):
        docs = [docs]
    for di, doc in enumerate(docs):
        doc_metadata = get_doc_metadata(doc)
        doc_stats = {'doc_num': di+1}
        doc_stats.update(doc_metadata)
        if use_region_level is True:
            regions_stats = get_doc_region_stats(doc, doc_stats,
                                                 line_width_boundary_points=line_width_boundary_points,
                                                 line_bin_width=line_bin_width, max_bin=max_bin,
                                                 stop_words=stop_words, max_word_length=max_word_length,
                                                 use_re_word_boundaries=use_re_word_boundaries)
            for field in regions_stats:
                docs_stats[field].extend(regions_stats[field])
        else:
            doc_stats.update(doc.stats)
            doc_line_stats = get_doc_line_stats(doc, line_width_boundary_points=line_width_boundary_points,
                                                line_bin_width=line_bin_width, max_bin=max_bin)
            doc_word_stats = get_doc_word_stats(doc, stop_words=stop_words, max_word_length=max_word_length,
                                                use_re_word_boundaries=use_re_word_boundaries)
            doc_stats.update(doc_word_stats)
            doc_stats.update(doc_line_stats)
            for field in doc_stats:
                docs_stats[field].append(doc_stats[field])
    return docs_stats


def get_doc_stats_old(pagexml_docs: Union[pdm.PageXMLRegion, List[pdm.PageXMLRegion]],
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
    :type pagexml_docs: PageXMLRegion
    :param line_width_boundary_points: a list of points indicating boundaries between categories of
        line widths
    :type line_width_boundary_points: List[int]
    :param stop_words: a list of stopwords to include in number of stopwords the scan statistics
    :type stop_words: List[str],
    :param max_word_length: max word length above which words are considered oversized
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
    if isinstance(pagexml_docs, pdm.PageXMLRegion):
        pagexml_docs = [pagexml_docs]
    for pi, pagexml_doc in enumerate(pagexml_docs):
        pagexml_doc_stats['doc_id'].append(pagexml_doc.id)
        pagexml_doc_stats['doc_num'].append(pi + 1)
        pagexml_doc_stats['doc_width'].append(pagexml_doc.coords.width if pagexml_doc.coords else None)
        pagexml_doc_stats['doc_height'].append(pagexml_doc.coords.height if pagexml_doc.coords else None)
        lines = [line for line in pagexml_doc.get_lines() if line.text is not None]
        words = text_stats.get_doc_words(pagexml_doc, use_re_word_boundaries=use_re_word_boundaries)
        word_stats = text_stats.get_word_cat_stats(words, stop_words=stop_words,
                                                   max_word_length=max_word_length)
        wpl_stats = text_stats.get_words_per_line(lines, alpha_words_only=False)
        awpl_stats = text_stats.get_words_per_line(lines, alpha_words_only=True)
        # for field in pagexml_doc.stats:
        for field in DEFAULT_ELEMENTS:
            pagexml_doc_stats[field].append(pagexml_doc.stats[field] if field in pagexml_doc.stats else 0)
        for word_cat in word_stats:
            pagexml_doc_stats[word_cat].append((word_stats[word_cat]))
        for wpl_cat in text_stats.wpl_cat_range.values():
            pagexml_doc_stats[f'words_per_line_{wpl_cat}'].append(wpl_stats[wpl_cat])
        for wpl_cat in text_stats.wpl_cat_range.values():
            pagexml_doc_stats[f'alpha_words_per_line_{wpl_cat}'].append(awpl_stats[wpl_cat])
        if line_width_boundary_points is None:
            bin_width = pagexml_doc.coords.width / 5
            line_width_boundary_points = [point for point in np.arange(bin_width, pagexml_doc.coords.width, bin_width)]
        line_width_stats = layout_stats.get_line_width_stats(lines, line_width_boundary_points)
        for line_width_range in line_width_stats:
            pagexml_doc_stats[f'line_width_range_{line_width_range}'].append(line_width_stats[line_width_range])
    return pagexml_doc_stats
