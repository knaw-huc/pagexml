from typing import Dict, List
import pagexml.model.physical_document_model as pdm
import pagexml.analysis.text_stats as text_stats
import pagexml.analysis.layout_stats as layout_stats


def get_scan_stats(scan: pdm.PageXMLScan, line_width_boundary_points: List[int],
                   stop_words: List[str] = None,
                   max_word_length: int = 30, scan_num: int = None,
                   use_re_word_boundaries: bool = False) -> Dict[str, int]:
    """Generate basic statistics for a PageXML scan object (number of text regions, lines,
    words, etc.)

    :param scan: a PageXML scan object
    :type scan: PageXMLScan
    :param line_width_boundary_points: a list of points indicating boundaries between categories of
    line widths
    :type line_width_boundary_points: List[int]
    :param stop_words: a list of stopwords to include in number of stopwords the scan statistics
    :type stop_words: List[str],
    :param max_word_length: max word length above which words are considered over sized
    :type max_word_length: int
    :param scan_num: the number of a scan in a sequence of scans
    :type scan_num: int
    :param use_re_word_boundaries: flag whether to use RegEx word boundaries for word count
    :type use_re_word_boundaries: bool
    :return: a dictionary with scan statistics
    :rtype: Dict[str, int]
    """
    lines = [line for line in scan.get_lines() if line.text is not None]
    words = text_stats.get_scan_words(scan, use_re_word_boundaries=use_re_word_boundaries)
    word_stats = text_stats.get_word_cat_stats(words, stop_words=stop_words,
                                               max_word_length=max_word_length)
    scan_stats = scan.stats
    scan_stats['scan_num'] = scan_num
    scan_stats.update(word_stats)
    line_width_stats = layout_stats.get_line_width_stats(lines, line_width_boundary_points)
    for line_width_cat in line_width_stats:
        scan_stats[f'line_width_cat_{line_width_cat}'] = line_width_stats[line_width_cat]
    return scan_stats
