import copy
import gzip
import re
import string
from collections import Counter
from enum import Enum
from typing import Dict, Generator, List, Set, Tuple, Union

import numpy as np

import pagexml.analysis.layout_stats as summarise
import pagexml.analysis.text_stats as text_stats
import pagexml.helper.text_helper as text_helper
import pagexml.model.physical_document_model as pdm


def is_point_inside(point: Tuple[int, int], element: pdm.PageXMLDoc) -> bool:
    x, y = point
    if x < element.coords.left or x > element.coords.right:
        return False
    if y < element.coords.top or y > element.coords.bottom:
        return False
    return True


class RegionType(Enum):

    POINT = 1
    HLINE = 2
    VLINE = 3
    BOX = 4


def get_region_type(element: pdm.PageXMLDoc) -> RegionType:
    if element.coords.height == 0:
        if element.coords.width == 0:
            return RegionType.POINT
        else:
            return RegionType.HLINE
    elif element.coords.width == 0:
        return RegionType.VLINE
    else:
        return RegionType.BOX


def same_point(point1: Tuple[int, int], point2: Tuple[int, int]) -> bool:
    """Check if two points are the same."""
    return point1[0] == point2[0] and point1[1] == point2[1]


def regions_overlap(region1: pdm.PageXMLDoc, region2: pdm.PageXMLDoc,
                    threshold: float = 0.5) -> bool:
    """Check if two regions have overlapping coordinates.

    Assumption: points are pixels, so regions with at least one point have at least
    a width, height and area of 1."""
    if region1.coords is None or region2.coords is None:
        return False

    height1 = region1.coords.height + 1
    width1 = region1.coords.width + 1
    height2 = region2.coords.height + 1
    width2 = region2.coords.width + 1

    v_overlap = pdm.get_vertical_overlap(region1, region2)
    h_overlap = pdm.get_horizontal_overlap(region1, region2)

    if v_overlap / height1 > threshold:
        if h_overlap / width1 > threshold:
            return True
    if v_overlap / height2 > threshold:
        if h_overlap / width2 > threshold:
            return True
    else:
        return False


def sort_regions_in_reading_order(doc: pdm.PageXMLDoc) -> List[pdm.PageXMLTextRegion]:
    """Sort text regions in reading order. If an explicit reading order is given,
    that is used, otherwise, text regions are sorted top to bottom, left to right."""
    doc_text_regions: List[pdm.PageXMLTextRegion] = []
    if doc.reading_order and hasattr(doc, 'text_regions') and doc.text_regions:
        text_region_ids = [region for _index, region in sorted(doc.reading_order.items(), key=lambda x: x[0])]
        return [tr for tr in sorted(doc.text_regions, key=lambda x: text_region_ids.index(x.id))]
    if hasattr(doc, 'columns') and sorted(doc.columns):
        doc_text_regions.extend(doc.columns)
    if hasattr(doc, 'text_regions') and doc.text_regions:
        doc_text_regions.extend(doc.text_regions)
    if hasattr(doc, 'extra') and doc.extra:
        doc_text_regions.extend(doc.extra)
    if doc_text_regions:
        sub_text_regions = []
        for text_region in sorted(doc_text_regions, key=lambda x: (x.coords.top, x.coords.left)):
            sub_text_regions += sort_regions_in_reading_order(text_region)
        return sub_text_regions
    elif isinstance(doc, pdm.PageXMLTextRegion):
        return [doc]
    else:
        return []


def horizontal_group_lines(lines: List[pdm.PageXMLTextLine]) -> List[List[pdm.PageXMLTextLine]]:
    """Sort lines of a text region vertically as a list of lists,
    with adjacent lines grouped in inner lists."""
    if len(lines) == 0:
        return []
    # First, sort lines vertically
    vertically_sorted = [line for line in sorted(lines, key=lambda line: line.coords.top) if line.text is not None]
    if len(vertically_sorted) == 0:
        # for line in lines:
        #     print(line.coords.box, line.text)
        return []
    # Second, group adjacent lines in vertical line stack
    horizontally_grouped_lines = [[vertically_sorted[0]]]
    rest_lines = vertically_sorted[1:]
    if len(vertically_sorted) > 1:
        for li, curr_line in enumerate(rest_lines):
            prev_line = horizontally_grouped_lines[-1][-1]
            if curr_line.is_below(prev_line):
                horizontally_grouped_lines.append([curr_line])
            elif curr_line.is_next_to(prev_line):
                horizontally_grouped_lines[-1].append(curr_line)
            else:
                horizontally_grouped_lines.append([curr_line])
    # Third, sort adjecent lines horizontally
    for line_group in horizontally_grouped_lines:
        line_group.sort(key=lambda line: line.coords.left)
    return horizontally_grouped_lines


def merge_sets(sets: List[Set[any]], min_overlap: int = 1) -> List[Set[any]]:
    merged_sets = []

    while len(sets) > 0:
        current_set = sets.pop(0)
        merged_set = set(current_set)

        i = 0
        while i < len(sets):
            if len(merged_set.intersection(sets[i])) >= min_overlap:
                merged_set.update(sets[i])
                sets.pop(i)
            else:
                i += 1

        merged_sets.append(merged_set)

    return merged_sets


def merge_textregions(text_regions: List[pdm.PageXMLTextRegion],
                      metadata: dict = None, doc_id: str = None) -> Union[pdm.PageXMLTextRegion, None]:
    """Merge two text_regions into one, sorting lines by baseline height."""
    if len(text_regions) == 0:
        return None
    merged_lines = [line for tr in text_regions for line in tr.get_lines()]
    merged_lines = list(set(merged_lines))
    sorted_lines = sorted(merged_lines, key=lambda x: x.baseline.y)
    merged_coords = pdm.parse_derived_coords(sorted_lines)
    merged_tr = pdm.PageXMLTextRegion(doc_id=doc_id, doc_type='index_text_region',
                                      metadata=metadata, coords=merged_coords,
                                      lines=sorted_lines)
    if doc_id is None:
        merged_tr.set_derived_id(text_regions[0].parent.id)
    return merged_tr


def horizontally_merge_lines(lines: List[pdm.PageXMLTextLine]) -> List[pdm.PageXMLTextLine]:
    """Sort lines vertically and merge horizontally adjacent lines."""
    horizontally_grouped_lines = horizontal_group_lines(lines)
    horizontally_merged_lines = []
    for line_group in horizontally_grouped_lines:
        coords = pdm.parse_derived_coords(line_group)
        baseline = pdm.Baseline([point for line in line_group for point in line.baseline.points])
        line = pdm.PageXMLTextLine(metadata=line_group[0].metadata, coords=coords, baseline=baseline,
                                   text=' '.join([line.text for line in line_group]))
        line.set_derived_id(line_group[0].metadata['parent_id'])
        horizontally_merged_lines.append(line)
    return horizontally_merged_lines


def sort_lines_in_reading_order(doc: pdm.PageXMLTextRegion,
                                row_order: bool = False,
                                reading_direction: str = 'ltr') -> Generator[pdm.PageXMLTextLine, None, None]:
    if row_order is True:
        return sort_lines_in_row_reading_order(doc, reading_direction=reading_direction)
    else:
        return sort_lines_in_column_reading_order(doc, reading_direction=reading_direction)


def sort_lines_in_column_reading_order(doc: pdm.PageXMLDoc,
                                       reading_direction: str = 'ltr') -> Generator[pdm.PageXMLTextLine, None, None]:
    """Sort the lines of a pdm.PageXML document in reading order.
    Reading order is: columns from left to right, text regions in columns from top to bottom,
    lines in text regions from top to bottom, and when (roughly) adjacent, from left to right."""
    for text_region in sort_regions_in_reading_order(doc):
        if text_region.main_type == 'column':
            text_region.metadata['column_id'] = text_region.id
        for line in text_region.lines:
            if line.metadata is None:
                line.metadata = {'id': line.id, 'type': ['pagexml', 'line'], 'parent_id': text_region.id}
            if 'column_id' in text_region.metadata and 'column_id' not in line.metadata:
                line.metadata['column_id'] = text_region.metadata['column_id']
        for line in sort_lines_in_reading_direction(text_region.lines, reading_direction=reading_direction):
            yield line


def sort_lines_in_row_reading_order(doc: pdm.PageXMLTextRegion,
                                    reading_direction: str = 'ltr') -> Generator[pdm.PageXMLTextLine, None, None]:
    """Sort the lines of a pdm.PageXML document in row order.
    Row order is: lines from top to bottom, and when (roughly) adjacent, in the
    given reading direction."""
    return sort_lines_in_reading_direction(doc.get_lines(), reading_direction=reading_direction)


def sort_lines_in_reading_direction(lines: List[pdm.PageXMLTextLine],
                                    reading_direction: str = 'ltr') -> Generator[pdm.PageXMLTextLine, None, None]:
    stacked_lines = horizontal_group_lines(lines)
    for lines in stacked_lines:
        if reading_direction == 'ltr':
            stacked_lines = sorted(lines, key=lambda x: x.coords.left)
        elif reading_direction == 'rtl':
            stacked_lines = sorted(lines, key=lambda x: x.coords.right, reverse=True)
        else:
            raise ValueError(f'invalid reading direction {reading_direction}, should be "ltr" or "rtl"')
        for line in stacked_lines:
            yield line


def combine_adjacent_lines(lines: List[pdm.PageXMLTextLine], reading_direction: str,
                           avg_char_width: float):
    if reading_direction not in {'ltr', 'rtl'}:
        raise ValueError(f'invalid reading direction {reading_direction}, should be "ltr" or "rtl"')
    prev_line = None
    line_string = ''
    for curr_line in lines:
        line_text = curr_line.text if curr_line.text is not None else ''
        infix_whitespace = ""
        if prev_line is not None:
            if reading_direction == 'ltr':
                indent = curr_line.coords.left - prev_line.coords.right
            else:
                indent = prev_line.coords.left - curr_line.coords.right
            if indent > 0 and avg_char_width > 0:
                infix_whitespace = " " * int(float(indent) / avg_char_width)
        if reading_direction == 'ltr':
            line_string = line_string + infix_whitespace + line_text
        else:
            line_string = line_text + infix_whitespace + line_string
        prev_line = curr_line
    return line_string


def print_textregion_stats(text_region: pdm.PageXMLTextRegion) -> None:
    """Print statistics on the textual content of a text region.

    :param text_region: a TextRegion object that contains TextLines
    :type text_region: PageXMLTextRegion
    """
    avg_line_distance = summarise.get_textregion_avg_line_distance(text_region)
    avg_char_width = summarise.get_textregion_avg_char_width(text_region)
    avg_line_width_chars = summarise.get_textregion_avg_line_width(text_region, unit="char")
    avg_line_width_pixels = summarise.get_textregion_avg_line_width(text_region, unit="pixel")
    print("\n--------------------------------------")
    print("Document info")
    print(f"  {'id:': <30}{text_region.id}")
    print(f"  {'type:': <30}{text_region.type}")
    stats = text_region.stats
    for element_type in stats:
        element_string = f'number of {element_type}:'
        print(f'  {element_string: <30}{stats[element_type]:>6.0f}')
    print(f"  {'avg. distance between lines:': <30}{avg_line_distance: >6.0f}")
    print(f"  {'avg. char width:': <30}{avg_char_width: >6.0f}")
    print(f"  {'avg. chars per line:': <30}{avg_line_width_chars: >6.0f}")
    print(f"  {'avg. pixels per line:': <30}{avg_line_width_pixels: >6.0f}")
    print("--------------------------------------\n")


def pretty_print_textregion(text_region: pdm.PageXMLTextRegion,
                            reading_direction: str = 'ltr', print_stats: bool = False) -> None:
    """Pretty print the text of a text region, using indentation and
    vertical space based on the average character width and average
    distance between lines. If no corresponding images of the PageXML
    are available, this can serve as a visual approximation to reveal
    the page layout.

    :param text_region: a TextRegion object that contains TextLines
    :type text_region: PageXMLTextRegion
    :param reading_direction: option to set reading direction left-to-right (default) or right-to-left
    :param print_stats: flag to print text_region statistics if set to True
    :type print_stats: bool
    """
    if print_stats:
        print_textregion_stats(text_region)
    avg_line_distance = summarise.get_textregion_avg_line_distance(text_region)
    avg_char_width = summarise.get_textregion_avg_char_width(text_region)
    pretty_string = ''
    lines = [line for line in sort_lines_in_reading_order(text_region, reading_direction=reading_direction)]
    min_left = min([line.coords.left for line in lines])
    max_right = max([line.coords.right for line in lines])
    stacked_lines = horizontal_group_lines(lines)
    prev_stack = None
    for curr_stack in stacked_lines:
        line_string = combine_adjacent_lines(curr_stack, reading_direction=reading_direction,
                                             avg_char_width=avg_char_width)
        if reading_direction == 'ltr':
            indent = curr_stack[0].coords.left - min_left
        else:
            indent = max_right - curr_stack[0].coords.right
        preceding_whitespace = " " * int(float(indent) / avg_char_width) if avg_char_width > 0 else ""
        if reading_direction == 'ltr':
            pretty_string += f"{preceding_whitespace}{line_string}\n"
        else:
            pretty_string += f"{line_string}{preceding_whitespace}\n"
        if prev_stack is not None:
            distances = summarise.compute_baseline_distances(prev_stack, curr_stack)
            if np.median(distances) > avg_line_distance * 1.2:
                pretty_string += '\n'
        prev_stack = curr_stack
    print(pretty_string)


def line_ends_with_word_break(curr_line: pdm.PageXMLTextLine, next_line: pdm.PageXMLTextLine,
                              word_freq: Counter = None) -> bool:
    if not next_line or not next_line.text:
        # if the next line has no text, it has no first word to join with the last word of the current line
        return False
    if not curr_line.text[-1] in string.punctuation:
        # if the current line does not end with punctuation, we assume, the last word is not hyphenated
        return False
    match = re.search(r"(\w+)\W+$", curr_line.text)
    if not match:
        # if the current line has no word immediately before the punctuation, we assume there is no word break
        return False
    last_word = match.group(1)
    match = re.search(r"^(\w+)", next_line.text)
    if not match:
        # if the next line does not start with a word, we assume it should not be joined to the last word
        # on the current line
        return False
    next_word = match.group(1)
    if curr_line.text[-1] == "-":
        # if the current line ends in a proper hyphen, we assume it should be joined to the first
        # word on the next line
        return True
    if not word_freq:
        # if no word_freq counter is given, we cannot compare frequencies, so assume the words should
        # not be joined
        return False
    joint_word = last_word + next_word
    if word_freq[joint_word] == 0:
        return False
    if word_freq[joint_word] > 0 and word_freq[last_word] * word_freq[next_word] == 0:
        return True
    pmi = word_freq[joint_word] * sum(word_freq.values()) / (word_freq[last_word] * word_freq[next_word])
    if pmi > 1:
        return True
    if word_freq[joint_word] > word_freq[last_word] and word_freq[joint_word] > word_freq[next_word]:
        return True
    elif word_freq[next_word] < word_freq[joint_word] <= word_freq[last_word]:
        print("last word:", last_word, word_freq[last_word])
        print("next word:", next_word, word_freq[next_word])
        print("joint word:", joint_word, word_freq[joint_word])
        return True
    else:
        return False


def pagexml_to_line_format(pagexml_doc: pdm.PageXMLTextRegion) -> Generator[Tuple[str, str, str], None, None]:
    for line in pagexml_doc.get_lines():
        yield pagexml_doc.id, line.id, line.text


def write_pagexml_to_line_format(pagexml_docs: List[pdm.PageXMLTextRegion], output_file: str) -> None:
    with gzip.open(output_file, 'wt') as fh:
        for pagexml_doc in pagexml_docs:
            for doc_id, line_id, line_text in pagexml_to_line_format(pagexml_doc):
                fh.write(f"{doc_id}\t{line_id}\t{line_text}\n")


def read_line_format_file(line_format_files: Union[str, List[str]],
                          headers: List[str] = None,
                          has_header: bool = False) -> Generator[Tuple[str, str, str], None, None]:
    if isinstance(line_format_files, str):
        line_format_files = [line_format_files]
    for line_format_file in line_format_files:
        with gzip.open(line_format_file, 'rt') as fh:
            if has_header is True or headers is None:
                header_line = next(fh)
                headers = header_line.strip().split('\t')
            for li, line in enumerate(fh):
                row = line.strip().split('\t')
                if headers is None:
                    yield row
                else:
                    if len(row) > len(headers):
                        raise IndexError(
                            f"Missing columns. Header has {len(headers)} columns while line {li+1} in row "
                            f"has {len(row)} columns")
                    yield {header: row[hi] if len(row) > hi else None for hi, header in enumerate(headers)}


class LineIterable:

    def __init__(self, line_format_files: Union[str, List[str]], headers: List[str] = None):
        self.line_format_files = line_format_files
        self.headers = headers

    def __iter__(self):
        line_iterator = read_line_format_file(line_format_files=self.line_format_files,
                                              headers=self.headers)
        for line in line_iterator:
            yield line


def make_line_text(line: pdm.PageXMLTextLine, do_merge: bool,
                   end_word: str, merge_word: str,
                   word_break_chars: Union[str, Set[str], List[str]] = '-') -> str:
    line_text = line.text
    if len(line_text) >= 2 and line_text[-1] in word_break_chars and line_text[-2] in word_break_chars:
        # remove the redundant line break char
        line_text = line_text[:-1]
    if do_merge:
        if line_text[-1] in word_break_chars and merge_word.startswith(end_word) is False:
            # the merge word does not contain a line break char, so remove it from the line
            # before adding it to the text
            line_text = line_text[:-1]
        else:
            # the line contains no line break char or the merge word contains the hyphen as
            # well, so leave it in.
            line_text = line.text
    else:
        # no need to merge so add line with trailing whitespace
        if line_text[-1] in word_break_chars and len(line_text) >= 2 and line_text[-2] != ' ':
            # the line break char at the end is trailing, so disconnect it from the preceding word
            line_text = line_text[:-1] + f' {line_text[-1]} '
        else:
            line_text = line_text + ' '
    return line_text


def make_line_range(text: str, line: pdm.PageXMLTextLine, line_text: str) -> Dict[str, any]:
    len_line = len(line_text) if line_text is not None else 0
    return {
        "start": len(text), "end": len(text) + len_line,
        "line_id": line.id,
        "parent_id": line.metadata["parent_id"] if "parent_id" in line.metadata else None
    }


def make_text_region_text(lines: List[pdm.PageXMLTextLine],
                          word_break_chars: Union[str, Set[str], List[str]] = '-',
                          wbd: text_stats.WordBreakDetector = None) -> Tuple[Union[str, None], List[Dict[str, any]]]:
    """Turn the text lines in a region into a single paragraph of text, with a list of line ranges
    that indicates how the text of each line corresponds to character offsets in the paragraph.

    :param lines: a list of PageXML text lines belonging to the same text region
    :type lines: List[PageXMLTextLine]
    :param word_break_chars: a lsit of characters that signal a word-break
    :type word_break_chars: List[str]
    :param wbd: a line break detector object
    :type wbd: LineBreakDetector
    :return: a paragraph of text and a list of line ranges that indicates how the text of each line
        corresponds to character offsets in the paragraph.
    :rtype: Tuple[str, List[Dict[str, any]]
    """
    if wbd is not None and wbd.word_break_chars is not None:
        word_break_chars = set([char for char in wbd.word_break_chars])
    text = ''
    line_ranges = []
    lines = [line for line in lines if line.text is not None and line.text != '']
    if len(lines) == 0:
        return None, []
    prev_line = lines[0]
    prev_words = text_helper.get_line_words(prev_line.text, word_break_chars=word_break_chars) \
        if prev_line.text else []
    if len(lines) > 1:
        remove_prefix_word_break = False
        for curr_line in lines[1:]:
            if curr_line.text is None or curr_line.text == '':
                do_merge = False
                merge_word = None
                curr_words = []
                prev_line_text = prev_line.text if prev_line.text else ''
            else:
                curr_words = text_helper.get_line_words(curr_line.text,
                                                        word_break_chars=word_break_chars)
                if prev_line.text is not None:
                    do_merge, merge_word = text_stats.determine_word_break(curr_words, prev_words,
                                                                           wbd=wbd,
                                                                           word_break_chars=word_break_chars,
                                                                           debug=False)
                    # print(do_merge, merge_word)
                    prev_line_text = make_line_text(prev_line, do_merge, prev_words[-1], merge_word,
                                                    word_break_chars=word_break_chars)
                    if remove_prefix_word_break and prev_line_text.startswith('„'):
                        prev_line_text = prev_line_text[1:]
                    if '„' in word_break_chars and prev_words[-1].endswith('„') and curr_line.text.startswith('„'):
                        remove_prefix_word_break = True
                    else:
                        remove_prefix_word_break = False
                    # print(prev_line_text)
                else:
                    prev_line_text = ''
            line_range = make_line_range(text, prev_line, prev_line_text)
            line_ranges.append(line_range)
            text += prev_line_text

            prev_words = curr_words
            prev_line = curr_line
    # add the last line (without adding trailing whitespace)
    line_range = make_line_range(text, prev_line, prev_line.text)
    line_ranges.append(line_range)
    if prev_line.text is not None:
        text += prev_line.text
    return text, line_ranges


def merge_lines(lines: List[pdm.PageXMLTextLine], remove_word_break: bool = False,
                word_break_char: str = '-') -> pdm.PageXMLTextLine:
    """Returns a PageXMLTextline object that is the merge of a list of PageXMLTextlines.

    :param lines: a list of PageXML text lines
    :type lines: List[PageXMLTextline]
    :param remove_word_break: flag indicating whether line break characters should be removed
    :type remove_word_break: bool
    :param word_break_char: the character that is used as a line break
    :type word_break_char: str
    :return: a PageXML text line object
    :rtype: PageXMLTextline
    """
    coords = pdm.parse_derived_coords(lines)
    text = ''
    for li, curr_line in enumerate(lines):
        if remove_word_break and len(text) > 0 and text.endswith(word_break_char):
            if curr_line.text[0].islower():
                # remove hyphen
                text = text[:-1]
        text += curr_line.text
    return pdm.PageXMLTextLine(metadata=copy.deepcopy(lines[0].metadata),
                               coords=coords, text=text)


