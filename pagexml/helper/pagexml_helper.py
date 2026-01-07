import copy
import gzip
import re
import string
from collections import Counter
from collections import defaultdict
from enum import Enum
from typing import Dict, Generator, List, Set, Tuple, Union

import numpy as np

import pagexml.analysis.layout_stats as summarise
import pagexml.analysis.text_stats as text_stats
import pagexml.helper.text_helper as text_helper
import pagexml.model.coords
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


def get_doc_indent_left_right(doc, doc_indent: int = 0):
    return doc.coords.left + doc_indent, doc.coords.right - doc_indent


def regions_overlap(region1: pdm.PageXMLDoc, region2: pdm.PageXMLDoc,
                    threshold: float = 0.5, debug: int = 0) -> bool:
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

    if debug > 4:
        print(f"pagexml.pagexml_helper.regions_overlap\n\tregion 1: {region1.id}\n\tregion 2: {region2.id}")
        print(f"\th_overlap: {h_overlap}\tv_overlap: {v_overlap}")

    if v_overlap / height1 > threshold:
        if h_overlap / width1 > threshold:
            if debug > 4:
                print(f'\tboxes 1 and 2: {region1.coords.box} {region2.coords.box}')
                print(f'\toverlap True based on height1 and width1')
            return True
    if v_overlap / height2 > threshold:
        if h_overlap / width2 > threshold:
            if debug > 4:
                print(f'\tboxes 1 and 2: {region1.coords.box} {region2.coords.box}')
                print(f'\toverlap True based on height2 and width2')
            return True
    if debug > 4:
        print(f'\toverlap False')
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


def horizontal_group_lines(lines: List[pdm.PageXMLTextLine],
                           debug: int = 0) -> List[List[pdm.PageXMLTextLine]]:
    """Sort lines of a text region vertically as a list of lists,
    with adjacent lines grouped in inner lists."""
    if len(lines) == 0:
        return []
    # First, sort lines vertically
    vertically_sorted = [line for line in sorted(lines, key=lambda line: line.baseline.top) if line.text is not None]
    if len(vertically_sorted) == 0:
        # for line in lines:
        #     print(line.coords.box, line.text)
        return []
    # Second, group adjacent lines in vertical line stack
    horizontally_grouped_lines = [[vertically_sorted[0]]]
    rest_lines = vertically_sorted[1:]
    if len(vertically_sorted) > 1:
        for li, curr_line in enumerate(rest_lines):
            prev_group = horizontally_grouped_lines[-1]
            prev_line = prev_group[-1]
            if debug > 0:
                print(f"prev_line: {prev_line.id} {make_coords_string(prev_line)}")
                print(f"curr_line: {curr_line.id} {make_coords_string(curr_line)}")
            if any(curr_line.is_below(pl, direct_only=True) for pl in prev_group):
                if debug > 0:
                    print("curr is directly below one of prev group - make new group")
                horizontally_grouped_lines.append([curr_line])
            elif curr_line.is_below(prev_line, direct_only=False):
                if debug > 0:
                    print("curr is below prev - make new group")
                horizontally_grouped_lines.append([curr_line])
            elif curr_line.is_next_to(prev_line):
                if debug > 0:
                    print(f"curr is next to prev - add to last group")
                horizontally_grouped_lines[-1].append(curr_line)
            else:
                if debug > 0:
                    print(f"curr is not next to prev - make new group")
                horizontally_grouped_lines.append([curr_line])
    # Third, sort adjecent lines horizontally
    for line_group in horizontally_grouped_lines:
        line_group.sort(key=lambda line: line.coords.left)
    return horizontally_grouped_lines


def horizontally_group_lines(lines: List[pdm.PageXMLTextLine],
                             debug: int = 0) -> List[List[pdm.PageXMLTextLine]]:
    """Wraps `horizontal_group_lines` but with more appropriate naming."""
    return horizontal_group_lines(lines, debug=debug)


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


def check_lines_are_from_same_table(lines: List[pdm.PageXMLTextLine]) -> pdm.PageXMLTableRegion:
    first_line = lines[0]
    first_cell = first_line.parent
    if not isinstance(first_cell, pdm.PageXMLTableCell):
        raise TypeError(f"parent of first line in list is not of class PageXMLTableCell, "
                        f"but instead of {first_cell.__class__.__name__}")
    first_row = first_cell.parent
    if not isinstance(first_row, pdm.PageXMLTableRow):
        raise TypeError(f"parent of cell of first line in list is not of class PageXMLTableRow, "
                        f"but instead of {first_row.__class__.__name__}")
    table = first_row.parent
    if not isinstance(table, pdm.PageXMLTableRegion):
        raise TypeError(f"parent of row of first line in list is not of class PageXMLTableRegion, "
                        f"but instead of {table.__class__.__name__}")
    if any(line.parent.parent.parent != table for line in lines):
        tables = set(line.parent.parent.parent for line in lines)
        raise ValueError(f"lines come from multiple tables: {[table.id for table in tables]}")
    return table


def derive_table_region_from_lines(lines: List[pdm.PageXMLTextLine]) -> pdm.PageXMLTableRegion:
    if len(lines) < 0:
        raise ValueError(f"cannot derive text_region from empty list of lines.")
    table = check_lines_are_from_same_table(lines)
    row_lines = defaultdict(lambda: defaultdict(list))
    for line in lines:
        row_lines[line.parent.parent][line.parent].append(copy_line(line))
    new_table = copy_table_region(table)
    new_table.rows = []
    for row in row_lines:
        new_row = copy_row(row)
        new_row.set_derived_id(new_table.id)
        new_row.cells = []
        for cell in row_lines[row]:
            new_cell = copy_cell(cell)
            new_cell.lines = row_lines[row][cell]
            new_cell.coords = pdm.parse_derived_coords(new_cell.lines)
            new_cell.set_derived_id(new_row.id)
            new_row.cells.append(new_cell)
        new_row.coords = pdm.parse_derived_coords(new_row.cells)
        new_table.rows.append(new_row)
    new_table.coords = pdm.parse_derived_coords(new_table.rows)
    new_table.set_derived_id(table.parent.id)
    pdm.set_parentage(new_table)
    return new_table


def derive_text_region_from_lines(lines: List[pdm.PageXMLTextLine], parent: pdm.PageXMLRegion = None) -> pdm.PageXMLTextRegion:
    if len(lines) < 0:
        raise ValueError(f"cannot derive text_region from empty list of lines.")
    lines = [copy_line(line) for line in lines]
    coords = pdm.parse_derived_coords(lines)
    tr = pdm.PageXMLTextRegion(metadata=copy.deepcopy(lines[0].metadata), coords=coords, lines=lines)
    if parent:
        tr.parent = parent
        tr.set_derived_id(parent.id)
        tr.set_as_parent(tr.lines)
    return tr


def merge_textregions(text_regions: List[pdm.PageXMLTextRegion],
                      metadata: dict = None, doc_id: str = None) -> Union[pdm.PageXMLTextRegion, None]:
    """Merge two text_regions into one, sorting lines by baseline height."""
    if len(text_regions) == 0:
        return None
    merged_lines = [line for tr in text_regions for line in tr.get_lines()]
    merged_lines = list(set(merged_lines))
    sorted_lines = sorted(merged_lines, key=lambda x: x.baseline.y)
    merged_coords = pagexml.model.coords.parse_derived_coords(sorted_lines)
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
        coords = pagexml.model.coords.parse_derived_coords(line_group)
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
                            f"Missing columns. Header has {len(headers)} columns while line {li + 1} in row "
                            f"has {len(row)} columns")
                    yield {header: row[hi] if len(row) > hi else None for hi, header in enumerate(headers)}


def get_custom_tags(doc: pdm.PageXMLDoc) -> List[Dict[str, any]]:
    """
    Get all custom tags and their textual values from a PageXMLDoc.

    This function assumes that the PageXML document is generated with
    input of some `custom_tags` in the parse_pagexml_file function.
    This helper retrieves those tags from all TextLines and finds the
    corresponding text from their offset and length. It returns a
    dictionary with the tag type, the textual value, region and line
    id, and the offset and length.

    :param doc: A PageXMLDoc
    :type doc: pdm.PageXMLDoc
    :return: List of custom tags
    :rtype: List[Dict[str, any]]
    """
    custom_tags = []

    if hasattr(doc, 'text_regions'):
        for region in doc.text_regions:
            for line in region.lines:
                for tag_el in line.metadata.get("custom_tags", []):
                    tag = tag_el["type"]
                    offset = tag_el["offset"]
                    length = tag_el["length"]

                    value = line.text[offset:offset + length]

                    custom_tags.append({
                        "type": tag,
                        "value": value,
                        "region_id": region.id,
                        "line_id": line.id,
                        "offset": offset,
                        "length": length,
                    })

    return custom_tags


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
    :param word_break_chars: a list of characters that signal a word-break
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
    coords = pagexml.model.coords.parse_derived_coords(lines)
    text = ''
    for li, curr_line in enumerate(lines):
        if remove_word_break and len(text) > 0 and text.endswith(word_break_char):
            if curr_line.text[0].islower():
                # remove hyphen
                text = text[:-1]
        text += curr_line.text
    return pdm.PageXMLTextLine(metadata=copy.deepcopy(lines[0].metadata),
                               coords=coords, text=text)


def make_baseline_string(line: pdm.PageXMLTextLine):
    b = line.baseline
    return f"{b.left: >4}-{b.right: <4}\t{b.top: >4}-{b.bottom: <4}"


def make_coords_string(line: pdm.PageXMLTextLine):
    c = line.coords
    return f"{c.left: >4}-{c.right: <4}\t{c.top: >4}-{c.bottom: <4}"


def translate_point(point: Tuple[int, int], translate_by: Tuple[int, int]) -> Tuple[int, int]:
    """Translate a 2D point"""
    return point[0] + translate_by[0], point[1] + translate_by[1]


def rescale_point(point: Tuple[int, int], rescale_by: float) -> Tuple[int, int]:
    """Rescale a 2D point"""
    return int(point[0] * rescale_by), int(point[1] * rescale_by)


def check_transform_is_valid(rescale_by: float = None, translate_by: Tuple[int, int] = None):
    if rescale_by is not None:
        if not isinstance(rescale_by, int) and not isinstance(rescale_by, float):
            raise TypeError(f"rescale_by must be an integer, not {type(rescale_by)}")
    if translate_by is not None:
        if not isinstance(translate_by[0], int) or not isinstance(translate_by[1], int):
            raise TypeError(f"translate_by must be a tuple of integers, "
                            f"not {type(translate_by[0])}, {type(translate_by[1])}")


def transform_doc_coords(doc, rescale_by: float = None, translate_by: Tuple[int, int] = None,
                         in_place: bool = False):
    """Rescale all coordinates and baseline points of a document,
    including all those of its child elements."""
    check_transform_is_valid(rescale_by, translate_by)
    if in_place:
        new_doc = doc
    else:
        # new_doc = copy.deepcopy(doc)
        new_doc = copy_pagexml_doc(doc)
    if new_doc.coords is None:
        new_coords = None
    else:
        new_points = [point for point in doc.coords.points]
        if rescale_by is None or rescale_by == 1.0:
            pass
        else:
            new_points = [rescale_point(point, rescale_by) for point in doc.coords.points]
        if translate_by is None or translate_by == (0, 0):
            pass
        else:
            new_points = [translate_point(point, translate_by) for point in doc.coords.points]
        new_coords = pdm.Coords(new_points)
    for child in new_doc.children:
        transform_doc_coords(child, rescale_by, translate_by, in_place=True)
    new_doc.coords = new_coords
    if hasattr(doc, 'baseline'):
        if doc.baseline is None:
            new_doc.baseline = None
        else:
            new_points = [point for point in doc.baseline.points]
            if rescale_by is not None:
                new_points = [rescale_point(point, rescale_by) for point in doc.baseline.points]
            if translate_by is not None:
                new_points = [translate_point(point, translate_by) for point in doc.baseline.points]
            new_doc.baseline = pdm.Coords(new_points)
    if rescale_by is not None and hasattr(doc, 'xheight') and doc.xheight is not None:
        doc.xheight = int(doc.xheight * rescale_by)
    return new_doc


def copy_pagexml_doc(doc: pdm.PageXMLDoc) -> pdm.PageXMLDoc:
    parent = doc.parent
    doc.parent = None
    new_doc = copy.deepcopy(doc)
    new_doc.parent = parent
    return new_doc


def copy_doc(doc: pdm.PageXMLDoc) -> pdm.PageXMLDoc:
    if isinstance(doc, pdm.PageXMLScan):
        return copy_scan(doc)
    if isinstance(doc, pdm.PageXMLPage):
        return copy_page(doc)
    if isinstance(doc, pdm.PageXMLColumn):
        return copy_column(doc)
    if isinstance(doc, pdm.PageXMLTextRegion):
        return copy_text_region(doc)
    if isinstance(doc, pdm.PageXMLTextLine):
        return copy_line(doc)
    if isinstance(doc, pdm.PageXMLWord):
        return copy_word(doc)
    if isinstance(doc, pdm.PageXMLDoc):
        return copy.deepcopy(doc)
    else:
        raise TypeError(f"doc must be an instance of pdm.PageXMLDoc and its sub-classes, not {type(doc)}")


def copy_scan(scan: pdm.PageXMLScan) -> pdm.PageXMLScan:
    new_scan = pdm.PageXMLScan(doc_id=scan.id,
                               doc_type=copy.deepcopy(scan.type),
                               metadata=copy.deepcopy(scan.metadata),
                               coords=copy.deepcopy(scan.coords),
                               text_regions=[copy_text_region(tr) for tr in scan.text_regions])
    new_scan.type = copy.deepcopy(scan.type)
    return new_scan


def copy_page(page: pdm.PageXMLPage) -> pdm.PageXMLPage:
    new_page = pdm.PageXMLPage(doc_id=page.id,
                               doc_type=copy.deepcopy(page.type),
                               metadata=copy.deepcopy(page.metadata),
                               coords=copy.deepcopy(page.coords),
                               text_regions=[copy_text_region(tr) for tr in page.text_regions],
                               extra=[copy_region(r) for r in page.extra],
                               columns=[copy_column(col) for col in page.columns])
    new_page.type = copy.deepcopy(page.type)
    return new_page


def copy_column(col: pdm.PageXMLColumn) -> pdm.PageXMLColumn:
    new_col = pdm.PageXMLColumn(doc_id=col.id,
                                doc_type=copy.deepcopy(col.type),
                                metadata=copy.deepcopy(col.metadata),
                                coords=copy.deepcopy(col.coords),
                                text_regions=[copy_text_region(tr) for tr in col.text_regions])
    new_col.type = copy.deepcopy(col.type)
    return new_col


def copy_region(region: pdm.PageXMLRegion) -> pdm.PageXMLRegion:
    """
    print(f"pagexml_helper.copy_region - region: {region.id} {region.__class__.__name__}")
    print(f"\tPageXMLTextRegion: {isinstance(region, pdm.PageXMLTextRegion)}")
    print(f"\tPageXMLTableRegion: {isinstance(region, pdm.PageXMLTableRegion)}")
    print(f"\tPageXMLEmptyRegion: {isinstance(region, pdm.PageXMLEmptyRegion)}")
    """
    if isinstance(region, pdm.PageXMLTextRegion):
        return copy_text_region(region)
    if isinstance(region, pdm.PageXMLTableRegion):
        return copy_table_region(region)
    if isinstance(region, pdm.PageXMLEmptyRegion):
        return copy_empty_region(region)
    new_region = pdm.PageXMLRegion(doc_id=region.id, doc_type=copy.deepcopy(region.type),
                                   metadata=copy.deepcopy(region.metadata), coords=copy.deepcopy(region.coords),
                                   text_regions=[copy_text_region(tr) for tr in region.text_regions],
                                   table_regions=[copy_table_region(tr) for tr in region.table_regions],
                                   empty_regions=[copy_empty_region(r) for r in region.empty_regions],
                                   orientation=region.orientation, reading_order=region.reading_order,
                                   reading_order_attributes=region.reading_order_attributes)
    return new_region


def copy_empty_region(region: pdm.PageXMLEmptyRegion):
    new_region = pdm.PageXMLEmptyRegion(doc_id=region.id, doc_type=copy.deepcopy(region.type),
                                        metadata=copy.deepcopy(region.metadata), coords=copy.deepcopy(region.coords),
                                        orientation=region.orientation)
    return new_region


def copy_text_region(tr: pdm.PageXMLTextRegion) -> pdm.PageXMLTextRegion:
    new_tr = pdm.PageXMLTextRegion(doc_id=tr.id,
                                   doc_type=copy.deepcopy(tr.type),
                                   metadata=copy.deepcopy(tr.metadata),
                                   coords=copy.deepcopy(tr.coords),
                                   lines=[copy_line(line) for line in tr.lines],
                                   text_regions=[copy_text_region(tr) for tr in tr.text_regions])
    new_tr.type = copy.deepcopy(tr.type)
    return new_tr


def copy_table_region(tr: pdm.PageXMLTableRegion) -> pdm.PageXMLTableRegion:
    new_tr = pdm.PageXMLTableRegion(doc_id=tr.id,
                                    doc_type=copy.deepcopy(tr.type),
                                    metadata=copy.deepcopy(tr.metadata),
                                    coords=copy.deepcopy(tr.coords),
                                    rows=[copy_row(row) for row in tr.rows])
    new_tr.type = copy.deepcopy(tr.type)
    return new_tr


def copy_row(row: pdm.PageXMLTableRow) -> pdm.PageXMLTableRow:
    new_row = pdm.PageXMLTableRow(doc_id=row.id, metadata=row.metadata, coords=row.coords,
                                  attrs=row.attrs, cells=[copy_cell(cell) for cell in row.cells],
                                  orientation=row.orientation)
    return new_row


def copy_cell(cell: pdm.PageXMLTableCell) -> pdm.PageXMLTableCell:
    new_cell = pdm.PageXMLTableCell(doc_id=cell.id, metadata=cell.metadata, coords=cell.coords,
                                    attrs=cell.attrs, lines=[copy_line(line) for line in cell.lines],
                                    orientation=cell.orientation)
    new_cell.col = cell.col
    return new_cell


def copy_line(line: pdm.PageXMLTextLine) -> pdm.PageXMLTextLine:
    new_line = pdm.PageXMLTextLine(doc_id=line.id,
                                   doc_type=copy.deepcopy(line.type),
                                   metadata=copy.deepcopy(line.metadata),
                                   coords=copy.deepcopy(line.coords), baseline=copy.deepcopy(line.baseline),
                                   text=line.text,
                                   words=[copy_word(word) for word in line.words] if line.words else None)
    new_line.type = copy.deepcopy(line.type)
    return new_line


def copy_word(word: pdm.PageXMLWord) -> pdm.PageXMLWord:
    new_word = pdm.PageXMLWord(doc_id=word.id,
                               doc_type=copy.deepcopy(word.type),
                               metadata=copy.deepcopy(word.metadata), conf=word.conf,
                               coords=copy.deepcopy(word.coords), text=word.text)
    new_word.type = copy.deepcopy(word.type)
    return new_word
