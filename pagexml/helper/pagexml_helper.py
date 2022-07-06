from typing import Dict, Generator, List, Tuple, Union
from collections import Counter
import copy
import re
import gzip
import string

import numpy as np

import pagexml.model.physical_document_model as pdm
import pagexml.analysis.layout_stats as summarise
import pagexml.analysis.text_stats as text_stats
import pagexml.helper.text_helper as text_helper


def sort_regions_in_reading_order(doc: pdm.PageXMLDoc) -> List[pdm.PageXMLTextRegion]:
    doc_text_regions: List[pdm.PageXMLTextRegion] = []
    if doc.reading_order and hasattr(doc, 'text_regions') and doc.text_regions:
        text_region_ids = [region for _index, region in sorted(doc.reading_order.items(), key=lambda x: x[0])]
        return [tr for tr in sorted(doc.text_regions, key=lambda x: text_region_ids.index(x.id))]
    if hasattr(doc, 'columns') and sorted(doc.columns):
        doc_text_regions = doc.columns
    elif hasattr(doc, 'text_regions') and doc.text_regions:
        doc_text_regions = doc.text_regions
    if doc_text_regions:
        sub_text_regions = []
        for text_region in sorted(doc_text_regions, key=lambda x: x.coords.left):
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
        for line in lines:
            print(line.coords.box, line.text)
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


def sort_lines_in_reading_order(doc: pdm.PageXMLDoc) -> Generator[pdm.PageXMLTextLine, None, None]:
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
        stacked_lines = horizontal_group_lines(text_region.lines)
        for lines in stacked_lines:
            for line in sorted(lines, key=lambda x: x.coords.left):
                yield line


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


def pretty_print_textregion(text_region: pdm.PageXMLTextRegion, print_stats: bool = False) -> None:
    """Pretty print the text of a text region, using indentation and
    vertical space based on the average character width and average
    distance between lines. If no corresponding images of the PageXML
    are available, this can serve as a visual approximation to reveal
    the page layout.

    :param text_region: a TextRegion object that contains TextLines
    :type text_region: PageXMLTextRegion
    :param print_stats: flag to print text_region statistics if set to True
    :type print_stats: bool
    """
    if print_stats:
        print_textregion_stats(text_region)
    avg_line_distance = summarise.get_textregion_avg_line_distance(text_region)
    avg_char_width = summarise.get_textregion_avg_char_width(text_region)
    for ti, tr in enumerate(text_region.get_inner_text_regions()):
        if len(tr.lines) < 2:
            continue
        for li, curr_line in enumerate(tr.lines[:-1]):
            next_line = tr.lines[li + 1]
            left_indent = (curr_line.coords.left - tr.coords.left)
            if left_indent > 0 and avg_char_width > 0:
                preceding_whitespace = " " * int(float(left_indent) / avg_char_width)
            else:
                preceding_whitespace = ""
            distances = summarise.compute_baseline_distances(curr_line.baseline, next_line.baseline)
            if curr_line.text is None:
                print()
            else:
                print(preceding_whitespace, curr_line.text)
            if np.median(distances) > avg_line_distance * 1.2:
                print()
    print()


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


def json_to_pagexml_word(json_doc: dict) -> pdm.PageXMLWord:
    word = pdm.PageXMLWord(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                           text=json_doc['text'])
    return word


def json_to_pagexml_line(json_doc: dict) -> pdm.PageXMLTextLine:
    words = [json_to_pagexml_word(word) for word in json_doc['words']] if 'words' in json_doc else []
    reading_order = json_doc['reading_order'] if 'reading_order' in json_doc else {}
    try:
        line = pdm.PageXMLTextLine(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                                   coords=pdm.Coords(json_doc['coords']), baseline=pdm.Baseline(json_doc['baseline']),
                                   text=json_doc['text'], words=words, reading_order=reading_order)
        return line
    except TypeError:
        print(json_doc['baseline'])
        raise


def json_to_pagexml_text_region(json_doc: dict) -> pdm.PageXMLTextRegion:
    text_regions = [json_to_pagexml_text_region(text_region) for text_region in json_doc['text_regions']] \
        if 'text_regions' in json_doc else []
    lines = [json_to_pagexml_line(line) for line in json_doc['lines']] if 'lines' in json_doc else []
    reading_order = json_doc['reading_order'] if 'reading_order' in json_doc else {}

    text_region = pdm.PageXMLTextRegion(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                                        coords=pdm.Coords(json_doc['coords']), text_regions=text_regions, lines=lines,
                                        reading_order=reading_order)
    pdm.set_parentage(text_region)
    return text_region


def json_to_pagexml_column(json_doc: dict) -> pdm.PageXMLColumn:
    text_regions = [json_to_pagexml_text_region(text_region) for text_region in json_doc['text_regions']] \
        if 'text_regions' in json_doc else []
    lines = [json_to_pagexml_line(line) for line in json_doc['lines']] if 'lines' in json_doc else []
    reading_order = json_doc['reading_order'] if 'reading_order' in json_doc else {}

    column = pdm.PageXMLColumn(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                               coords=pdm.Coords(json_doc['coords']), text_regions=text_regions, lines=lines,
                               reading_order=reading_order)
    pdm.set_parentage(column)
    return column


def json_to_columns_regions_lines(json_doc: dict) -> Tuple[list, list, list, dict, pdm.Coords]:
    columns = [json_to_pagexml_column(column) for column in json_doc['columns']] if 'columns' in json_doc else []
    text_regions = [json_to_pagexml_text_region(text_region) for text_region in json_doc['text_regions']] \
        if 'text_regions' in json_doc else []
    lines = [json_to_pagexml_line(line) for line in json_doc['lines']] if 'lines' in json_doc else []
    reading_order = json_doc['reading_order'] if 'reading_order' in json_doc else {}
    coords = pdm.Coords(json_doc['coords']) if 'coords' in json_doc else None
    return columns, text_regions, lines, reading_order, coords


def json_to_pagexml_page(json_doc: dict) -> pdm.PageXMLPage:
    extra = [json_to_pagexml_text_region(text_region) for text_region in json_doc['extra']] \
        if 'extra' in json_doc else []
    columns, text_regions, lines, reading_order, coords = json_to_columns_regions_lines(json_doc)
    page = pdm.PageXMLPage(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                           coords=coords, extra=extra, columns=columns,
                           text_regions=text_regions, lines=lines,
                           reading_order=reading_order)
    pdm.set_parentage(page)
    return page


def json_to_pagexml_scan(json_doc: dict) -> pdm.PageXMLScan:
    pages = [json_to_pagexml_page(page) for page in json_doc['pages']] if 'pages' in json_doc else []
    columns, text_regions, lines, reading_order, coords = json_to_columns_regions_lines(json_doc)
    scan = pdm.PageXMLScan(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                           coords=coords, pages=pages, columns=columns,
                           text_regions=text_regions, lines=lines, reading_order=reading_order)
    pdm.set_parentage(scan)
    return scan


def json_to_pagexml_doc(json_doc: dict) -> pdm.PageXMLDoc:
    if 'pagexml_doc' not in json_doc['type']:
        raise TypeError('json_doc is not of type "pagexml_doc".')
    if 'scan' in json_doc['type']:
        return json_to_pagexml_scan(json_doc)
    if 'page' in json_doc['type']:
        return json_to_pagexml_page(json_doc)
    if 'column' in json_doc['type']:
        return json_to_pagexml_column(json_doc)
    if 'text_region' in json_doc['type']:
        return json_to_pagexml_text_region(json_doc)
    if 'line' in json_doc['type']:
        return json_to_pagexml_line(json_doc)
    if 'word' in json_doc['type']:
        return json_to_pagexml_word(json_doc)


def pagexml_to_line_format(pagexml_doc: pdm.PageXMLTextRegion) -> Generator[Tuple[str, str, str], None, None]:
    for line in pagexml_doc.get_lines():
        yield pagexml_doc.id, line.id, line.text


def write_pagexml_to_line_format(pagexml_docs: List[pdm.PageXMLTextRegion], output_file: str) -> None:
    with gzip.open(output_file, 'wt') as fh:
        for pagexml_doc in pagexml_docs:
            for doc_id, line_id, line_text in pagexml_to_line_format(pagexml_doc):
                fh.write(f"{doc_id}\t{line_id}\t{line_text}\n")


def read_line_format_file(line_format_files: Union[str, List[str]]) -> Generator[Tuple[str, str, str], None, None]:
    if isinstance(line_format_files, str):
        line_format_files = [line_format_files]
    for line_format_file in line_format_files:
        with gzip.open(line_format_file, 'rt') as fh:
            for line in fh:
                yield line.strip().split('\t')


def make_line_text(line: pdm.PageXMLTextLine, do_merge: bool,
                   end_word: str, merge_word: str, line_break_char: str = '-') -> str:
    line_text = line.text
    if len(line_text) >= 2 and line_text.endswith(line_break_char*2):
        # remove the redundant line break char
        line_text = line_text[:-1]
    if do_merge:
        if line_text[-1] == line_break_char and merge_word.startswith(end_word) is False:
            # the merge word does not contain a line break char, so remove it from the line
            # before adding it to the text
            line_text = line_text[:-1]
        else:
            # the line contains no line break char or the merge word contains the hyphen as
            # well, so leave it in.
            line_text = line.text
    else:
        # no need to meed so add line with trailing whitespace
        if line_text[-1] == line_break_char and len(line_text) >= 2 and line_text[-2] != ' ':
            # the line break char at the end is trailing, so disconnect it from the preceding word
            line_text = line_text[:-1] + f' {line_break_char} '
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
                          lbd: text_stats.LineBreakDetector) -> Tuple[Union[str, None], List[Dict[str, any]]]:
    text = ''
    line_ranges = []
    if len(lines) == 0:
        return None, []
    prev_line = lines[0]
    prev_words = text_helper.get_line_words(prev_line.text) if prev_line.text else []
    if len(lines) > 1:
        for curr_line in lines[1:]:
            if curr_line.text is None:
                curr_words = []
                prev_line_text = prev_line.text if prev_line.text else ''
            else:
                curr_words = text_helper.get_line_words(curr_line.text,
                                                        line_break_char=lbd.line_break_char)
                if prev_line.text is not None:
                    do_merge, merge_word = text_stats.determine_line_break(lbd, curr_words, prev_words)
                    prev_line_text = make_line_text(prev_line, do_merge, prev_words[-1], merge_word)
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


def merge_lines(lines: List[pdm.PageXMLTextLine], remove_line_break: bool = False,
                line_break_char: str = '-'):
    coords = pdm.parse_derived_coords(lines)
    text = ''
    for li, curr_line in enumerate(lines):
        if remove_line_break and len(text) > 0 and text.endswith(line_break_char):
            if curr_line.text[0].islower():
                # remove hyphen
                text = text[:-1]
        text += curr_line.text
    return pdm.PageXMLTextLine(metadata=copy.deepcopy(lines[0].metadata),
                               coords=coords, text=text)
