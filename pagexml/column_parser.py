import copy
import re
from collections import Counter
from typing import Dict, List, Tuple

import pagexml.model.physical_document_model as pdm
import pagexml.helper.pagexml_helper as pagexml_helper


def within_column(line: pdm.PageXMLTextLine, column_range: Dict[str, int],
                  overlap_threshold: float = 0.5):
    """Determine if a given line is within the horizontal range of a column."""
    start = max([line.coords.left, column_range["start"]])
    end = min([line.coords.right, column_range["end"]])
    overlap = end - start if end > start else 0
    return overlap / line.coords.width > overlap_threshold


def find_overlapping_columns(columns: List[pdm.PageXMLColumn]):
    columns.sort()
    merge_sets = []
    for ci, curr_col in enumerate(columns[:-1]):
        next_col = columns[ci+1]
        if pdm.is_horizontally_overlapping(curr_col, next_col):
            for merge_set in merge_sets:
                if curr_col in merge_set:
                    merge_set.append(next_col)
                    break
            else:
                merge_sets.append([curr_col, next_col])
    return merge_sets


#################################################
# Identifying columns using pixel distributions #
#################################################


def compute_pixel_dist(lines: List[pdm.PageXMLTextLine]) -> Counter:
    """Count how many lines are above each horizontal pixel coordinate."""
    pixel_dist = Counter()
    for line in lines:
        pixel_dist.update([pixel for pixel in range(line.coords.left, line.coords.right + 1)])
    return pixel_dist


def new_gap_pixel_interval(pixel: int) -> dict:
    return {"start": pixel, "end": pixel}


def determine_freq_gap_interval(pixel_dist: Counter, gap_threshold: int) -> list:
    common_pixels = sorted([pixel for pixel, freq in pixel_dist.items()])
    gap_pixel_intervals = []
    if len(common_pixels) == 0:
        return gap_pixel_intervals
    curr_interval = new_gap_pixel_interval(common_pixels[0])
    prev_interval_end = 0
    for curr_index, curr_pixel in enumerate(common_pixels[:-1]):
        next_pixel = common_pixels[curr_index + 1]
        if next_pixel - curr_pixel < gap_threshold:
            curr_interval["end"] = next_pixel
        else:
            if curr_interval["start"] - prev_interval_end < gap_threshold:
                continue
            gap_pixel_intervals += [curr_interval]
            prev_interval_end = curr_interval["end"]
            curr_interval = new_gap_pixel_interval(next_pixel)
    gap_pixel_intervals += [curr_interval]
    return gap_pixel_intervals


def find_column_gaps(lines: List[pdm.PageXMLTextLine], gap_threshold: int = 50):
    gap_pixel_dist = compute_pixel_dist(lines)
    gap_pixel_intervals = determine_freq_gap_interval(gap_pixel_dist, gap_threshold)
    return gap_pixel_intervals


def column_bounding_box_surrounds_lines(column: pdm.PageXMLColumn) -> bool:
    """Check if the column coordinates contain the coordinate
    boxes of the column lines."""
    for line in column.get_lines():
        if not pagexml_helper.elements_overlap(column, line, threshold=0.6):
            return False
    return True


def is_text_column(column: pdm.PageXMLColumn) -> bool:
    """Check if there is at least one alpha-numeric word on the page."""
    # num_chars = 0
    num_alpha_words = 0
    for line in column.get_lines():
        if line.text:
            try:
                words = [word for word in re.split(r'\W+', line.text.strip()) if len(word) > 1]
            except re.error:
                print(line.text)
                raise
            num_alpha_words += len(words)
            # num_chars += len(line.text)
    # return num_chars >= 20
    return num_alpha_words > 0


def is_full_text_column(column: pdm.PageXMLColumn,
                        page: pdm.PageXMLTextRegion = None,
                        num_page_cols: int = 2) -> bool:
    """Check if a page column is a full-text column (running from top to bottom of page)."""
    between_cols_margin = 300 * (num_page_cols - 1)
    if page is None and column.parent is not None:
        page = column.parent
    if page is None:
        raise ValueError(f'no information on parent of column {column.id}')
    lines = page.get_lines()
    left = min([line.coords.left for line in lines])
    right = min([line.coords.right for line in lines])
    page_text_width = right - left
    full_column_text_width = (page_text_width - between_cols_margin) / num_page_cols
    if column.coords.width < full_column_text_width - 80:
        # narrow column is not a normal text column
        return False
    if column.coords.height > 2500:
        # full page-height column
        return True
    if column.coords.height / column.stats['lines'] > 100:
        # lines are far apart, probably something wrong
        return False
    if column.coords.width > 700 and column.stats['lines'] > 30:
        return True


def is_noise_column(column: pdm.PageXMLColumn) -> bool:
    """Check if columns contains only very short lines."""
    for line in column.get_lines():
        if line.text and len(line.text) > 3:
            return False
    return True


def is_header_footer_column(column: pdm.PageXMLColumn) -> bool:
    """Check if a column is a header or footer."""
    if column.coords.top <= 600:
        if column.coords.bottom > 600:
            # column is too low for top margin header
            return False
    if column.coords.bottom >= 3200:
        if column.coords.top < 3200:
            # column is too high for bottom margin footer
            return False
    if is_text_column(column):
        return False
    if column.stats['lines'] > 4:
        return False
    if column.coords.width > 500 and column.coords.height > 150:
        return False
    return True


def determine_column_type(column: pdm.PageXMLColumn) -> str:
    """Determine whether a column is a full-text column, margin column
    or extra text column."""
    if is_full_text_column(column):
        return 'full_text'
    elif is_text_column(column):
        return 'extra_text'
    elif is_header_footer_column(column):
        return 'header_footer'
    elif is_noise_column(column):
        return 'noise_column'
    else:
        print('Bounding box:', column.coords.box)
        print('Stats:', column.stats)
        num_chars = 0
        for line in column.get_lines():
            print(line.coords.box, line.text)
            num_chars += len(line.text)
        print('num_chars:', num_chars)
        raise TypeError('unknown column type')


def make_derived_column(lines: List[pdm.PageXMLTextLine], metadata: dict, page_id: str) -> pdm.PageXMLColumn:
    """Make a new PageXMLColumn based on a set of lines, column metadata and a page_id."""
    coords = pdm.parse_derived_coords(lines)
    column = pdm.PageXMLColumn(metadata=metadata, coords=coords, lines=lines)
    column.set_derived_id(page_id)
    return column


def merge_columns(columns: List[pdm.PageXMLColumn],
                  doc_id: str, metadata: dict) -> pdm.PageXMLColumn:
    """Merge two columns into one, sorting lines by baseline height."""
    merged_lines = [line for col in columns for line in col.get_lines()]
    merged_lines = list(set(merged_lines))
    sorted_lines = sorted(merged_lines, key=lambda x: x.baseline.y)
    merged_coords = pdm.parse_derived_coords(sorted_lines)
    merged_col = pdm.PageXMLColumn(doc_id=doc_id, doc_type='index_column',
                                   metadata=metadata, coords=merged_coords,
                                   lines=merged_lines)
    return merged_col


def sort_lines_in_column_ranges(lines: List[pdm.PageXMLTextLine],
                                column_ranges: List[Dict[str, int]],
                                overlap_threshold: float,
                                debug: bool = False) -> Tuple[List[List[pdm.PageXMLTextLine]], List[pdm.PageXMLTextLine]]:
    column_lines = [[] for _ in range(len(column_ranges))]
    extra_lines = []
    append_count = 0
    for line in lines:
        index = None
        for column_range in column_ranges:
            if line.coords.width == 0:
                continue
            if within_column(line, column_range, overlap_threshold=overlap_threshold):
                index = column_ranges.index(column_range)
                column_lines[index].append(line)
                append_count += 1
        if index is None:
            extra_lines.append(line)
            append_count += 1
    if debug:
        print('RANGE SPLIT num_lines:', len(lines), 'append_count:', append_count)
        for ci, lines in enumerate(column_lines):
            print('\tcolumn', ci, '\tlines:', len(lines))
        print('\textra lines:', len(extra_lines))
    return column_lines, extra_lines


def merge_overlapping_columns(text_region: pdm.PageXMLTextRegion, columns: List[pdm.PageXMLColumn]):
    # column range may have expanded with lines partially overlapping initial range
    # check which extra lines should be added to columns
    merge_sets = find_overlapping_columns(columns)
    merge_cols = {col for merge_set in merge_sets for col in merge_set}
    non_overlapping_cols = [col for col in columns if col not in merge_cols]
    for merge_set in merge_sets:
        merged_col = merge_columns(merge_set, "temp_id", merge_set[0].metadata)
        if text_region.parent and text_region.parent.id:
            merged_col.set_derived_id(text_region.parent.id)
            merged_col.set_parent(text_region.parent)
        else:
            merged_col.set_derived_id(text_region.id)
        non_overlapping_cols.append(merged_col)
    return non_overlapping_cols


def make_column_range_columns(text_region: pdm.PageXMLTextRegion,
                              column_lines: List[List[pdm.PageXMLTextLine]]) -> List[pdm.PageXMLColumn]:
    columns = []
    for lines in column_lines:
        if len(lines) == 0:
            continue
        coords = pdm.parse_derived_coords(lines)
        column = pdm.PageXMLColumn(doc_type=copy.deepcopy(text_region.type),
                                   metadata=copy.deepcopy(text_region.metadata),
                                   coords=coords, lines=lines)
        if text_region.parent and text_region.parent.id:
            column.set_derived_id(text_region.parent.id)
            column.set_parent(text_region.parent)
        else:
            column.set_derived_id(text_region.id)
        columns.append(column)
    columns = merge_overlapping_columns(text_region, columns)
    return columns


def handle_extra_lines(text_region: pdm.PageXMLTextRegion,
                       columns: List[pdm.PageXMLColumn],
                       extra_lines: List[pdm.PageXMLTextLine],
                       gap_threshold: int = 50,
                       debug: bool = False):
    non_col_lines = []
    if debug:
        print("NUM COLUMNS:", len(columns))
        print("EXTRA LINES BEFORE:", len(extra_lines))
        for line in extra_lines:
            print('\tEXTRA LINE:', line.text, line.coords)
    append_count = 0
    for line in extra_lines:
        best_overlap = 0
        best_column = None
        for column in columns:
            # print("EXTRA LINE CHECKING OVERLAP:", line.coords.left, line.coords.right,
            #       column.coords.left, column.coords.right)
            overlap = pdm.get_horizontal_overlap(line, column)
            # print('\tOVERLAP', overlap)
            if overlap > best_overlap:
                if best_column is None or column.coords.width < best_column.coords.width:
                    best_column = column
                    best_overlap = overlap
                    # print('\t\tBEST', best_column)
        if best_column is not None and pdm.is_horizontally_overlapping(line, best_column):
            best_column.lines.append(line)
            append_count += 1
            best_column.coords = pdm.parse_derived_coords(best_column.lines)
            if text_region.parent:
                best_column.set_derived_id(text_region.parent.id)
        else:
            # print(f"APPENDING NON-COL LINE: {line.coords.left}-{line.coords.right}\t{line.coords.y}\t{line.text}")
            non_col_lines.append(line)
            append_count += 1
    if debug is True:
        print('append_count:', append_count)
    extra_lines = non_col_lines
    if debug is True:
        print("EXTRA LINES AFTER:", len(extra_lines))
    extra = None
    if len(extra_lines) > 0:
        try:
            coords = pdm.parse_derived_coords(extra_lines)
        except BaseException:
            for line in extra_lines:
                print(line.coords.box, line.text)
            raise ValueError('Cannot generate column coords for extra lines')
        extra = pdm.PageXMLTextRegion(metadata=text_region.metadata, coords=coords,
                                      lines=extra_lines)
        if text_region.parent and text_region.parent.id:
            extra.set_derived_id(text_region.parent.id)
            extra.set_parent(text_region.parent)
        else:
            extra.set_derived_id(text_region.id)
        # for line in extra.lines:
        #     print(f"RETURNING EXTRA LINE: {line.coords.left}-{line.coords.right}\t{line.coords.y}\t{line.text}")
        if debug:
            print('SPLITTING EXTRA')
        extra_cols = split_lines_on_column_gaps(extra, gap_threshold=gap_threshold)
        for extra_col in extra_cols:
            if debug:
                print('\tEXTRA COL AFTER EXTRA SPLIT:', extra_col.stats)
            extra_col.set_parent(text_region.parent)
            if text_region.parent:
                extra_col.set_derived_id(text_region.parent.id)
        columns += extra_cols
        extra = None
    if extra is not None:
        print('source doc:', text_region.id)
        print(extra)
        raise TypeError(f'Extra is not None but {type(extra)}')
    return columns


def split_lines_on_column_gaps(text_region: pdm.PageXMLTextRegion,
                               gap_threshold: int = 50,
                               overlap_threshold: float = 0.5) -> List[pdm.PageXMLColumn]:
    """Takes a PageXMLTextRegion object and tries to split the lines into columns based
    on a minimum horizontal gap (in number of pixels) between columns.

    :param text_region: a text region with lines (as direct children or as deeper descendants).
    :type text_region: PageXMLTextRegion
    :param gap_threshold: the minimum number of horizontal pixels between columns of horizontally
    aligned lines to be considered a column boundary. Default is 50.
    :type gap_threshold: int
    :param overlap_threshold: the minimum overlap ratio between two lines to be considered
    horizontally aligned (i.e. part of the same 'column'). Default is 0.5, that is, two lines need
    to horizontally overlap at least 50% of the shortest line.
    :type overlap_threshold: float
    """
    column_ranges = find_column_gaps(text_region.get_lines(), gap_threshold=gap_threshold)
    column_ranges = [col_range for col_range in column_ranges if col_range["end"] - col_range["start"] >= 20]
    column_lines, extra_lines = sort_lines_in_column_ranges(text_region.get_lines(),
                                                            column_ranges,
                                                            overlap_threshold)
    columns = make_column_range_columns(text_region, column_lines)
    columns = handle_extra_lines(text_region, columns, extra_lines)
    return columns
