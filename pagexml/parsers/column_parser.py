import copy
from typing import  List
from collections import Counter

import pagexml.model.physical_document_model as pdm


def compute_text_pixel_dist(lines: List[pdm.PageXMLTextLine]) -> Counter:
    """Count how many lines are above each horizontal pixel coordinate."""
    pixel_dist = Counter()
    for line in lines:
        if line.coords is None and line.baseline is None:
            continue
        left, right = None, None
        if line.coords:
            left = line.coords.left
            right = line.coords.right
        if line.baseline:
            if left is None or line.baseline.left > left:
                left = line.coords.left
            if right is None or line.baseline.right > right:
                right = line.coords.right
        pixel_dist.update([pixel for pixel in range(left, right + 1)])
    return pixel_dist


def new_text_pixel_interval(pixel: int) -> pdm.Interval:
    return pdm.Interval('text_pixel', pixel, pixel)


def find_column_ranges(lines: List[pdm.PageXMLTextLine], min_column_lines: int = 2,
                       min_gap_width: int = 20, min_column_width: int = 20,
                       debug: int = 0) -> List[pdm.Interval]:
    text_pixel_dist = compute_text_pixel_dist(lines)
    common_pixels = sorted([pixel for pixel, freq in text_pixel_dist.items() if freq >= min_column_lines])
    if debug > 2:
        print("determine_column_ranges - common_pixels:", common_pixels)
    column_ranges = []
    if len(common_pixels) == 0:
        return column_ranges
    curr_text_interval = new_text_pixel_interval(common_pixels[0])
    if debug > 2:
        print("determine_column_ranges - curr_text_interval:", curr_text_interval)
    prev_interval_end = 0
    for curr_index, curr_pixel in enumerate(common_pixels[:-1]):
        next_pixel = common_pixels[curr_index + 1]
        if debug > 2:
            print("determine_column_ranges - curr:", curr_pixel, "next:", next_pixel, "start:",
                  curr_text_interval.start, "end:", curr_text_interval.end, "prev_end:", prev_interval_end)
        if next_pixel - curr_pixel < min_gap_width:
            curr_text_interval = pdm.Interval('text_pixel', curr_text_interval.start, next_pixel)
        else:
            if curr_text_interval.start - prev_interval_end < min_gap_width:
                if debug > 2:
                    print("determine_column_ranges - skipping interval:", curr_text_interval, "\tcurr_pixel:",
                          curr_pixel, "next_pixel:", next_pixel)
                continue
            if debug > 2:
                print("determine_column_ranges - adding interval:", curr_text_interval, "\tcurr_pixel:",
                      curr_pixel, "next_pixel:", next_pixel)
            if curr_text_interval.end - curr_text_interval.start >= min_column_width:
                column_ranges += [curr_text_interval]
                prev_interval_end = curr_text_interval.end
            curr_text_interval = new_text_pixel_interval(next_pixel)
    if curr_text_interval.end - curr_text_interval.start >= min_column_width:
        column_ranges += [curr_text_interval]
    return column_ranges


def find_column_gaps(lines: List[pdm.PageXMLTextLine],
                     min_column_lines: int = 2, min_gap_width: int = 20,
                     min_column_width: int = 20, debug: int = 0):
    column_ranges = find_column_ranges(lines, min_column_lines=min_column_lines,
                                       min_gap_width=min_gap_width, min_column_width=min_column_width,
                                       debug=debug)
    if len(column_ranges) < 2:
        return []
    gap_ranges = []
    for ci, curr_range in enumerate(column_ranges[:-1]):
        next_range = column_ranges[ci+1]
        if next_range.start - curr_range.end >= min_gap_width:
            gap_range = pdm.Interval('text_pixel', start=curr_range.end, end=next_range.start)
            gap_ranges.append(gap_range)
    return gap_ranges


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


def merge_columns(columns: List[pdm.PageXMLColumn],
                  doc_id: str, metadata: dict, lines_only: bool = False) -> pdm.PageXMLColumn:
    """Merge two columns into one, sorting lines by baseline height."""
    if lines_only is True:
        merged_lines = [line for col in columns for line in col.get_lines()]
        merged_lines = list(set(merged_lines))
        sorted_lines = sorted(merged_lines, key=lambda x: x.baseline.y)
        merged_coords = pdm.parse_derived_coords(sorted_lines)
        merged_col = pdm.PageXMLColumn(doc_id=doc_id,
                                       metadata=metadata, coords=merged_coords,
                                       lines=merged_lines)
    else:
        merged_trs = [tr for col in columns for tr in col.text_regions]
        sorted_trs = sorted(merged_trs, key=lambda x: x.coords.y)
        merged_lines = [line for col in columns for line in col.lines]
        sorted_lines = sorted(merged_lines, key=lambda x: x.baseline.y)
        try:
            merged_coords = pdm.parse_derived_coords(sorted_trs + sorted_lines)
        except IndexError:
            print(f"pagexml_helper.merge_column - Error deriving coords from trs and lines:")
            print(f"    number of trs: {len(sorted_trs)}")
            print(f"    number of lines: {len(sorted_lines)}")
            for tr in sorted_trs:
                print(f"\ttr {tr.id}\tnumber of points: {len(tr.coords.points)}")
            for line in sorted_lines:
                print(f"\tline {line.id}\tnumber of points: {len(line.coords.points)}")
            raise
        merged_col = pdm.PageXMLColumn(doc_id=doc_id,
                                       metadata=metadata, coords=merged_coords,
                                       text_regions=sorted_trs, lines=sorted_lines)

    for col in columns:
        for col_type in col.types:
            if col_type not in merged_col.type:
                merged_col.add_type(col_type)
    return merged_col


def add_line_to_column(line: pdm.PageXMLTextLine, column: pdm.PageXMLColumn) -> None:
    for tr in column.text_regions:
        if pdm.is_horizontally_overlapping(line, tr, threshold=0.1) and \
                pdm.is_vertically_overlapping(line, tr, threshold=0.1):
            tr.lines.append(line)
            tr.set_as_parent(tr.lines)
            tr.lines.sort()
            return None
    new_tr = pdm.PageXMLTextRegion(metadata=copy.deepcopy(column.metadata),
                                   coords=pdm.parse_derived_coords([line]),
                                   lines=[line])
    new_tr.set_derived_id(column.metadata['scan_id'])
    column.text_regions.append(new_tr)
    column.set_as_parent([new_tr])
    column.text_regions.sort()


def split_lines_on_column_gaps(text_region: pdm.PageXMLTextRegion,
                               min_column_lines: int = 2,
                               min_gap_width: int = 20,
                               min_column_width: int = 20,
                               overlap_threshold: float = 0.5,
                               ignore_bad_coordinate_lines: bool = True,
                               debug: int = 0) -> List[pdm.PageXMLColumn]:
    lines = [line for line in text_region.get_lines()]
    if 'scan_id' not in text_region.metadata:
        raise KeyError(f'no "scan_id" in text_region {text_region.id}')
    column_ranges = find_column_ranges(lines, min_column_lines=min_column_lines, min_gap_width=min_gap_width,
                                       min_column_width=min_column_width, debug=debug-1)
    if debug > 0:
        print('split_lines_on_column_gaps - text_region:', text_region.id, text_region.stats)
        print("COLUMN RANGES:", column_ranges)
    column_lines = [[] for _ in range(len(column_ranges))]
    extra_lines = []
    num_lines = text_region.stats['lines']
    append_count = 0
    for line in lines:
        index = None
        for column_range in column_ranges:
            if line.coords.width == 0:
                if debug:
                    print("ZERO WIDTH LINE:", line.coords.box, line.text)
                continue

            if pdm.within_interval(line, column_range, overlap_threshold=overlap_threshold):
                index = column_ranges.index(column_range)
                column_lines[index].append(line)
                append_count += 1
        if index is None:
            extra_lines.append(line)
            append_count += 1
            # print(f"APPENDING EXTRA LINE: {line.coords.left}-{line.coords.right}\t{line.coords.y}\t{line.text}")
    columns = []
    if debug > 0:
        print('RANGE SPLIT num_lines:', num_lines, 'append_count:', append_count)
        for ci, lines in enumerate(column_lines):
            print('\tcolumn', ci, '\tlines:', len(lines))
        print('\textra lines:', len(extra_lines))
    for lines in column_lines:
        if len(lines) == 0:
            continue
        coords = pdm.parse_derived_coords(lines)
        tr = pdm.PageXMLTextRegion(doc_type=copy.deepcopy(text_region.type),
                                   metadata=copy.deepcopy(text_region.metadata),
                                   coords=copy.deepcopy(coords), lines=lines)
        tr.set_derived_id(text_region.metadata['scan_id'])
        tr.set_as_parent(lines)
        column = pdm.PageXMLColumn(doc_type=copy.deepcopy(text_region.type),
                                   metadata=copy.deepcopy(text_region.metadata),
                                   coords=copy.deepcopy(coords), text_regions=[tr])
        if text_region.parent and text_region.parent.id:
            column.set_derived_id(text_region.parent.id)
            column.set_parent(text_region.parent)
        else:
            column.set_derived_id(text_region.id)
        column.set_as_parent(column.text_regions)
        columns.append(column)
    # column range may have expanded with lines partially overlapping initial range
    # check which extra lines should be added to columns
    non_col_lines = []
    merge_sets = find_overlapping_columns(columns)
    merge_cols = {col for merge_set in merge_sets for col in merge_set}
    non_overlapping_cols = [col for col in columns if col not in merge_cols]
    for merge_set in merge_sets:
        if debug > 0:
            print("MERGING OVERLAPPING COLUMNS:", [col.id for col in merge_set])
        merged_col = merge_columns(merge_set, "temp_id", merge_set[0].metadata)
        if text_region.parent and text_region.parent.id:
            merged_col.set_derived_id(text_region.parent.id)
            merged_col.set_parent(text_region.parent)
        else:
            merged_col.set_derived_id(text_region.id)
        non_overlapping_cols.append(merged_col)
    columns = non_overlapping_cols
    if debug > 0:
        print("NUM COLUMNS:", len(columns))
        print("EXTRA LINES BEFORE:", len(extra_lines))
        for line in extra_lines:
            print('\tEXTRA LINE:', line.text)
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
            add_line_to_column(line, best_column)
            append_count += 1
            best_column.coords = pdm.parse_derived_coords(best_column.text_regions)
            if text_region.parent:
                best_column.set_derived_id(text_region.parent.id)
        else:
            # print(f"APPENDING NON-COL LINE: {line.coords.left}-{line.coords.right}\t{line.coords.y}\t{line.text}")
            non_col_lines.append(line)
            append_count += 1
    if debug > 0:
        print('append_count:', append_count)
    extra_lines = non_col_lines
    if debug > 0:
        print("EXTRA LINES AFTER:", len(extra_lines))
    extra = None
    if len(extra_lines) > 0:
        try:
            coords = pdm.parse_derived_coords(extra_lines)
        except BaseException:
            for line in extra_lines:
                print('\tproblem with coords of extra line:', line.id, line.coords.box, line.text)
                print('\tcoords:', line.coords)
                print('\tin text_region', text_region.id)
            coord_points = [point for line in extra_lines for point in line.coords.points]
            coords = pdm.Coords(coord_points)
            if ignore_bad_coordinate_lines is False:
                raise ValueError('Cannot generate column coords for extra lines')
        if coords is not None:
            extra = pdm.PageXMLTextRegion(metadata=copy.deepcopy(text_region.metadata), coords=coords,
                                          lines=extra_lines)
            if text_region.parent and text_region.parent.id:
                extra.set_derived_id(text_region.parent.id)
                extra.set_parent(text_region.parent)
            else:
                extra.set_derived_id(text_region.metadata['scan_id'])
            # for line in extra.lines:
            #     print(f"RETURNING EXTRA LINE: {line.coords.left}-{line.coords.right}\t{line.coords.y}\t{line.text}")
            if debug > 0:
                print('split_lines_on_column_gaps - SPLITTING EXTRA')
            if extra.id == text_region.id and len(columns) == 0:
                if debug > 0:
                    print('split_lines_on_column_gaps - extra equals text_region:')
                    print('\t', text_region.id, text_region.stats)
                    print('\t', extra.id, extra.stats)
                    print('split_lines_on_column_gaps - cannot split text_region, returning text_region')
                extra_cols = [extra]
            elif all([extra_stat == tr_stat for extra_stat, tr_stat in zip(extra.stats, text_region.stats)]):
                if debug > 0:
                    print('split_lines_on_column_gaps - extra equals text_region:')
                    print('\t', text_region.id, text_region.stats)
                    print('\t', extra.id, extra.stats)
                    print('split_lines_on_column_gaps - cannot split text_region, returning text_region')
                extra_cols = [extra]
            else:
                extra_cols = split_lines_on_column_gaps(extra, min_gap_width, debug=debug)
            for extra_col in extra_cols:
                if debug > 0:
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
    if debug > 3:
        print('\n------------\n')
        for col in columns:
            print(f"split_lines_on_column_gaps - number of lines directly under column {col.id}: {len(col.lines)}")
        print('\n------------\n')
    return columns
