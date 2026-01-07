import copy
from typing import Iterable, List, Union
from collections import Counter

import pagexml.helper.pagexml_helper as pagexml_helper
import pagexml.model.physical_document_model as pdm


def compute_text_pixel_dist(lines: List[pdm.PageXMLTextLine],
                            use_height: bool = False) -> Counter:
    """Count how many lines are above each horizontal pixel coordinate.
    Use `use_height` to use the height of lines in pixels instead of the
    number of lines."""
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
        if use_height is True:
            for pixel in range(left, right + 1):
                if line.xheight:
                    height = line.xheight
                else:
                    # bounding box usually is cut out quite generously
                    height = line.coords.height / 2
                pixel_dist[pixel] += height
        else:
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


def merge_columns(columns: List[pdm.PageXMLColumn], doc_id: str, metadata: dict,
                  lines_only: bool = False, debug: int = 0) -> pdm.PageXMLColumn:
    """Merge two columns into one, sorting lines by baseline height."""
    if lines_only is True:
        merged_lines = [line for col in columns for line in col.get_lines()]
        merged_lines = list(set(merged_lines))
        sorted_lines = sorted(merged_lines, key=lambda x: x.baseline.y)
        tr = pagexml_helper.derive_text_region_from_lines(sorted_lines)
        merged_col = derive_column_from_regions(tr, debug=debug)
    else:
        merged_trs = [tr for col in columns for tr in col.text_regions]
        sorted_trs = sorted(merged_trs, key=lambda x: x.coords.y)
        merged_col = derive_column_from_regions(sorted_trs, debug=debug)
    if doc_id:
        merged_col.doc_id = doc_id
    if metadata:
        merged_col.metadata = metadata

    for col in columns:
        for col_type in col.types:
            if col_type not in merged_col.type:
                merged_col.add_type(col_type)
    return merged_col


def add_line_to_column(line: pdm.PageXMLTextLine, column: pdm.PageXMLColumn) -> None:
    """Add a PageXMLTextLine to a PageXMLColumn, assign to the appropriate text region
    or create a new text region for it."""
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


def sort_lines_on_column_ranges(text_region: pdm.PageXMLTextRegion, lines: List[pdm.PageXMLTextLine],
                                column_ranges: List[pdm.Interval], overlap_threshold: float,
                                debug: int = 0):
    """Map each line of a set of lines to the corresponding horizontally overlapping column range."""
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
    if debug > 0:
        print('RANGE SPLIT num_lines:', num_lines, 'append_count:', append_count)
        for ci, lines in enumerate(column_lines):
            print('\tcolumn', ci, '\tlines:', len(lines))
        print('\textra lines:', len(extra_lines))
    return column_lines, extra_lines


def derive_column_from_regions(regions: Union[pdm.PageXMLRegion, List[pdm.PageXMLRegion]],
                               debug: int = 0):
    text_regions, table_regions, empty_regions = [], [], []
    if isinstance(regions, Iterable) is False:
        regions = [regions]
    for region in regions:
        if isinstance(region, pdm.PageXMLTextRegion):
            text_regions.append(pagexml_helper.copy_text_region(region))
        elif isinstance(region, pdm.PageXMLTableRegion):
            table_regions.append(pagexml_helper.copy_table_region(region))
        elif isinstance(region, pdm.PageXMLEmptyRegion):
            empty_regions.append(pagexml_helper.copy_empty_region(region))
        else:
            print(f"column_parser.derive_column_from_regions - region: {region.id} {region.__class__.__name__}")
            raise ValueError(f"Generating a column from a generic PageXMLRegion is not implemented. "
                             f"Please use PageXMLTextRegion, PageXMLTableRegion or PagexmlEmptyRegion "
                             f"instead.")
    if debug > 0:
        print(f"derive_column_from_regions - region ids: {[r.id for r in regions]}")
    try:
        coords = pdm.parse_derived_coords(regions)
    except BaseException as err:
        for region in regions:
            print(region.coords.box_string)
        raise
    column = pdm.PageXMLColumn(metadata=copy.deepcopy(regions[0].metadata),
                               coords=coords, text_regions=text_regions,
                               table_regions=table_regions, empty_regions=empty_regions)
    if regions[0].parent:
        column.set_parent(regions[0].parent)
    if regions[0].parent and regions[0].parent.id:
        column.set_derived_id(regions[0].parent.id)
    else:
        column.set_derived_id(regions[0].id)
    column.set_as_parent(column.regions)
    return column


def normalise_columns(columns: List[pdm.PageXMLColumn], debug: int = 0) -> List[pdm.PageXMLColumn]:
    """Normalise a list of columns by merging columns with horizontal overlap."""
    merge_sets = find_overlapping_columns(columns)
    merge_cols = {col for merge_set in merge_sets for col in merge_set}
    non_overlapping_cols = [col for col in columns if col not in merge_cols]
    for merge_set in merge_sets:
        if debug > 0:
            print("MERGING OVERLAPPING COLUMNS:", [col.id for col in merge_set])
        first_col = merge_set[0]
        merged_col = merge_columns(merge_set, "temp_id", first_col.metadata)
        if first_col.parent:
            merged_col.set_parent(first_col.parent)
        if first_col.parent and first_col.parent.id:
            merged_col.set_derived_id(first_col.parent.id)
        else:
            merged_col.set_derived_id(first_col.id)
        non_overlapping_cols.append(merged_col)
    columns = non_overlapping_cols
    return sorted(columns)


def derive_column_from_lines(text_region: pdm.PageXMLTextRegion,
                             column_lines: List[List[pdm.PageXMLTextLine]],
                             debug: int = 0) -> List[pdm.PageXMLColumn]:
    """Generate PageXMLColumn instances per set of horizontally grouped lines."""
    columns = []
    for lines in column_lines:
        if len(lines) == 0:
            continue
        coords = pdm.parse_derived_coords(lines)
        tr = pdm.PageXMLTextRegion(doc_type=copy.deepcopy(text_region.type),
                                   metadata=copy.deepcopy(text_region.metadata),
                                   coords=copy.deepcopy(coords), lines=lines)
        tr.set_derived_id(text_region.metadata['scan_id'])
        tr.set_as_parent(lines)
        column = derive_column_from_regions(tr, debug=debug)
        columns.append(column)
    # column range may have expanded with lines partially overlapping initial range
    return columns


def add_extra_lines_to_columns(text_region: pdm.PageXMLTextRegion, columns: List[pdm.PageXMLColumn],
                               extra_lines: List[pdm.PageXMLTextLine], debug: int = 0):
    """"Assign extra lines not yet belonging to any column to its overlapping column, returning
    any lines that do not overlap with any column."""
    # check which extra lines should be added to columns
    non_col_lines = []
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
        print("EXTRA LINES AFTER:", len(extra_lines))
    return non_col_lines


def add_extra_text_regions(text_region: pdm.PageXMLTextRegion, columns: List[pdm.PageXMLColumn],
                           extra_lines: List[pdm.PageXMLTextLine], min_gap_width: int,
                           ignore_bad_coordinate_lines: bool, debug: int = 0):
    """Create additional text regions for lines that are not assigned to any columns."""
    extra = None
    extra_regions = []
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
                extra_trs = [extra]
            elif all([extra_stat == tr_stat for extra_stat, tr_stat in zip(extra.stats, text_region.stats)]):
                if debug > 0:
                    print('split_lines_on_column_gaps - extra equals text_region:')
                    print('\t', text_region.id, text_region.stats)
                    print('\t', extra.id, extra.stats)
                    print('split_lines_on_column_gaps - cannot split text_region, returning text_region')
                extra_trs = [extra]
            else:
                extra_trs = split_lines_on_column_gaps(extra, min_gap_width, debug=debug)
            for extra_col in extra_trs:
                if debug > 0:
                    print('\tEXTRA COL AFTER EXTRA SPLIT:', extra_col.stats)
                extra_col.set_parent(text_region.parent)
                if text_region.parent:
                    extra_col.set_derived_id(text_region.parent.id)
            extra_regions += extra_trs
            extra = None
    if extra is not None:
        print('source doc:', text_region.id)
        print(extra)
        raise TypeError(f'Extra is not None but {type(extra)}')
    return extra_regions


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
    column_lines, extra_lines = sort_lines_on_column_ranges(text_region, lines, column_ranges,
                                                            overlap_threshold, debug=debug)
    columns = derive_column_from_lines(text_region, column_lines, debug=debug)
    extra_lines = add_extra_lines_to_columns(text_region, columns, extra_lines, debug=debug)
    extra_regions = add_extra_text_regions(text_region, columns, extra_lines, min_gap_width,
                                           ignore_bad_coordinate_lines, debug=debug)
    columns.extend(extra_regions)
    if debug > 3:
        print('\n------------\n')
        for col in columns:
            print(f"split_lines_on_column_gaps - number of lines directly under column {col.id}: {len(col.get_lines())}")
        print('\n------------\n')
    num_lines = [col.stats['lines'] for col in columns]
    if sum(num_lines) != len(lines):
        raise ValueError(f"Columns have a different number of lines ({' + '.join(num_lines)} = {sum(num_lines)}) "
                         f"from text region ({len(lines)}). This is probably an issue with this function, rather "
                         f"than with the input region.")
    return columns
