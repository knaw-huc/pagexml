from typing import Dict, Generator, List, Tuple, Union
from collections import Counter
from collections import defaultdict

import numpy as np

import pagexml.parser as pagexml_parser
import pagexml.model.physical_document_model as pdm


def same_column(element1: pdm.PageXMLDoc, element2: pdm.PageXMLDoc) -> bool:
    """Check if two PageXML elements are part of the same column."""
    if (
            'scan_id' in element1.metadata
            and 'scan_id' in element2.metadata
            and element1.metadata['scan_id'] != element2.metadata['scan_id']
    ):
        return False
    if 'column_id' in element1.metadata and 'column_id' in element2.metadata:
        return element1.metadata['column_id'] == element2.metadata['column_id']
    else:
        # check if the two lines have a horizontal overlap that is more than 50% of the width of line 1
        # Note: this doesn't work for short adjacent lines within the same column
        return pdm.horizontal_overlap(element1.coords, element2.coords) > (element1.coords.w / 2)


def get_baseline_y(line: pdm.PageXMLTextLine) -> List[int]:
    """Return the Y/vertical coordinates of a text line's baseline."""
    if line_starts_with_big_capital(line):
        return [point[1] for point in line.baseline.points if point[1] < line.baseline.bottom - 20]
    else:
        return [point[1] for point in line.baseline.points]


def line_starts_with_big_capital(line: pdm.PageXMLTextLine) -> bool:
    """Determine is a line start with a capital in a larger font than the rest,
    which is aligned at the top, so sticks out at the bottom."""
    # The vertical distance between the lowest and highest baseline point (height) should be large
    if line.baseline.h < 30:
        return False
    lowest_point = find_lowest_point(line)
    # The lowest point should be left-aligned with the sentence.
    return lowest_point[0] - line.baseline.left <= 100


def find_lowest_point(line: pdm.PageXMLTextLine) -> Tuple[int, int]:
    """Find the first baseline point that corresponds to the lowest vertical point.

    :param line: a PageXML TextLine object with baseline information
    :type line: PageXMLTextLine
    :return: the left most point that has the lowest vertical coordinate
    :rtype: Tuple[int, int]
    """
    for point in line.baseline.points:
        if point[1] == line.baseline.bottom:
            return point


def interpolate_points(p1: Tuple[int, int], p2: Tuple[int, int],
                       step: int = 50) -> Generator[Dict[int, int], None, None]:
    """Determine the x coordinates between a pair of points on a baseline
    and calculate their corresponding y coordinates.
    :param p1: a 2D point
    :type p1: Tuple[int, int]
    :param p2: a 2D point
    :type p2: Tuple[int, int]
    :param step: the step size in pixels for interpolation
    :type step: int
    :return: a generator of interpolated points based on step size
    :rtype: Generator[Dict[int, int], None, None]
    """
    if p1[0] > p2[0]:
        # p2 should be to the right of p1
        p1, p2 = p2, p1
    start_x = p1[0] + step - (p1[0] % step)
    end_x = p2[0] - (p2[0] % step)
    if p2[0] == p1[0]:
        # points 1 and 2 have the same x coordinate
        # so there is nothing to interpolate
        return None
    delta_y = (p1[1] - p2[1]) / (p2[0] - p1[0])
    for int_x in range(start_x, end_x + 1, step):
        int_y = p1[1] - int((int_x - p1[0]) * delta_y)
        yield int_x, int_y


def interpolate_baseline_points(points: List[Tuple[int, int]],
                                step: int = 50) -> Dict[int, int]:
    """Determine the x coordinates between each pair of subsequent
    points on a baseline and calculate their corresponding y coordinates.

    :param points: the list of points of a baseline object
    :type points: List[Tuple[int, int]]
    :param step: the step size in pixels for interpolation
    :type step: int
    :return: a dictionary of interpolated points based on step size
    :rtype: Dict[int, int]
    """
    interpolated_baseline_points = {}
    # iterate over each subsequent pair of baseline points
    for ci, curr_point in enumerate(points[:-1]):
        next_point = points[ci + 1]
        if next_point[0] == curr_point[0]:
            # skip pair when they have the same x coordinate
            continue
        # interpolate points between the current and next points using step as size
        for int_x, int_y in interpolate_points(curr_point, next_point, step=step):
            interpolated_baseline_points[int_x] = int_y
    return interpolated_baseline_points


def compute_baseline_distances(baseline1: pdm.Baseline, baseline2: pdm.Baseline,
                               step: int = 50) -> np.ndarray:
    """Compute the vertical distance between two baselines, based on
    their horizontal overlap, using a fixed step size. Interpolated
    points will be generated at fixed increments of step size for
    both baselines, so they have points with corresponding x
    coordinates to calculate the distance.

    If two lines have no horizontal overlap, it returns a list with
    a single distance between the average heights of the two baselines

    :param baseline1: the first Baseline object to compare
    :type baseline1: Baseline
    :param baseline2: the second Baseline object to compare
    :type baseline2: Baseline
    :param step: the step size in pixels for interpolation
    :type step: int
    :return: a list of vertical distances based on horizontal overlap
    :rtype: List[int]
    """
    if baseline1 is None or baseline2 is None:
        return np.array([])
    b1_points = interpolate_baseline_points(baseline1.points, step=step)
    b2_points = interpolate_baseline_points(baseline2.points, step=step)
    distances = np.array([abs(b2_points[curr_x] - b1_points[curr_x]) for curr_x in b1_points
                          if curr_x in b2_points])
    if len(distances) == 0:
        avg1 = average_baseline_height(baseline1)
        avg2 = average_baseline_height(baseline2)
        distances = np.array([abs(avg1 - avg2)])
    return distances


def average_baseline_height(baseline: pdm.Baseline) -> int:
    """Compute the average (mean) baseline height for comparing lines that
    are not horizontally aligned.

    :param baseline: the baseline of a TextLine
    :type baseline: Baseline
    :return: the average (mean) baseline height across all its baseline points
    :rtype: int
    """
    total_avg = 0
    # iterate over each subsequent pair of baseline points
    for ci, curr_point in enumerate(baseline.points[:-1]):
        next_point = baseline.points[ci + 1]
        segment_avg = (curr_point[1] + next_point[1]) / 2
        # segment contributes its average height times its width
        total_avg += segment_avg * (next_point[0] - curr_point[0])

    # average is total of average heights divided by total width
    total_width = (baseline.points[-1][0] - baseline.points[0][0])
    if total_width != 0:
        return int(total_avg / total_width)
    # this should not happen, but if it does we need to calculate
    # the average differently, to avoid a division by zero error
    print(f"total_avg={total_avg}")
    print(f"baseline={baseline}")
    print(f"baseline.points[-1][0]={baseline.points[-1][0]}")
    xcoords = [p[0] for p in baseline.points]
    left_x = min(xcoords)
    right_x = max(xcoords)
    if left_x != right_x:
        return int(total_avg / (right_x - left_x))
    else:
        return int(total_avg)


def get_textregion_line_distances(text_region: pdm.PageXMLTextRegion) -> List[np.ndarray]:
    """Returns a list of line distance numpy arrays. For each line, its distance
    to the next at 50 pixel intervals is computed and stored in an numpy ndarray.

    :param text_region: a TextRegion object that contains TextLines
    :type text_region: PageXMLTextRegion
    :return: a list of numpy ndarrays of line distances
    :rtype: List[np.ndarray]
    """
    all_distances: List[np.ndarray] = []
    text_regions = text_region.get_inner_text_regions()
    for ti, curr_tr in enumerate(text_regions):
        above_next_tr = False
        next_tr = None
        if ti + 1 < len(text_regions):
            # check if the next textregion is directly below the current one
            next_tr = text_regions[ti + 1]
            above_next_tr = same_column(curr_tr, next_tr)
        for li, curr_line in enumerate(curr_tr.lines):
            next_line = None
            if li + 1 < len(curr_tr.lines):
                next_line = curr_tr.lines[li + 1]
            elif above_next_tr and next_tr.lines:
                # if the next textregion is directly below this one, include the distance
                # of this textregion's last line and the next textregion's first line
                next_line = next_tr.lines[0]
            if next_line:
                distances = compute_baseline_distances(curr_line.baseline, next_line.baseline)
                all_distances.append(distances)
    return all_distances


def get_textregion_avg_line_distance(text_region: pdm.PageXMLTextRegion,
                                     avg_type: str = "macro") -> float:
    """Returns the median distance between subsequent lines in a
    textregion object. If the textregion contains smaller textregions, it only
    considers line distances between lines within the same column (i.e. only
    lines from textregions that are horizontally aligned.

    By default the macro-average is returned.

    :param text_region: a TextRegion object that contains TextLines
    :type text_region: PageXMLTextRegion
    :param avg_type: the type of averging to apply (macro or micro)
    :type avg_type: str
    :return: the median distance between horizontally aligned lines
    :rtype: float
    """
    if avg_type not in ["micro", "macro"]:
        raise ValueError(f'Invalid avg_type "{avg_type}", must be "macro" or "micro"')
    all_distances = get_textregion_line_distances(text_region)
    if len(all_distances) == 0:
        return 0
    if avg_type == "micro":
        return float(np.median(np.concatenate(all_distances)))
    else:
        return float(np.median(np.array([distances.mean() for distances in all_distances])))


def get_textregion_avg_char_width(text_region: pdm.PageXMLTextRegion) -> float:
    """Return the estimated average (mean) character width, determined as the sum
    of the width of text lines divided by the sum of the number of characters
    of all text lines.

    :param text_region: a TextRegion object that contains TextLines
    :type text_region: PageXMLTextRegion
    :return: the average (mean) character width
    :rtype: float
    """
    total_chars = 0
    total_text_width = 0
    for tr in text_region.get_inner_text_regions():
        for line in tr.lines:
            if line.text is None:
                continue
            total_chars += len(line.text)
            total_text_width += line.coords.width
    return total_text_width / total_chars if total_chars else 0.0


def get_textregion_avg_line_width(text_region: pdm.PageXMLTextRegion, unit: str = "char") -> float:
    """Return the estimated average (mean) character width, determined as the sum
    of the width of text lines divided by the sum of the number of characters
    of all text lines.

    :param text_region: a TextRegion object that contains TextLines
    :type text_region: PageXMLTextRegion
    :param unit: the unit to measure line width, either char or pixel
    :type unit: str
    :return: the average (mean) character width
    :rtype: float
    """
    if unit not in {'char', 'pixel'}:
        raise ValueError(f'Invalid unit "{unit}", must be "char" (default) or "pixel"')
    total_lines = 0
    total_line_width = 0
    for tr in text_region.get_inner_text_regions():
        for line in tr.lines:
            if line.text is None:
                # skip non-text lines
                continue
            total_lines += 1
            total_line_width += len(line.text) if unit == 'char' else line.coords.width
    return total_line_width / total_lines if total_lines > 0 else 0.0


def compute_textregion_distance(tr1: pdm.PageXMLTextRegion,
                                tr2: pdm.PageXMLTextRegion) -> Union[int, float]:
    if pdm.is_vertically_overlapping(tr1, tr2):
        return 0
    elif tr1.coords.top > tr2.coords.top:
        tr1, tr2 = tr2, tr1
    if len(tr1.lines) > 0 and len(tr2.lines) > 0:
        prev_line = tr1.lines[-1]
        curr_line = tr2.lines[0]
        distances = compute_baseline_distances(prev_line.baseline, curr_line.baseline)
        return float(np.median(distances))
    else:
        return tr2.coords.top - tr1.coords.bottom


def compute_lines_stats(lines: List[pdm.PageXMLTextLine],
                        stats: Dict[str, Dict[str, Counter]]) -> None:
    prev_line = None
    for curr_line in sorted(lines):
        stats["line"]["height"].update([curr_line.coords.h])
        stats["line"]["width"].update([curr_line.coords.w])
        stats["line"]["words"].update([curr_line.num_words])
        if isinstance(prev_line, pdm.PageXMLTextLine):
            distances = compute_baseline_distances(prev_line.baseline, curr_line.baseline)
            if len(distances) == 0:
                continue
            try:
                stats["line"]["distance"].update([np.median(distances)])
            except TypeError:
                print(prev_line.baseline)
                print(curr_line.baseline)
                print(distances, type(distances))
                raise
        prev_line = curr_line


def compute_textregions_stats(text_regions: List[pdm.PageXMLTextRegion],
                              stats: Dict[str, Dict[str, Counter]]) -> None:
    prev_tr = None
    for curr_tr in sorted(text_regions):
        if isinstance(prev_tr, pdm.PageXMLTextRegion) and pdm.is_horizontally_overlapping(curr_tr, prev_tr):
            tr_dist = compute_textregion_distance(prev_tr, curr_tr)
            stats["textregion"]["vertical_dist"].update([tr_dist])
        stats["textregion"]["height"].update([curr_tr.coords.h])
        stats["textregion"]["width"].update([curr_tr.coords.w])
        tr_stats = curr_tr.stats
        for field in tr_stats:
            if field == "text_regions":
                continue
            stats["textregion"][f"{field}"].update([tr_stats[field]])
        if len(curr_tr.lines) > 0:
            compute_lines_stats(curr_tr.lines, stats)
        prev_tr = curr_tr


def compute_columns_stats(columns: List[pdm.PageXMLColumn],
                          stats: Dict[str, Dict[str, Counter]]):
    for column in columns:
        stats["column"]["height"].update([column.coords.h])
        stats["column"]["width"].update([column.coords.w])
        column_stats = column.stats
        for field in column_stats:
            stats["column"][f"{field}"].update([column_stats[field]])
        if len(column.text_regions) > 0:
            compute_textregions_stats(column.text_regions, stats)
    return stats


def compute_pages_stats(pages: List[pdm.PageXMLPage], stats: Dict[str, Dict[str, Counter]]):
    for page in pages:
        stats["page"]["height"].update([page.coords.h])
        stats["page"]["width"].update([page.coords.w])
        page_stats = page.stats
        for field in page_stats:
            stats["page"][f"{field}"].update([page_stats[field]])
        if len(page.columns) > 0:
            compute_columns_stats(page.columns, stats)
        if len(page.text_regions) > 0:
            compute_textregions_stats(page.text_regions, stats)
    return stats


def compute_scans_stats(scans: List[pdm.PageXMLScan], stats: Dict[str, Dict[str, Counter]]):
    for scan in scans:
        stats["scan"]["height"].update([scan.coords.h])
        stats["scan"]["width"].update([scan.coords.w])
        scan_stats = scan.stats
        for field in scan_stats:
            if field in {'columns', 'extra', 'pages'}:
                continue
            stats["scan"][f"{field}"].update([scan_stats[field]])
        if len(scan.pages) > 0:
            compute_pages_stats(scan.pages, stats)
        if len(scan.columns) > 0:
            compute_columns_stats(scan.columns, stats)
        if len(scan.text_regions) > 0:
            compute_textregions_stats(scan.text_regions, stats)
    return stats


def compute_pagexml_stats(docs: List[pdm.PageXMLDoc]) -> Dict[str, Dict[str, Counter]]:
    """Compute statistics on the numbers of PageXML elements that are part of a given
    list of PageXMLDoc objects.

    :param docs: a list of PageXMLDoc objects
    :type docs: List[PageXMLDoc]
    :return: A nested dictionary of statistic per PageXML element type
    :rtype: Dict[str, Dict[str, Counter]]
    """
    stats = defaultdict(lambda: defaultdict(Counter))
    type_docs = defaultdict(list)
    for doc in docs:
        type_docs[doc.__class__.__name__].append(doc)
    for doc_type in type_docs:
        if doc_type == 'PageXMLScan':
            compute_scans_stats(type_docs[doc_type], stats)
        elif doc_type == 'PageXMLPage':
            compute_pages_stats(type_docs[doc_type], stats)
        elif doc_type == 'PageXMLColumn':
            compute_columns_stats(type_docs[doc_type], stats)
        elif doc_type == 'PageXMLTextRegion':
            compute_textregions_stats(type_docs[doc_type], stats)
        elif doc_type == 'PageXMLTextLine':
            compute_lines_stats(type_docs[doc_type], stats)
    return stats


def get_line_widths(pagexml_files: List[str], line_width_bin_size: int = 50) -> List[int]:
    """Return a list of line widths for the lines in a list of PageXML files.

    :param pagexml_files: a list of PageXML filepaths
    :type pagexml_files: List[str]
    :param line_width_bin_size: the bin size for grouping lines (default is 50 pixels)
    :type line_width_bin_size: int
    :return: a list of line widths
    :rtype: List[int]
    """
    line_widths = []
    for inv_file in pagexml_files:
        scan = pagexml_parser.parse_pagexml_file(pagexml_file=inv_file)
        line_widths += [int(line.coords.w / line_width_bin_size) * line_width_bin_size for line in scan.get_lines()]
    return line_widths


def find_line_width_boundary_points(line_widths: List[int], line_bin_size: int = 50,
                                    min_ratio: float = 0.5) -> List[int]:
    """Find the minima in the distribution of line widths relative to the peaks in the distribution.
    These minima represent the boundaries between clusters of lines within the same line width
    intervals.

    :param line_widths: a list of PageXML text line widths
    :type line_widths: List[int]
    :param line_bin_size: the bin size for grouping lines to establish the line width distribution (default 50 pixels)
    :type line_bin_size: int
    :param min_ratio: the minimum ratio between a peak frequency and it's neighbouring minimum to determine
    if the minimum is a category boundary
    :type min_ratio: float
    :return: A list of category boundary points
    :rtype: List[int]
    """
    width_freq = Counter(line_widths)
    boundary_points = []
    total_widths = sum(width_freq.values())
    max_width = max(width_freq.keys())
    max_freq = max(width_freq.values())
    curr_max_freq = 0
    curr_min_freq = max_freq + 1
    curr_max_width = None
    curr_min_width = None
    prev_freq = 0

    for w in range(0, max_width + 1, line_bin_size):
        f = width_freq[w]
        if f > curr_max_freq:
            curr_max_freq = f
            curr_max_width = w
        if f < prev_freq and f < curr_min_freq:
            curr_min_freq = f
            curr_min_width = w
        if f > prev_freq and f > curr_min_freq:
            if (curr_max_freq - curr_min_freq) / curr_max_freq > min_ratio:
                boundary_points.append((curr_min_width, curr_min_freq))
                curr_max_freq = 0
                curr_max_width = 0
                curr_min_freq = max_freq + 1
        # print(w, f, prev_freq, curr_min_freq, curr_max_freq, boundary_points)
        prev_freq = f
    return [bp[0] for bp in boundary_points]


def categorise_line_width(line: pdm.PageXMLTextLine, boundary_points: List[int]) -> int:
    """Categorise a line based on its width and a list of line width boundary points."""
    for bi, boundary_point in enumerate(boundary_points):
        if boundary_point > line.coords.w:
            return bi
    return len(boundary_points)


def get_line_width_stats(lines: List[pdm.PageXMLTextLine], boundary_points: List[int]) -> Counter:
    """Return a Counter object with statistics of the number of lines categorised according
    to a list of category break points (line widths that are the boundary between categories
    of line width).

    :param lines: A list of PageXML text lines
    :type lines: List[PageXMLTextLine]
    :param boundary_points: A list of line width category boundary points
    :type boundary_points: List[int]
    :return: A counter with the number of lines per line width interval
    :rtype: Counter
    """
    line_width_stats = Counter()
    for line_width_cat in range(0, len(boundary_points) + 1):
        line_width_stats[line_width_cat] = 0
    line_width_stats.update([categorise_line_width(line, boundary_points) for line in lines])
    return line_width_stats
