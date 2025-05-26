from typing import Dict, List, Tuple, Union

import numpy as np

import pagexml.model.physical_document_model as pdm
from pagexml.analysis.layout_stats import get_line_widths, find_line_width_boundary_points
from pagexml.helper.pagexml_helper import regions_overlap


def map_aspect_ratio(wh_ratio: float) -> str:
    """Determine whether a width/height ratio represents a shape that is
    portrait, landscape or square."""
    if wh_ratio < 0.90:
        return 'portrait'
    elif wh_ratio > 1.10:
        return 'landscape'
    else:
        return 'square'


def determine_main_text_line_width(inventory_docs: List[Union[pdm.PageXMLScan, pdm.PageXMLPage]]):
    trs = [tr for doc in inventory_docs for tr in doc.get_regions()]
    if len(trs) == 0:
        return -1, -1
    # print(f'inventory_stats.determine_main_text_line_width - number of trs: {len(trs)}')
    main_text_trs = [tr for tr in trs if is_main_text_paragraph(tr)]
    # print(f'inventory_stats.determine_main_text_line_width - number of main_text_trs: {len(main_text_trs)}')
    if len(main_text_trs) > 0:
        main_text_widths = np.array(get_line_widths(main_text_trs))
    else:
        main_text_widths = np.array(get_line_widths(trs))
    # print(f'inventory_stats.determine_main_text_line_width - number of main_text_widths: {len(main_text_widths)}')
    boundary_points = find_line_width_boundary_points(main_text_widths, line_bin_size=50, debug=0, min_peak_frac=0.1)
    main_text_min_width = main_text_widths.mean() - main_text_widths.std()
    main_text_max_width = main_text_widths.mean() + main_text_widths.std()
    if len(boundary_points) > 0:
        print(f"Warning - determine_main_text_line_width - line width distribution of "
              f"lines in text paragraphs has multiple peaks")
    return main_text_min_width, main_text_max_width


def is_main_text_paragraph(text_region: pdm.PageXMLTextRegion, min_words: int = 10) -> bool:
    return text_region.stats['words'] > min_words


def is_main_text_line(line: pdm.PageXMLTextLine, main_text_min_width: float,
                      main_text_max_width: float) -> bool:
    """Determine whether a line is a main-text line."""
    return main_text_min_width <= line.coords.width <= main_text_max_width


def get_page_main_text_range(pages: List[pdm.PageXMLPage]):
    # split pages into even and odd (verso and recto)
    pages_even = [page for page in pages if page.metadata['page_side'] == 'even']
    pages_odd = [page for page in pages if page.metadata['page_side'] == 'odd']

    # get all text regions
    trs_even = [tr for page in pages_even for tr in page.get_regions()]
    trs_odd = [tr for page in pages_odd for tr in page.get_regions()]

    # filter regions that are full_text paragraphs (at least 10 words)
    main_text_trs_even = [tr for tr in trs_even if is_main_text_paragraph(tr)]
    main_text_trs_odd = [tr for tr in trs_odd if is_main_text_paragraph(tr)]

    # determine width of full-text lines in full text paragraphs
    main_text_min_width, main_text_max_width = determine_main_text_line_width(pages)

    # get all full-text lines on even and odd sides
    main_text_lines_even = [line for tr in main_text_trs_even for line in tr.lines if
                            is_main_text_line(line, main_text_min_width, main_text_max_width)]
    main_text_lines_odd = [line for tr in main_text_trs_odd for line in tr.lines if
                           is_main_text_line(line, main_text_min_width, main_text_max_width)]

    # determine mean left and right coordinates of full-text lines on
    # even and odd sided pages
    lefts_even = np.array([line.coords.left for line in main_text_lines_even])
    lefts_odd = np.array([line.coords.left for line in main_text_lines_odd])
    rights_even = np.array([line.coords.right for line in main_text_lines_even])
    rights_odd = np.array([line.coords.right for line in main_text_lines_odd])
    if len(lefts_even) == 0 or len(lefts_odd) == 0:
        main_text_ranges = {
            'even': pdm.Interval('even', 0, 99999),
            'odd': pdm.Interval('odd', 0, 99999)
        }
        print(f"no left-right values for inventory {pages[0].metadata['scan_id']}")
    else:
        main_text_ranges = {
            'even': pdm.Interval('even', int(lefts_even.mean()), int(rights_even.mean())),
            'odd': pdm.Interval('odd', int(lefts_odd.mean()), int(rights_odd.mean())),
        }
    return main_text_ranges


def set_top_bottom_text_columns(scans: List[pdm.PageXMLDoc]):
    """Determine the average relative top and bottom heights of full-text columns.

    TODO: check for multi-modal distributions.
    """
    para_tops = []
    para_bottoms = []
    for scan in scans:
        para_trs = [tr for tr in scan.text_regions if is_main_text_paragraph(tr)]
        if len(para_trs) == 0:
            continue
        para_top = min(tr.coords.top / scan.coords.bottom for tr in para_trs)
        para_bottom = max(tr.coords.bottom / scan.coords.bottom for tr in para_trs)
        para_tops.append(para_top)
        para_bottoms.append(para_bottom)
    para_tops = np.array(para_tops)
    para_bottoms = np.array(para_bottoms)
    para_tops_mean = para_tops.mean() if len(para_tops) > 0 else 0.0
    para_bottoms_mean = para_bottoms.mean() if len(para_bottoms) > 0 else 1.0
    inv_para_top = max(0.0, para_tops_mean - para_tops.std())
    inv_para_bottom = min(1.0, para_bottoms_mean + para_bottoms.std())
    return inv_para_top, inv_para_bottom


def get_page_main_text_top_bottom(page: pdm.PageXMLPage,
                                  region_min_words: int = 10) -> Tuple[int, int]:
    para_trs = [tr for tr in page.text_regions if tr.stats['words'] > region_min_words]
    if len(para_trs) == 0:
        return 0, 0
    main_text_top = min(tr.coords.top / page.coords.bottom for tr in para_trs)
    main_text_bottom = max(tr.coords.bottom / page.coords.bottom for tr in para_trs)
    return main_text_top, main_text_bottom


def make_main_text_column(main_text_range: pdm.Interval,
                          main_text_top: int, main_text_bottom: int):
    points = [
        (main_text_range.start, main_text_top), (main_text_range.end, main_text_top),
        (main_text_range.end, main_text_bottom), (main_text_range.start, main_text_bottom)
    ]
    coords = pdm.Coords(points)
    return pdm.PageXMLColumn(coords=coords)


def mbs(doc: pdm.PageXMLDoc):
    c = doc.coords
    return f"{c.left: >4}-{c.right: <4} - {c.top: >4}-{c.bottom: <4}"


def get_page_main_text_regions(page: pdm.PageXMLPage, main_text_range: pdm.Interval,
                               main_text_top: int, main_text_bottom: int):
    main_text_column = make_main_text_column(main_text_range, main_text_top, main_text_bottom)
    # print(f"main_text_column.coords.box: {mbs(main_text_column)}")
    main_text_trs: List[pdm.PageXMLTextRegion] = []
    for tr in page.text_regions:
        # print(f"    tr.coords.box: {mbs(tr)}\t{regions_overlap(tr, main_text_column, threshold=0.7)}")
        if regions_overlap(tr, main_text_column, threshold=0.7):
            main_text_trs.append(tr)
    return main_text_trs


def is_full_text_page(page: pdm.PageXMLPage, inv_full_text_top: float, inv_full_text_bottom: float) -> bool:
    page_text_top, page_text_bottom = get_page_main_text_top_bottom(page)
    if (page_text_top - inv_full_text_top) > 0.10:
        return False
    if (inv_full_text_bottom - page_text_bottom) / inv_full_text_bottom > 0.10:
        return False
    return True
