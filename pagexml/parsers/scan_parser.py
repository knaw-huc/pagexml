import copy
from collections import defaultdict
from typing import List, Tuple

import numpy as np

import pagexml.model.physical_document_model as pdm
import pagexml.helper.pagexml_helper as pagexml_helper
import pagexml.parsers.column_parser as column_parser


def set_average_scan_width(scans: List[pdm.PageXMLScan]):
    widths = np.array([scan.coords.width for scan in scans])
    heights = np.array([scan.coords.height for scan in scans])
    avg_width = widths.mean()
    avg_height = heights.mean()
    for scan in scans:
        scan.metadata['avg_scan_width'] = avg_width
        scan.metadata['avg_scan_height'] = avg_height


def get_page_split_widths(scan: pdm.PageXMLScan, separation_point: int = None, page_overlap: int = 0) -> Tuple[int, int, int, int]:
    if hasattr(scan, 'coords') is False or scan.coords is None or scan.coords.points is None:
        print(f'ERROR determining scan width in get_page_split_widths for scan {scan.id}')
    even_start = scan.coords.left
    odd_end = scan.coords.width
    if separation_point is None:
        separation_point = odd_end / 2
    odd_start = separation_point - page_overlap
    even_end = separation_point + page_overlap
    return even_start, int(even_end), int(odd_start), odd_end


def initialize_pagexml_page(scan_doc: pdm.PageXMLScan, side: str,
                            page_start: int, page_end: int, debug: int = 0) -> pdm.PageXMLPage:
    """Initialize a pagexml page type document based on the scan metadata."""
    metadata = copy.copy(scan_doc.metadata)
    if 'doc_type' in metadata:
        del metadata['doc_type']
    metadata['type'] = 'page'
    metadata['page_side'] = side
    region = [page_start, scan_doc.coords.top, page_end, scan_doc.coords.bottom]
    if 'scan_num' in scan_doc.metadata:
        if side == 'odd':
            metadata['page_num'] = scan_doc.metadata['scan_num'] * 2 - 1
            metadata['page_id'] = f"{scan_doc.metadata['scan_id']}-page-{metadata['page_num']}"
        elif side == 'even':
            metadata['page_num'] = scan_doc.metadata['scan_num'] * 2 - 2
            metadata['page_id'] = f"{scan_doc.metadata['scan_id']}-page-{metadata['page_num']}"
        else:
            metadata['page_num'] = scan_doc.metadata['scan_num'] * 2 - 2
            metadata['page_id'] = f"{scan_doc.metadata['scan_id']}-page-{metadata['page_num']}-extra"
    else:
        metadata['page_id'] = f"{scan_doc.metadata['scan_id']}-page-{side}"
    metadata['scan_id'] = scan_doc.metadata['scan_id']
    points = [
        (region[0], region[1]), (region[2], region[1]),
        (region[2], region[3]), (region[0], region[3])
    ]
    if debug > 0:
        print(f"scan_parser.initialize_pagexml_page - region: {region}")
        print(f"scan_parser.initialize_pagexml_page - points: {points}")
    coords = pdm.Coords(points)
    page_doc = pdm.PageXMLPage(doc_id=metadata['page_id'], metadata=metadata, coords=coords,
                               text_regions=[])
    page_doc.set_parent(scan_doc)
    if page_doc.id is None:
        page_doc.set_derived_id(scan_doc.id)
    return page_doc


def initialize_scan_pages(scan: pdm.PageXMLScan, separation_point: int = None, page_overlap: int = 0):
    even_start, even_end, odd_start, odd_end = get_page_split_widths(scan, separation_point=separation_point,
                                                                     page_overlap=page_overlap)
    page_even = initialize_pagexml_page(scan, 'even', even_start, even_end)
    page_odd = initialize_pagexml_page(scan, 'odd', odd_start, odd_end)
    pages = [page_even, page_odd]
    if scan.coords.width > odd_end:
        extra_start = odd_end
        extra_end = scan.coords.width
        page_extra = initialize_pagexml_page(scan, 'extra', extra_start, extra_end)
        pages.append(page_extra)
    return pages


def sort_docs_on_separation_point(docs: List[pdm.PageXMLDoc],
                                  separation_point: int, doc_indent: int):
    docs_left, docs_right, docs_mid = [], [], []
    for doc in docs:
        left, right = pagexml_helper.get_doc_indent_left_right(doc, doc_indent=doc_indent)
        if left > separation_point:
            docs_right.append(doc)
        elif right < separation_point:
            docs_left.append(doc)
        else:
            docs_mid.append(doc)
    return docs_left, docs_right, docs_mid


def get_parent_region(doc: pdm.PageXMLDoc):
    if doc.parent is None:
        return None
    region_types = [pdm.PageXMLTextRegion, pdm.PageXMLTableRegion, pdm.PageXMLEmptyRegion]
    parent = doc.parent
    while parent is not None:
        if any(isinstance(parent, region_type) for region_type in region_types):
            return parent
        parent = parent.parent
    return None


def group_lines_by_parent_regions(lines: List[pdm.PageXMLTextLine]):
    trs = []
    tr_lines = defaultdict(list)
    for line in lines:
        parent_region = get_parent_region(line)
        tr_lines[parent_region].append(line)
    for tr in tr_lines:
        if isinstance(tr, pdm.PageXMLTextRegion):
            new_tr = pagexml_helper.derive_text_region_from_lines(tr_lines[tr], parent=tr.parent)
            assert isinstance(new_tr, pdm.PageXMLTextRegion), "region type changed"
            new_tr.set_derived_id(tr.parent.id)
            trs.append(new_tr)
        elif isinstance(tr, pdm.PageXMLTableRegion):
            new_table = pagexml_helper.derive_table_region_from_lines(tr_lines[tr])
            assert isinstance(new_table, pdm.PageXMLTableRegion), "region type changed"
            new_table.set_derived_id(tr.parent.id)
            trs.append(new_table)
        else:
            raise TypeError(f"parent of line is neither PageXMLTextRegion nor "
                            f"PageXMLTableRegion, but {tr.__class__.__name__}")
    return trs


def group_regions_by_column(regions: List[pdm.PageXMLRegion]):
    """
    for region in regions:
        print(f"scan_parser.group_regions_by_column - region: {region.id} {region.__class__.__name__}")
    """
    columns = [column_parser.derive_column_from_regions(region) for region in regions]
    return column_parser.normalise_columns(columns)


def adjust_page_left(page: pdm.PageXMLPage, indent: int = 0):
    """Adjust the left coordinate of a split page based on the left-most text region.
    If the left point of a page is to the right of the left-most text region, adjust
    the page left coordinate (and its width) to the left-most point of the left-most
    region. Optionally, add an extra indent to create a left margin between the page
    boundary and the left-most text region."""
    trs = page.get_textual_regions()
    if len(trs) == 0:
        return None
    text_left = min([tr.coords.left for tr in trs])
    if page.coords.left > text_left - indent:
        # old_box_string = page.coords.box_string
        old_left = page.coords.left
        new_left = text_left - indent
        points = [p if p[0] != old_left else (new_left, p[1]) for p in page.coords.points]
        page.coords = pdm.Coords(points=points)


def split_scan_pages_with_separation_point(scan: pdm.PageXMLScan, separation_point: int,
                                           doc_indent: int = 0):
    """Split the text lines of a scan into verso and recto pages using a separation
    (and an optional line indentation).

    """
    trs = [pagexml_helper.copy_region(tr) for tr in scan.get_regions()]
    for tr in trs:
        tr.parent = scan
    trs_verso, trs_recto, trs_mid = sort_docs_on_separation_point(trs, separation_point,
                                                                  doc_indent=doc_indent)
    trs_mid_lines = [line for tr in trs_mid for line in tr.lines]
    """
    print(f"\nscan_parser.split_scan_pages_with_separation_points - BEFORE extending")
    for region in trs_verso:
        print(f"scan_parser.split_scan_pages_with_separation_points - "
              f"verso region: {region.id} {region.__class__.__name__}")
    for region in trs_mid:
        print(f"scan_parser.split_scan_pages_with_separation_points - "
              f"mid region: {region.id} {region.__class__.__name__}")
    for region in trs_recto:
        print(f"scan_parser.split_scan_pages_with_separation_points - "
              f"recto region: {region.id} {region.__class__.__name__}")
    """
    lines_verso, lines_recto, lines_mid = sort_docs_on_separation_point(trs_mid_lines, separation_point,
                                                                        doc_indent=doc_indent)
    trs_verso.extend(group_lines_by_parent_regions(lines_verso))
    trs_recto.extend(group_lines_by_parent_regions(lines_recto))
    trs_mid = group_lines_by_parent_regions(lines_mid)
    """
    print(f"\nscan_parser.split_scan_pages_with_separation_points - AFTER extending")
    for region in trs_verso:
        print(f"scan_parser.split_scan_pages_with_separation_points - "
              f"verso region: {region.id} {region.__class__.__name__}")
    for region in trs_mid:
        print(f"scan_parser.split_scan_pages_with_separation_points - "
              f"mid region: {region.id} {region.__class__.__name__}")
    for region in trs_recto:
        print(f"scan_parser.split_scan_pages_with_separation_points - "
              f"recto region: {region.id} {region.__class__.__name__}")
    """
    page_verso = initialize_pagexml_page(scan, 'verso', scan.coords.left, separation_point)
    page_recto = initialize_pagexml_page(scan, 'recto', separation_point, scan.coords.right)
    columns_verso = group_regions_by_column(trs_verso)
    columns_recto = group_regions_by_column(trs_recto)
    for col in columns_verso:
        page_verso.add_child(col)
    for col in columns_recto:
        page_recto.add_child(col)
    for tr in trs_mid:
        overlap_verso = pdm.get_horizontal_overlap(page_verso, tr)
        overlap_recto = pdm.get_horizontal_overlap(page_recto, tr)
        col = page_verso.columns[-1] if overlap_verso > overlap_recto else page_recto.columns[0]
        col.add_child(tr)
    verso_lines = len(page_recto.get_lines())
    recto_lines = len(page_verso.get_lines())
    page_lines = sum([verso_lines, recto_lines])
    if page_lines != scan.stats['lines']:
        raise ValueError(f"number of lines in pages verso ({verso_lines}) and recto ({recto_lines}) "
                         f"is not the same as in scan ({scan.stats['lines']}). This is probably an "
                         f"error of this function. ")
    adjust_page_left(page_verso)
    adjust_page_left(page_recto)
    if scan.coords.width < separation_point:
        return page_verso, None
    else:
        return page_verso, page_recto


def split_scan_pages(scan: pdm.PageXMLScan, separation_point: int = None, page_overlap: int = 0):
    pages = initialize_scan_pages(scan, separation_point=separation_point, page_overlap=page_overlap)
    trs: List[pdm.PageXMLDoc] = []
    trs.extend(scan.text_regions)
    trs.extend(scan.table_regions)
    for tr in scan.get_regions():
        max_overlap = 0
        best_page = None
        for page in pages:
            if pdm.is_horizontally_overlapping(tr, page):
                overlap = min([tr.coords.right, page.coords.right]) - max([tr.coords.left, page.coords.left])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_page = page
        if best_page is None:
            for p in pages:
                print(f"page {p.id} - {p.coords.box}")
            print(f"text_region {tr.id} - {tr.coords.box}")
            raise ValueError(f"None of the initialized pages overlaps with scan text_region {tr.id}")
        else:
            best_page.add_child(tr)
    return pages
