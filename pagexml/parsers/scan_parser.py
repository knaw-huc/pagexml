import copy
from typing import List, Tuple

import numpy as np

import pagexml.model.physical_document_model as pdm


def set_average_scan_width(scans: List[pdm.PageXMLScan]):
    widths = np.array([scan.coords.width for scan in scans])
    heights = np.array([scan.coords.height for scan in scans])
    avg_width = widths.mean()
    avg_height = heights.mean()
    for scan in scans:
        scan.metadata['avg_scan_width'] = avg_width
        scan.metadata['avg_scan_height'] = avg_height


def get_page_split_widths(scan: pdm.PageXMLScan, page_overlap: int = 0) -> Tuple[int, int, int, int]:
    if hasattr(scan, 'coords') is False or scan.coords is None or scan.coords.points is None:
        print(f'ERROR determining scan width in get_page_split_widths for scan {scan.id}')
    scan_width = scan.coords.width
    if 'avg_scan_width' in scan.metadata:
        scan_width_ratio = scan.coords.width / scan.metadata['avg_scan_width']
        if scan_width_ratio < 0.8 or scan_width_ratio > 1.25:
            scan_width = scan.metadata['avg_scan_width']
    even_start = scan.coords.left
    odd_end = scan_width
    odd_start = scan_width / 2 - page_overlap
    even_end = scan_width / 2 + page_overlap
    return even_start, int(even_end), int(odd_start), odd_end


def initialize_pagexml_page(scan_doc: pdm.PageXMLScan, side: str,
                            page_start: int, page_end: int) -> pdm.PageXMLPage:
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
        (region[2], region[3]), (region[0], region[1])
    ]
    coords = pdm.Coords(points)
    page_doc = pdm.PageXMLPage(doc_id=metadata['page_id'], metadata=metadata, coords=coords,
                               text_regions=[])
    page_doc.set_parent(scan_doc)
    return page_doc


def initialize_scan_pages(scan: pdm.PageXMLScan, page_overlap: int = 0):
    even_start, even_end, odd_start, odd_end = get_page_split_widths(scan, page_overlap=page_overlap)
    page_even = initialize_pagexml_page(scan, 'even', even_start, even_end)
    page_odd = initialize_pagexml_page(scan, 'odd', odd_start, odd_end)
    pages = [page_even, page_odd]
    if scan.coords.width > odd_end:
        extra_start = odd_end
        extra_end = scan.coords.width
        page_extra = initialize_pagexml_page(scan, 'extra', extra_start, extra_end)
        pages.append(page_extra)
    return pages


def split_scan_pages(scan: pdm.PageXMLScan, page_overlap: int = 0):
    pages = initialize_scan_pages(scan, page_overlap=page_overlap)
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
