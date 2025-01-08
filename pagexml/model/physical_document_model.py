from __future__ import annotations

from typing import List, Union

from pagexml.model.coords import Baseline, Coords
from pagexml.model.basic_document_model import StructureDoc
from pagexml.model.pagexml_document_model import PageXMLDoc, PageXMLTextLine, PageXMLTextRegion
from pagexml.model.pagexml_document_model import get_horizontal_overlap, get_vertical_overlap
from pagexml.model.pagexml_document_model import sort_lines
from pagexml.model.pagexml_document_model import is_vertically_overlapping, is_horizontally_overlapping
from pagexml.model.pagexml_document_model import get_vertical_diff, get_horizontal_diff
from pagexml.model.pagexml_document_model import get_vertical_diff_ratio, get_horizontal_diff_ratio
from pagexml.model.pagexml_document_model import PageXMLWord, PageXMLColumn, PageXMLPage, PageXMLScan


def combine_doc_types(doc_type1: Union[str, List[str], None],
                      doc_type2: Union[str, List[str], None]):
    if doc_type1 is None:
        return doc_type2
    elif doc_type2 is None:
        return doc_type1

    # make a new list of combined type of doc_type1 so the original doc_type1 is unaffected
    if isinstance(doc_type1, str):
        combined_type = [doc_type1]
    else:
        combined_type = [dt for dt in doc_type1]

    # make sure doc_type2 is a list
    if isinstance(doc_type2, str):
        doc_type2 = [doc_type2]

    # add new doc 2 types to the combined type
    for dt2 in doc_type2:
        if dt2 not in doc_type1:
            combined_type.append(dt2)
    return combined_type


def set_parentage(parent_doc: StructureDoc):
    if isinstance(parent_doc, PageXMLScan) or hasattr(parent_doc, 'pages') and parent_doc.pages:
        parent_doc.set_as_parent(parent_doc.pages)
        for page in parent_doc.pages:
            set_parentage(page)
    if isinstance(parent_doc, PageXMLPage) or hasattr(parent_doc, 'columns') and parent_doc.columns:
        parent_doc.set_as_parent(parent_doc.columns)
        for column in parent_doc.columns:
            set_parentage(column)
    if isinstance(parent_doc, PageXMLColumn) or hasattr(parent_doc, 'text_regions') and parent_doc.text_regions:
        parent_doc.set_as_parent(parent_doc.text_regions)
        for text_region in parent_doc.text_regions:
            set_parentage(text_region)
    if hasattr(parent_doc, 'lines') and parent_doc.lines:
        parent_doc.set_as_parent(parent_doc.lines)
        for line in parent_doc.lines:
            set_parentage(line)
    if hasattr(parent_doc, 'words') and parent_doc.words:
        parent_doc.set_as_parent(parent_doc.words)
        for word in parent_doc.words:
            set_parentage(word)
    if isinstance(parent_doc, PageXMLWord):
        pass


def in_same_column(element1: PageXMLDoc, element2: PageXMLDoc) -> bool:
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
        return get_horizontal_overlap(element1, element2) > (element1.coords.w / 2)


def has_baseline(doc: PageXMLDoc) -> bool:
    if isinstance(doc, PageXMLTextLine):
        return doc.baseline is not None
    else:
        return False


def is_below(region1: PageXMLTextRegion, region2: PageXMLTextRegion, margin: int = 20) -> bool:
    if is_horizontally_overlapping(region1, region2):
        return region1.coords.top > region2.coords.bottom - margin
    else:
        return False


def is_next_to(region1: PageXMLTextRegion, region2: PageXMLTextRegion, margin: int = 20) -> bool:
    if is_vertically_overlapping(region1, region2):
        return region1.coords.left > region2.coords.right - margin
    else:
        return False


def horizontal_distance(doc1: PageXMLDoc, doc2: PageXMLDoc):
    if doc1.coords.right < doc2.coords.left:
        # doc1 is to the left of doc2
        return doc2.coords.left - doc1.coords.right
    elif doc1.coords.left > doc2.coords.right:
        # doc1 is to the right of doc2
        return doc1.coords.left - doc2.coords.right
    else:
        # doc1 and doc2 horizontally overlap
        return 0


def vertical_distance(doc1: PageXMLDoc, doc2: PageXMLDoc):
    if doc1.coords.bottom < doc2.coords.top:
        # doc1 is above doc2
        return doc2.coords.top - doc1.coords.bottom
    elif doc1.coords.top > doc2.coords.bottom:
        # doc1 is below doc2
        return doc1.coords.top - doc2.coords.bottom
    else:
        # doc1 and doc2 vertically overlap
        return 0


def get_horizontal_overlap_ratio(doc1: PageXMLDoc, doc2: PageXMLDoc) -> float:
    horizontal_overlap = get_horizontal_overlap(doc1, doc2)
    max_right = max(doc1.coords.right, doc2.coords.right)
    min_left = min(doc1.coords.left, doc2.coords.left)
    return horizontal_overlap / (max_right - min_left)


def get_vertical_overlap_ratio(doc1: PageXMLDoc, doc2: PageXMLDoc) -> float:
    vertical_overlap = get_vertical_overlap(doc1, doc2)
    max_bottom = max(doc1.coords.bottom, doc2.coords.bottom)
    min_top = min(doc1.coords.top, doc2.coords.top)
    return vertical_overlap / (max_bottom - min_top)


