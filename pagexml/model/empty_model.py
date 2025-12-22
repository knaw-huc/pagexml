from collections import defaultdict
from typing import Dict, List

import pagexml.helper.pagexml_helper as pagexml_helper
import pagexml.helper.spatial_helper as spatial_helper
import pagexml.model.pagexml_document_model as pdm


def get_region_alignment(doc: pdm.PageXMLTextRegion = None,
                         regions: List[pdm.PageXMLTextRegion] = None):
    if doc is None and regions is None:
        raise ValueError(f"must pass 'doc' or 'regions'.")
    if doc is not None and regions is not None:
        raise ValueError(f"must pass either 'doc' or 'regions', not both.")
    if doc is not None:
        regions = [region for region in doc.text_regions]
    rel_loc = {}
    aligned = {
        'left': defaultdict(list),
        'right': defaultdict(list),
        'above': defaultdict(list),
        'below': defaultdict(list),
    }
    for ci, curr_tr in enumerate(regions):
        if curr_tr == regions[-1]:
            break
        for next_tr in regions[ci+1:]:
            rel_hloc = spatial_helper.get_relative_hloc(curr_tr, next_tr)
            rel_vloc = spatial_helper.get_relative_vloc(curr_tr, next_tr)
            inv_rel_hloc = spatial_helper.invert_rel_loc(rel_hloc)
            inv_rel_vloc = spatial_helper.invert_rel_loc(rel_vloc)
            if rel_hloc == 'over' and rel_vloc == 'over':
                raise ValueError(f"regions overlap both horizontally and vertically:"
                                 f"\n\t{curr_tr}\n\t{next_tr}")
            rel_loc[(curr_tr, next_tr)] = (rel_hloc, rel_vloc)
            rel_loc[(next_tr, curr_tr)] = (inv_rel_hloc, inv_rel_vloc)
            aligned[rel_vloc][curr_tr].append(next_tr)
    return rel_loc, aligned

def get_region_top_line(region: pdm.PageXMLRegion):
    return pdm.Coords([(region.coords.left, region.coords.top), (region.coords.right, region.coords.top)])


def get_region_bottom_line(region: pdm.PageXMLRegion):
    return pdm.Coords([(region.coords.left, region.coords.bottom), (region.coords.right, region.coords.bottom)])


def get_region_horizontal_lines(region: pdm.PageXMLTextRegion, doc: pdm.PageXMLTextRegion,
                                aligned: Dict[str, Dict[pdm.PageXMLTextRegion, List[pdm.PageXMLTextRegion]]]):
    """Determine the top left, top right, bottom left, bottom right lines between a region
    and it's neighbours."""
    if region not in doc.text_regions:
        raise ValueError(f"region {region.id} is not direct text regions of doc {doc.id}.")

    doc_left, doc_right = doc.coords.left, doc.coords.right

    top_line = get_region_top_line(region)
    bottom_line = get_region_bottom_line(region)

    if len(aligned['left'][region]) == 0:
        top_left_boundary = doc_left
        bottom_left_boundary = doc_left
    else:
        top_left_aligned = spatial_helper.get_regions_horizontal_over_line(top_line, aligned['left'][region])
        right_most_top_aligned = top_left_aligned.index(max([tr.coords.right for tr in top_left_aligned]))
        top_left_boundary = right_most_top_aligned.coords.right
        bottom_left_aligned = spatial_helper.get_regions_horizontal_over_line(bottom_line, aligned['left'][region])
        right_most_bottom_aligned = bottom_left_aligned.index(max([tr.coords.right for tr in bottom_left_aligned]))
        bottom_left_boundary = right_most_bottom_aligned.coords.right
    if len(aligned['right'][region]) == 0:
        top_right_boundary = doc_right
        bottom_right_boundary = doc_right
    else:
        top_right_aligned = spatial_helper.get_regions_horizontal_over_line(top_line, aligned['right'][region])
        left_most_top_aligned = top_right_aligned.index(min([tr.coords.left for tr in top_right_aligned]))
        top_right_boundary = left_most_top_aligned.coords.left
        bottom_right_aligned = spatial_helper.get_regions_horizontal_over_line(bottom_line, aligned['right'][region])
        left_most_bottom_aligned = bottom_right_aligned.index(min([tr.coords.left for tr in bottom_right_aligned]))
        bottom_right_boundary = left_most_bottom_aligned.coords.left
    # top_left_line = pdm.Coords([(top_left_boundary, region.coords.top), (region.coords.left, region.coords.top)])
    # top_right_line = pdm.Coords([(top_right_boundary, region.coords.top), (region.coords.right, region.coords.top)])
    top_line = pdm.Coords([(top_left_boundary, region.coords.top), (top_right_boundary, region.coords.top)])
    bottom_line = pdm.Coords([(bottom_left_boundary, region.coords.bottom), (bottom_right_boundary, region.coords.bottom)])
    # bottom_left_line = pdm.Coords([(bottom_left_boundary, region.coords.bottom), (region.coords.left, region.coords.bottom)])
    # bottom_right_line = pdm.Coords([(bottom_right_boundary, region.coords.bottom), (region.coords.right, region.coords.bottom)])
    # return top_left_line, top_right_line, bottom_left_line, bottom_right_line
    return top_line, bottom_line


def get_doc_horizontal_lines(doc: pdm.PageXMLTextRegion):
    top_line = get_region_top_line(doc)
    bottom_line = get_region_bottom_line(doc)
    horizontal_lines = [top_line, bottom_line]
    rel_loc, aligned = get_region_alignment(doc=doc)
    for curr_tr in sorted(doc.text_regions, key = lambda tr: tr.coords.top):
        # top_left, top_right, bottom_left, bottom_right = get_region_horizontal_lines(curr_tr, doc, aligned)
        top_line, bottom_line = get_region_horizontal_lines(curr_tr, doc, aligned)
        horizontal_lines.extend([top_line, bottom_line])



def get_empty_regions(doc: pdm.PageXMLTextRegion):
    # make sure there are no overlapping regions
    return None
