from typing import List, Union

from shapely import Polygon

import pagexml.model.pagexml_document_model as pdm
from pagexml.model.coords import parse_derived_coords


INVERSE = {
    'above': 'below',
    'below': 'above',
    'left': 'right',
    'right': 'left',
    'over': 'over'
}

def regions_poly_overlap(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion):
    poly1 = Polygon(region1.coords.points)
    poly2 = Polygon(region2.coords.points)
    return poly1.intersects(poly2)


def regions_box_overlap_horizontally(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion,
                                     debug: int = 0):
    max_left = max(region1.coords.left, region2.coords.left)
    min_right = min(region1.coords.right, region2.coords.right)
    if debug > 1:
        print(f"  r1.left: {region1.coords.left}\tr2.left: {region2.coords.left}\tmax_left: {max_left}")
        print(f"  r1.right: {region1.coords.right}\tr2.right: {region2.coords.right}\tmin_right: {min_right}")
    return min_right > max_left


def regions_box_overlap_vertically(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion,
                                   debug: int = 0):
    max_top = max(region1.coords.top, region2.coords.top)
    min_bottom = min(region1.coords.bottom, region2.coords.bottom)
    if debug > 1:
        print(f"    r1.top: {region1.coords.top}\tr2.top: {region2.coords.top}\tmax_top: {max_top}")
        print(f"    r1.bottom: {region1.coords.bottom}\tr2.bottom: {region2.coords.bottom}\tmin_bottom: {min_bottom}")
    return min_bottom > max_top


def regions_box_overlap(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion,
                        debug: int = 0):
    if debug > 1:
        print(f"  r1 box: {region1.coords.box}")
        print(f"  r2 box: {region2.coords.box}")
    if regions_box_overlap_horizontally(region1, region2, debug=debug) is False:
        return False
    if regions_box_overlap_vertically(region1, region2, debug=debug) is False:
        return False
    return True


def region_touches_left(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion,
                        max_diff_abs: int = None, max_diff_rel: float = None):
    """Check if the left side of region1 touches the right side of region 2.
    That is, region 2 is on the left side of region 1 and they are not overlapping
    but touching. Use `max_diff_abs` to allow for a small absolute margin or
    `max_diff_rel` for a relative margin."""
    diff = abs(region1.coords.left - region2.coords.right)
    rel_diff = diff / region1.coords.width
    return region_diff_is_touch(diff, rel_diff, max_diff_abs, max_diff_rel)


def region_touches_right(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion,
                         max_diff_abs: int = None, max_diff_rel: float = None):
    """Check if the right side of region1 touches the left side of region 2.
    That is, region 2 is on the right side of region 1 and they are not overlapping
    but touching. Use `max_diff_abs` to allow for a small absolute margin or
    `max_diff_rel` for a relative margin."""
    diff = abs(region1.coords.right - region2.coords.left)
    rel_diff = diff / region1.coords.width
    return region_diff_is_touch(diff, rel_diff, max_diff_abs, max_diff_rel)


def region_touches_above(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion,
                         max_diff_abs: int = None, max_diff_rel: float = None):
    """Check if the top side of region1 touches the bottom side of region 2.
    That is, region 2 is above region 1 and they are not overlapping
    but touching. Use `max_diff_abs` to allow for a small absolute margin or
    `max_diff_rel` for a relative margin."""
    diff = abs(region1.coords.top - region2.coords.bottom)
    rel_diff = diff / region1.coords.height
    return region_diff_is_touch(diff, rel_diff, max_diff_abs, max_diff_rel)


def region_touches_below(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion,
                         max_diff_abs: int = None, max_diff_rel: float = None):
    """Check if the bottom side of region1 touches the top side of region 2.
    That is, region 2 is below region 1 and they are not overlapping
    but touching. Use `max_diff_abs` to allow for a small absolute margin or
    `max_diff_rel` for a relative margin."""
    diff = abs(region1.coords.bottom - region2.coords.top)
    rel_diff = diff / region1.coords.height
    return region_diff_is_touch(diff, rel_diff, max_diff_abs, max_diff_rel)


def region_diff_is_touch(diff: int, rel_diff: float, max_diff_abs: int = None,
                         max_diff_rel: float = None):
    if max_diff_abs is not None:
        return diff <= max_diff_abs
    elif max_diff_rel is not None:
        return rel_diff <= max_diff_rel
    else:
        return diff == 0


def get_relative_hloc(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion):
    if region2.coords.right <= region1.coords.left:
        return 'left'
    elif region2.coords.left >= region1.coords.right:
        return 'right'
    else:
        return 'over'


def get_relative_vloc(region1: pdm.PageXMLRegion, region2: pdm.PageXMLRegion):
    if region2.coords.bottom <= region1.coords.top:
        return 'above'
    elif region2.coords.top >= region1.coords.bottom:
        return 'below'
    else:
        return 'over'


def invert_rel_loc(rel_loc: str):
    if rel_loc not in INVERSE:
        raise ValueError(f"invalid relative location '{rel_loc}', "
                         f"must be one of {list(INVERSE.keys())}.")
    return INVERSE[rel_loc]


def line_and_region_align_horizontally(line: pdm.Coords,
                                       text_region: pdm.PageXMLRegion):
    """Determine whether a horizontal line and region
    align at the top or bottom of the region."""
    return text_region.coords.top == line.top or text_region.coords.bottom == line.top


def line_and_region_overlap_horizontally(line: pdm.Coords, text_region: pdm.PageXMLRegion):
    return text_region.coords.top < line.top < text_region.coords.bottom


def get_regions_horizontal_over_line(line: pdm.Coords,
                                     text_regions: List[pdm.PageXMLRegion],
                                     boundary_thickness: int = 0):
    return [tr for tr in text_regions if line_and_region_align_horizontally(line, tr)]


def make_empty_region(left: int, top: int, right: int, bottom: int, region_id: str):
    coords = pdm.Coords([
        (left, top), (right, top), (right, bottom), (left, bottom)
    ])
    return pdm.PageXMLRegion(doc_id=region_id, doc_type='empty_region', coords=coords)


def make_region_neighbours(inner_region: pdm.PageXMLRegion, outer_region: pdm.PageXMLRegion,
                           debug: int = 0):
    neighbour_regions = []
    ic = inner_region.coords
    oc = outer_region.coords
    if debug > 0:
        print(f"inner.coords: {ic.box}")
        print(f"outer.coords: {oc.box}")
    # check if we need to make an above region
    if inner_region.coords.top > outer_region.coords.top:
        above_region = make_empty_region(oc.left, oc.top, oc.right, ic.top, region_id=f'above_{inner_region.id}')
        neighbour_regions.append(above_region)
    # check if we need to make a below region
    if inner_region.coords.bottom < outer_region.coords.bottom:
        below_region = make_empty_region(oc.left, ic.bottom, oc.right, oc.bottom, region_id=f'below_{inner_region.id}')
        neighbour_regions.append(below_region)
    # check if we need to make a left region
    if inner_region.coords.left > outer_region.coords.left:
        left_region = make_empty_region(oc.left, ic.top, ic.left, ic.bottom, region_id=f'left_{inner_region.id}')
        neighbour_regions.append(left_region)
    # check if we need to make a right region
    if inner_region.coords.right < outer_region.coords.right:
        right_region = make_empty_region(ic.right, ic.top, oc.right, ic.bottom, region_id=f'right_{inner_region.id}')
        neighbour_regions.append(right_region)
    return neighbour_regions


def make_empty_regions(doc: pdm.PageXMLTextRegion, debug: int = 0):
    if len(doc.lines) > 0:
        return []
    if len(doc.text_regions) == 0:
        return []
    empty_region = make_empty_region(doc.coords.left, doc.coords.top,
                                     doc.coords.right, doc.coords.bottom,
                                     f"empty_{doc.id}")
    region_exists = set()
    candidate_regions = [empty_region]
    non_overlapping_regions: List[pdm.PageXMLRegion] = [tr for tr in doc.text_regions]
    region_exists.add(empty_region.coords.box_string)
    empty_regions = []
    if debug > 0:
        print(f"initial doc: {doc.id}, candidate_regions: {len(candidate_regions)}, empty: {len(empty_regions)}")
    while len(candidate_regions) > 0:
        candidate_region = candidate_regions.pop(0)
        overlapping_regions = [tr for tr in non_overlapping_regions if regions_box_overlap(tr, candidate_region, debug=debug)]
        if len(overlapping_regions) == 0:
            if debug > 0:
                cr = candidate_region
                print(f"adding empty region: {cr.id} {cr.coords.box_string}")
            empty_regions.append(candidate_region)
            non_overlapping_regions.append(candidate_region)
        else:
            tr = overlapping_regions[0]
            new_regions = make_region_neighbours(tr, candidate_region)
            candidate_regions.extend(new_regions)
        if debug > 0:
            print(f"current candidate: {candidate_region.id}, {candidate_region.coords.box_string} "
                  f"candidate_regions: {len(candidate_regions)}, "
                  f"empty: {len(empty_regions)}")
            for cr in candidate_regions:
                print(f"\tcandidate: {cr.id}\t{cr.coords.box_string}")
    return empty_regions
