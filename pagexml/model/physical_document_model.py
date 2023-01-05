from __future__ import annotations
from typing import Dict, List, Set, Tuple, Union
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull


def parse_points(points: Union[str, List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    """Parse a string of PageXML image coordinates into a list of coordinates."""
    if isinstance(points, str):
        points = [point.split(',') for point in points.split(' ')]
        return [(int(point[0]), int(point[1])) for point in points if len(point) == 2]
    elif isinstance(points, list):
        if len(points) == 0:
            raise IndexError("point list cannot be empty")
        for point in points:
            if not isinstance(point, list) and not isinstance(point, tuple):
                print(point)
                print(type(point))
                raise TypeError("List of points must be list of tuples with (int, int)")
            if not isinstance(point[0], int) or not isinstance(point[1], int):
                raise TypeError("List of points must be list of tuples with (int, int)")
        return points


class Coords:

    def __init__(self, points: Union[str, List[Tuple[int, int]]]):
        self.points: List[Tuple[int, int]] = parse_points(points)
        self.point_string = " ".join(
            ",".join([str(point[0]), str(point[1])]) for point in self.points
        )

        self.x = min([point[0] for point in self.points])
        self.y = min(point[1] for point in self.points)
        self.w = max(point[0] for point in self.points) - self.x
        self.h = max([point[1] for point in self.points]) - self.y
        self.type = "coords"

    def __repr__(self):
        return f'{self.__class__.__name__}(points="{self.point_string}")'

    def __str__(self):
        return self.__repr__()

    @property
    def json(self):
        return {
            'type': self.type,
            'points': self.points
        }

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def height(self):
        return self.h

    @property
    def width(self):
        return self.w

    @property
    def box(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


class Baseline(Coords):

    def __init__(self, points: Union[str, List[Tuple[int, int]]]):
        super().__init__(points)
        self.type = "baseline"


def find_baseline_overlap_start_indexes(baseline1: Baseline, baseline2: Baseline) -> Tuple[int, int]:
    """Find the first point in each baseline where the two start to horizontally overlap."""
    baseline1_start_index = 0
    baseline2_start_index = 0
    for bi1, p1 in enumerate(baseline1.points):
        if bi1 < len(baseline1.points) - 1 and baseline1.points[bi1 + 1][0] < baseline2.points[0][0]:
            continue
        baseline1_start_index = bi1
        break
    for bi2, p2 in enumerate(baseline2.points):
        if bi2 < len(baseline2.points) - 1 and baseline2.points[bi2 + 1][0] < baseline1.points[0][0]:
            continue
        baseline2_start_index = bi2
        break
    return baseline1_start_index, baseline2_start_index


def baseline_is_below(baseline1: Baseline, baseline2: Baseline) -> bool:
    """Test if baseline 1 is directly below baseline 2"""
    num_below = 0
    num_overlap = 0
    # find the indexes of the first baseline points where the two lines horizontally overlap
    index1, index2 = find_baseline_overlap_start_indexes(baseline1, baseline2)
    while True:
        # check if the current baseline point of line 1 is below that of the one of line 2
        if baseline1.points[index1][1] > baseline2.points[index2][1]:
            num_below += 1
        num_overlap += 1
        # Check which baseline index to move forward for the next test
        if baseline1.points[index1][0] <= baseline2.points[index2][0]:
            # if current point of baseline 1 is to the left of the current point of baseline 2
            # move to the next point of baseline 1
            index1 += 1
        else:
            # otherwise, move to the next points of baseline 2
            index2 += 1
        if len(baseline1.points) == index1 or len(baseline2.points) == index2:
            # if the end of one of the baselines is reached, counting is done
            break
    # baseline 1 is below baseline 2 if the majority of
    # the horizontally overlapping points is below
    return num_below / num_overlap > 0.5


def has_baseline(doc: PageXMLDoc) -> bool:
    if isinstance(doc, PageXMLTextLine):
        return doc.baseline is not None
    else:
        return False


def get_horizontal_overlap(doc1: PageXMLDoc, doc2: PageXMLDoc) -> int:
    if isinstance(doc1, PageXMLTextLine) and isinstance(doc2, PageXMLTextLine) and\
            doc1.baseline is not None and doc2.baseline is not None:
        overlap_left = max([doc1.baseline.left, doc2.baseline.left])
        overlap_right = min([doc1.baseline.right, doc2.baseline.right])
    else:
        overlap_left = max([doc1.coords.left, doc2.coords.left])
        overlap_right = min([doc1.coords.right, doc2.coords.right])
    return overlap_right - overlap_left if overlap_right > overlap_left else 0


def get_vertical_overlap(doc1: PageXMLDoc, doc2: PageXMLDoc) -> int:
    overlap_top = max([doc1.coords.top, doc2.coords.top])
    overlap_bottom = min([doc1.coords.bottom, doc2.coords.bottom])
    return overlap_bottom - overlap_top if overlap_bottom > overlap_top else 0


def is_vertically_overlapping(region1: PageXMLDoc,
                              region2: PageXMLDoc,
                              threshold: float = 0.5) -> bool:
    if region1.coords is None:
        raise ValueError(f"No coords for {region1.id}")
    elif region2.coords is None:
        raise ValueError(f"No coords for {region2.id}")
    if region1.coords.height == 0 and region2.coords.height == 0:
        return False
    elif region1.coords.height == 0:
        return region2.coords.top <= region1.coords.top <= region2.coords.bottom
    elif region2.coords.height == 0:
        return region1.coords.top <= region2.coords.top <= region1.coords.bottom
    v_overlap = get_vertical_overlap(region1, region2)
    return v_overlap / min(region1.coords.height, region2.coords.height) > threshold


def is_horizontally_overlapping(region1: PageXMLDoc,
                                region2: PageXMLDoc,
                                threshold: float = 0.5) -> bool:
    if region1.coords is None:
        raise ValueError(f"No coords for {region1.id}")
    elif region2.coords is None:
        raise ValueError(f"No coords for {region2.id}")
    h_overlap = get_horizontal_overlap(region1, region2)
    if region1.coords.width == 0 and region2.coords.width == 0:
        return False
    elif region1.coords.width == 0:
        return region2.coords.left <= region1.coords.left <= region2.coords.right
    elif region2.coords.width == 0:
        return region1.coords.left <= region2.coords.left <= region1.coords.right
    return h_overlap / min(region1.coords.width, region2.coords.width) > threshold


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


def get_horizontal_diff(doc1: PageXMLDoc, doc2: PageXMLDoc) -> int:
    if isinstance(doc1, PageXMLTextLine) and isinstance(doc2, PageXMLTextLine) and \
            doc1.baseline is not None and doc2.baseline is not None:
        return abs(doc1.baseline.left - doc2.baseline.left)
    else:
        return abs(doc1.coords.left - doc2.coords.left)


def get_horizontal_diff_ratio(doc1: PageXMLDoc, doc2: PageXMLDoc) -> float:
    horizontal_diff = get_horizontal_diff(doc1, doc2)
    max_right = max(doc1.coords.right, doc2.coords.right)
    min_left = min(doc1.coords.left, doc2.coords.left)
    return horizontal_diff / (max_right - min_left)


def get_horizontal_overlap_ratio(doc1: PageXMLDoc, doc2: PageXMLDoc) -> float:
    horizontal_overlap = get_horizontal_overlap(doc1, doc2)
    max_right = max(doc1.coords.right, doc2.coords.right)
    min_left = min(doc1.coords.left, doc2.coords.left)
    return horizontal_overlap / (max_right - min_left)


def get_vertical_diff(doc1: PageXMLDoc, doc2: PageXMLDoc) -> int:
    if isinstance(doc1, PageXMLTextLine) and isinstance(doc2, PageXMLTextLine) and \
            doc1.baseline is not None and doc2.baseline is not None:
        return abs(doc1.baseline.top - doc2.baseline.top)
    else:
        return abs(doc1.coords.top - doc2.coords.top)


def get_vertical_diff_ratio(doc1: PageXMLDoc, doc2: PageXMLDoc) -> float:
    vertical_diff = get_vertical_diff(doc1, doc2)
    max_bottom = max(doc1.coords.bottom, doc2.coords.bottom)
    min_top = min(doc1.coords.top, doc2.coords.top)
    return vertical_diff / (max_bottom - min_top)


def get_vertical_overlap_ratio(doc1: PageXMLDoc, doc2: PageXMLDoc) -> float:
    vertical_overlap = get_vertical_overlap(doc1, doc2)
    max_bottom = max(doc1.coords.bottom, doc2.coords.bottom)
    min_top = min(doc1.coords.top, doc2.coords.top)
    return vertical_overlap / (max_bottom - min_top)


def sort_lines(line1: PageXMLTextLine, line2: PageXMLTextLine, as_column: bool = True):
    if get_horizontal_overlap(line1, line2):
        if get_vertical_overlap(line1, line2):
            # check which orientation dominates the difference
            horizontal_ratio = get_horizontal_diff_ratio(line1, line2)
            vertical_ratio = get_vertical_diff_ratio(line1, line2)
            if vertical_ratio < 0.2 and horizontal_ratio > 0.8:
                return line1.coords.left < line2.coords.left
            else:
                return line1.coords.top < line2.coords.top
        else:
            return line1.is_below(line2) is False
    elif get_vertical_overlap(line1, line2):
        return line1.coords.left < line2.coords.left
    elif as_column is True:
        # assume lines in a single column, so read from top to bottom
        return line1.coords.top < line2.coords.top
    else:
        # assume lines in multiple columns, so read from left to right
        return line1.coords.left < line2.coords.left


def parse_derived_coords(document_list: list) -> Coords:
    """Derive scan coordinates for a composite document based on the list of documents it contains.
    A convex hull is drawn around all points of all contained documents."""
    return coords_list_to_hull_coords([document.coords for document in document_list])


def coords_list_to_hull_coords(coords_list):
    points = np.array([point for coords in coords_list for point in coords.points])
    edges = points_to_hull_edges(points)
    hull_points = edges_to_hull_points(edges)
    return Coords(hull_points)


def points_to_hull_edges(points):
    hull = ConvexHull(points)
    edges = defaultdict(dict)
    for simplex in hull.simplices:
        p1 = (int(points[simplex, 0][0]), int(points[simplex, 1][0]))
        p2 = (int(points[simplex, 0][1]), int(points[simplex, 1][1]))
        edges[p2][p1] = 1
        edges[p1][p2] = 1
    return edges


def edges_to_hull_points(edges):
    nodes = list(edges.keys())
    curr_point = sorted(nodes)[0]
    sorted_nodes = [curr_point]
    while len(sorted_nodes) < len(nodes):
        for next_point in edges[curr_point]:
            if next_point not in sorted_nodes:
                sorted_nodes.append(next_point)
                curr_point = next_point
                break
    return sorted_nodes


class StructureDoc:

    def __init__(self, doc_id: Union[None, str] = None, doc_type: Union[None, str, List[str]] = None,
                 metadata: Dict[str, any] = None, reading_order: Dict[int, str] = None):
        self.id = doc_id
        self.type = doc_type
        self.main_type = 'doc'
        self.metadata = metadata if metadata else {}
        self.reading_order: Dict[int, str] = reading_order if reading_order else {}
        self.reading_order_number = {}
        self.parent: Union[StructureDoc, None] = None

    def set_parent(self, parent: StructureDoc):
        """Set parent document and add metadata of parent to this document's metadata"""
        self.parent = parent
        self.add_parent_id_to_metadata()

    def add_type(self, doc_type: Union[str, List[str]]) -> None:
        doc_types = [doc_type] if isinstance(doc_type, str) else doc_type
        if isinstance(self.type, str):
            self.type = [self.type]
        for doc_type in doc_types:
            if doc_type not in self.type:
                self.type.append(doc_type)

    def remove_type(self, doc_type: Union[str, List[str]]) -> None:
        doc_types = [doc_type] if isinstance(doc_type, str) else doc_type
        if isinstance(self.type, str):
            self.type = [self.type]
        for doc_type in doc_types:
            if doc_type in self.type:
                self.type.remove(doc_type)
        if len(self.type) == 1:
            self.type = self.type[0]

    def has_type(self, doc_type: str) -> bool:
        if isinstance(self.type, str):
            return doc_type == self.type
        else:
            return doc_type in self.type

    @property
    def types(self) -> Set[str]:
        if isinstance(self.type, str):
            return {self.type}
        else:
            return set(self.type)

    def set_as_parent(self, children: List[StructureDoc]):
        """Set this document as parent of a list of child documents"""
        for child in children:
            child.set_parent(self)

    def add_parent_id_to_metadata(self):
        if self.parent:
            self.metadata['parent_type'] = self.parent.main_type
            self.metadata['parent_id'] = self.parent.id
            if hasattr(self.parent, 'main_type'):
                self.metadata[f'{self.parent.main_type}_id'] = self.parent.id

    @property
    def json(self) -> Dict[str, any]:
        json_data = {
            'id': self.id,
            'type': self.type,
            'metadata': self.metadata
        }
        if self.reading_order:
            json_data['reading_order'] = self.reading_order
        return json_data


class PhysicalStructureDoc(StructureDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None, reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type=doc_type, metadata=metadata, reading_order=reading_order)
        self.coords: Union[None, Coords] = coords
        self.main_type = 'physical_structure_doc'

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        if self.coords:
            doc_json['coords'] = self.coords.points
        return doc_json

    def set_derived_id(self, parent_id: str):
        box_string = f"{self.coords.x}-{self.coords.y}-{self.coords.w}-{self.coords.h}"
        self.id = f"{parent_id}-{self.main_type}-{box_string}"
        self.metadata['id'] = self.id


class LogicalStructureDoc(StructureDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, lines: List[PageXMLTextLine] = None,
                 text_regions: List[PageXMLTextRegion] = None, reading_order: Dict[int, str] = None):
        super().__init__(doc_id, doc_type, metadata, reading_order=reading_order)
        self.lines: List[PageXMLTextLine] = lines if lines else []
        self.text_regions: List[PageXMLTextRegion] = text_regions if text_regions else []
        self.logical_parent: Union[StructureDoc, None] = None

    def set_logical_parent(self, parent: StructureDoc):
        """Set parent document and add metadata of parent to this document's metadata"""
        self.logical_parent = parent
        self.add_logical_parent_id_to_metadata()

    def add_logical_parent_id_to_metadata(self):
        if self.logical_parent:
            self.metadata['logical_parent_type'] = self.logical_parent.main_type
            self.metadata['logical_parent_id'] = self.logical_parent.id
            if hasattr(self.logical_parent, 'main_type'):
                self.metadata[f'{self.logical_parent.main_type}_id'] = self.logical_parent.id


class PageXMLDoc(PhysicalStructureDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None, reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type="pagexml_doc", metadata=metadata, reading_order=reading_order)
        self.coords: Union[None, Coords] = coords
        self.add_type(doc_type)
        self.main_type = 'pagexml_doc'


class PageXMLWord(PageXMLDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 conf: float = None, text: str = None):
        super().__init__(doc_id, "word", metadata, coords)
        self.conf = conf
        self.text = text
        self.main_type = 'word'
        if doc_type:
            self.add_type(doc_type)

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        doc_json['text'] = self.text
        if self.conf:
            doc_json['conf'] = self.conf
        return doc_json


class PageXMLTextLine(PageXMLDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 baseline: Baseline = None, xheight: int = None,
                 text: str = None, words: List[PageXMLWord] = None,
                 reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type="line", metadata=metadata,
                         coords=coords, reading_order=reading_order)
        self.main_type = 'line'
        self.text: Union[None, str] = text
        self.xheight: Union[None, int] = xheight
        self.baseline: Union[None, Baseline] = baseline
        self.words: List[PageXMLWord] = words if words else []
        self.metadata['type'] = 'line'
        self.set_as_parent(self.words)
        if doc_type:
            self.add_type(doc_type)

    def __lt__(self, other: PageXMLTextLine):
        """For sorting text lines. Assumptions: reading from left to right,
        top to bottom. If two lines are horizontally overlapping, sort from
        top to bottom, even if the upper lines is more horizontally indented."""
        if other == self:
            return False
        return sort_lines(self, other, as_column=True)

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        doc_json['text'] = self.text
        if self.baseline:
            doc_json['baseline'] = self.baseline.points
        if self.words:
            doc_json['words'] = [word.json for word in self.words]
        if self.xheight:
            doc_json['xheight'] = self.xheight
        return doc_json

    def get_words(self):
        if self.words:
            return self.words
        elif self.text:
            return self.text.split(' ')
        else:
            return []

    @property
    def num_words(self):
        return len(self.get_words())

    def is_below(self, other: PageXMLTextLine) -> bool:
        """Test if the baseline of this line is directly below the baseline of the other line."""
        # if there is no horizontal overlap, this line is not directly below the other
        if not get_horizontal_overlap(self, other):
            # print("NO HORIZONTAL OVERLAP")
            return False
        # if the bottom of this line is above the top of the other line, this line is above the other
        if self.baseline.bottom < other.baseline.top:
            # print("BOTTOM IS ABOVE TOP")
            return False
        # if most of this line's baseline points are not below most the other's baseline points
        # this line is not below the other
        if baseline_is_below(self.baseline, other.baseline):
            # print("BASELINE IS BELOW")
            return True
        return False

    def is_next_to(self, other: PageXMLTextLine) -> bool:
        """Test if this line is vertically aligned with the other line."""
        if get_vertical_overlap(self, other) == 0:
            # print("NO VERTICAL OVERLAP")
            return False
        if get_horizontal_overlap(self, other) > 40:
            # print("TOO MUCH HORIZONTAL OVERLAP", horizontal_overlap(self.coords, other.coords))
            return False
        if self.baseline.top > other.baseline.bottom + 10:
            # print("VERTICAL BASELINE GAP TOO BIG")
            return False
        elif self.baseline.bottom < other.baseline.top - 10:
            return False
        else:
            return True


class PageXMLTextRegion(PageXMLDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 text_regions: List[PageXMLTextRegion] = None,
                 lines: List[PageXMLTextLine] = None, text: str = None,
                 orientation: float = None, reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type="text_region", metadata=metadata,
                         coords=coords, reading_order=reading_order)
        self.main_type = 'text_region'
        self.text_regions: List[PageXMLTextRegion] = text_regions if text_regions else []
        self.lines: List[PageXMLTextLine] = lines if lines else []
        self.orientation: Union[None, float] = orientation
        self.reading_order_number = {}
        self.text = text
        if self.reading_order:
            self.set_text_regions_in_reader_order()
        if doc_type:
            self.add_type(doc_type)

    def __lt__(self, other: PageXMLTextRegion):
        """For sorting text regions. Assumptions: reading from left to right,
        top to bottom. If two regions are horizontally overlapping, sort from
        top to bottom, even if the upper region is more horizontally indented."""
        if other == self:
            return False
        if is_horizontally_overlapping(self, other):
            return self.coords.top < other.coords.top
        else:
            return self.coords.left < other.coords.left

    def add_child(self, child: PageXMLDoc):
        child.set_parent(self)
        if isinstance(child, PageXMLTextLine):
            self.lines.append(child)
        elif isinstance(child, PageXMLTextRegion):
            self.text_regions.append(child)
        else:
            raise TypeError(f'unknown child type: {child.__class__.__name__}')

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        if self.text:
            doc_json['text'] = self.text
        if self.lines:
            doc_json['lines'] = [line.json for line in self.lines]
        if self.text_regions:
            doc_json['text_regions'] = [text_region.json for text_region in self.text_regions]
        if self.orientation:
            doc_json['orientation'] = self.orientation
        doc_json['stats'] = self.stats
        return doc_json

    def get_text_regions_in_reading_order(self):
        if not self.reading_order:
            return self.text_regions
        tr_ids = list({region_id: None for _index, region_id in sorted(self.reading_order.items(), key=lambda x: x[0])})
        tr_map = {}
        for text_region in self.text_regions:
            if text_region.id not in tr_ids:
                print("reading order:", self.reading_order)
                raise KeyError(f"text_region with id {text_region.id} is not listed in reading_order")
            tr_map[text_region.id] = text_region
        return [tr_map[tr_id] for tr_id in tr_ids if tr_id in tr_map]

    def set_text_regions_in_reader_order(self):
        tr_ids = [tr.id for tr in self.text_regions]
        for order_number in self.reading_order:
            text_region_id = self.reading_order[order_number]
            self.reading_order_number[text_region_id] = order_number
        for tr_id in tr_ids:
            if tr_id not in self.reading_order_number:
                # there is a text_region that was not in the original PageXML output:
                # ignore reading order
                self.reading_order = None
                return None
        self.text_regions = self.get_text_regions_in_reading_order()

    def get_inner_text_regions(self) -> List[PageXMLTextRegion]:
        text_regions: List[PageXMLTextRegion] = []
        for text_region in self.text_regions:
            if text_region.text_regions:
                text_regions += text_region.get_inner_text_regions()
            elif text_region.lines:
                text_regions.append(text_region)
        if not self.text_regions and self.lines:
            text_regions.append(self)
        return text_regions

    def get_lines(self) -> List[PageXMLTextLine]:
        lines: List[PageXMLTextLine] = []
        if self.text_regions:
            if self.reading_order:
                for tr in sorted(self.text_regions, key=lambda t: self.reading_order_number[t.id]):
                    lines += tr.get_lines()
            else:
                for text_region in sorted(self.text_regions):
                    lines += text_region.get_lines()
        if self.lines:
            lines += self.lines
        return lines

    def get_words(self) -> Union[List[str], List[PageXMLWord]]:
        words = []
        if self.text is not None:
            return self.text.split(' ')
        if self.lines:
            for line in self.get_lines():
                if line.words:
                    words += line.words
                elif line.text:
                    words += line.text.split(' ')
        if self.text_regions:
            for tr in self.text_regions:
                words += tr.get_words()
        return words

    @property
    def num_lines(self):
        return len(self.get_lines())

    @property
    def num_words(self):
        return len(self.get_words())

    @property
    def num_text_regions(self):
        return len(self.text_regions)

    @property
    def stats(self):
        return {
            'lines': self.num_lines,
            'words': self.num_words,
            'text_regions': self.num_text_regions
        }


class PageXMLColumn(PageXMLTextRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 text_regions: List[PageXMLTextRegion] = None, lines: List[PageXMLTextLine] = None,
                 reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type="column", metadata=metadata, coords=coords, lines=lines,
                         text_regions=text_regions, reading_order=reading_order)
        self.main_type = 'column'
        if doc_type:
            self.add_type(doc_type)

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        doc_json['stats'] = self.stats
        return doc_json

    @property
    def stats(self):
        stats = super().stats
        return stats


class PageXMLPage(PageXMLTextRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 columns: List[PageXMLColumn] = None, text_regions: List[PageXMLTextRegion] = None,
                 extra: List[PageXMLTextRegion] = None, lines: List[PageXMLTextLine] = None,
                 reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type="page", metadata=metadata, coords=coords, lines=lines,
                         text_regions=text_regions, reading_order=reading_order)
        self.main_type = 'page'
        self.columns: List[PageXMLColumn] = columns if columns else []
        self.extra: List[PageXMLTextRegion] = extra if extra else []
        self.set_as_parent(self.columns)
        self.set_as_parent(self.extra)
        if doc_type:
            self.add_type(doc_type)

    def get_lines(self):
        lines = []
        if self.columns:
            # First, add lines from columns
            for column in sorted(self.columns):
                lines += column.get_lines()
            # Second, add lines from text_regions
            for tr in self.extra:
                lines += tr.get_lines()
        elif self.text_regions:
            if self.reading_order:
                for tr in sorted(self.text_regions, key=lambda t: self.reading_order_number[t]):
                    lines += tr.get_lines()
            else:
                for tr in sorted(self.text_regions):
                    lines += tr.get_lines()
        return lines

    def add_child(self, child: PageXMLDoc, as_extra: bool = False):
        child.set_parent(self)
        if isinstance(child, PageXMLColumn):
            self.columns.append(child)
        elif isinstance(child, PageXMLTextLine):
            self.lines.append(child)
        elif isinstance(child, PageXMLTextRegion):
            if as_extra:
                self.extra.append(child)
            else:
                self.text_regions.append(child)
        else:
            raise TypeError(f'unknown child type: {child.__class__.__name__}')

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        # if self.lines:
        #    doc_json['lines'] = [line.json for line in self.lines]
        # if self.text_regions:
        #     doc_json['text_regions'] = [text_region.json for text_region in self.text_regions]
        if self.columns:
            doc_json['columns'] = [column.json for column in self.columns]
        if self.extra:
            doc_json['extra'] = [text_region.json for text_region in self.extra]
        doc_json['stats'] = self.stats
        return doc_json

    @property
    def stats(self):
        """Pages diverge from other types since they have columns and extra
        text regions, or plain text regions, so have their own way of calculating
        stats."""
        lines = self.get_lines()
        stats = {
            "words": sum([len(line.get_words()) for line in lines]),
            "lines": len(lines)
        }
        if self.columns:
            stats['columns'] = len(self.columns)
            stats['extra'] = len(self.extra)
        elif self.text_regions:
            stats['text_regions'] = len(self.text_regions)
        return stats


class PageXMLScan(PageXMLTextRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 pages: List[PageXMLPage] = None, columns: List[PageXMLColumn] = None,
                 text_regions: List[PageXMLTextRegion] = None, lines: List[PageXMLTextLine] = None,
                 reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type="scan", metadata=metadata, coords=coords, lines=lines,
                         text_regions=text_regions, reading_order=reading_order)
        self.main_type = 'scan'
        self.pages: List[PageXMLPage] = pages if pages else []
        self.columns: List[PageXMLColumn] = columns if columns else []
        self.set_as_parent(self.pages)
        self.set_as_parent(self.columns)
        if doc_type:
            self.add_type(doc_type)

    def add_child(self, child: PageXMLDoc):
        child.set_parent(self)
        if isinstance(child, PageXMLPage):
            self.pages.append(child)
        elif isinstance(child, PageXMLColumn):
            self.columns.append(child)
        elif isinstance(child, PageXMLTextRegion):
            self.text_regions.append(child)
        elif isinstance(child, PageXMLTextLine):
            self.lines.append(child)

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        # if self.lines:
        #     doc_json['lines'] = [line.json for line in self.lines]
        # if self.text_regions:
        #     doc_json['text_regions'] = [text_region.json for text_region in self.text_regions]
        if self.columns:
            doc_json['columns'] = [line.json for line in self.columns]
        if self.pages:
            doc_json['pages'] = [line.json for line in self.pages]
        doc_json['stats'] = self.stats
        return doc_json

    @property
    def stats(self):
        stats = super().stats
        stats['columns'] = len([column for page in self.pages for column in page.columns])
        stats['extra'] = len([text_region for page in self.pages for text_region in page.extra])
        stats['pages'] = len(self.pages)
        return stats


def set_parentage(parent_doc: StructureDoc):
    if hasattr(parent_doc, 'pages') and parent_doc.pages:
        parent_doc.set_as_parent(parent_doc.pages)
        for page in parent_doc.pages:
            set_parentage(page)
    if hasattr(parent_doc, 'columns') and parent_doc.columns:
        parent_doc.set_as_parent(parent_doc.columns)
        for column in parent_doc.columns:
            set_parentage(column)
    if hasattr(parent_doc, 'text_regions') and parent_doc.text_regions:
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
