from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from scipy.spatial import QhullError, ConvexHull


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
        """Coordinates of a PageXML region based on a set of points."""
        self.points: List[Tuple[int, int]] = parse_points(points)
        self.point_string = " ".join(
            ",".join([str(point[0]), str(point[1])]) for point in self.points
        )

        self.x = min([point[0] for point in self.points])
        self.y = min([point[1] for point in self.points])
        self.w = max([point[0] for point in self.points]) - self.x
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


def parse_derived_coords(document_list: list) -> Coords:
    """Derive scan coordinates for a composite document based on the list of documents it contains.
    A convex hull is drawn around all points of all contained documents."""
    try:
        return coords_list_to_hull_coords([document.coords for document in document_list])
    except (IndexError, QhullError) as err:
        print('pagexml.model.physical_document_model.parse_derived_coords - '
              'Error with coords in list of documents with the following ids:\n',
              [doc.id for doc in document_list])
        raise


def coords_list_to_hull_coords(coords_list):
    points = [point for coords in coords_list for point in coords.points]
    if len(points) <= 2:
        return Coords(points)
    try:
        edges = points_to_hull_edges(points)
        hull_points = edges_to_hull_points(edges)
        return Coords(hull_points)
    except (IndexError, QhullError):
        print('pagexml.model.physical_document_model.coords_list_to_hull_coords - IndexError')
        print('coords in coords_list:', [coords for coords in coords_list])
        print('points derived from list of coords:', points)
        raise


def points_to_hull_edges(points: List[Tuple[int, int]]):
    points_array = np.array(points)
    hull = ConvexHull(points_array)
    edges = defaultdict(dict)
    for simplex in hull.simplices:
        p1 = (int(points_array[simplex, 0][0]), int(points_array[simplex, 1][0]))
        p2 = (int(points_array[simplex, 0][1]), int(points_array[simplex, 1][1]))
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


