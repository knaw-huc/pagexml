from __future__ import annotations

from typing import Union, List, Dict, Set, Tuple

from pagexml.model.coords import Coords, points_to_hull_edges, edges_to_hull_points
from shapely import Polygon


class StructureDoc:

    def __init__(self, doc_id: Union[None, str] = None, doc_type: Union[None, str, List[str]] = None,
                 main_type: Union[None, str] = None,
                 metadata: Dict[str, any] = None, reading_order: Dict[int, str] = None):
        self.id = doc_id
        self.type = "structure_doc"
        self.metadata = metadata if metadata else {}
        self.main_type = 'structure_doc'
        if doc_type is not None:
            self.add_type(doc_type)
            if isinstance(doc_type, str):
                self.main_type = doc_type
        if main_type is not None:
            self.main_type = main_type
        self.domain = None
        self.reading_order: Dict[int, str] = reading_order if reading_order else {}
        self.reading_order_number = {}
        self.parent: Union[StructureDoc, None] = None
        self.logical_parent: Union[StructureDoc, None] = None

    def set_parent(self, parent: StructureDoc):
        """Set parent document and add metadata of parent to this document's metadata"""
        self.parent = parent
        self.add_parent_id_to_metadata()

    def add_type(self, doc_type: Union[str, List[str]]) -> None:
        doc_types = [doc_type] if isinstance(doc_type, str) else doc_type
        if isinstance(self.type, str):
            self.type = [self.type]
        elif isinstance(self.type, set):
            self.type = list(self.type)
        for doc_type in doc_types:
            if doc_type not in self.type:
                self.type.append(doc_type)

    def remove_type(self, doc_type: Union[str, List[str]]) -> None:
        doc_types = [doc_type] if isinstance(doc_type, str) else doc_type
        if isinstance(self.type, str):
            self.type = [self.type]
        elif isinstance(self.type, set):
            self.type = list(self.type)
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
        if self.logical_parent:
            self.metadata['logical_parent_type'] = self.logical_parent.main_type
            self.metadata['logical_parent_id'] = self.logical_parent.id
            if hasattr(self.logical_parent, 'main_type'):
                self.metadata[f'{self.logical_parent.main_type}_id'] = self.logical_parent.id

    @property
    def json(self) -> Dict[str, any]:
        json_data = {
            'id': self.id,
            'type': list(self.type) if isinstance(self.type, set) else self.type,
            'main_type': self.main_type,
            'domain': self.domain,
            'metadata': self.metadata
        }
        if self.reading_order:
            json_data['reading_order'] = self.reading_order
        return json_data


class PhysicalStructureDoc(StructureDoc):

    def __init__(self, doc_id: str = None,
                 doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None,
                 coords: Coords = None,
                 reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type='physical_structure_doc', metadata=metadata, reading_order=reading_order)
        self.coords: Union[None, Coords] = coords
        self._area = None
        if doc_type:
            self.main_type = doc_type
            self.add_type(doc_type)
        self.domain = 'physical'

    @property
    def aspect_ratio(self):
        return self.coords.width / self.coords.height

    @property
    def area(self):
        """Returns the size of the area represented by the convex hull of the coordinates.

         The area is calculated the first time this function is called and stored in a
         private property for later calls. The reason to not call it at object instantiation
         is that it probably not often needed and only computing it when needed is more
         efficient."""
        if self._area is None:
            if self.coords is None:
                self._area = 0
            else:
                self._area = poly_area(self.coords.points)
        return self._area

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        doc_json['domain'] = self.domain
        if self.coords:
            doc_json['coords'] = self.coords.points
        return doc_json

    @property
    def children(self):
        return []

    def set_derived_id(self, parent_id: str):
        box_string = f"{self.coords.x}-{self.coords.y}-{self.coords.w}-{self.coords.h}"
        self.id = f"{parent_id}-{self.main_type}-{box_string}"
        # self.metadata['id'] = self.id

    def add_parent_id_to_metadata(self):
        if self.parent:
            self.metadata['parent_type'] = self.parent.main_type
            self.metadata['parent_id'] = self.parent.id
            if hasattr(self.parent, 'main_type') and self.parent.main_type is not None:
                self.metadata[f'{self.parent.main_type}_id'] = self.parent.id


class EmptyRegionDoc(PhysicalStructureDoc):

    def __init__(self, doc_id: str = None, doc_type: str = None, metadata: Dict[str, any] = None,
                 coords: Coords = None):
        super().__init__(doc_id=doc_id, doc_type=doc_type, metadata=metadata, coords=coords)
        self.add_type('empty')
        if doc_type is None:
            self.main_type = 'empty'


def poly_area(points: Union[List[Tuple[int, int]], Coords, PhysicalStructureDoc],
              debug: int = 0):
    """Compute the surface area of a polygon represented by a set of Points."""
    if isinstance(points, PhysicalStructureDoc):
        points = points.coords.points
    elif isinstance(points, Coords):
        points = points.points
    elif points is None:
        return 0
    if len(points) <= 2:
        # two points represent a line, which has an area of zero
        return 0
    hull_edges = points_to_hull_edges(points)
    hull_points = edges_to_hull_points(hull_edges)
    if debug > 0:
        print(f'physical_document_model.poly_area - hull_points: {hull_points}')
    polygon = Polygon(hull_points)
    return polygon.area