from __future__ import annotations

from typing import Union, List, Dict

from pagexml.model.basic_document_model import StructureDoc
from pagexml.model.pagexml_document_model import PageXMLTextLine, PageXMLTextRegion


class LogicalStructureDoc(StructureDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, lines: List[PageXMLTextLine] = None,
                 text_regions: List[PageXMLTextRegion] = None, reading_order: Dict[int, str] = None):
        super().__init__(doc_id, doc_type="logical_structure_doc", metadata=metadata, reading_order=reading_order)
        self.lines: List[PageXMLTextLine] = lines if lines else []
        self.text_regions: List[PageXMLTextRegion] = text_regions if text_regions else []
        self.logical_parent: Union[StructureDoc, None] = None
        if doc_type:
            self.add_type(doc_type)
            self.main_type = doc_type
        self.domain = "logical"

    def set_logical_parent(self, parent: StructureDoc):
        """Set parent document and add metadata of parent to this document's metadata"""
        self.logical_parent = parent
        self.add_logical_parent_id_to_metadata()

    def set_as_logical_parent(self, children: Union[StructureDoc, List[StructureDoc]]):
        if isinstance(children, StructureDoc):
            children = [children]
        for child in children:
            child.parent = self

    def add_logical_parent_id_to_metadata(self):
        if self.logical_parent:
            self.metadata['logical_parent_type'] = self.logical_parent.main_type
            self.metadata['logical_parent_id'] = self.logical_parent.id
            if hasattr(self.logical_parent, 'main_type'):
                self.metadata[f'{self.logical_parent.main_type}_id'] = self.logical_parent.id
