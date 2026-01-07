from __future__ import annotations

import json
import csv
from typing import Union, List, Dict, Set

from lxml import etree
from pagexml.model.basic_document_model import PhysicalStructureDoc
from pagexml.model.coords import Coords, Baseline, baseline_is_below, parse_derived_coords
from pagexml.model.xml import make_empty_pagexml, add_pagexml_sub_element
import pagexml.model.xml as xml


CHILD_PROPERTIES = [
    'pages', 'columns',
    'text_regions', 'table_regions', 'empty_regions',
    'rows', 'cells', 'lines', 'words'
]


class PageXMLDoc(PhysicalStructureDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None,
                 attrs: Dict[str, any] = None,
                 coords: Coords = None, reading_order: Dict[int, str] = None,
                 reading_order_attributes: Dict[str, any] = None, orientation: float = None):
        if doc_type is None:
            doc_type = 'pagexml_doc'
        super().__init__(doc_id=doc_id, doc_type=doc_type, metadata=metadata,
                         reading_order=reading_order, coords=coords)
        self.reading_order_attributes = reading_order_attributes
        self.orientation = orientation
        self.pagexml_type = None
        self.attrs = attrs if attrs else {}
        self.type = doc_type
        # self.add_type('pagexml_doc')

    @property
    def stats(self):
        return {}

    @property
    def custom(self):
        if 'custom_attributes' in self.metadata:
            return self.metadata['custom_attributes']
        else:
            return {}

    @property
    def json(self):
        doc_json = super(PageXMLDoc, self).json
        doc_json['attributes'] = self.attrs
        doc_json['reading_order_attributes'] = self.reading_order_attributes
        return doc_json

    def add_to_pagexml(self, parent: etree.Element):
        return parent

    def _to_pagexml(self, page_xml):
        pass

    def to_pagexml(self, tostring: bool = False):
        image_filename = self.metadata['scan_id'] if 'scan_id' in self.metadata else None
        image_width = get_image_width(self)
        image_height = get_image_height(self)
        doc_pagexml = make_empty_pagexml(metadata=self.metadata, imageFilename=image_filename,
                                         imageWidth=image_width, imageHeight=image_height)
        page_xml = doc_pagexml.find(xml.PAGE + 'Page')
        self._to_pagexml(page_xml)
        if tostring is True:
            xml_string = etree.tostring(doc_pagexml, pretty_print=True, encoding="UTF-8", xml_declaration=True)
            return xml_string.decode()
        else:
            return doc_pagexml


def get_doc_attributes(doc: PageXMLDoc):
    attributes = {}
    if doc.orientation:
        attributes['orientation'] = doc.orientation
    return attributes


def get_image_width(doc: PageXMLDoc):
    if 'scan_width' in doc.metadata:
        return int(doc.metadata['scan_width'])
    elif doc.coords is not None:
        return doc.coords.width
    else:
        return None


def get_image_height(doc: PageXMLDoc):
    if 'scan_height' in doc.metadata:
        return int(doc.metadata['scan_height'])
    elif doc.coords is not None:
        return doc.coords.height
    else:
        return None


class PageXMLWord(PageXMLDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 baseline: Baseline = None,
                 conf: float = None, text: str = None):
        super().__init__(doc_id=doc_id, doc_type="word", attrs=attrs,
                         metadata=metadata, coords=coords)
        self.conf = conf
        self.baseline = baseline
        self.text = text
        self.main_type = 'word'
        self.pagexml_type = 'Word'
        if doc_type:
            self.add_type(doc_type)

    def __repr__(self):
        content_string = f"id={self.id}, type={self.type}, text={self.text}"
        if self.conf is not None:
            content_string += f", conf={self.conf}"
        return f"{self.__class__.__name__}({content_string})"

    @property
    def stats(self):
        return {
            'words': 1 if self.text is not None else 0,
            'chars': len(self.text) if self.text is not None else 0
        }

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        doc_json['text'] = self.text
        if self.conf:
            doc_json['conf'] = self.conf
        return doc_json

    def add_to_pagexml(self, parent: etree.Element = None):
        add_pagexml_sub_element(parent, 'Word', sub_id=self.id, custom=self.custom, coords=self.coords,
                                baseline=self.baseline, text=self.text, conf=self.conf)

    def _to_pagexml(self, page_xml: etree.Element):
        # make dummy line and region
        line = PageXMLTextLine(coords=self.coords, words=[self], text=self.text)
        tr = PageXMLTextRegion(coords=self.coords, lines=[line])
        # add region to page and line to region
        tr_xml = add_pagexml_sub_element(page_xml, 'TextRegion', sub_id=tr.id, custom=tr.custom,
                                         coords=tr.coords)
        line_xml = add_pagexml_sub_element(tr_xml, 'TextLine', sub_id=line.id, custom=line.custom,
                                           coords=line.coords)
        # finally, add word itself to line
        self.add_to_pagexml(line_xml)


class PageXMLTextLine(PageXMLDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 baseline: Baseline = None, xheight: int = None,
                 conf: float = None, text: str = None, words: List[PageXMLWord] = None,
                 reading_order: Dict[int, str] = None,
                 reading_order_attributes: Dict[str, any] = None):
        super().__init__(doc_id=doc_id, doc_type="line", attrs=attrs, metadata=metadata,
                         coords=coords, reading_order=reading_order,
                         reading_order_attributes=reading_order_attributes)
        self.main_type = 'line'
        self.pagexml_type = 'TextLine'
        self.conf = conf
        self.text: Union[None, str] = text
        self.xheight: Union[None, int] = xheight
        self.baseline: Union[None, Baseline] = baseline
        self.words: List[PageXMLWord] = words if words else []
        self.metadata['type'] = 'line'
        self.set_as_parent(self.words)
        if doc_type:
            self.add_type(doc_type)

    def __repr__(self):
        content_string = (f"\n\tid={self.id}, \n\ttype={self.type}, \n\ttext=\"{self.text}\" "
                          f"\n\tconf={self.conf}\n")
        return f"{self.__class__.__name__}({content_string})"

    def __lt__(self, other: PageXMLTextLine):
        """For sorting text lines. Assumptions: reading from left to right,
        top to bottom. If two lines are horizontally overlapping, sort from
        top to bottom, even if the upper lines is more horizontally indented."""
        if other == self:
            return False
        return sort_lines(self, other, as_column=True)

    @property
    def text_height(self):
        # Assume box cutout has top and bottom 25% around average character height
        return self.xheight if self.xheight else self.coords.height / 2

    @property
    def length(self):
        return len(self.text) if self.text is not None else 0

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        doc_json['text'] = self.text
        if self.conf is not None:
            doc_json['conf'] = self.conf
        if self.baseline:
            doc_json['baseline'] = self.baseline.points
        if self.words:
            doc_json['words'] = [word.json for word in self.words]
        if self.xheight:
            doc_json['xheight'] = self.xheight
        return doc_json

    @property
    def children(self):
        return self.words

    @property
    def stats(self):
        return {
            'lines': 1 if self.text is not None else 0,
            'words': self.num_words,
            'chars': self.length
        }

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

    def is_below(self, other: PageXMLTextLine, direct_only: bool = True) -> bool:
        """Test if the baseline of this line is directly below the baseline of the other line."""
        # if there is no horizontal overlap, this line is not directly below the other
        if direct_only and not get_horizontal_overlap(self, other):
            # print("pagexml.pdm - NO HORIZONTAL OVERLAP")
            return False
        if not direct_only and not get_horizontal_overlap(self, other):
            vertical_overlap = get_vertical_overlap(self, other)
            min_height = min(self.text_height, other.text_height)
            if vertical_overlap / min_height > 0.5:
                # print("pagexml.pdm - NO HORIZONTAL OVERLAP, LINES VERTICALLY OVERLAP")
                return False
        # if the bottom of this line is above the top of the other line, this line is above the other
        if self.baseline.bottom < other.baseline.top:
            # print("pagexml.pdm - BOTTOM OF SELF IS ABOVE TOP OF OTHER")
            return False
        # if most of this line's baseline points are not below most the other's baseline points
        # this line is not below the other
        if baseline_is_below(self.baseline, other.baseline):
            # print("pagexml.pdm - BASELINE OF SELF IS BELOW BASELINE OF OTHER")
            return True
        # print("pagexml.pdm - SELF IS ADJACENT TO OTHER")
        return False

    def is_next_to(self, other: PageXMLTextLine, debug: int = 0) -> bool:
        """Test if this line is vertically aligned with the other line."""
        vertical_overlap = get_vertical_overlap(self, other)
        min_height = min(self.text_height, other.text_height)
        if vertical_overlap == 0:
            if debug > 0:
                print("NO VERTICAL OVERLAP")
            return False
        if get_horizontal_overlap(self, other) > 40:
            if debug > 0:
                print("TOO MUCH HORIZONTAL OVERLAP", get_horizontal_overlap(self, other))
            return False
        """
        if self.baseline.top > other.baseline.bottom + 10:
            if debug > 0:
                print("VERTICAL BASELINE GAP TOO BIG")
            return False
        elif self.baseline.bottom < other.baseline.top - 10:
            if debug > 0:
                print("VERTICAL BASELINE GAP TOO BIG")
            return False
        """
        if vertical_overlap / min_height > 0.5:
            # print("pagexml.pdm - NO HORIZONTAL OVERLAP, LINES VERTICALLY OVERLAP")
            return True
        else:
            return False

    def add_to_pagexml(self, parent: etree.Element = None):
        line_xml = add_pagexml_sub_element(parent, 'TextLine', sub_id=self.id, custom=self.custom,
                                           coords=self.coords, baseline=self.baseline,
                                           text=self.text, conf=self.conf)
        for word in self.words:
            word.add_to_pagexml(line_xml)

    def _to_pagexml(self, page_xml: etree.Element):
        # make dummy region
        tr = PageXMLTextRegion(coords=self.coords, lines=[self])
        # add region to page
        tr_xml = add_pagexml_sub_element(page_xml, 'TextRegion', sub_id=tr.id, custom=tr.custom,
                                         coords=tr.coords)
        # finally, add line itself to text region
        self.add_to_pagexml(tr_xml)


def get_num_columns(rows: List[PageXMLTableRow]):
    return max(len(row) for row in rows) if len(rows) > 0 else 0


class PageXMLRegion(PageXMLDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 orientation: float = None, reading_order: Dict[int, str] = None,
                 text_regions: List[PageXMLTextRegion] = None,
                 table_regions: List[PageXMLTableRegion] = None,
                 empty_regions: List[PageXMLEmptyRegion] = None,
                 reading_order_attributes: Dict[str, any] = None):
        super().__init__(doc_id=doc_id, doc_type=doc_type, attrs=attrs, metadata=metadata,
                         coords=coords, reading_order=reading_order,
                         reading_order_attributes=reading_order_attributes, orientation=orientation)
        self.main_type = doc_type if doc_type is not None else 'region'
        self.text_regions: List[PageXMLTextRegion] = text_regions if text_regions is not None else []
        self.table_regions: List[PageXMLTableRegion] = table_regions if table_regions is not None else []
        self.empty_regions: List[PageXMLEmptyRegion] = empty_regions if empty_regions is not None else []
        for table in self.table_regions:
            for row in table.rows:
                if len(row) < table.num_columns:
                    row.pad_columns(table.num_columns)
        regions = self.text_regions + self.table_regions + self.empty_regions
        for region in regions:
            region.set_parent(self)

    def __repr__(self):
        content_string = f"\n\tid={self.id}, \n\ttype={self.type}, "
        return f"{self.__class__.__name__}({content_string}\n)"

    def __lt__(self, other: PageXMLTextRegion):
        """For sorting text regions. Assumptions: reading from left to right (western bias),
        top to bottom. If two regions are horizontally overlapping, sort from
        top to bottom, even if the upper region is more horizontally indented."""
        if other == self:
            return False
        if is_horizontally_overlapping(self, other):
            return self.coords.top < other.coords.top
        else:
            return self.coords.left < other.coords.left

    @property
    def regions(self):
        regions = []
        regions.extend(self.text_regions)
        regions.extend(self.table_regions)
        regions.extend(self.empty_regions)
        return sorted(regions)

    def add_child(self, child: PageXMLDoc):
        child.set_parent(self)
        if isinstance(child, PageXMLTextRegion):
            self.text_regions.append(child)
        elif isinstance(child, PageXMLTableRegion):
            self.table_regions.append(child)
        elif isinstance(child, PageXMLEmptyRegion):
            self.empty_regions.append(child)
        else:
            raise TypeError(f'unknown child type: {child.__class__.__name__}')
        self.set_as_parent([child])
        self.coords = parse_derived_coords(self.regions)

    @property
    def children(self):
        return self.regions

    @property
    def num_text_regions(self):
        return len(self.text_regions)

    @property
    def num_table_regions(self):
        return len(self.get_table_regions())

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        if self.text_regions:
            doc_json['text_regions'] = [text_region.json for text_region in self.text_regions]
        if self.table_regions:
            doc_json['table_regions'] = [table_region.json for table_region in self.table_regions]
        if self.empty_regions:
            doc_json['empty_regions'] = [empty_region.json for empty_region in self.empty_regions]
        if self.orientation:
            doc_json['orientation'] = self.orientation
        if self.reading_order_attributes:
            doc_json['reading_order_attributes'] = self.reading_order_attributes
        if self.reading_order:
            doc_json['reading_order'] = self.reading_order
        doc_json['stats'] = self.stats
        return doc_json

    def get_text_regions_in_reading_order(self):
        if not self.reading_order:
            return self.text_regions
        tr_ids = list({region_id: None for _index, region_id in sorted(self.reading_order.items(), key=lambda x: x[0])})
        tr_map = {}
        for text_region in self.text_regions:
            # if text_region.id not in tr_ids:
            #     print("reading order:", self.reading_order)
            #     raise KeyError(f"text_region with id {text_region.id} is not listed in reading_order")
            tr_map[text_region.id] = text_region
        return [tr_map[tr_id] for tr_id in tr_ids if tr_id in tr_map]

    def set_text_regions_in_reading_order(self):
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

    def get_all_text_regions(self):
        text_regions: Set[PageXMLTextRegion] = set()
        for text_region in self.text_regions:
            text_regions.add(text_region)
            if text_region.text_regions:
                text_regions += text_region.get_all_text_regions()
        return text_regions

    def get_inner_text_regions(self) -> List[PageXMLTextRegion]:
        text_regions: List[PageXMLTextRegion] = []
        for text_region in self.text_regions:
            if text_region.text_regions:
                text_regions += text_region.get_inner_text_regions()
            elif text_region.lines:
                text_regions.append(text_region)
        if not self.text_regions and isinstance(self, PageXMLTextRegion):
            text_regions.append(self)
        return text_regions

    def get_table_regions(self):
        table_regions = [tr for tr in self.table_regions]
        for tr in self.text_regions:
            table_regions.extend(tr.get_table_regions())
        return table_regions

    def get_textual_regions(self):
        return [region for region in self.regions if is_textual_region(region)]

    def get_regions(self, ignore_reading_order: bool = False) -> List[PageXMLDoc]:
        all_regions = self.regions
        if self.reading_order and not ignore_reading_order:
            ordered_regions = [tr for tr in all_regions if tr.id in self.reading_order]
            unordered_regions = [tr for tr in all_regions if tr.id not in self.reading_order]
        else:
            ordered_regions = []
            unordered_regions = all_regions
        all_regions = sorted(ordered_regions, key=lambda t: self.reading_order_number[t.id])
        all_regions.extend(unordered_regions)
        return all_regions

    def get_lines(self, ignore_reading_order: bool = False):
        lines = []
        for tr in self.get_textual_regions():
            lines.extend(tr.get_lines())
        return lines

    @property
    def stats(self):
        stats = {'lines': 0, 'words': 0, 'chars': 0}
        if self.text_regions:
            stats['text_regions'] = len(self.text_regions)
        if self.table_regions:
            stats['table_regions'] = len(self.table_regions)
        if self.empty_regions:
            stats['empty_regions'] = len(self.empty_regions)
        for region in self.regions:
            region_stats = region.stats
            for field in region_stats:
                stats[field] = stats.get(field, 0) + region_stats[field]
        return stats


class PageXMLEmptyRegion(PageXMLRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None, orientation: float = None):
        super(PageXMLEmptyRegion, self).__init__(doc_id=doc_id, doc_type='empty_region', metadata=metadata,
                                                 coords=coords, attrs=attrs, orientation=orientation)
        self.main_type = 'empty_region'
        if doc_type:
            self.add_type(doc_type)

    @property
    def stats(self):
        return super(PageXMLEmptyRegion, self).stats


class PageXMLTableRegion(PageXMLRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 rows: List[PageXMLTableRow] = None, orientation: float = None):
        super().__init__(doc_id=doc_id, doc_type="table_region", attrs=attrs, metadata=metadata,
                         coords=coords, reading_order=None,
                         reading_order_attributes=None, orientation=orientation)
        self.main_type = 'table_region'
        self.rows: List[PageXMLTableRow] = rows if rows is not None else []
        self.orientation: Union[None, float] = orientation
        self.reading_order_number = {}
        if doc_type:
            self.add_type(doc_type)
        self.empty_regions = []

    def __getitem__(self, idx):
        return self.rows[idx]

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        stats = json.dumps(self.stats)
        content_string = f"\n\tid={self.id}, \n\ttype={self.type}, \n\tstats={stats}"
        return f"{self.__class__.__name__}({content_string}\n)"

    @property
    def children(self):
        return self.rows

    @property
    def num_columns(self):
        return get_num_columns(self.rows)

    @property
    def values(self):
        return [row.values for row in self.rows]

    @property
    def shape(self):
        return len(self.rows), self.num_columns

    def cell(self, row: int, cell: int):
        return self.rows[row].cells[cell]

    def get_lines(self):
        return [line for row in self.rows for line in row.get_lines()]

    def get_words(self):
        return [word for row in self.rows for word in row.get_words()]

    @property
    def num_rows(self):
        return len(self.rows)

    @property
    def num_cells(self):
        return sum(row.num_cells for row in self.rows)

    @property
    def num_lines(self):
        return len(self.get_lines())

    @property
    def num_words(self):
        return len(self.get_words())

    @property
    def num_chars(self):
        return sum(line.length for line in self.get_lines())

    def to_csv(self, csv_file: str, separator: str = '\t'):
        with open(csv_file, 'w') as fh:
            table_writer = csv.writer(fh, delimiter=separator, quotechar='|')
            for row in self.rows:
                table_writer.writerows(row.values)

    @property
    def stats(self):
        return {
            'rows': self.num_rows,
            'cells': self.num_cells,
            'lines': self.num_lines,
            'words': self.num_words,
            'chars': self.num_chars
        }

    @property
    def json(self):
        doc_json = super(PageXMLTableRegion, self).json
        doc_json['num_rows'] = self.shape[0]
        doc_json['num_cols'] = self.shape[1]
        doc_json['rows'] = [row.json for row in self.rows]
        if self.orientation:
            doc_json['orientation'] = self.orientation
        doc_json['stats'] = self.stats
        return doc_json

    def add_to_pagexml(self, parent: etree.Element = None):
        attributes = {}
        if self.orientation:
            attributes['orientation'] = self.orientation
        add_pagexml_sub_element(parent, 'TableRegion', sub_id=self.id, custom=self.custom,
                                coords=self.coords, attributes=attributes)

    def _to_pagexml(self, page_xml: etree.Element):
        self.add_to_pagexml(page_xml)


def check_cell_row_consistency(cells: List[PageXMLTableCell]):
    row_idxs = [cell.row for cell in cells]
    if len(set(row_idxs)) > 1:
        message = f"Cannot make a row of cells with different row indexes:"
        for cell in cells:
            message += f"\n\tcell id '{cell.id}'\tcol: {cell.col}"
        raise ValueError(message)


class PageXMLTableRow(PageXMLRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 cells: List[PageXMLTableCell] = None, orientation: float = None):
        super().__init__(doc_id=doc_id, doc_type="table_row", attrs=attrs, metadata=metadata,
                         coords=coords, reading_order=None,
                         reading_order_attributes=None, orientation=orientation)
        self.main_type = 'table_row'
        self.cells: List[Union[PageXMLTableCell, None]] = cells if cells is not None else []
        self.column_cells = []
        if cells is not None:
            check_cell_row_consistency(cells)
            for cell in self.cells:
                if cell.col > len(self.column_cells):
                    self.pad_columns(cell.col)
                self.column_cells.append(cell)
            # also set the row index
            self.row_idx = cells[0].row
        self.orientation: Union[None, float] = orientation
        self.reading_order_number = {}
        if doc_type:
            self.add_type(doc_type)
        self.empty_regions = []

    def __getitem__(self, col_idx):
        if self.column_cells[col_idx] is None:
            return PageXMLTableCell(row=self.row_idx, col=col_idx, doc_type='empty_cell')
        return self.column_cells[col_idx]

    def __len__(self):
        return len(self.cells)

    def __repr__(self):
        stats = json.dumps(self.stats)
        content_string = f"\n\tid={self.id}, \n\ttype={self.main_type}, \n\tstats={stats}"
        return f"{self.__class__.__name__}({content_string}\n)"

    def pad_columns(self, col_idx: int):
        padding = [None] * (col_idx - len(self.column_cells))
        self.column_cells.extend(padding)

    @property
    def children(self):
        return self.cells

    @property
    def num_columns(self):
        if isinstance(self.parent, PageXMLTableRegion):
            return self.parent.num_columns
        else:
            return max(cell.col for cell in self.cells)

    @property
    def values(self):
        values = ['' if cell is None else cell.value for cell in self.column_cells]
        # values: List[Union[None, str]] = [cell if cell is None else cell.value for cell in self.column_cells]
        # print(f"num_columns: {num_columns}\tlen(values): {len(values)}")
        # print(f"values step 1: {values}")
        if len(values) < self.num_columns:
            values.extend([''] * (self.num_columns - len(values)))
        # print(f"values step 2: {values}")
        return values

    def get_lines(self):
        return [line for cell in self.cells for line in cell.get_lines()]

    def get_words(self):
        return [word for cell in self.cells for word in cell.get_words()]

    @property
    def num_cells(self):
        return len(self.cells)

    @property
    def num_lines(self):
        return len(self.get_lines())

    @property
    def num_words(self):
        return len(self.get_words())

    @property
    def num_chars(self):
        return sum(line.length for line in self.get_lines())

    @property
    def stats(self):
        return {
            'cells': self.num_cells,
            'lines': self.num_lines,
            'words': self.num_words,
            'chars': self.num_chars
        }

    @property
    def json(self):
        doc_json = super(PageXMLTableRow, self).json
        doc_json['num_cols'] = len(self.column_cells)
        doc_json['cells'] = [cell.json for cell in self.cells]
        if self.orientation:
            doc_json['orientation'] = self.orientation
        doc_json['stats'] = self.stats
        return doc_json

    def add_to_pagexml(self, parent: etree.Element = None):
        for cell in self.cells:
            cell.add_to_pagexml(parent)

    def _to_pagexml(self, page_xml: etree.Element):
        tr_xml = add_pagexml_sub_element(page_xml, 'TableRegion', custom=self.custom,
                                         coords=self.coords)
        self.add_to_pagexml(tr_xml)


class PageXMLTableCell(PageXMLRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 row: int = None, col: int = None, row_span: int = None, cell_span: int = None,
                 header: bool = None, cornerpoints: Union[List[int], str] = None,
                 lines: List[PageXMLTextLine] = None, orientation: float = None):
        super().__init__(doc_id=doc_id, doc_type="table_cell", attrs=attrs, metadata=metadata,
                         coords=coords, reading_order=None,
                         reading_order_attributes=None, orientation=orientation)
        self.main_type = 'table_cell'
        self.lines: List[PageXMLTextLine] = lines if lines is not None else []
        # Initial value is concatenated text of lines, but can be overwritten by user
        # with e.g. interpreted/evaluated text
        self.value = " ".join([line.text for line in self.lines])
        self.row = row
        self.col = col
        self.row_span = row_span
        self.cell_span = cell_span
        self.header = header
        self.cornerpoints = cornerpoints
        self.orientation: Union[None, float] = orientation
        self.reading_order_number = {}
        if doc_type:
            self.add_type(doc_type)
        self.empty_regions = []

    def __repr__(self):
        stats = json.dumps(self.stats)
        content_string = (f"\n\tid={self.id}, \n\ttype={self.main_type}, "
                          f"\n\trow={self.row}, col={self.col}\n\tstats={stats}")
        return f"{self.__class__.__name__}({content_string}\n)"

    def get_lines(self):
        return self.lines

    def get_words(self):
        words = []
        if self.lines:
            for line in self.lines:
                if line.words:
                    words += line.words
                elif line.text:
                    words += line.text.split(' ')
        return words

    @property
    def num_lines(self):
        return len(self.lines)

    @property
    def num_words(self):
        return len(self.get_words())

    @property
    def num_chars(self):
        return sum(line.length for line in self.get_lines())

    @property
    def stats(self):
        return {
            'lines': self.num_lines,
            'words': self.num_words,
            'chars': self.num_chars
        }

    @property
    def json(self):
        doc_json = super().json
        doc_json['col'] = self.col
        doc_json['cell_span'] = self.cell_span
        doc_json['row_span'] = self.row_span
        doc_json['lines'] = [line.json for line in self.lines]
        if self.cornerpoints:
            doc_json['cornerpoints'] = self.cornerpoints
        if self.orientation:
            doc_json['orientation'] = self.orientation
        doc_json['stats'] = self.stats
        return doc_json

    def add_to_pagexml(self, parent: etree.Element = None):
        attributes = get_doc_attributes(self)
        cell_xml = add_pagexml_sub_element(parent, 'TableCell', sub_id=self.id, coords=self.coords,
                                           custom=self.custom, attributes=attributes)
        for line in self.lines:
            line.add_to_pagexml(cell_xml)
        if self.cornerpoints:
            if isinstance(self.cornerpoints, str):
                text = self.cornerpoints
            else:
                text = ' '.join(str(point) for point in self.cornerpoints)
        else:
            text = ''
        add_pagexml_sub_element(cell_xml, 'CornerPts', text=text)

    def _to_pagexml(self, page_xml: etree.Element):
        tr_xml = add_pagexml_sub_element(page_xml, 'TableRegion', custom=self.custom,
                                         coords=self.coords)
        self.add_to_pagexml(tr_xml)


class PageXMLTextRegion(PageXMLRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 text_regions: List[PageXMLTextRegion] = None,
                 lines: List[PageXMLTextLine] = None, text: str = None,
                 orientation: float = None, reading_order: Dict[int, str] = None,
                 reading_order_attributes: Dict[str, any] = None):
        super().__init__(doc_id=doc_id, doc_type="text_region", attrs=attrs, metadata=metadata,
                         coords=coords, reading_order=reading_order, orientation=orientation,
                         text_regions=text_regions,
                         reading_order_attributes=reading_order_attributes)
        self.main_type = 'text_region'
        self.lines: List[PageXMLTextLine] = lines if lines is not None else []
        self.reading_order_number = {}
        self.text = text
        if self.lines is not None:
            self.set_as_parent(self.lines)
        if self.lines is not None:
            self.set_as_parent(self.lines)
        if self.text_regions is not None:
            self.set_as_parent(self.text_regions)
        if self.reading_order:
            self.set_text_regions_in_reading_order()
        if doc_type:
            self.add_type(doc_type)
        self.empty_regions = []

    def __repr__(self):
        stats = json.dumps(self.stats)
        content_string = f"\n\tid={self.id}, \n\ttype={self.type}, \n\tstats={stats}"
        return f"{self.__class__.__name__}({content_string}\n)"

    def add_child(self, child: PageXMLDoc):
        child.set_parent(self)
        if isinstance(child, PageXMLTextLine):
            self.lines.append(child)
        elif isinstance(child, PageXMLTextRegion):
            self.text_regions.append(child)
            self.set_as_parent([child])
        else:
            raise TypeError(f'unknown child type: {child.__class__.__name__}')
        self.coords = parse_derived_coords(self.text_regions + self.lines)

    @property
    def children(self):
        child_elements: List[PageXMLDoc] = [tr for tr in self.text_regions]
        child_elements.extend([line for line in self.lines])
        return child_elements

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        if self.text:
            doc_json['text'] = self.text
        if self.lines:
            doc_json['lines'] = [line.json for line in self.lines]
        return doc_json

    def get_lines(self, ignore_reading_order: bool = False) -> List[PageXMLTextLine]:
        lines: List[PageXMLTextLine] = []
        all_regions = self.get_regions(ignore_reading_order=ignore_reading_order)
        for tr in all_regions:
            if isinstance(tr, PageXMLTextRegion):
                lines.extend(tr.get_lines())
        if self.lines:
            lines += self.lines
        return lines

    def get_words(self, ignore_reading_order: bool = False) -> Union[List[str], List[PageXMLWord]]:
        words = []
        if self.text is not None:
            return self.text.split(' ')
        for line in self.get_lines(ignore_reading_order=ignore_reading_order):
            if line.words:
                words += line.words
            elif line.text:
                words += line.text.split(' ')
        return words

    @property
    def num_lines(self):
        return len(self.get_lines())

    @property
    def num_words(self):
        return len(self.get_words())

    @property
    def num_chars(self):
        return sum(line.length for line in self.get_lines())

    @property
    def stats(self):
        stats = {
            'text_regions': self.num_text_regions,
            'lines': self.num_lines,
            'words': self.num_words,
            'chars': self.num_chars,
        }
        if self.table_regions:
            stats['table_regions'] = len(self.table_regions)
        return stats

    def add_to_pagexml(self, parent: etree.Element = None):
        attributes = get_doc_attributes(self)
        tr_xml = add_pagexml_sub_element(parent, 'TextRegion', sub_id=self.id, custom=self.custom,
                                         coords=self.coords, attributes=attributes)
        for line in self.lines:
            line.add_to_pagexml(tr_xml)
        for sub_tr in self.text_regions:
            sub_tr.add_to_pagexml(tr_xml)
        for sub_tr in self.table_regions:
            sub_tr.add_to_pagexml(tr_xml)

    def _to_pagexml(self, page_xml: etree.Element):
        self.add_to_pagexml(page_xml)


class PageXMLColumn(PageXMLRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 text_regions: List[PageXMLTextRegion] = None,
                 table_regions: List[PageXMLTableRegion] = None,
                 empty_regions: List[PageXMLEmptyRegion] = None,
                 reading_order: Dict[int, str] = None,
                 reading_order_attributes: Dict[str, any] = None,
                 orientation: float = None):
        super().__init__(doc_id=doc_id, doc_type="column", attrs=attrs,
                         metadata=metadata, coords=coords,
                         text_regions=text_regions, table_regions=table_regions,
                         empty_regions=empty_regions,
                         orientation=orientation, reading_order=reading_order,
                         reading_order_attributes=reading_order_attributes)
        self.main_type = 'column'
        if doc_type:
            self.add_type(doc_type)

    @property
    def json(self) -> Dict[str, any]:
        doc_json = super().json
        doc_json['stats'] = self.stats
        return doc_json

    @property
    def children(self):
        return self.regions

    @property
    def stats(self):
        stats = super().stats
        return stats

    def add_to_pagexml(self, parent: etree.Element = None):
        attributes = get_doc_attributes(self)
        attributes['type'] = 'column'
        column_xml = add_pagexml_sub_element(parent, 'TextRegion', sub_id=self.id, custom=self.custom,
                                             coords=self.coords, attributes=attributes)
        for tr in self.text_regions:
            tr.add_to_pagexml(column_xml)
        for sub_tr in self.table_regions:
            sub_tr.add_to_pagexml(column_xml)

    def _to_pagexml(self, page_xml: etree.Element):
        self.add_to_pagexml(page_xml)


def is_textual_region(region: PageXMLDoc):
    return isinstance(region, PageXMLTextRegion) or isinstance(region, PageXMLTableRegion)


class PageXMLPage(PageXMLRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 columns: List[PageXMLColumn] = None,
                 text_regions: List[PageXMLTextRegion] = None,
                 table_regions: List[PageXMLTableRegion] = None,
                 empty_regions: List[PageXMLEmptyRegion] = None,
                 extra: List[PageXMLRegion] = None,
                 orientation: float = None,
                 reading_order: Dict[int, str] = None,
                 reading_order_attributes: Dict[str, any] = None):
        super().__init__(doc_id=doc_id, doc_type="page", attrs=attrs,
                         metadata=metadata, coords=coords,
                         text_regions=text_regions, table_regions=table_regions,
                         empty_regions=empty_regions,
                         orientation=orientation, reading_order=reading_order,
                         reading_order_attributes=reading_order_attributes)
        self.main_type = 'page'
        self.columns: List[PageXMLColumn] = columns if columns else []
        self.extra: List[PageXMLRegion] = extra if extra else []
        self.set_as_parent(self.columns)
        self.set_as_parent(self.extra)
        if doc_type:
            self.add_type(doc_type)
        if coords is None:
            self.coords = parse_derived_coords(self.columns + self.regions + self.extra)

    def get_lines(self, ignore_reading_order: bool = False):
        lines = []
        all_regions = self.get_regions(ignore_reading_order=ignore_reading_order)
        for tr in all_regions:
            if hasattr(tr, 'get_lines'):
                lines += tr.get_lines()
        return lines

    def add_child(self, child: PageXMLDoc, as_extra: bool = False):
        child.set_parent(self)
        if as_extra and (isinstance(child, PageXMLColumn) or isinstance(child, PageXMLTextRegion)):
            self.extra.append(child)
        elif isinstance(child, PageXMLColumn) or child.__class__.__name__ == 'PageXMLColumn':
            self.columns.append(child)
        elif isinstance(child, PageXMLTextRegion):
            self.text_regions.append(child)
        else:
            raise TypeError(f'unknown child type: {child.__class__.__name__}')
        derived_coords = parse_derived_coords([self] + self.extra + self.columns + self.text_regions)
        self.coords = derived_coords

    @property
    def regions(self):
        regions = [region for col in self.columns for region in col.get_regions()]
        regions.extend(region for region in self.extra)
        return sorted(regions)

    def get_textual_regions(self):
        text_regions = [tr for col in self.columns for tr in col.get_textual_regions()]
        text_regions.extend([r for r in self.extra if is_textual_region(r)])
        return text_regions

    def get_text_regions(self):
        text_regions = [tr for col in self.columns for tr in col.text_regions]
        text_regions.extend([r for r in self.extra if isinstance(r, PageXMLTextRegion)])
        return text_regions

    def get_text_regions_in_reading_order(self, include_extra: bool = True):
        text_regions = []
        if len(self.text_regions) > 0:
            text_regions.extend(self.text_regions)
        if hasattr(self, 'columns'):
            for col in sorted(self.columns):
                text_regions.extend(col.get_text_regions_in_reading_order())
        if include_extra and hasattr(self, 'extra'):
            text_regions.extend(sorted(self.extra))
        return text_regions

    def get_inner_text_regions(self) -> List[PageXMLTextRegion]:
        text_regions = self.get_all_text_regions()
        inner_trs = []
        for tr in text_regions:
            inner_trs.extend(tr.get_inner_text_regions())
        return inner_trs

    def get_table_regions(self):
        table_regions = []
        if self.table_regions:
            table_regions.extend(table_regions)
        for column in self.columns:
            table_regions.extend(column.get_table_regions())
        for tr in self.text_regions:
            table_regions.extend(tr.get_table_regions())
        for tr in self.extra:
            table_regions.extend(tr.get_table_regions())
        return table_regions

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
    def children(self):
        child_elements: List[PageXMLDoc] = []
        child_elements.extend([col for col in self.columns])
        child_elements.extend([tr for tr in self.text_regions])
        child_elements.extend([tr for tr in self.table_regions])
        child_elements.extend([tr for tr in self.empty_regions])
        return child_elements

    @property
    def stats(self):
        """Pages diverge from other types since they have columns and extra
        text regions, or plain text regions, so have their own way of calculating
        stats."""
        lines = self.get_lines()
        stats = {
            "lines": len(lines),
            "words": sum([len(line.get_words()) for line in lines]),
            "chars": sum(line.length for line in lines)
        }
        if self.columns:
            stats['columns'] = len(self.columns)
            stats['text_regions'] = sum(col.num_text_regions for col in self.columns)
            stats['table_regions'] = sum(col.num_table_regions for col in self.columns)
        if self.extra:
            stats['extra'] = len(self.extra)
        if self.text_regions:
            stats['text_regions'] += self.num_text_regions
        if self.table_regions:
            stats['table_regions'] += self.num_table_regions
        return stats

    def add_to_pagexml(self, parent: etree.Element = None):
        attributes = get_doc_attributes(self)
        attributes['type'] = 'page'
        page_xml = add_pagexml_sub_element(parent, 'TextRegion', sub_id=self.id, custom=self.custom,
                                           coords=self.coords, attributes=attributes)
        for column in self.text_regions:
            column.add_to_pagexml(page_xml)
        for tr in self.text_regions:
            tr.add_to_pagexml(page_xml)
        for sub_tr in self.table_regions:
            sub_tr.add_to_pagexml(page_xml)
        for tr in self.extra:
            tr.add_to_pagexml(page_xml)

    def _to_pagexml(self, page_xml: etree.Element):
        self.add_to_pagexml(page_xml)


class PageXMLScan(PageXMLRegion):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 attrs: Dict[str, any] = None,
                 pages: List[PageXMLPage] = None, columns: List[PageXMLColumn] = None,
                 text_regions: List[PageXMLTextRegion] = None,
                 table_regions: List[PageXMLTableRegion] = None,
                 orientation: float = None,
                 reading_order: Dict[int, str] = None,
                 reading_order_attributes: Dict[str, any] = None):
        super().__init__(doc_id=doc_id, doc_type="scan", attrs=attrs,
                         metadata=metadata, coords=coords,
                         text_regions=text_regions, table_regions=table_regions,
                         orientation=orientation, reading_order=reading_order,
                         reading_order_attributes=reading_order_attributes)
        self.main_type = 'scan'
        self.pages: List[PageXMLPage] = pages if pages else []
        self.columns: List[PageXMLColumn] = columns if columns else []
        self.set_as_parent(self.pages)
        self.set_as_parent(self.columns)
        if doc_type:
            self.add_type(doc_type)
        self.set_scan_id_as_metadata()

    def add_child(self, child: PageXMLDoc):
        child.set_parent(self)
        if isinstance(child, PageXMLPage):
            self.pages.append(child)
        elif isinstance(child, PageXMLColumn):
            self.columns.append(child)
        elif isinstance(child, PageXMLTextRegion):
            self.text_regions.append(child)

    def set_scan_id_as_metadata(self):
        self.metadata['scan_id'] = self.id
        for tr in self.get_all_text_regions():
            tr.metadata['scan_id'] = self.id
        for region in self.regions:
            for line in region.get_lines():
                line.metadata['scan_id'] = self.id
            for word in region.get_words():
                if isinstance(word, PageXMLWord):
                    word.metadata['scan_id'] = self.id

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
    def children(self):
        child_elements: List[PageXMLDoc] = []
        child_elements.extend([page for page in self.pages])
        child_elements.extend([column for column in self.columns])
        child_elements.extend([tr for tr in self.text_regions])
        child_elements.extend([tr for tr in self.table_regions])
        child_elements.extend([tr for tr in self.empty_regions])
        return child_elements

    @property
    def stats(self):
        stats = super().stats
        stats['columns'] = len([column for page in self.pages for column in page.columns])
        stats['extra'] = len([text_region for page in self.pages for text_region in page.extra])
        stats['pages'] = len(self.pages)
        return stats

    def add_to_pagexml(self, scan_xml: etree.Element = None):
        if self.orientation:
            scan_xml.attrib['orientation'] = self.orientation
        if self.reading_order:
            xml.add_reading_order(scan_xml, self.reading_order,
                                  reading_order_attributes=self.reading_order_attributes)
        for tr in self.text_regions:
            tr.add_to_pagexml(scan_xml)
        for sub_tr in self.table_regions:
            sub_tr.add_to_pagexml(scan_xml)

    def _to_pagexml(self, page_xml: etree.Element):
        self.add_to_pagexml(page_xml)


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


def has_baseline(doc: PhysicalStructureDoc):
    if hasattr(doc, 'baseline'):
        return doc.baseline is not None
    else:
        return False


def get_horizontal_overlap(doc1: PageXMLDoc, doc2: PageXMLDoc, debug: int = 0) -> int:
    if debug > 4:
        dc1 = doc1.coords
        dc2 = doc2.coords
        print(f"doc1 - left: {dc1.left} right: {dc1.right} width: {dc1.width}")
        print(f"doc2 - left: {dc2.left} right: {dc2.right} width: {dc2.width}")
    if doc1.coords.width == 0:
        return 1 if doc2.coords.left <= doc1.coords.left <= doc2.coords.right else 0
    if doc2.coords.width == 0:
        return 1 if doc1.coords.left <= doc2.coords.left <= doc1.coords.right else 0
    if has_baseline(doc1) and has_baseline(doc2):
        max_left = max([doc1.baseline.left, doc2.baseline.left])
        min_right = min([doc1.baseline.right, doc2.baseline.right])
    else:
        max_left = max([doc1.coords.left, doc2.coords.left])
        min_right = min([doc1.coords.right, doc2.coords.right])
    return max(0, min_right - max_left)
    # return min_right - max_left + 1 if min_right >= max_left else 0


def get_line_top_bottom(line: PageXMLTextLine):
    if has_baseline(line):
        line_bottom = line.baseline.bottom
        line_height = line.xheight if line.xheight else int(line.coords.height / 2)
        line_top = line.baseline.top - line_height
    else:
        # assume the box cut around a line has 25% of its height below
        # the baseline
        line_bottom = line.coords.bottom - line.coords.height * 0.25
        line_top = line.coords.top + line.coords.height * 0.25
    return line_top, line_bottom


def get_vertical_overlap(doc1: PageXMLDoc, doc2: PageXMLDoc) -> int:
    if doc1.coords.height == 0:
        return 1 if doc2.coords.top <= doc1.coords.top <= doc2.coords.bottom else 0
    if doc2.coords.height == 0:
        return 1 if doc1.coords.top <= doc2.coords.top <= doc1.coords.bottom else 0
    if isinstance(doc1, PageXMLTextLine) and isinstance(doc2, PageXMLTextLine):
        doc1_top, doc1_bottom = get_line_top_bottom(doc1)
        doc2_top, doc2_bottom = get_line_top_bottom(doc2)
        return min(doc1_bottom, doc2_bottom) - max(doc1_top, doc2_top)
    else:
        overlap_top = max([doc1.coords.top, doc2.coords.top])
        overlap_bottom = min([doc1.coords.bottom, doc2.coords.bottom])
    return overlap_bottom - overlap_top + 1 if overlap_bottom >= overlap_top else 0


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


def get_horizontal_diff_ratio(doc1: PageXMLDoc, doc2: PageXMLDoc) -> float:
    horizontal_diff = get_horizontal_diff(doc1, doc2)
    max_right = max(doc1.coords.right, doc2.coords.right)
    min_left = min(doc1.coords.left, doc2.coords.left)
    return horizontal_diff / (max_right - min_left)


def get_vertical_diff_ratio(doc1: PageXMLDoc, doc2: PageXMLDoc) -> float:
    vertical_diff = get_vertical_diff(doc1, doc2)
    max_bottom = max(doc1.coords.bottom, doc2.coords.bottom)
    min_top = min(doc1.coords.top, doc2.coords.top)
    return vertical_diff / (max_bottom - min_top)


def get_vertical_diff(doc1: PageXMLDoc, doc2: PageXMLDoc) -> int:
    if isinstance(doc1, PageXMLTextLine) and isinstance(doc2, PageXMLTextLine) and \
            doc1.baseline is not None and doc2.baseline is not None:
        return abs(doc1.baseline.top - doc2.baseline.top)
    else:
        return abs(doc1.coords.top - doc2.coords.top)


def get_horizontal_diff(doc1: PageXMLDoc, doc2: PageXMLDoc) -> int:
    if isinstance(doc1, PageXMLTextLine) and isinstance(doc2, PageXMLTextLine) and \
            doc1.baseline is not None and doc2.baseline is not None:
        return abs(doc1.baseline.left - doc2.baseline.left)
    else:
        return abs(doc1.coords.left - doc2.coords.left)
