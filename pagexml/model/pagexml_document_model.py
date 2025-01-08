from __future__ import annotations

import json
from typing import Union, List, Dict, Set

from lxml import etree
from pagexml.model.basic_document_model import PhysicalStructureDoc
from pagexml.model.coords import Coords, Baseline, baseline_is_below, parse_derived_coords
from pagexml.model.xml import make_empty_pagexml, add_pagexml_sub_element
import pagexml.model.xml as xml


class PageXMLDoc(PhysicalStructureDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None,
                 coords: Coords = None, reading_order: Dict[int, str] = None):
        if doc_type is None:
            doc_type = 'pagexml_doc'
        super().__init__(doc_id=doc_id, doc_type=doc_type, metadata=metadata, reading_order=reading_order)
        self.coords: Union[None, Coords] = coords
        self.pagexml_type = None
        self.add_type('pagexml_doc')

    @property
    def stats(self):
        return {}

    @property
    def custom(self):
        if 'custom_attributes' in self.metadata:
            return self.metadata['custom_attributes']
        else:
            return {}

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
                 baseline: Baseline = None,
                 conf: float = None, text: str = None):
        super().__init__(doc_id=doc_id, doc_type="word", metadata=metadata, coords=coords)
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
                 baseline: Baseline = None, xheight: int = None,
                 conf: float = None, text: str = None, words: List[PageXMLWord] = None,
                 reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type="line", metadata=metadata,
                         coords=coords, reading_order=reading_order)
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
        content_string = f"id={self.id}, type={self.type}, text=\"{self.text}\" conf={self.conf}"
        return f"{self.__class__.__name__}({content_string})"

    def __lt__(self, other: PageXMLTextLine):
        """For sorting text lines. Assumptions: reading from left to right,
        top to bottom. If two lines are horizontally overlapping, sort from
        top to bottom, even if the upper lines is more horizontally indented."""
        if other == self:
            return False
        return sort_lines(self, other, as_column=True)

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
    def stats(self):
        return {
            'words': self.num_words
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


class PageXMLTextRegion(PageXMLDoc):

    def __init__(self, doc_id: str = None, doc_type: Union[str, List[str]] = None,
                 metadata: Dict[str, any] = None, coords: Coords = None,
                 text_regions: List[PageXMLTextRegion] = None,
                 lines: List[PageXMLTextLine] = None, text: str = None,
                 orientation: float = None, reading_order: Dict[int, str] = None):
        super().__init__(doc_id=doc_id, doc_type="text_region", metadata=metadata,
                         coords=coords, reading_order=reading_order)
        self.main_type = 'text_region'
        self.text_regions: List[PageXMLTextRegion] = text_regions if text_regions is not None else []
        self.lines: List[PageXMLTextLine] = lines if lines is not None else []
        self.orientation: Union[None, float] = orientation
        self.reading_order_number = {}
        self.text = text
        if self.lines is not None:
            self.set_as_parent(self.lines)
        if self.lines is not None:
            self.set_as_parent(self.lines)
        if self.text_regions is not None:
            self.set_as_parent(self.text_regions)
        if self.reading_order:
            self.set_text_regions_in_reader_order()
        if doc_type:
            self.add_type(doc_type)
        self.empty_regions = []

    def __repr__(self):
        stats = json.dumps(self.stats)
        content_string = f"\n\tid={self.id}, \n\ttype={self.type}, \n\tstats={stats}"
        return f"{self.__class__.__name__}({content_string}\n)"

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
            self.set_as_parent([child])
        else:
            raise TypeError(f'unknown child type: {child.__class__.__name__}')
        self.coords = parse_derived_coords(self.text_regions + self.lines)

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
            # if text_region.id not in tr_ids:
            #     print("reading order:", self.reading_order)
            #     raise KeyError(f"text_region with id {text_region.id} is not listed in reading_order")
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
        if not self.text_regions and self.lines:
            text_regions.append(self)
        return text_regions

    def get_lines(self) -> List[PageXMLTextLine]:
        lines: List[PageXMLTextLine] = []
        if self.text_regions:
            if self.reading_order and all([tr.id in self.reading_order for tr in self.text_regions]):
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
            for line in self.lines:
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

    def add_to_pagexml(self, parent: etree.Element = None):
        tr_xml = add_pagexml_sub_element(parent, 'TextRegion', sub_id=self.id, custom=self.custom,
                                         coords=self.coords)
        for line in self.lines:
            line.add_to_pagexml(tr_xml)
        for sub_tr in self.text_regions:
            sub_tr.add_to_pagexml(tr_xml)

    def _to_pagexml(self, page_xml: etree.Element):
        self.add_to_pagexml(page_xml)


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

    def add_to_pagexml(self, parent: etree.Element = None):
        column_attrib = {'type': 'column'}
        column_xml = add_pagexml_sub_element(parent, 'TextRegion', sub_id=self.id, custom=self.custom,
                                             coords=self.coords, attributes=column_attrib)
        for tr in self.text_regions:
            tr.add_to_pagexml(column_xml)


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
        if self.extra:
            for tr in self.extra:
                lines += tr.get_lines()
        if self.text_regions:
            if self.reading_order and all([tr.id in self.reading_order for tr in self.text_regions]):
                for tr in sorted(self.text_regions, key=lambda t: self.reading_order_number[t]):
                    lines += tr.get_lines()
            else:
                for tr in sorted(self.text_regions):
                    lines += tr.get_lines()
        if self.lines:
            raise AttributeError(f'page {self.id} has lines as direct property')
        return lines

    def add_child(self, child: PageXMLDoc, as_extra: bool = False):
        child.set_parent(self)
        if as_extra and (isinstance(child, PageXMLColumn) or isinstance(child, PageXMLTextRegion)):
            self.extra.append(child)
        elif isinstance(child, PageXMLColumn) or child.__class__.__name__ == 'PageXMLColumn':
            self.columns.append(child)
        elif isinstance(child, PageXMLTextLine):
            self.lines.append(child)
        elif isinstance(child, PageXMLTextRegion):
            self.text_regions.append(child)
        else:
            raise TypeError(f'unknown child type: {child.__class__.__name__}')
        self.coords = parse_derived_coords(self.extra + self.columns + self.text_regions + self.lines)

    def get_all_text_regions(self):
        text_regions = [tr for col in self.columns for tr in col.text_regions]
        text_regions.extend([tr for tr in self.extra])
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
        if self.extra:
            stats['extra'] = len(self.extra)
        if self.text_regions:
            stats['text_regions'] = len(self.text_regions)
        return stats

    def add_to_pagexml(self, parent: etree.Element = None):
        page_attrib = {'type': 'page'}
        page_xml = add_pagexml_sub_element(parent, 'TextRegion', sub_id=self.id, custom=self.custom,
                                           coords=self.coords, attributes=page_attrib)
        for column in self.text_regions:
            column.add_to_pagexml(page_xml)
        for tr in self.text_regions:
            tr.add_to_pagexml(page_xml)


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
        self.set_scan_id_as_metadata()

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

    def set_scan_id_as_metadata(self):
        self.metadata['scan_id'] = self.id
        for tr in self.get_all_text_regions():
            tr.metadata['scan_id'] = self.id
        for line in self.get_lines():
            line.metadata['scan_id'] = self.id
        for word in self.get_words():
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
    def stats(self):
        stats = super().stats
        stats['columns'] = len([column for page in self.pages for column in page.columns])
        stats['extra'] = len([text_region for page in self.pages for text_region in page.extra])
        stats['pages'] = len(self.pages)
        return stats

    def add_to_pagexml(self, scan_xml: etree.Element = None):
        for tr in self.text_regions:
            tr.add_to_pagexml(scan_xml)


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


def get_horizontal_overlap(doc1: PageXMLDoc, doc2: PageXMLDoc) -> int:
    if has_baseline(doc1) and has_baseline(doc2):
        overlap_left = max([doc1.baseline.left, doc2.baseline.left])
        overlap_right = min([doc1.baseline.right, doc2.baseline.right])
    else:
        overlap_left = max([doc1.coords.left, doc2.coords.left])
        overlap_right = min([doc1.coords.right, doc2.coords.right])
    return overlap_right - overlap_left + 1 if overlap_right >= overlap_left else 0


def get_vertical_overlap(doc1: PageXMLDoc, doc2: PageXMLDoc) -> int:
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
