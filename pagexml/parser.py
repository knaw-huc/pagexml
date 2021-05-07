import re
from datetime import datetime
from typing import List, Dict, Union

import xmltodict
from dateutil.parser import parse as date_parse

from pagexml.model.physical_document_model import Baseline, Coords, parse_derived_coords
from pagexml.model.physical_document_model import PageXMLScan, PageXMLTextLine, PageXMLTextRegion, PageXMLWord


def parse_coords(coords: dict) -> Union[Coords, None]:
    if coords is None:
        return None
    if '@points' in coords:
        return Coords(points=coords['@points'])
    else:
        return None


def parse_baseline(baseline: dict) -> Baseline:
    return Baseline(points=baseline['@points'])


def parse_line_words(textline: dict) -> List[PageXMLWord]:
    words: List[PageXMLWord] = []
    if "Word" not in textline:
        return words
    if isinstance(textline["Word"], dict):
        textline["Word"] = [textline["Word"]]
    for word_dict in textline["Word"]:
        if isinstance(word_dict["TextEquiv"]["Unicode"], str):
            unicode_string = word_dict["TextEquiv"]["Unicode"]
        else:
            unicode_string = word_dict["TextEquiv"]["Unicode"]['#text']
        try:
            word = PageXMLWord(text=unicode_string,
                               doc_id=word_dict['@id'] if '@id' in word_dict else None,
                               metadata=parse_custom_metadata(word_dict) if '@custom' in word_dict else None,
                               coords=parse_coords(word_dict["Coords"]),
                               conf=word_dict["TextEquiv"]["@conf"] if "@conf" in word_dict["TextEquiv"] else None)
            words.append(word)
        except TypeError:
            print('Unexpected format for Word Unicode representation:', word_dict)
            raise
    return words


def parse_textline(textline: dict) -> PageXMLTextLine:
    if 'TextEquiv' not in textline:
        text = None
    elif isinstance(textline['TextEquiv'], str):
        text = textline['TextEquiv']
    elif 'Unicode' in textline['TextEquiv']:
        text = textline['TextEquiv']['Unicode']
    elif 'PlainText' in textline['TextEquiv']:
        text = textline['TextEquiv']['PlainText']
    else:
        text = None
    line = PageXMLTextLine(xheight=int(textline['@xheight']) if '@xheight' in textline else None,
                           doc_id=textline['@id'] if '@id' in textline else None,
                           metadata=parse_custom_metadata(textline) if '@custom' in textline else None,
                           coords=parse_coords(textline['Coords']),
                           baseline=parse_baseline(textline.get('Baseline')),
                           text=text,
                           words=parse_line_words(textline))
    return line


def parse_textline_list(textline_list: list) -> List[PageXMLTextLine]:
    return [parse_textline(textline) for textline in textline_list]


def parse_custom_metadata_element(custom_string: str, custom_field: str) -> Dict[str, str]:
    match = re.search(r'\b' + custom_field + r' {(.*?)}', custom_string)
    if not match:
        print(custom_string)
        raise ValueError('Invalid structure metadata in custom attribute.')
    structure_parts = match.group(1).strip().split(';')
    metadata = {}
    for part in structure_parts:
        if part == '':
            continue
        field, value = part.split(':')
        metadata[field] = value
    return metadata


def parse_custom_metadata(text_element: Dict[str, any]) -> Dict[str, any]:
    """Parse custom metadata, like readingOrder, structure."""
    metadata = {}
    if '@custom' not in text_element:
        return metadata
    if 'readingOrder {' in text_element['@custom']:
        metadata['reading_order'] = parse_custom_metadata_element(text_element['@custom'], 'readingOrder')
    if 'structure {' in text_element['@custom']:
        metadata['structure'] = parse_custom_metadata_element(text_element['@custom'], 'structure')
        if 'type' in metadata['structure']:
            metadata['type'] = metadata['structure']['type']
    return metadata


def parse_textregion(text_region_dict: dict) -> PageXMLTextRegion:
    text_region = PageXMLTextRegion(
        doc_id=text_region_dict['@id'] if '@id' in text_region_dict else None,
        orientation=float(text_region_dict['@orientation']) if '@orientation' in text_region_dict else None,
        coords=parse_coords(text_region_dict['Coords']) if 'Coords' in text_region_dict else None,
        metadata=parse_custom_metadata(text_region_dict) if '@custom' in text_region_dict else None,
    )
    if text_region.metadata and 'type' in text_region.metadata:
        text_region.add_type(text_region.metadata['type'])
    for child in text_region_dict:
        if child == 'TextLine':
            if isinstance(text_region_dict['TextLine'], list):
                text_region.lines = parse_textline_list(text_region_dict['TextLine'])
            else:
                text_region.lines = [parse_textline(text_region_dict['TextLine'])]
            if not text_region.coords:
                text_region.coords = parse_derived_coords(text_region.lines)
        if child == 'TextRegion':
            if isinstance(text_region_dict['TextRegion'], list):
                text_region.text_regions = parse_textregion_list(text_region_dict['TextRegion'])
            else:
                text_region.text_regions = [parse_textregion(text_region_dict['TextRegion'])]
            if not text_region.coords:
                text_region.coords = parse_derived_coords(text_region.text_regions)
    return text_region


def parse_textregion_list(textregion_dict_list: list) -> List[PageXMLTextRegion]:
    return [parse_textregion(textregion_dict) for textregion_dict in textregion_dict_list]


def parse_page_metadata(metadata_json: dict) -> dict:
    metadata = {}
    for field in metadata_json:
        if not metadata_json[field]:
            continue
        if field in ['Created', 'LastChange']:
            if metadata_json[field].isdigit():
                metadata[field] = datetime.fromtimestamp(int(metadata_json[field]) / 1000)
            else:
                try:
                    metadata[field] = date_parse(metadata_json[field])
                except ValueError:
                    print('Date format deviation')
                    print(metadata_json)
                    metadata[field] = date_parse(metadata_json[field])
        elif isinstance(metadata_json[field], dict):
            metadata[field] = metadata_json[field]
        elif metadata_json[field].isdigit():
            metadata[field] = int(metadata_json[field])
        elif metadata_json[field]:
            metadata[field] = metadata_json[field]
    return metadata


def parse_page_image_size(page_json: dict) -> Coords:
    w = int(page_json['@imageWidth'])
    h = int(page_json['@imageHeight'])
    points = [(0, 0), (w, 0), (w, h), (0, h)]
    return Coords(points=points)


def parse_page_reading_order(page_json: dict) -> dict:
    order_dict = page_json['ReadingOrder']
    reading_order = {}
    if order_dict is None:
        return {}
    if 'OrderedGroup' in order_dict and 'RegionRefIndexed' in order_dict['OrderedGroup']:
        if isinstance(order_dict['OrderedGroup']['RegionRefIndexed'], list):
            group_list = order_dict['OrderedGroup']['RegionRefIndexed']
            for region_ref in group_list:
                reading_order[int(region_ref['@index'])] = region_ref['@regionRef']
        else:
            group_item = order_dict['OrderedGroup']['RegionRefIndexed']
            reading_order[int(group_item['@index'])] = group_item['@regionRef']
    return reading_order


def read_pagexml_file(pagexml_file: str) -> str:
    with open(pagexml_file, 'rt') as fh:
        return fh.read()


def parse_pagexml_file(pagexml_file: str, pagexml_data: Union[str, None] = None) -> PageXMLScan:
    """Read PageXML from file (or passed separately if read from elsewhere, e.g. tarball)
    and return a PageXMLScan object."""
    if not pagexml_data:
        pagexml_data = read_pagexml_file(pagexml_file)
    scan_json = xmltodict.parse(pagexml_data)
    scan_doc = parse_pagexml_json(scan_json)
    scan_doc.metadata['filename'] = pagexml_file
    return scan_doc


def parse_pagexml_json(scan_json: dict) -> PageXMLScan:
    """Parse a JSON/xmltodict representation of a PageXML file and return a PageXMLScan object."""
    coords, text_regions = None, None
    metadata = {}
    if 'Metadata' in scan_json['PcGts'] and scan_json['PcGts']['Metadata']:
        metadata = parse_page_metadata(scan_json['PcGts']['Metadata'])
    if 'xmlns' in scan_json['PcGts']:
        metadata['namespace'] = scan_json['PcGts']['xmlns']
    scan_json = scan_json['PcGts']['Page']
    if scan_json['@imageWidth'] != '0' and scan_json['@imageHeight'] != '0':
        coords = parse_page_image_size(scan_json)
    if 'TextRegion' in scan_json:
        if isinstance(scan_json['TextRegion'], list):
            text_regions = parse_textregion_list(scan_json['TextRegion'])
        else:
            text_regions = [parse_textregion(scan_json['TextRegion'])]
    if 'ReadingOrder' in scan_json and scan_json['ReadingOrder']:
        reading_order = parse_page_reading_order(scan_json)
    else:
        reading_order = {}
    scan_doc = PageXMLScan(metadata=metadata, coords=coords, text_regions=text_regions, reading_order=reading_order)
    return scan_doc
