from typing import Generator, List, Dict, Union
from datetime import datetime
import glob
import re

import xmltodict
from dateutil.parser import parse as date_parse

from pagexml.model.physical_document_model import Baseline, Coords, parse_derived_coords
from pagexml.model.physical_document_model import PageXMLScan, PageXMLTextLine, PageXMLTextRegion, PageXMLWord


def parse_coords(coords: dict) -> Union[Coords, None]:
    if coords is None:
        return None
    if '@points' in coords and coords['@points'] != '':
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
        elif isinstance(word_dict["TextEquiv"]["Unicode"], dict):
            unicode_string = word_dict["TextEquiv"]["Unicode"]['#text']
        else:
            unicode_string = ""
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


def parse_text_equiv(text_equiv: dict) -> Union[str, None]:
    if isinstance(text_equiv, str):
        return text_equiv
    elif 'Unicode' in text_equiv:
        return text_equiv['Unicode']
    elif 'PlainText' in text_equiv:
        return text_equiv['PlainText']
    else:
        return None


def parse_textline(textline: dict) -> PageXMLTextLine:
    text = parse_text_equiv(textline['TextEquiv']) if 'TextEquiv' in textline else None
    return PageXMLTextLine(
        xheight=int(textline['@xheight']) if '@xheight' in textline else None,
        doc_id=textline['@id'] if '@id' in textline else None,
        metadata=parse_custom_metadata(textline)
        if '@custom' in textline
        else None,
        coords=parse_coords(textline['Coords']),
        baseline=parse_baseline(textline['Baseline'])
        if 'Baseline' in textline
        else None,
        text=text,
        words=parse_line_words(textline),
    )


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
        if child == 'TextEquiv':
            text_region.text = parse_text_equiv(text_region_dict[child])
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
        else:
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
                if '@regionRef' in region_ref:
                    reading_order[int(region_ref['@index'])] = region_ref['@regionRef']
        else:
            group_item = order_dict['OrderedGroup']['RegionRefIndexed']
            if '@regionRef' in group_item:
                reading_order[int(group_item['@index'])] = group_item['@regionRef']
    return reading_order


def parse_pagexml_json(pagexml_file: str, scan_json: dict) -> PageXMLScan:
    """Parse a JSON/xmltodict representation of a PageXML file and return a PageXMLScan object."""
    doc_id = pagexml_file
    coords, text_regions = None, None
    metadata = {}
    if 'Metadata' in scan_json['PcGts'] and scan_json['PcGts']['Metadata']:
        metadata = parse_page_metadata(scan_json['PcGts']['Metadata'])
    if 'xmlns' in scan_json['PcGts']:
        metadata['namespace'] = scan_json['PcGts']['xmlns']
    scan_json = scan_json['PcGts']['Page']
    if '@imageFilename' in scan_json and scan_json['@imageFilename'] is not None:
        doc_id = scan_json['@imageFilename']
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
    return PageXMLScan(
        doc_id=doc_id,
        metadata=metadata,
        coords=coords,
        text_regions=text_regions,
        reading_order=reading_order,
    )


def read_pagexml_file(pagexml_file: str, encoding: str = 'utf-8') -> str:
    """Return the content of a PageXML file as text string."""
    with open(pagexml_file, 'rt', encoding=encoding) as fh:
        return fh.read()


def parse_pagexml_file(pagexml_file: str, pagexml_data: Union[str, None] = None,
                       encoding: str = 'utf-8') -> PageXMLScan:
    """Read PageXML from file (or passed separately if read from elsewhere, e.g. tarball)
    and return a PageXMLScan object."""
    if not pagexml_data:
        pagexml_data = read_pagexml_file(pagexml_file, encoding=encoding)
    scan_json = xmltodict.parse(pagexml_data)
    scan_doc = parse_pagexml_json(pagexml_file, scan_json)
    scan_doc.metadata['filename'] = pagexml_file
    return scan_doc


def parse_pagexml_files(pagexml_files: List[str],
                        ignore_errors: bool = False,
                        encoding: str = 'utf-8') -> Generator[PageXMLScan, None, None]:
    """Parse a list of PageXML files and return each as a PageXMLScan object."""
    for pagexml_file in pagexml_files:
        try:
            yield parse_pagexml_file(pagexml_file, encoding=encoding)
        except (KeyError, AttributeError, IndexError, ValueError, TypeError):
            if ignore_errors:
                print(f'Skipping file with parser error: {pagexml_file}')
                continue
            else:
                raise


def read_pagexml_dirs(pagexml_dirs: str) -> List[str]:
    """Return a list of all (Page)XML files within a list of directories."""
    pagexml_files = []
    if isinstance(pagexml_dirs, str):
        pagexml_dirs = [pagexml_dirs]
    for pagexml_dir in pagexml_dirs:
        pagexml_files += glob.glob(pagexml_dir + "**/*.xml", recursive=True)
    return pagexml_files
