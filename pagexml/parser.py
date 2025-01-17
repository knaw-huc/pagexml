import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Generator, Iterable, List, Tuple, Union
from xml.parsers import expat

import xmltodict
from dateutil.parser import parse as date_parse
from tqdm import tqdm

import pagexml.model.physical_document_model as pdm
from pagexml.helper.file_helper import read_page_archive_file
from pagexml.model.physical_document_model import Baseline, Coords
from pagexml.model.coords import parse_derived_coords


def parse_coords(coords: dict) -> Union[Coords, None]:
    if coords is None:
        return None
    if '@points' in coords and coords['@points'] != '':
        try:
            return Coords(points=coords['@points'])
        except ValueError as err:
            message = f'{err} in "@points": "{coords["@points"]}"'
            raise ValueError(message)
    else:
        return None


def parse_baseline(baseline: dict) -> Baseline:
    if baseline['@points'] == "":
        raise ValueError('Empty points attribute in baseline element')
    return Baseline(points=baseline['@points'])


def parse_line_words(textline: dict) -> List[pdm.PageXMLWord]:
    words: List[pdm.PageXMLWord] = []
    if "Word" not in textline:
        return words
    if isinstance(textline["Word"], dict):
        textline["Word"] = [textline["Word"]]
    for word_dict in textline["Word"]:
        if 'TextEquiv' not in word_dict or word_dict['TextEquiv'] is None:
            continue
        if isinstance(word_dict["TextEquiv"]["Unicode"], str):
            unicode_string = word_dict["TextEquiv"]["Unicode"]
        elif isinstance(word_dict["TextEquiv"]["Unicode"], dict):
            unicode_string = word_dict["TextEquiv"]["Unicode"]['#text']
        else:
            unicode_string = ""
        try:
            conf = None
            custom = parse_custom_metadata(word_dict)
            if word_dict["TextEquiv"] is not None:
                if "@conf" in word_dict["TextEquiv"]:
                    conf = word_dict["TextEquiv"]["@conf"]
            word = pdm.PageXMLWord(text=unicode_string,
                                   doc_id=word_dict['@id'] if '@id' in word_dict else None,
                                   metadata=custom,
                                   coords=parse_coords(word_dict["Coords"]),
                                   conf=conf)
            words.append(word)
        except TypeError:
            print('Unexpected format for Word Unicode representation:', word_dict)
            raise
    return words


def parse_text_equiv(text_equiv: dict) -> Union[str, None]:
    if isinstance(text_equiv, str):
        return text_equiv
    elif text_equiv is None:
        return None
    elif 'Unicode' in text_equiv:
        return text_equiv['Unicode']
    elif 'PlainText' in text_equiv:
        return text_equiv['PlainText']
    else:
        return None


def parse_textline(textline: dict, custom_tags: Iterable = None) -> pdm.PageXMLTextLine:
    text = parse_text_equiv(textline['TextEquiv']) if 'TextEquiv' in textline else None
    try:
        return pdm.PageXMLTextLine(
            xheight=int(textline['@xheight']) if '@xheight' in textline else None,
            doc_id=textline['@id'] if '@id' in textline else None,
            metadata=parse_custom_metadata(textline, custom_tags=custom_tags)
            if '@custom' in textline
            else None,
            coords=parse_coords(textline['Coords']),
            baseline=parse_baseline(textline['Baseline'])
            if 'Baseline' in textline
            else None,
            text=text,
            conf=parse_conf(textline['TextEquiv']) if 'TextEquiv' in textline else None,
            words=parse_line_words(textline),
        )
    except ValueError as err:
        message = f'Error parsing TextLine:\n{json.dumps(textline, indent=4)}\n{err}'
        raise ValueError(message)


def parse_conf(text_element: dict) -> Union[float, None]:
    if text_element and '@conf' in text_element:
        if text_element['@conf'] == '':
            return None
        else:
            return float(text_element['@conf'])
    else:
        return None


def parse_textline_list(textline_list: list, custom_tags: Iterable = None) -> List[pdm.PageXMLTextLine]:
    """Parse a list TextLine dictionaries into a list of PageXMLTextLine objects."""
    if isinstance(textline_list, dict):
        textline_list = [textline_list]
    return [parse_textline(textline, custom_tags) for textline in textline_list]


def parse_custom_metadata_element(custom_string: str, custom_field: str) -> Dict[str, str]:
    """Parse a custom metadata element from the custom attribute string.

    Deprecated and kept for backwards compatibility. Please use parse_custom_attribute and
    parse_custom_attribute_part."""
    match = re.search(r'\b' + custom_field + r' {(.*?)}', custom_string)
    if not match:
        print(f'pagexml.parser.parse_custom_metadata_element - custom_string:\n\n{custom_string}\n')
        raise ValueError('Invalid structure metadata in custom attribute.')
    metadata = parse_custom_attribute_parts(match.group(1))
    return metadata


def parse_custom_metadata_element_list(custom_string: str, custom_field: str) -> List[Dict[str, str]]:
    """Parse a repeated custom metadata element from the custom attribute string.

    Deprecated and kept for backwards compatibility. Please use parse_custom_attribute and
    parse_custom_attribute_part."""
    metadata_list = []

    matches = re.finditer(r'\b(' + custom_field + r') {(.*?)}', custom_string)

    for match in matches:
        tag = match.group(1)
        metadata = parse_custom_attribute_parts(match.group(2))
        metadata['type'] = tag
        metadata_list.append(metadata)

    return metadata_list


def parse_custom_attributes(custom_string: str) -> List[Dict[str, any]]:
    """Parse the custom attribute string of a PageXML element."""

    matches = re.finditer(r'\b(\w+) {(.*?)}', custom_string)
    custom_attributes = []
    for match in matches:
        attribute = parse_custom_attribute_parts(match.group(2))
        attribute['tag_name'] = match.group(1)
        custom_attributes.append(attribute)
    return custom_attributes


def parse_custom_attribute_parts(attribute_string: str) -> Dict[str, any]:
    """Parse the string of custom attributes into a dictionary.

    Assumptions:

    1. attributes are always and only separated by semicolons (;)
    2. attribute key/value pairs are always separated by a colon (:)
    3. there is no nesting of attributes. The attributes are a flat list
    4. attribute values contain only alphanumeric characters, no punctuation
       or quotes, whitespace other symbols
    """
    structure_parts = attribute_string.strip().split(';')
    metadata = {}
    for part in structure_parts:
        if part == '':
            continue
        field, value = part.split(':')

        field = field.strip()
        value = value.strip()

        if field in ('offset', 'length', 'index'):
            metadata[field] = int(value)
        else:
            metadata[field] = value
        # Update 2025-01-06: remove the text field to stick as close to the PageXML as possible
        """
        if 'offset' in metadata and 'length' in metadata:
            offset = metadata['offset']
            length = metadata['length']
            if element_text is not None:
                metadata['text'] = element_text[offset:offset+length]
        """
    return metadata


def parse_custom_metadata(text_element: Dict[str, any],
                          custom_tags: Iterable = None) -> Dict[str, any]:
    """Parse custom metadata, like readingOrder, structure, textStyle, unclear, abbrev."""
    if '@custom' not in text_element:
        return {}
    metadata = {
        'custom_attributes': parse_custom_attributes(text_element['@custom'])
    }
    if 'readingOrder {' in text_element['@custom']:
        metadata['reading_order'] = parse_custom_metadata_element(text_element['@custom'], 'readingOrder')
    if 'structure {' in text_element['@custom']:
        metadata['structure'] = parse_custom_metadata_element(text_element['@custom'], 'structure')
        if 'type' in metadata['structure']:
            metadata['type'] = metadata['structure']['type']
    if 'textStyle {' in text_element['@custom']:
        metadata['text_style'] = parse_custom_metadata_element_list(text_element['@custom'], 'textStyle')
    if custom_tags:
        regex_tags = r'(?:' + '|'.join(custom_tags) + r')'
        metadata['custom_tags'] = parse_custom_metadata_element_list(text_element['@custom'], regex_tags)
    return metadata


def parse_textregion(text_region_dict: dict,
                     custom_tags: Iterable = None) -> Union[pdm.PageXMLTextRegion, None]:
    text_region = pdm.PageXMLTextRegion(
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
            text_region.lines = parse_textline_list(text_region_dict['TextLine'], custom_tags)
            text_region.set_as_parent(text_region.lines)
            if not text_region.coords:
                text_region.coords = parse_derived_coords(text_region.lines)
        if child == 'TextRegion':
            text_region.text_regions = []
            if isinstance(text_region_dict['TextRegion'], list) is False:
                text_region_dict['TextRegion'] = [text_region_dict['TextRegion']]
            for tr in parse_textregion_list(text_region_dict['TextRegion'], custom_tags):
                if tr is not None:
                    text_region.text_regions.append(tr)
            text_region.set_as_parent(text_region.text_regions)
            if not text_region.coords:
                text_region.coords = parse_derived_coords(text_region.text_regions)
    if text_region.coords is None:
        stats = text_region.stats
        if sum([stats[field] for field in stats]) == 0:
            return None
    return text_region


def parse_textregion_list(textregion_dict_list: list,
                          custom_tags: Iterable = None) -> List[pdm.PageXMLTextRegion]:
    return [parse_textregion(textregion_dict, custom_tags) for textregion_dict in textregion_dict_list]


def parse_table_cell(table_cell_dict: Dict[str, any], custom_tags: Iterable = None) -> pdm.PageXMLTableCell:
    lines = []
    if 'TextLine' in table_cell_dict:
        lines = parse_textline_list(table_cell_dict['TextLine'], custom_tags)
    table_cell = pdm.PageXMLTableCell(
        doc_id=table_cell_dict['@id'],
        row=int(table_cell_dict['@row']) if '@row' in table_cell_dict else None,
        col=int(table_cell_dict['@col']) if '@col' in table_cell_dict else None,
        row_span=int(table_cell_dict['@rowSpan']) if '@rowSpan' in table_cell_dict else None,
        cell_span=int(table_cell_dict['@cellSpan']) if '@cellSpan' in table_cell_dict else None,
        header=table_cell_dict['@header'] if '@header' in table_cell_dict else None,
        orientation=float(table_cell_dict['@orientation']) if '@orientation' in table_cell_dict else None,
        coords=parse_coords(table_cell_dict['Coords']) if 'Coords' in table_cell_dict else None,
        metadata=parse_custom_metadata(table_cell_dict) if '@custom' in table_cell_dict else None,
        cornerpoints=parse_corner_points(table_cell_dict['CornerPts']) if 'CornerPts' in table_cell_dict else None,
        lines=lines
    )
    return table_cell


def parse_corner_points(cornerpoints_dict: Dict[str, any],
                        warnings: bool = False) -> Tuple[int, int, int, int]:
    if isinstance(cornerpoints_dict, str):
        cornerpoints = cornerpoints_dict
    elif isinstance(cornerpoints_dict, dict):
        cornerpoints = cornerpoints_dict['#text']
    else:
        raise TypeError(f"invalid cornerpoints: {cornerpoints_dict}")
    points = cornerpoints.split()
    if len(points) != 4 or not all(point.isdigit() for point in points):
        if warnings is True:
            print(f"content of CornerPts is not a list of 4 integers")
        return cornerpoints
    p1, p2, p3, p4 = [int(point) for point in points]
    return p1, p2, p3, p4


def parse_tableregion(table_region_dict, custom_tags: Iterable = None):
    table_region = pdm.PageXMLTableRegion(
        doc_id=table_region_dict['@id'] if '@id' in table_region_dict else None,
        orientation=float(table_region_dict['@orientation']) if '@orientation' in table_region_dict else None,
        coords=parse_coords(table_region_dict['Coords']) if 'Coords' in table_region_dict else None,
        metadata=parse_custom_metadata(table_region_dict) if '@custom' in table_region_dict else None,
    )
    if table_region.metadata and 'type' in table_region.metadata:
        table_region.add_type(table_region.metadata['type'])
    cells = []
    for child in table_region_dict:
        if child == 'TableCell':
            table_cell_dicts = table_region_dict['TableCell']
            for table_cell_dict in table_cell_dicts:
                cell = parse_table_cell(table_cell_dict, custom_tags)
                cells.append(cell)
    table_region.rows = make_rows_from_cells(cells)
    table_region.set_as_parent(table_region.rows)
    return table_region


def make_rows_from_cells(cells: List[pdm.PageXMLTableCell]) -> List[pdm.PageXMLTableRow]:
    row_cells = defaultdict(list)
    rows = []
    for cell in cells:
        row_cells[cell.row].append(cell)
    for row_id in row_cells:
        row_coords = parse_derived_coords(row_cells[row_id])
        table_row = pdm.PageXMLTableRow(doc_id=row_id, coords=row_coords, cells=row_cells[row_id])
        table_row.set_as_parent(table_row.cells)
        rows.append(table_row)
    return rows


def parse_tableregion_list(tableregion_dict_list: list,
                           custom_tags: Iterable = None) -> List[pdm.PageXMLTableRegion]:
    return [parse_tableregion(tableregion_dict, custom_tags) for tableregion_dict in tableregion_dict_list]


def parse_page_metadata(metadata_json: dict) -> dict:
    metadata = {}
    for field in metadata_json:
        if not metadata_json[field]:
            continue
        if field in ['Created', 'LastChange']:
            if metadata_json[field].isdigit():
                metadata[field] = datetime.fromtimestamp(int(metadata_json[field]) / 1000).isoformat()
            else:
                try:
                    metadata[field] = date_parse(metadata_json[field]).isoformat()
                except ValueError:
                    print('Date format deviation')
                    print(metadata_json)
                    metadata[field] = date_parse(metadata_json[field]).isoformat()
        elif isinstance(metadata_json[field], dict):
            metadata[field] = metadata_json[field]
        elif hasattr(metadata_json[field], 'isdigit') and metadata_json[field].isdigit():
            metadata[field] = int(metadata_json[field])
        else:
            metadata[field] = metadata_json[field]
    return metadata


def parse_page_image_size(page_json: dict) -> Coords:
    w = int(page_json['@imageWidth'])
    h = int(page_json['@imageHeight'])
    points = [(0, 0), (w, 0), (w, h), (0, h)]
    return Coords(points=points)


def parse_page_reading_order(page_json: dict) -> Tuple[Dict[int, any], Dict[str, any]]:
    order_dict = page_json['ReadingOrder']
    reading_order = {}
    if order_dict is None:
        return {}, {}
    reading_order_attribs = {}
    if 'OrderedGroup' in order_dict and 'RegionRefIndexed' in order_dict['OrderedGroup']:
        region_ref_indexed = order_dict['OrderedGroup']['RegionRefIndexed']
        if isinstance(region_ref_indexed, list):
            group_list = region_ref_indexed
        else:
            group_list = [region_ref_indexed]
        for region_ref in group_list:
            if '@regionRef' in region_ref:
                reading_order[int(region_ref['@index'])] = region_ref['@regionRef']
        if '@id' in order_dict['OrderedGroup']:
            reading_order_attribs['id'] = order_dict['OrderedGroup']['@id']
        if '@caption' in order_dict['OrderedGroup']:
            reading_order_attribs['caption'] = order_dict['OrderedGroup']['@caption']
    elif 'UnorderedGroup' in order_dict:
        # unordered means no order is established, so ignore
        pass
    return reading_order, reading_order_attribs


def parse_pagexml_json(pagexml_file: str, scan_json: dict, custom_tags: Iterable = None) -> pdm.PageXMLScan:
    """Parse a JSON/xmltodict representation of a PageXML file and return a PageXMLScan object."""
    doc_id = pagexml_file
    coords, text_regions = None, None
    metadata = {}
    text_regions = []
    table_regions = []
    reading_order, reading_order_attributes = {}, {}
    if 'PcGts' not in scan_json:
        # print('SCAN_JSON:', scan_json)
        raise TypeError(f'Not a PageXML file: {pagexml_file}')
    if 'Metadata' in scan_json['PcGts'] and scan_json['PcGts']['Metadata']:
        metadata = parse_page_metadata(scan_json['PcGts']['Metadata'])
    if 'xmlns' in scan_json['PcGts']:
        metadata['namespace'] = scan_json['PcGts']['xmlns']
    scan_json = scan_json['PcGts']['Page']
    if '@imageFilename' in scan_json and scan_json['@imageFilename'] is not None:
        doc_id = scan_json['@imageFilename']
    if scan_json['@imageWidth'] != '0' and scan_json['@imageHeight'] != '0':
        coords = parse_page_image_size(scan_json)
        metadata['scan_width'] = int(scan_json['@imageWidth'])
        metadata['scan_height'] = int(scan_json['@imageHeight'])
    if 'TextRegion' in scan_json:
        if isinstance(scan_json['TextRegion'], list) is False:
            scan_json['TextRegion'] = [scan_json['TextRegion']]
        for tr in parse_textregion_list(scan_json['TextRegion'], custom_tags=custom_tags):
            if tr is not None:
                text_regions.append(tr)
    if 'TableRegion' in scan_json:
        if isinstance(scan_json['TableRegion'], list) is False:
            scan_json['TableRegion'] = [scan_json['TableRegion']]
        for tr in parse_tableregion_list(scan_json['TableRegion'], custom_tags=custom_tags):
            if tr is not None:
                table_regions.append(tr)
    if 'ReadingOrder' in scan_json and scan_json['ReadingOrder']:
        reading_order, reading_order_attributes = parse_page_reading_order(scan_json)
    scan_doc = pdm.PageXMLScan(
        doc_id=doc_id,
        metadata=metadata,
        coords=coords,
        text_regions=text_regions,
        table_regions=table_regions,
        reading_order=reading_order,
        reading_order_attributes=reading_order_attributes
    )
    return scan_doc


def read_pagexml_file(pagexml_file: str, encoding: str = 'utf-8') -> str:
    """Return the content of a PageXML file as text string."""
    with open(pagexml_file, 'rt', encoding=encoding) as fh:
        return fh.read()


def parse_pagexml_file(pagexml_file: str, pagexml_data: Union[str, None] = None,
                       custom_tags: Iterable = None, encoding: str = 'utf-8') -> pdm.PageXMLScan:
    """Read PageXML from file (or content of file passed separately if read from elsewhere,
    e.g. tarball) and return a PageXMLScan object.

    :param pagexml_file: filepath to a PageXML file
    :type pagexml_file: str
    :param pagexml_data: string representation of PageXML document (corresponding to the content of pagexml_file)
    :type pagexml_data: str
    :param custom_tags: list of custom tags to be parsed in the metadata
    :type custom_tags: list
    :param encoding: the encoding of the file (default utf-8)
    :type encoding: str
    :return: a pdm.PageXMLScan object
    :rtype: PageXMLScan
    """
    if not pagexml_data:
        pagexml_data = read_pagexml_file(pagexml_file, encoding=encoding)
    scan_json = xmltodict.parse(pagexml_data)
    try:
        scan_doc = parse_pagexml_json(pagexml_file, scan_json, custom_tags=custom_tags)
    except BaseException:
        print(f'Error parsing file {pagexml_file}')
        raise
    scan_doc.metadata['filename'] = pagexml_file
    return scan_doc


def parse_pagexml_files(pagexml_files: List[str],
                        ignore_errors: bool = False,
                        encoding: str = 'utf-8') -> Generator[pdm.PageXMLScan, None, None]:
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


def read_pagexml_dirs(pagexml_dirs: Union[str, List[str]]) -> List[str]:
    """Return a list of all (Page)XML files within a list of directories.

    :param pagexml_dirs: a list of directories containing PageXML files.
    :type pagexml_dirs: Union[str, List[str]]
    """
    pagexml_files = []
    if isinstance(pagexml_dirs, str):
        pagexml_dirs = [pagexml_dirs]
    for pagexml_dir in pagexml_dirs:
        pagexml_files += glob.glob(pagexml_dir + "**/*.xml", recursive=True)
    return pagexml_files


def parse_pagexml_files_from_directory(pagexml_directories: List[str],
                                       show_progress: bool = False) -> Generator[pdm.PageXMLScan, None, None]:
    """Parse PageXML files from one or more directories.

    :param pagexml_directories: the name of one or more directories containing uncompressed PageXML files
    :type pagexml_directories: List[str]
    :param show_progress: flag to determine whether a TQDM progress bar is shown
    :type show_progress: bool
    :return: a generator that yields a tuple of archived file name and content
    :rtype: Generator[Tuple[str, str], None, None]
    """
    if isinstance(pagexml_directories, str):
        pagexml_directories = [pagexml_directories]
    for pagexml_directory in pagexml_directories:
        # print('dir:', pagexml_directory)
        dir_files = glob.glob(os.path.join(pagexml_directory, '**/*.xml'), recursive=True)
        # print('num files:', len(dir_files))
        pagexml_files = [fname for fname in dir_files if fname.endswith('.xml')]
        if show_progress is True:
            for pagexml_file in tqdm(pagexml_files, desc=f'Parsing files from directory {pagexml_directory}'):
                yield parse_pagexml_file(pagexml_file)
        else:
            for pagexml_file in pagexml_files:
                yield parse_pagexml_file(pagexml_file)


def parse_pagexml_files_from_archive(archive_file: str, ignore_errors: bool = False,
                                     silent_mode: bool = False,
                                     encoding: str = 'utf-8') -> Generator[pdm.PageXMLScan, None, None]:
    """Parse a list of PageXML files from an archive (e.g. zip, tar) and return each
    PageXML file as a PageXMLScan object.

    :param archive_file: filepath of an archive (zip, tar) containing PageXML files
    :type archive_file: str
    :param ignore_errors: whether to ignore errors when parsing individual PageXML files
    :type ignore_errors: bool
    :param silent_mode: whether to ignore errors warnings when parsing individual PageXML files
    :type silent_mode: bool
    :param encoding: the encoding of the file (default utf-8)
    :type encoding: str
    :return: a PageXMLScan object
    :rtype: PageXMLScan
    """
    for pagefile_info, pagefile_data in read_page_archive_file(archive_file):
        try:
            scan = parse_pagexml_file(pagefile_info['archived_filename'], pagexml_data=pagefile_data,
                                      encoding=encoding)
            scan.metadata['pagefile_info'] = pagefile_info
            yield scan
        except expat.ExpatError:
            if pagefile_info['archived_filename'].endswith('.xml') is False:
                continue
            else:
                print('Error parsing file', pagefile_info['archived_filename'])
                raise
        except (KeyError, AttributeError, IndexError,
                ValueError, TypeError, FileNotFoundError,
                expat.ExpatError) as err:
            if ignore_errors is True:
                if silent_mode is False:
                    print(f"Skipping file with parser error: {pagefile_info['archived_filename']}")
                    print(err)
                continue
            else:
                raise


def get_json_element(json_doc: dict, element_name: str, default_value: Union[None, dict, list] = None):
    return json_doc[element_name] if element_name in json_doc else default_value


def json_to_region_metadata(json_doc: dict):
    reading_order = get_json_element(json_doc, 'reading_order', default_value={})
    reading_order_attributes = get_json_element(json_doc, 'reading_order_attributes', default_value={})
    orientation = get_json_element(json_doc, 'orientation')
    return reading_order, reading_order_attributes, orientation


def json_to_pagexml_word(json_doc: dict) -> pdm.PageXMLWord:
    word = pdm.PageXMLWord(doc_id=json_doc['id'], doc_type=json_doc['type'],
                           metadata=json_doc['metadata'], text=json_doc['text'],
                           conf=json_doc['conf'] if 'conf' in json_doc else None)
    return word


def json_to_pagexml_line(json_doc: dict) -> pdm.PageXMLTextLine:
    words = [json_to_pagexml_word(word) for word in json_doc['words']] if 'words' in json_doc else []
    reading_order, reading_order_attributes, orientation = json_to_region_metadata(json_doc)
    try:
        line = pdm.PageXMLTextLine(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                                   coords=pdm.Coords(json_doc['coords']), baseline=pdm.Baseline(json_doc['baseline']),
                                   text=json_doc['text'], conf=json_doc['conf'] if 'conf' in json_doc else None,
                                   words=words, reading_order=reading_order,
                                   reading_order_attributes=reading_order_attributes)
        return line
    except TypeError:
        print(json_doc['baseline'])
        raise


def json_to_pagexml_text_region(json_doc: dict) -> pdm.PageXMLTextRegion:
    text_regions = [json_to_pagexml_text_region(text_region) for text_region in
                    get_json_element(json_doc, 'text_regions', default_value=[])]
    lines = [json_to_pagexml_line(line) for line in get_json_element(json_doc, 'lines', default_value=[])]
    reading_order, reading_order_attributes, orientation = json_to_region_metadata(json_doc)

    text_region = pdm.PageXMLTextRegion(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                                        coords=pdm.Coords(json_doc['coords']), text_regions=text_regions, lines=lines,
                                        orientation=orientation, reading_order=reading_order,
                                        reading_order_attributes=reading_order_attributes)
    pdm.set_parentage(text_region)
    return text_region


def json_to_pagexml_table_cell(json_doc: dict) -> pdm.PageXMLTableCell:
    lines = [json_to_pagexml_line(table_line) for table_line in json_doc['lines']] \
        if 'lines' in json_doc else []
    orientation = get_json_element(json_doc, 'orientation')
    cornerpoints = get_json_element(json_doc, 'cornerpoints')

    table_cell = pdm.PageXMLTableCell(doc_id=json_doc['id'], doc_type=json_doc['type'],
                                      metadata=json_doc['metadata'], coords=pdm.Coords(json_doc['coords']),
                                      lines=lines, orientation=orientation, cornerpoints=cornerpoints,
                                      col=json_doc['col'], cell_span=json_doc['cell_span'],
                                      row_span=json_doc['row_span'])
    pdm.set_parentage(table_cell)
    return table_cell


def json_to_pagexml_table_row(json_doc: dict) -> pdm.PageXMLTableRow:
    table_cells = [json_to_pagexml_table_cell(table_cell) for table_cell in
                   get_json_element(json_doc, 'cells', default_value=[])]
    orientation = get_json_element(json_doc, 'orientation')

    table_row = pdm.PageXMLTableRow(doc_id=json_doc['id'], doc_type=json_doc['type'],
                                    metadata=json_doc['metadata'], coords=pdm.Coords(json_doc['coords']),
                                    cells=table_cells, orientation=orientation)
    pdm.set_parentage(table_row)
    return table_row


def json_to_pagexml_table_region(json_doc: dict) -> pdm.PageXMLTableRegion:
    rows_json = get_json_element(json_doc, 'rows', default_value=[])
    table_rows = [json_to_pagexml_table_row(row_json) for row_json in rows_json]
    orientation = get_json_element(json_doc, 'orientation')

    table_region = pdm.PageXMLTableRegion(doc_id=json_doc['id'], doc_type=json_doc['type'],
                                          metadata=json_doc['metadata'], coords=pdm.Coords(json_doc['coords']),
                                          rows=table_rows, orientation=orientation)
    pdm.set_parentage(table_region)
    return table_region


def json_to_regions(json_doc: dict) -> Tuple[List[pdm.PageXMLTextRegion], List[pdm.PageXMLTableRegion]]:
    text_regions = [json_to_pagexml_text_region(text_region) for text_region in
                    get_json_element(json_doc, 'text_regions', default_value=[])]
    table_regions = [json_to_pagexml_table_region(table_region) for table_region in
                     get_json_element(json_doc, 'table_regions', default_value=[])]
    return text_regions, table_regions


def json_to_pagexml_column(json_doc: dict) -> pdm.PageXMLColumn:
    text_regions, table_regions = json_to_regions(json_doc)
    lines = [json_to_pagexml_line(line) for line in json_doc['lines']] if 'lines' in json_doc else []
    reading_order, reading_order_attributes, orientation = json_to_region_metadata(json_doc)

    column = pdm.PageXMLColumn(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                               coords=pdm.Coords(json_doc['coords']), orientation=orientation,
                               reading_order=reading_order, reading_order_attributes=reading_order_attributes,
                               text_regions=text_regions, table_regions=table_regions,
                               lines=lines)
    pdm.set_parentage(column)
    return column


def json_to_column_container(json_doc: dict) -> tuple:
    columns = [json_to_pagexml_column(column) for column in json_doc['columns']] if 'columns' in json_doc else []
    text_regions, table_regions = json_to_regions(json_doc)
    lines = [json_to_pagexml_line(line) for line in get_json_element(json_doc, 'lines', default_value=[])]
    coords = pdm.Coords(json_doc['coords']) if 'coords' in json_doc else None
    return columns, text_regions, table_regions, lines, coords


def json_to_pagexml_page(json_doc: dict) -> pdm.PageXMLPage:
    extra = [json_to_pagexml_text_region(tr) for tr in get_json_element(json_doc, 'extra', default_value=[])]
    columns, text_regions, table_regions, lines, coords = json_to_column_container(json_doc)
    reading_order, reading_order_attributes, orientation = json_to_region_metadata(json_doc)
    page = pdm.PageXMLPage(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                           coords=coords, extra=extra, columns=columns,
                           text_regions=text_regions, table_regions=table_regions, lines=lines,
                           orientation=orientation, reading_order=reading_order,
                           reading_order_attributes=reading_order_attributes)
    pdm.set_parentage(page)
    return page


def json_to_pagexml_scan(json_doc: dict) -> pdm.PageXMLScan:
    pages = [json_to_pagexml_page(page) for page in json_doc['pages']] if 'pages' in json_doc else []
    columns, text_regions, table_regions, lines, coords = json_to_column_container(json_doc)
    reading_order, reading_order_attributes, orientation = json_to_region_metadata(json_doc)
    scan = pdm.PageXMLScan(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                           coords=coords, pages=pages, columns=columns,
                           text_regions=text_regions, table_regions=table_regions, lines=lines,
                           orientation=orientation, reading_order=reading_order,
                           reading_order_attributes=reading_order_attributes)
    pdm.set_parentage(scan)
    return scan


def json_to_pagexml_doc(json_doc: dict) -> pdm.PageXMLDoc:
    if 'pagexml_doc' not in json_doc['type']:
        raise TypeError('json_doc is not of type "pagexml_doc".')
    if 'scan' in json_doc['type']:
        return json_to_pagexml_scan(json_doc)
    if 'page' in json_doc['type']:
        return json_to_pagexml_page(json_doc)
    if 'column' in json_doc['type']:
        return json_to_pagexml_column(json_doc)
    if 'text_region' in json_doc['type']:
        return json_to_pagexml_text_region(json_doc)
    if 'line' in json_doc['type']:
        return json_to_pagexml_line(json_doc)
    if 'word' in json_doc['type']:
        return json_to_pagexml_word(json_doc)


def parse_pagexml_from_json(pagexml_json: Union[str, Dict[str, any]]) -> pdm.PageXMLDoc:
    """Turn a JSON representation of a PageXML document into an instance from
    the physical document model."""
    if isinstance(pagexml_json, str):
        pagexml_json = json.loads(pagexml_json)
    return json_to_pagexml_doc(pagexml_json)
