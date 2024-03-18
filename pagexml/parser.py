import glob
import json
import os
import re
from datetime import datetime
from typing import Generator, List, Dict, Union, Iterable
from xml.parsers import expat

import xmltodict
from dateutil.parser import parse as date_parse
from tqdm import tqdm

import pagexml.model.physical_document_model as pdm
from pagexml.helper.file_helper import read_page_archive_file
from pagexml.model.physical_document_model import Baseline, Coords, parse_derived_coords


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
            if word_dict["TextEquiv"] is not None:
                if "@conf" in word_dict["TextEquiv"]:
                    conf = word_dict["TextEquiv"]["@conf"]
            word = pdm.PageXMLWord(text=unicode_string,
                                   doc_id=word_dict['@id'] if '@id' in word_dict else None,
                                   metadata=parse_custom_metadata(word_dict) if '@custom' in word_dict else None,
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


def parse_textline(textline: dict, custom_tags: Iterable = []) -> pdm.PageXMLTextLine:
    text = parse_text_equiv(textline['TextEquiv']) if 'TextEquiv' in textline else None
    try:
        return pdm.PageXMLTextLine(
            xheight=int(textline['@xheight']) if '@xheight' in textline else None,
            doc_id=textline['@id'] if '@id' in textline else None,
            metadata=parse_custom_metadata(textline, custom_tags)
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


def parse_textline_list(textline_list: list, custom_tags: Iterable = []) -> List[pdm.PageXMLTextLine]:
    return [parse_textline(textline, custom_tags) for textline in textline_list]


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


def parse_custom_metadata_element_list(custom_string: str, custom_field: str) -> List[Dict[str, str]]:
    metadata_list = []

    matches = re.finditer(r'\b(' + custom_field + r') {(.*?)}', custom_string)

    for match in matches:
        tag = match.group(1)
        metadata = {"type": tag}
        structure_parts = match.group(2).strip().split(';')

        for part in structure_parts:
            if part == '':
                continue
            field, value = part.split(':')

            field = field.strip()
            value = value.strip()

            if field in ('offset', 'length'):
                metadata[field] = int(value)
            else:
                metadata[field] = value

        metadata_list.append(metadata)

    return metadata_list


def parse_custom_metadata(text_element: Dict[str, any], custom_tags: Iterable = []) -> Dict[str, any]:
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
    if 'textStyle {' in text_element['@custom']:
        metadata['text_style'] = parse_custom_metadata_element_list(text_element['@custom'], 'textStyle')
    if custom_tags:
        regex_tags = r'(?:' + '|'.join(custom_tags) + r')'
        metadata['custom_tags'] = parse_custom_metadata_element_list(text_element['@custom'], regex_tags)
    return metadata


def parse_textregion(text_region_dict: dict, custom_tags: Iterable = []) -> Union[pdm.PageXMLTextRegion, None]:
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
            if isinstance(text_region_dict['TextLine'], list):
                text_region.lines = parse_textline_list(text_region_dict['TextLine'], custom_tags)
            else:
                text_region.lines = [parse_textline(text_region_dict['TextLine'], custom_tags)]
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


def parse_textregion_list(textregion_dict_list: list, custom_tags: Iterable = []) -> List[pdm.PageXMLTextRegion]:
    return [parse_textregion(textregion_dict, custom_tags) for textregion_dict in textregion_dict_list]


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


def parse_pagexml_json(pagexml_file: str, scan_json: dict, custom_tags: Iterable = []) -> pdm.PageXMLScan:
    """Parse a JSON/xmltodict representation of a PageXML file and return a PageXMLScan object."""
    doc_id = pagexml_file
    coords, text_regions = None, None
    metadata = {}
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
    if 'TextRegion' in scan_json:
        text_regions = []
        if isinstance(scan_json['TextRegion'], list) is False:
            scan_json['TextRegion'] = [scan_json['TextRegion']]
        for tr in parse_textregion_list(scan_json['TextRegion'], custom_tags=custom_tags):
            if tr is not None:
                text_regions.append(tr)
    if 'ReadingOrder' in scan_json and scan_json['ReadingOrder']:
        reading_order = parse_page_reading_order(scan_json)
    else:
        reading_order = {}
    scan_doc = pdm.PageXMLScan(
        doc_id=doc_id,
        metadata=metadata,
        coords=coords,
        text_regions=text_regions,
        reading_order=reading_order,
    )
    return scan_doc


def read_pagexml_file(pagexml_file: str, encoding: str = 'utf-8') -> str:
    """Return the content of a PageXML file as text string."""
    with open(pagexml_file, 'rt', encoding=encoding) as fh:
        return fh.read()


def parse_pagexml_file(pagexml_file: str, pagexml_data: Union[str, None] = None, custom_tags: Iterable = {},
                       encoding: str = 'utf-8') -> pdm.PageXMLScan:
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


def json_to_pagexml_word(json_doc: dict) -> pdm.PageXMLWord:
    word = pdm.PageXMLWord(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                           text=json_doc['text'], conf=json_doc['conf'] if 'conf' in json_doc else None)
    return word


def json_to_pagexml_line(json_doc: dict) -> pdm.PageXMLTextLine:
    words = [json_to_pagexml_word(word) for word in json_doc['words']] if 'words' in json_doc else []
    reading_order = json_doc['reading_order'] if 'reading_order' in json_doc else {}
    try:
        line = pdm.PageXMLTextLine(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                                   coords=pdm.Coords(json_doc['coords']), baseline=pdm.Baseline(json_doc['baseline']),
                                   text=json_doc['text'], conf=json_doc['conf'] if 'conf' in json_doc else None,
                                   words=words, reading_order=reading_order)
        return line
    except TypeError:
        print(json_doc['baseline'])
        raise


def json_to_pagexml_text_region(json_doc: dict) -> pdm.PageXMLTextRegion:
    text_regions = [json_to_pagexml_text_region(text_region) for text_region in json_doc['text_regions']] \
        if 'text_regions' in json_doc else []
    lines = [json_to_pagexml_line(line) for line in json_doc['lines']] if 'lines' in json_doc else []
    reading_order = json_doc['reading_order'] if 'reading_order' in json_doc else {}

    text_region = pdm.PageXMLTextRegion(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                                        coords=pdm.Coords(json_doc['coords']), text_regions=text_regions, lines=lines,
                                        reading_order=reading_order)
    pdm.set_parentage(text_region)
    return text_region


def json_to_pagexml_column(json_doc: dict) -> pdm.PageXMLColumn:
    text_regions = [json_to_pagexml_text_region(text_region) for text_region in json_doc['text_regions']] \
        if 'text_regions' in json_doc else []
    lines = [json_to_pagexml_line(line) for line in json_doc['lines']] if 'lines' in json_doc else []
    reading_order = json_doc['reading_order'] if 'reading_order' in json_doc else {}

    column = pdm.PageXMLColumn(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                               coords=pdm.Coords(json_doc['coords']), text_regions=text_regions, lines=lines,
                               reading_order=reading_order)
    pdm.set_parentage(column)
    return column


def json_to_column_container(json_doc: dict) -> tuple:
    columns = [json_to_pagexml_column(column) for column in json_doc['columns']] if 'columns' in json_doc else []
    text_regions = [json_to_pagexml_text_region(text_region) for text_region in json_doc['text_regions']] \
        if 'text_regions' in json_doc else []
    lines = [json_to_pagexml_line(line) for line in json_doc['lines']] if 'lines' in json_doc else []
    reading_order = json_doc['reading_order'] if 'reading_order' in json_doc else {}

    coords = pdm.Coords(json_doc['coords']) if 'coords' in json_doc else None
    return columns, text_regions, lines, reading_order, coords


def json_to_pagexml_page(json_doc: dict) -> pdm.PageXMLPage:
    extra = [json_to_pagexml_text_region(text_region) for text_region in json_doc['extra']] \
        if 'extra' in json_doc else []
    columns, text_regions, lines, reading_order, coords = json_to_column_container(json_doc)
    page = pdm.PageXMLPage(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                           coords=coords, extra=extra, columns=columns,
                           text_regions=text_regions, lines=lines,
                           reading_order=reading_order)
    pdm.set_parentage(page)
    return page


def json_to_pagexml_scan(json_doc: dict) -> pdm.PageXMLScan:
    pages = [json_to_pagexml_page(page) for page in json_doc['pages']] if 'pages' in json_doc else []
    columns, text_regions, lines, reading_order, coords = json_to_column_container(json_doc)
    scan = pdm.PageXMLScan(doc_id=json_doc['id'], doc_type=json_doc['type'], metadata=json_doc['metadata'],
                           coords=coords, pages=pages, columns=columns,
                           text_regions=text_regions, lines=lines, reading_order=reading_order)
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
