from typing import Dict, Iterable, Union

from lxml import etree

from pagexml.model.coords import Coords
from pagexml.model.coords import Baseline


PAGE_NAMESPACE = "https://www.primaresearch.org/schema/PAGE/gts/pagecontent/2019-07-15"
PAGE = "{%s}" % PAGE_NAMESPACE
PAGE_SCHEMA_LOC = ("https://www.primaresearch.org/schema/PAGE/gts/pagecontent/2019-07-15 "
                   "https://www.primaresearch.org/schema/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd")
NS_XSI = "{http://www.w3.org/2001/XMLSchema-instance}"
NSMAP = {None: PAGE_NAMESPACE}

VALID_TAGS = {
    'PcGts', 'Metadata', 'Page',
    # Metadata elements
    'Creator', 'Created', 'LastChange', 'Comments', 'UserDefined', 'MetadataItem',
    # Page elements
    'Border', 'PrintSpace', 'Layers', 'Relations', 'Labels',
    # reading order
    'ReadingOrder', 'OrderedGroup', 'UnorderedGroup', 'RegionRefIndexed',
    # Region types
    'TextRegion', 'ImageRegion', 'LineDrawingRegion', 'GraphicRegion', 'ChartRegion', 'TableRegion',
    'MapRegion', 'SeparatorRegion', 'MathsRegion', 'ChemRegion', 'MusicRegion', 'AdvertRegion',
    'NoiseRegion', 'UnknownRegion', 'CustomRegion',
    # Text elements
    'TextLine', 'Word',
    'Coords', 'Baseline',
    'TextEquiv', 'TextStyle', 'Unicode', 'PlainText'
}

VALID_NS_TAGS = {PAGE + tag for tag in VALID_TAGS}


def prettyprint(element: etree.Element, **kwargs):
    xml = etree.tostring(element, pretty_print=True, **kwargs)
    print(xml.decode(), end='')


def add_namespace(name: str, ns: str = 'PAGE'):
    if ns == 'PAGE':
        return PAGE + name


def namespaced_tags(ns: str, tags: Union[str, Iterable[str]]):
    if isinstance(tags, str):
        tags = [tags]
    return [ns + tag for tag in tags]


def make_empty_pagexml(metadata: Dict[str, any] = None, imageFilename: str = '',
                       imageWidth: int = None, imageHeight: int = None, page_attributes: Dict[str, any] = None):
    pcgts = etree.Element(PAGE + 'PcGts', nsmap=NSMAP)
    pcgts.set(NS_XSI + "schemaLocation", PAGE_SCHEMA_LOC)
    metadata_ele = etree.SubElement(pcgts, PAGE + 'Metadata')
    if metadata is not None:
        for field in ['Creator', 'Created', 'LastChange']:
            if field in metadata:
                metadata_ele.set(field, metadata[field])
    if page_attributes is None:
        page_attributes = {
            'imageFilename': imageFilename,
            'imageWidth': str(imageWidth) if imageWidth is not None else '',
            'imageHeight': str(imageHeight) if imageHeight is not None else ''
        }
    page = add_pagexml_sub_element(pcgts, 'Page')
    for key in page_attributes:
        if page_attributes[key] is not None:
            page.set(key, str(page_attributes[key]))
    return pcgts


def make_custom_string(custom):
    element_strings = []
    for custom_element in custom:
        tag_string = ''
        tag_fields = [field for field in custom_element if field != 'tag_name']
        for field in tag_fields:
            tag_string += f"{field}:{custom_element[field]};"
        element_string = custom_element['tag_name'] + ' {' + tag_string + '} '
        element_strings.append(element_string)

    return ' '.join(element_strings)


def make_pagexml_element(name: str, ele_id: str = None, custom: Dict[str, any] = None,
                         coords: Coords = None, baseline: Baseline = None, text: str = None,
                         conf: float = None, attributes: Dict[str, any] = None):
    element = etree.Element(PAGE + name, nsmap=NSMAP)
    if ele_id is not None:
        element.set('id', ele_id)
    if custom is not None:
        custom_string = make_custom_string(custom)
        element.set('custom', custom_string)
    if attributes is not None:
        for key in attributes:
            element.set(key, str(attributes[key]))
    if coords is not None:
        add_pagexml_coords(element, coords)
    if baseline is not None:
        add_pagexml_baseline(element, baseline)
    if text is not None:
        add_pagexml_text(element, coords=coords, baseline=baseline, text=text, conf=conf)
    return element


def is_valid_element_name(tag: str):
    return tag in VALID_NS_TAGS


def is_valid_pagexml_sub_element(parent_tag: str, child_tag: str):
    if is_valid_element_name(parent_tag) is False:
        raise ValueError(f"invalid parent_tag '{parent_tag}'")
    if is_valid_element_name(child_tag) is False:
        raise ValueError(f"invalid child_tag '{child_tag}'")

    parent_tag = parent_tag.replace(PAGE, '')
    child_tag = child_tag.replace(PAGE, '')

    if parent_tag == 'TextRegion':
        return child_tag in {'TextRegion', 'TextLine', 'TextEquiv', 'Coords'}
    elif parent_tag == 'TextLine':
        return child_tag in {'Word', 'TextEquiv', 'TextStyle', 'Coords', 'Baseline'}
    elif parent_tag == 'Word':
        return child_tag in {'TextEquiv', 'TextStyle', 'Coords'}
    elif parent_tag == 'TextEquiv':
        return child_tag in {'Unicode', 'PlainText'}
    elif parent_tag == 'PcGts':
        return child_tag in {'Page', 'Metadata'}
    elif parent_tag == 'Page':
        return child_tag in {'ReadingOrder', 'TextRegion', 'TableRegion', 'ChartRegion', 'MapRegion'}
    elif parent_tag == 'ReadingOrder':
        return child_tag in {'OrderedGroup'}
    elif parent_tag == 'OrderedGroup':
        return child_tag in {'RegionRefIndexed'}
    elif parent_tag == 'Metadata':
        return child_tag in {'Creator', 'Created', 'LastChange', 'Comments', 'UserDefined', 'MetadataItem'}
    elif parent_tag in {'Unicode', 'PlainText', 'TextStyle', 'Coords', 'Baseline', 'RegionRefIndexed'}:
        return False
    else:
        raise ValueError(f"No valid relationships set for parent_tag '{parent_tag}'")


def is_pagexml_singleton_relation(parent_tag: str, child_tag: str):
    if is_valid_element_name(parent_tag) is False:
        raise ValueError(f"invalid parent_tag '{parent_tag}'")
    if is_valid_element_name(child_tag) is False:
        raise ValueError(f"invalid child_tag '{child_tag}'")

    if parent_tag == PAGE + 'PcGts' and child_tag in namespaced_tags(PAGE, {'Metadata', 'Page'}):
        return True
    if parent_tag == PAGE + 'Metadata' and child_tag in namespaced_tags(PAGE, {'LastChanged', 'Created',
                                                                               'Comments', 'UserDefined'}):
        return True
    if parent_tag == PAGE + 'Page' and child_tag in namespaced_tags(PAGE, {'ReadingOrder', 'Border', 'PrintSpace'}):
        return True
    if parent_tag == PAGE + 'TextRegion' and child_tag in namespaced_tags(PAGE, 'Coords'):
        return True
    if parent_tag == PAGE + 'TextLine' and child_tag in namespaced_tags(PAGE, {'Coords', 'Baseline', 'TextEquiv'}):
        return True
    if parent_tag == PAGE + 'Word' and child_tag in namespaced_tags(PAGE, {'Coords', 'TextEquiv'}):
        return True
    if parent_tag == PAGE + 'TextEquiv' and child_tag in namespaced_tags(PAGE, {'PlainText', 'Unicode'}):
        return True
    if parent_tag == PAGE + 'Page' and child_tag in namespaced_tags(PAGE, 'ReadingOrder'):
        return True


def add_pagexml_sub_element(parent: etree.Element, sub_name: str, sub_id: str = None,
                            **kwargs):
    if is_valid_pagexml_sub_element(parent.tag, PAGE + sub_name) is False:
        raise TypeError(f"parent '{parent.tag}' cannot have child '{PAGE + sub_name}' according to schema")
    if is_pagexml_singleton_relation(parent.tag, PAGE + sub_name):
        sub_element = parent.find(PAGE + sub_name)
        if sub_element is not None:
            raise ValueError(f"parent '{parent.tag}' can only have one child '{PAGE + sub_name}' according to schema")
    sub_element = make_pagexml_element(name=sub_name, ele_id=sub_id, **kwargs)
    parent.append(sub_element)
    return sub_element


def add_pagexml_coords(element: etree.Element, coords: Coords):
    if element.tag not in namespaced_tags(PAGE, {'TextRegion', 'TextLine', 'Word'}):
        raise TypeError(f"Cannot add Coords to '{element.tag}' element")
    if element.find(PAGE + 'Coords') is not None:
        raise ValueError(f"Cannot add more than one Coords element to '{element.tag}' element")
    attrib = {'points': coords.point_string}
    etree.SubElement(element, PAGE + 'Coords', attrib=attrib)


def add_pagexml_baseline(element: etree.Element, baseline: Coords):
    if element.tag not in namespaced_tags(PAGE, 'TextLine'):
        raise TypeError(f"Cannot add Baseline to '{element.tag}' element")
    if element.find(PAGE + 'Baseline') is not None:
        raise ValueError(f"Cannot add more than one Baseline element to '{element.tag}' element")
    attrib = {'points': baseline.point_string}
    etree.SubElement(element, PAGE + 'Baseline', attrib=attrib)


def add_pagexml_text(element: etree.Element, text: str,
                     coords: Coords = None, baseline: Baseline = None,
                     conf: float = None):
    attrib = {'conf': str(conf)}
    if element.tag in namespaced_tags(PAGE, {'TextLine', 'Word'}):
        text_element = element
    elif element.tag in namespaced_tags(PAGE, 'TextRegion'):
        text_element = add_pagexml_sub_element(element, 'TextLine', coords=coords, baseline=baseline,
                                               text=text, conf=conf)
    else:
        raise TypeError(f"Cannot add text to '{element.tag}' element")
    if element.find(PAGE + 'TextEquiv') is not None:
        raise ValueError(f"Cannot add more than one TextEquiv element to '{element.tag}' element")
    text_equiv = etree.SubElement(text_element, PAGE + 'TextEquiv', attrib=attrib)
    unicode = etree.SubElement(text_equiv, PAGE + 'Unicode')
    unicode.text = text
    plaintext = etree.SubElement(text_equiv, PAGE + 'PlainText')
    plaintext.text = text


def stringify_xml(page: etree.Element):
    xml_string = etree.tostring(page, pretty_print=True, encoding="UTF-8", xml_declaration=True)
    return xml_string.decode()
