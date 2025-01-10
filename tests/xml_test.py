from unittest import TestCase

from lxml import etree

from pagexml.model.coords import Coords, Baseline
import pagexml.model.physical_document_model as pdm
import pagexml.model.xml as xml


class TestTagValidity(TestCase):

    def test_page_with_namespace_is_valid(self):
        tag = xml.PAGE + 'Page'
        self.assertEqual(True, xml.is_valid_element_name(tag))

    def test_page_without_namespace_is_valid(self):
        tag = 'Page'
        self.assertEqual(False, xml.is_valid_element_name(tag))

    def test_bla_is_valid(self):
        tag = xml.PAGE + 'Bla'
        self.assertEqual(False, xml.is_valid_element_name(tag))


class TestSubTagValidity(TestCase):

    def test_page_and_textregion_are_valid(self):
        parent_tag = xml.PAGE + 'Page'
        child_tag = xml.PAGE + 'TextRegion'
        self.assertEqual(True, xml.is_valid_pagexml_sub_element(parent_tag, child_tag))

    def test_page_and_textline_are_invalid(self):
        parent_tag = xml.PAGE + 'Page'
        child_tag = xml.PAGE + 'TextLine'
        self.assertEqual(False, xml.is_valid_pagexml_sub_element(parent_tag, child_tag))


class TestEmptyPage(TestCase):

    def test_empty_pagexml_has_namespace(self):
        pagexml = xml.make_empty_pagexml()
        self.assertEqual(xml.PAGE_NAMESPACE, pagexml.nsmap[None])

    def test_empty_pagexml_has_metadata(self):
        pagexml = xml.make_empty_pagexml()
        metadata = pagexml.find(xml.PAGE + 'Metadata')
        self.assertNotEqual(None, metadata)

    def test_empty_pagexml_has_page(self):
        pagexml = xml.make_empty_pagexml()
        page = pagexml.find(xml.PAGE + 'Page')
        self.assertNotEqual(None, page)

    def test_empty_pagexml_can_set_image_details(self):
        image_details = {
            'imageFilename': 'image_001.xml',
            'imageWidth': 2000,
            'imageHeight': 1000
        }
        pagexml = xml.make_empty_pagexml(imageFilename=image_details['imageFilename'],
                                         imageWidth=image_details['imageWidth'],
                                         imageHeight=image_details['imageHeight'])
        page = pagexml.find(xml.PAGE + 'Page')
        for ki, key in enumerate(image_details):
            with self.subTest(ki):
                self.assertEqual(str(image_details[key]), page.attrib[key])


class TestStringify(TestCase):

    def test_stringify_returns_string(self):
        page = xml.make_empty_pagexml()
        xml_string = xml.stringify_xml(page)
        self.assertEqual(True, isinstance(xml_string, str))

    def test_stringify_returns_string_with_xml_declaration(self):
        page = xml.make_empty_pagexml()
        xml_string = xml.stringify_xml(page)
        self.assertEqual(True, xml_string.startswith("<?xml version"))

    def test_stringify_returns_string_with_valid_xml(self):
        page = xml.make_empty_pagexml()
        xml_string = xml.stringify_xml(page)
        error = None
        try:
            etree.fromstring(xml_string.encode())
        except BaseException as err:
            error = err
        self.assertEqual(None, error)


class TestMakeElements(TestCase):

    def setUp(self) -> None:
        self.element = xml.make_pagexml_element('TextLine')
        points = [(0, 0), (0, 100), (100, 100), (100, 0)]
        self.coords = Coords(points)
        self.baseline = Baseline([(10, 50), (40, 50), (70, 50), (100, 50)])

    def test_can_make_empty_element(self):
        element = xml.make_pagexml_element('TextRegion')
        self.assertEqual(True, element.tag.endswith('TextRegion'))

    def test_element_can_add_element(self):
        xml.add_pagexml_coords(self.element, self.coords)
        coords = self.element.find(xml.PAGE + 'Coords')
        self.assertNotEqual(None, coords)

    def test_element_can_add_coordinates(self):
        xml.add_pagexml_coords(self.element, self.coords)
        coords = self.element.find(xml.PAGE + 'Coords')
        self.assertEqual(True, coords.tag.endswith('Coords'))

    def test_element_cannot_add_multiple_coordinates(self):
        xml.add_pagexml_coords(self.element, self.coords)
        error = None
        try:
            xml.add_pagexml_coords(self.element, self.coords)
        except ValueError as err:
            error = err
        self.assertNotEqual(None, error)

    def test_element_can_add_coordinates_with_point_string(self):
        xml.add_pagexml_coords(self.element, self.coords)
        coords = self.element.find(xml.PAGE + 'Coords')
        self.assertEqual(self.coords.point_string, coords.attrib['points'])

    def test_element_can_add_baseline(self):
        xml.add_pagexml_baseline(self.element, self.baseline)
        baseline = self.element.find(xml.PAGE + 'Baseline')
        self.assertEqual(True, baseline.tag.endswith('Baseline'))

    def test_element_cannot_add_multiple_baseline(self):
        xml.add_pagexml_baseline(self.element, self.baseline)
        error = None
        try:
            xml.add_pagexml_baseline(self.element, self.baseline)
        except ValueError as err:
            error = err
        self.assertNotEqual(None, error)

    def test_element_can_add_baseline_with_point_string(self):
        xml.add_pagexml_baseline(self.element, self.baseline)
        baseline = self.element.find(xml.PAGE + 'Baseline')
        self.assertEqual(self.baseline.point_string, baseline.attrib['points'])


class TestReadingOrder(TestCase):

    def setUp(self) -> None:
        self.reading_order = {0: 'r1', 1: 'r2'}
        self.reading_order_attributes = {'id': 'group1'}
        self.pagexml_doc = xml.make_empty_pagexml()
        self.page_xml = self.pagexml_doc.find(f".//{xml.PAGE + 'Page'}")

    def test_can_add_reading_order_to_pcgts(self):
        xml.add_reading_order(self.pagexml_doc, self.reading_order)
        reading_xml = self.pagexml_doc.find(f".//{xml.PAGE + 'ReadingOrder'}")
        self.assertNotEqual(None, reading_xml)

    def test_can_add_reading_order_to_page(self):
        xml.add_reading_order(self.page_xml, self.reading_order)
        reading_xml = self.pagexml_doc.find(f".//{xml.PAGE + 'ReadingOrder'}")
        self.assertNotEqual(None, reading_xml)

    def test_can_add_reading_order_to_text_region(self):
        text_region = xml.make_pagexml_element('TextRegion', ele_id='tr1')
        xml.add_reading_order(text_region, self.reading_order)
        reading_xml = text_region.find(f".//{xml.PAGE + 'ReadingOrder'}")
        self.assertNotEqual(None, reading_xml)

    def test_cannot_add_reading_order_to_word(self):
        word = xml.make_pagexml_element('Word', ele_id='w1')
        self.assertRaises(ValueError, xml.add_reading_order, word, self.reading_order)

    def test_make_reading_order_adds_ordered_group(self):
        xml.add_reading_order(self.page_xml, self.reading_order)
        ordered_xml = self.page_xml.find(f".//{xml.PAGE + 'OrderedGroup'}")
        self.assertNotEqual(None, ordered_xml)

    def test_make_reading_order_adds_ordered_group_attributes(self):
        xml.add_reading_order(self.page_xml, self.reading_order,
                              reading_order_attributes=self.reading_order_attributes)
        ordered_xml = self.page_xml.find(f".//{xml.PAGE + 'OrderedGroup'}")
        self.assertEqual(self.reading_order_attributes, ordered_xml.attrib)

    def test_make_reading_order_adds_region_ref_indexed(self):
        xml.add_reading_order(self.page_xml, self.reading_order)
        indexed_xml = self.page_xml.find(f".//{xml.PAGE + 'RegionRefIndexed'}")
        self.assertNotEqual(None, indexed_xml)


class TestAddSubElements(TestCase):

    def test_element_cannot_add_multiple_of_singleton_element(self):
        element = xml.make_pagexml_element('Page')
        xml.add_pagexml_sub_element(element, 'ReadingOrder')
        error = None
        try:
            xml.add_pagexml_sub_element(element, 'ReadingOrder')
        except ValueError as err:
            error = err
        self.assertNotEqual(None, error)


class TestMakeTextEquiv(TestCase):

    def setUp(self) -> None:
        self.element = xml.make_pagexml_element('TextLine')
        self.text = "some text"

    def test_element_can_add_text_equiv(self):
        xml.add_pagexml_text(self.element, self.text, conf=0.96)
        child_tags = [child.tag for child in self.element]
        self.assertEqual(True, len(child_tags) > 0)
        self.assertEqual(True, all(tag.endswith('TextEquiv') for tag in child_tags))

    def test_element_cannot_add_multiple_text_equivs(self):
        xml.add_pagexml_text(self.element, self.text, conf=0.96)
        error = None
        try:
            xml.add_pagexml_text(self.element, self.text, conf=0.96)
        except ValueError as err:
            error = err
        self.assertNotEqual(None, error)

    def test_element_add_text_equiv_with_conf(self):
        xml.add_pagexml_text(self.element, self.text, conf=0.96)
        text_equiv = self.element.find(xml.PAGE + 'TextEquiv')
        self.assertEqual(str(0.96), text_equiv.attrib['conf'])

    def test_element_add_text_adds_equiv_types(self):
        xml.add_pagexml_text(self.element, self.text, conf=0.96)
        expected_tags = [xml.PAGE + 'Unicode', xml.PAGE + 'PlainText']
        text_equiv = self.element.find(xml.PAGE + 'TextEquiv')
        child_tags = [child.tag for child in text_equiv]
        self.assertEqual(expected_tags, child_tags)

    def test_element_add_text_adds_unicode(self):
        xml.add_pagexml_text(self.element, self.text, conf=0.96)
        text_equiv = self.element.find(xml.PAGE + 'TextEquiv')
        unicode = text_equiv.find(xml.PAGE + 'Unicode')
        self.assertNotEqual(None, unicode)
        self.assertEqual(self.text, unicode.text)

    def test_element_add_text_adds_plaintext(self):
        xml.add_pagexml_text(self.element, self.text, conf=0.96)
        plaintext = self.element.find(xml.PAGE + 'TextEquiv')
        plaintext = plaintext.find(xml.PAGE + 'PlainText')
        self.assertNotEqual(None, plaintext)
        self.assertEqual(self.text, plaintext.text)


class TestMakeWord(TestCase):

    def test_make_word_with_confidence(self):
        word = xml.make_pagexml_element('Word', text='word', conf=0.19)
        text_equiv = word.find(xml.PAGE + 'TextEquiv')
        self.assertEqual(False, 'conf' in word.attrib)
        self.assertEqual(True, 'conf' in text_equiv.attrib)

    def test_make_word_with_text(self):
        word = xml.make_pagexml_element('Word', text='word', conf=0.19)
        unicode = word.find(xml.PAGE + 'TextEquiv').find(xml.PAGE + 'Unicode')
        self.assertEqual('word', unicode.text)


class TestMakeTextLine(TestCase):

    def test_make_line_with_confidence(self):
        line = xml.make_pagexml_element('TextLine', text='line', conf=0.19)
        text_equiv = line.find(xml.PAGE + 'TextEquiv')
        self.assertEqual(False, 'conf' in line.attrib)
        self.assertEqual(True, 'conf' in text_equiv.attrib)

    def test_make_line_with_text(self):
        line = xml.make_pagexml_element('TextLine', text='line', conf=0.19)
        unicode = line.find(xml.PAGE + 'TextEquiv').find(xml.PAGE + 'Unicode')
        self.assertEqual('line', unicode.text)


class TestMakeTextRegion(TestCase):

    def test_can_make_textregion_with_id(self):
        tr = xml.make_pagexml_element('TextRegion', ele_id='tr-15')
        self.assertEqual('tr-15', tr.attrib['id'])

    def test_textregion_has_no_text_equiv(self):
        tr = xml.make_pagexml_element('TextRegion', text='region text')
        text_equiv = tr.find(xml.PAGE + 'TextEquiv')
        self.assertEqual(None, text_equiv)

    def test_making_textregion_with_text_adds_line(self):
        tr = xml.make_pagexml_element('TextRegion', text='region text')
        text_equiv = tr.find(xml.PAGE + 'TextLine')
        self.assertNotEqual(None, text_equiv)


class TestMakeTableRegion(TestCase):

    def test_can_make_tableregion_with_id(self):
        tr = xml.make_pagexml_element('TableRegion', ele_id='tr-15')
        self.assertEqual('tr-15', tr.attrib['id'])

    def test_tableregion_has_no_table_cell(self):
        tr = xml.make_pagexml_element('TableRegion')
        table_cell = tr.find(xml.PAGE + 'TableCell')
        self.assertEqual(None, table_cell)

    def test_tableregion_can_have_table_cell(self):
        tr = xml.make_pagexml_element('TableRegion')
        cell = xml.add_pagexml_sub_element(tr, 'TableCell')
        self.assertNotEqual(None, cell)

    def test_tableregion_cannot_have_textline(self):
        tr = xml.make_pagexml_element('TableRegion')
        self.assertRaises(TypeError, xml.add_pagexml_sub_element, tr, 'TextLine')

    def test_can_make_table_cell_with_row(self):
        attrs = {'row': 0, 'col': 0}
        tr = xml.make_pagexml_element('TableCell', attributes=attrs)
        self.assertEqual(str(attrs['row']), tr.attrib['row'])

    def test_can_make_table_cell_with_col(self):
        attrs = {'row': 0, 'col': 0}
        tr = xml.make_pagexml_element('TableCell', attributes=attrs)
        self.assertEqual(str(attrs['col']), tr.attrib['col'])

    def test_make_table_cell_adds_no_text_line(self):
        tr = xml.make_pagexml_element('TableCell')
        text_line = tr.find(xml.PAGE + 'TextLine')
        self.assertEqual(None, text_line)

    def test_make_table_cell_can_add_text_line(self):
        tr = xml.make_pagexml_element('TableCell')
        xml.add_pagexml_sub_element(tr, 'TextLine', text='first cell')
        text_line = tr.find(xml.PAGE + 'TextLine')
        self.assertNotEqual(None, text_line)

    def test_make_table_cell_can_add_cornerpoints(self):
        tr = xml.make_pagexml_element('TableCell')
        xml.add_pagexml_sub_element(tr, 'CornerPts', text='0 1 2 3')
        cp = tr.find(xml.PAGE + 'CornerPts')
        self.assertNotEqual(None, cp)

    """ skip because xml export shouldn't do this validation
    def test_make_table_cell_cannot_add_cornerpoints_with_text(self):
        tr = xml.make_pagexml_element('TableCell')
        self.assertRaises(ValueError, xml.add_pagexml_sub_element, tr, 'CornerPts', text='some text')
    """


class TestExportTableRegion(TestCase):

    def setUp(self) -> None:
        self.line = pdm.PageXMLTextLine(doc_id='line1', text='first cell')
        self.cell = pdm.PageXMLTableCell(doc_id='cell1', row=0, col=0,
                                         lines=[self.line], cornerpoints="1 2 3 4")
        self.row = pdm.PageXMLTableRow(doc_id='row1', cells=[self.cell])
        self.table = pdm.PageXMLTableRegion(doc_id='table1', rows=[self.row])

    def test_table_exports_to_page_with_table(self):
        pagexml = self.table.to_pagexml()
        table_region = pagexml.find(f".//{xml.PAGE + 'TableRegion'}")
        self.assertNotEqual(None, table_region)

    def test_cell_exports_to_page_with_table(self):
        pagexml = self.cell.to_pagexml()
        table_region = pagexml.find(f".//{xml.PAGE + 'TableRegion'}")
        self.assertNotEqual(None, table_region)

    def test_cell_exports_to_page_with_cell(self):
        pagexml = self.cell.to_pagexml()
        table_cell = pagexml.find(f".//{xml.PAGE + 'TableCell'}")
        self.assertNotEqual(None, table_cell)

    def test_cell_exports_to_page_with_text_line(self):
        pagexml = self.cell.to_pagexml()
        line = pagexml.find(f".//{xml.PAGE + 'TextLine'}")
        self.assertNotEqual(None, line)

    def test_cell_exports_to_page_with_cornerpoints(self):
        pagexml = self.cell.to_pagexml()
        cornerpoints = pagexml.find(f".//{xml.PAGE + 'CornerPts'}")
        self.assertNotEqual(None, cornerpoints)

    def test_row_exports_to_page_with_table(self):
        pagexml = self.row.to_pagexml()
        table_region = pagexml.find(f".//{xml.PAGE + 'TableRegion'}")
        self.assertNotEqual(None, table_region)

    def test_row_exports_to_page_with_cell(self):
        pagexml = self.row.to_pagexml()
        table_cell = pagexml.find(f".//{xml.PAGE + 'TableCell'}")
        self.assertNotEqual(None, table_cell)

    def test_row_exports_to_page_with_text_line(self):
        pagexml = self.row.to_pagexml()
        line = pagexml.find(f".//{xml.PAGE + 'TextLine'}")
        self.assertNotEqual(None, line)
