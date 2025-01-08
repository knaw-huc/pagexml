from unittest import TestCase

from lxml import etree

from pagexml.model.coords import Coords, Baseline
import pagexml.model.pagexml_document_model as pdm
import pagexml.model.xml as xml


class TestPageXMLWord(TestCase):

    def test_word_can_export_to_valid_pagexml(self) -> None:
        word = pdm.PageXMLWord(doc_id='word-1', text='word')
        word_xml = word.to_pagexml()
        self.assertEqual(xml.PAGE + 'PcGts', word_xml.tag)

    def test_word_pagexml_export_includes_word(self) -> None:
        word = pdm.PageXMLWord(doc_id='word-1', text='testword')
        doc_pagexml = word.to_pagexml()
        word_xml = doc_pagexml.find(f".//{xml.PAGE + 'Word'}")
        unicode_xml = word_xml.find(f".//{xml.PAGE + 'Unicode'}")
        self.assertEqual('testword', unicode_xml.text)

    def test_textline_can_export_to_valid_pagexml(self) -> None:
        line = pdm.PageXMLTextLine(doc_id='line-1', text='line')
        doc_pagexml = line.to_pagexml()
        line_xml = doc_pagexml.find(f".//{xml.PAGE + 'TextLine'}")
        self.assertNotEqual(None, line_xml)

    def test_textregion_can_export_to_valid_pagexml(self) -> None:
        tr = pdm.PageXMLTextRegion(doc_id='tr-1', text='tr')
        doc_pagexml = tr.to_pagexml()
        tr_xml = doc_pagexml.find(f".//{xml.PAGE + 'TextRegion'}")
        self.assertNotEqual(None, tr_xml)

    def test_column_can_export_to_valid_pagexml(self) -> None:
        col = pdm.PageXMLColumn(doc_id='col-1')
        doc_pagexml = col.to_pagexml()
        col_xml = doc_pagexml.find(f".//{xml.PAGE + 'TextRegion'}")
        self.assertEqual('column', col_xml.attrib['type'])
