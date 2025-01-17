import unittest

import pagexml.parser as parser
import pagexml.model.physical_document_model as pdm


class TestParser(unittest.TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example.xml'
        self.page_doc = parser.parse_pagexml_file(self.page_file, custom_tags=['date', 'place'])

    def test_reading_order_is_set(self):
        self.assertEqual(2, len(self.page_doc.reading_order))

    def test_reading_order_attributes_is_set(self):
        self.assertEqual(2, len(self.page_doc.reading_order_attributes))

    def test_parentage_is_set(self):
        for tr in self.page_doc.text_regions:
            self.assertEqual(True, tr.parent == self.page_doc)
            for line in tr.lines:
                self.assertEqual(True, line.parent == tr)

    def test_custom_attributes_are_property(self):
        for ti, tr in enumerate(self.page_doc.text_regions):
            with self.subTest(ti):
                self.assertTrue(len(tr.custom) > 0)

    def test_custom_index_property_is_integer(self):
        for ti, tr in enumerate(self.page_doc.text_regions):
            reading_orders = [c for c in tr.custom if c['tag_name'] == 'readingOrder']
            with self.subTest(ti):
                self.assertTrue(all(isinstance(tag['index'], int) for tag in reading_orders))

    def test_parsing_from_json_retains_stats(self):
        page_json = self.page_doc.json
        new_page = parser.parse_pagexml_from_json(page_json)
        for field in new_page.stats:
            self.assertEqual(True, new_page.stats[field] == self.page_doc.stats[field])

    def test_parsing_from_json_retains_type(self):
        page_json = self.page_doc.json
        new_page = parser.parse_pagexml_from_json(page_json)
        self.assertEqual(True, self.page_doc.main_type == new_page.main_type)

    def test_parsing_from_json_sets_parents(self):
        page_json = self.page_doc.json
        new_page = parser.parse_pagexml_from_json(page_json)
        for tr in new_page.text_regions:
            self.assertEqual(True, tr.parent == new_page)
            for line in tr.lines:
                self.assertEqual(True, line.parent == tr)

    def test_parsing_from_json_retains_table(self):
        scan_doc = parser.parse_pagexml_file('data/example_table.xml')
        doc_json = scan_doc.json
        new_scan = parser.json_to_pagexml_scan(doc_json)
        self.assertEqual(1, len(new_scan.table_regions))

    def test_parsing_from_json_retains_table(self):
        scan_doc = parser.parse_pagexml_file('data/example_table.xml')
        table = scan_doc.table_regions[0]
        doc_json = scan_doc.json
        new_scan = parser.json_to_pagexml_scan(doc_json)
        new_table = new_scan.table_regions[0]
        self.assertEqual(table.shape, new_table.shape)


class TestCustomParser(unittest.TestCase):

    def setUp(self) -> None:
        custom = ("readingOrder {index:0;} abbrev {offset:0; length:4;} "
                  "unclear {offset:0; length:4; continued:true;} "
                  "unclear {offset:4; length:5; continued:true;} "
                  "abbrev {offset:9; length:9;} "
                  "unclear {offset:9; length:9; continued:true;} "
                  "textStyle {offset:20; length:1;superscript:true;} "
                  "madeup {offset:20; length:1;imaginary_attribute:true;} "
                  "unclear {offset:18; length:33; continued:true;}")
        self.element = {
            '@custom': custom
        }

    def test_parse_metadata_element_list_single_type_as_list(self):
        metadata = parser.parse_custom_metadata_element_list(self.element['@custom'], 'readingOrder')
        self.assertEqual(1, len(metadata))

    def test_parse_metadata_element_list_multi_type_as_list(self):
        metadata = parser.parse_custom_metadata_element_list(self.element['@custom'], 'unclear')
        self.assertEqual(4, len(metadata))

    def test_parse_custom_attribute_part_returns_dict(self):
        attributes = parser.parse_custom_attribute_parts('offset:9; length:5')
        expected = {'offset': 9, 'length': 5}
        self.assertEqual(expected, attributes)

    """
    def test_parse_custom_attribute_part_returns_text_with_offset(self):
        text = "this is a text"
        attribute = parser.parse_custom_attribute_parts('offset:9; length:5', element_text=text)
        attrib_text = text[9:9+5]
        self.assertEqual(True, 'text' in attribute)
        self.assertEqual(attrib_text, attribute['text'])
    """

    def test_parse_custom_attributes_returns_list(self):
        attributes = parser.parse_custom_attributes('unclear {offset:9; length:5}')
        expected = [{'offset': 9, 'length': 5, 'tag_name': 'unclear'}]
        self.assertEqual(expected, attributes)

    def test_parse_custom_attributes_handles_semicolon_at_the_end(self):
        attributes = parser.parse_custom_attributes('unclear {offset:9; length:5;}')
        expected = [{'offset': 9, 'length': 5, 'tag_name': 'unclear'}]
        self.assertEqual(expected, attributes)

    def test_parse_custom_attributes_handles_arbitrary_whitespace(self):
        attributes = parser.parse_custom_attributes('unclear {offset: 9;  length :5 ;}')
        expected = [{'offset': 9, 'length': 5, 'tag_name': 'unclear'}]
        self.assertEqual(expected, attributes)

    def test_parse_custom_attributes_returns_list_with_repeated_elements(self):
        custom_string = 'unclear {offset:9; length:5} unclear {offset:16; length: 2;}'
        attributes = parser.parse_custom_attributes(custom_string)
        expected = [
            {'offset': 9, 'length': 5, 'tag_name': 'unclear'},
            {'offset': 16, 'length': 2, 'tag_name': 'unclear'}
        ]
        self.assertEqual(expected, attributes)

    def test_parse_custom_metadata_extracts_structure(self):
        custom = parser.parse_custom_metadata({"@custom": "structure {type: resolution}"})
        self.assertEqual(True, 'structure' in custom)
        expected = {'type': 'resolution'}
        self.assertEqual(expected, custom['structure'])
        expected = [{'type': 'resolution', 'tag_name': 'structure'}]
        self.assertEqual(expected, custom['custom_attributes'])

    def test_parse_custom_metadata_extracts_all_tag_types(self):
        custom = parser.parse_custom_metadata(self.element, custom_tags=['unclear'])
        tag_types = {'readingOrder', 'unclear', 'abbrev', 'textStyle', 'madeup'}
        self.assertEqual(tag_types, {attr['tag_name'] for attr in custom['custom_attributes']})

    """
    def test_parse_custom_metadata_returns_text_with_offset(self):
        text = "this is a text"
        custom = parser.parse_custom_metadata(self.element, element_text=text)
        attrib_text = text[0:4]
        unclears = [attr for attr in custom['custom_attributes'] if attr['tag_name'] == 'unclear']
        self.assertEqual(True, 'text' in unclears[0])
        self.assertEqual(attrib_text, unclears[0]['text'])
    """

    def test_parse_custom_metadata_extracts_unique_tag_as_dict(self):
        custom = parser.parse_custom_metadata(self.element)
        self.assertEqual(True, isinstance(custom['reading_order'], dict))

    def test_parse_custom_metadata_extracts_multiple_tags_of_same_type_as_list(self):
        custom = parser.parse_custom_metadata(self.element)
        unclear = [attr for attr in custom['custom_attributes'] if attr['tag_name'] == 'unclear']
        self.assertEqual(4, len(unclear))

    def test_parse_custom_metadata_extracts_all_tags(self):
        custom = parser.parse_custom_metadata(self.element)
        self.assertEqual(9, len(custom['custom_attributes']))


class TestTableParser(unittest.TestCase):

    def setUp(self) -> None:
        self.page_file = 'data/example_table.xml'
        self.page_doc = parser.parse_pagexml_file(self.page_file, custom_tags=['date', 'place'])

    def test_can_extract_table(self):
        self.assertEqual(1, len(self.page_doc.table_regions))

    def test_extracted_table_has_id(self):
        table = self.page_doc.table_regions[0]
        self.assertEqual('t1', table.id)

    def test_extracted_table_has_coords(self):
        table = self.page_doc.table_regions[0]
        self.assertTrue(len(table.coords.points) > 0)

    def test_extracted_table_has_rows(self):
        table = self.page_doc.table_regions[0]
        self.assertTrue(len(table.rows) > 0)

    def test_extracted_table_has_shape(self):
        table = self.page_doc.table_regions[0]
        self.assertEqual((len(table.rows), table.rows[0].num_columns), table.shape)

    def test_extracted_table_has_number_of_columns(self):
        table = self.page_doc.table_regions[0]
        self.assertTrue(len(table.rows) > 0)

    def test_extracted_table_can_return_cell_values(self):
        table = self.page_doc.table_regions[0]
        values = [cell_value for row_values in table.values for cell_value in row_values]
        self.assertTrue(all(isinstance(val, str) for val in values))

    def test_extracted_table_values_correspond_with_shape(self):
        table = self.page_doc.table_regions[0]
        values = [cell_value for row_values in table.values for cell_value in row_values]
        table_size = table.shape[0] * table.shape[1]
        self.assertEqual(table_size, len(values))

    def test_extracted_table_can_access_rows_by_index(self):
        table = self.page_doc.table_regions[0]
        row = table[2]
        self.assertTrue(isinstance(row, pdm.PageXMLTableRow))

    def test_extracted_table_row_has_cells(self):
        table = self.page_doc.table_regions[0]
        row = table[2]
        self.assertTrue(len(row.cells) > 0)

    def test_extracted_table_row_has_number_of_columns(self):
        table = self.page_doc.table_regions[0]
        row = table[2]
        self.assertTrue(2, row.num_columns)

    def test_extracted_table_row_can_access_cells_by_index(self):
        table = self.page_doc.table_regions[0]
        row = table[2]
        cell = row[1]
        self.assertTrue(isinstance(cell, pdm.PageXMLTableCell))

    def test_extracted_table_cell_has_text_lines(self):
        table = self.page_doc.table_regions[0]
        row = table[2]
        cell = row[0]
        self.assertEqual(1, len(cell.lines))

    def test_extracted_table_cell_sets_value(self):
        table = self.page_doc.table_regions[0]
        row = table[2]
        cell = row[0]
        self.assertEqual('3-8', cell.value)


if __name__ == '__main__':
    unittest.main()
