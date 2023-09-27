import unittest
from unittest.mock import Mock

from pagexml.model.physical_document_model import Coords, StructureDoc, PhysicalStructureDoc, LogicalStructureDoc


class TestCoords(unittest.TestCase):
    def test_valid_points(self):
        coords = Coords([(0, 0), (1, 1), (2, 2)])
        self.assertEqual([(0, 0), (1, 1), (2, 2)], coords.points)
        self.assertEqual(0, coords.left)
        self.assertEqual(0, coords.top)
        self.assertEqual(2, coords.right)
        self.assertEqual(2, coords.bottom)
        self.assertEqual(2, coords.width)
        self.assertEqual(2, coords.height)
        self.assertEqual({'x': 0, 'y': 0, 'w': 2, 'h': 2}, coords.box)
        self.assertEqual({'type': 'coords', 'points': [(0, 0), (1, 1), (2, 2)]}, coords.json)

    def test_point_string(self):
        coords = Coords([(0, 0), (1, 1), (2, 2)])
        self.assertEqual('0,0 1,1 2,2', coords.point_string)

    def test_invalid_points(self):
        with self.assertRaises(ValueError):
            coords = Coords('invalid points')

    def test_valid_points_from_str(self):
        coords = Coords('1216,1119 1205,1109 1202,1109 1198,1112 1195,1112 1191,1116 1164,1116 1160,1119 1147,1119'
                        ' 1143,1123 1126,1123 1123,1126 1102,1126 1098,1130 1074,1130 1071,1133 1016,1133 1012,1136'
                        ' 964,1136 961,1140 957,1140 954,1143 940,1143 937,1147 930,1147 926,1150 916,1150 912,1154'
                        ' 899,1154 895,1157 888,1157 885,1160 882,1160 878,1164 875,1164 857,1181 847,1181 840,1188'
                        ' 837,1188 833,1191 830,1191 826,1195 823,1195 820,1198 816,1198 813,1202 809,1202 795,1216'
                        ' 795,1229 799,1229 802,1233 813,1233 816,1236 875,1236 878,1240 895,1240 899,1243 923,1243'
                        ' 926,1247 1036,1247 1040,1243 1147,1243 1150,1240 1181,1240 1185,1236 1209,1236 1212,1233'
                        ' 1216,1233 1219,1229 1219,1226 1222,1222 1222,1216 1219,1212 1219,1209 1216,1205 1216,1150'
                        ' 1219,1147 1219,1143 1216,1140')
        x = [p[0] for p in coords.points]
        print(min(x), max(x))
        y = [p[1] for p in coords.points]
        print(min(y), max(y))

        self.assertEqual(795, coords.left)
        self.assertEqual(1109, coords.top)
        self.assertEqual(1222, coords.right)
        self.assertEqual(1247, coords.bottom)
        self.assertEqual(427, coords.width)
        self.assertEqual(138, coords.height)
        self.assertEqual({'x': 795, 'y': 1109, 'w': 427, 'h': 138}, coords.box)


class TestStructureDoc(unittest.TestCase):

    def test_init(self):
        doc = StructureDoc()
        self.assertIsNone(doc.id)
        self.assertIsNone(doc.type)
        self.assertEqual('doc', doc.main_type)
        self.assertEqual({}, doc.metadata)
        self.assertEqual({}, doc.reading_order)
        self.assertEqual({}, doc.reading_order_number)
        self.assertIsNone(doc.parent)

    def test_set_parent(self):
        parent_doc = StructureDoc(doc_id='parent_doc')
        child_doc = StructureDoc(doc_id='child_doc')

        child_doc.set_parent(parent_doc)

        self.assertEqual(parent_doc, child_doc.parent)
        self.assertEqual('doc', child_doc.metadata['parent_type'])
        self.assertEqual('parent_doc', child_doc.metadata['parent_id'])

    def test_add_type(self):
        doc = StructureDoc(doc_type='doc')

        # Add a new type
        doc.add_type('report')
        self.assertEqual(['doc', 'report'], doc.type)

        # Add the same type twice
        doc.add_type('doc')
        self.assertEqual(['doc', 'report'], doc.type)

        # Add multiple types at once
        doc.add_type(['pdf', 'ocr'])
        self.assertEqual(['doc', 'report', 'pdf', 'ocr'], doc.type)

    def test_remove_type(self):
        doc = StructureDoc(doc_type=['doc', 'report'])

        # Remove an existing type
        doc.remove_type('doc')
        self.assertEqual('report', doc.type)

        # Remove a non-existing type
        doc.remove_type('pdf')
        self.assertEqual('report', doc.type)

        # Remove multiple types at once
        doc.remove_type(['report', 'ocr'])
        self.assertEqual([], doc.type)

    def test_has_type(self):
        doc = StructureDoc(doc_type=['doc', 'report'])

        # Check for an existing type
        self.assertTrue(doc.has_type('doc'))

        # Check for a non-existing type
        self.assertFalse(doc.has_type('pdf'))

    def test_types(self):
        doc = StructureDoc(doc_type=['doc', 'report'])

        # Get all types
        self.assertEqual({'doc', 'report'}, doc.types)

    def test_set_as_parent(self):
        parent_doc = StructureDoc(doc_id='parent_doc')
        child_doc1 = StructureDoc(doc_id='child_doc1')
        child_doc2 = StructureDoc(doc_id='child_doc2')
        child_docs = [child_doc1, child_doc2]

        parent_doc.set_as_parent(child_docs)

        self.assertEqual(child_doc1.parent, parent_doc)
        self.assertEqual(child_doc2.parent, parent_doc)
        self.assertEqual(child_doc1.metadata['parent_type'], 'doc')
        self.assertEqual(child_doc1.metadata['parent_id'], 'parent_doc')
        self.assertEqual(child_doc2.metadata['parent_type'], 'doc')
        self.assertEqual(child_doc2.metadata['parent_id'], 'parent_doc')

    def test_add_parent_id_to_metadata(self):
        parent_doc = StructureDoc(doc_id='parent_doc')
        child_doc = StructureDoc(doc_id='child_doc')

        child_doc.set_parent(parent_doc)
        child_doc.add_parent_id_to_metadata()

        self.assertEqual('doc', child_doc.metadata['parent_type'])
        self.assertEqual('parent_doc', child_doc.metadata['parent_id'])
        self.assertEqual('parent_doc', child_doc.metadata['doc_id'])

    def test_json(self):
        doc = StructureDoc(doc_id='doc1', doc_type='book', metadata={'title': 'The Great Gatsby'})
        json_data = doc.json
        self.assertEqual('doc1', json_data['id'])
        self.assertEqual('book', json_data['type'])
        self.assertEqual({'title': 'The Great Gatsby'}, json_data['metadata'])
        self.assertEqual({}, json_data.get('reading_order', {}))

        doc.reading_order = {1: 'page1', 2: 'page2', 3: 'page3'}
        json_data = doc.json
        self.assertEqual({1: 'page1', 2: 'page2', 3: 'page3'}, json_data['reading_order'])


class TestPhysicalStructureDoc(unittest.TestCase):

    def setUp(self):
        self.metadata = {'author': 'Jane Doe'}
        self.coords = Coords([(0, 0), (0, 10), (10, 10), (10, 0)])
        self.doc = PhysicalStructureDoc(doc_id='doc1', doc_type='book', metadata=self.metadata, coords=self.coords)

    def test_init(self):
        self.assertEqual('doc1', self.doc.id)
        self.assertEqual('book', self.doc.type)
        self.assertEqual(self.metadata, self.doc.metadata)
        self.assertEqual(self.coords, self.doc.coords)

    def test_set_derived_id(self):
        parent = Mock(spec=StructureDoc)
        parent.id = 'parent_doc'
        self.doc.set_derived_id(parent.id)
        self.assertEqual('parent_doc-physical_structure_doc-0-0-10-10', self.doc.id)

    def test_json(self):
        expected_json = {
            'id': 'doc1',
            'type': 'book',
            'metadata': {'author': 'Jane Doe'},
            'coords': [(0, 0), (0, 10), (10, 10), (10, 0)]
        }
        self.assertEqual(expected_json, self.doc.json)


class TestLogicalStructureDoc(unittest.TestCase):
    def setUp(self):
        self.doc = LogicalStructureDoc(
            doc_id='doc_001',
            doc_type='article',
            metadata={'author': 'John Doe'},
            lines=[],
            text_regions=[]
        )

    def test_set_logical_parent(self):
        parent_doc = LogicalStructureDoc(
            doc_id='doc_002',
            doc_type='journal',
            metadata={'publisher': 'New York Times'},
            lines=[],
            text_regions=[]
        )
        self.doc.set_logical_parent(parent_doc)
        self.assertEqual(self.doc.logical_parent, parent_doc)
        self.assertIn('logical_parent_type', self.doc.metadata)
        self.assertEqual('doc', self.doc.metadata['logical_parent_type'])
        self.assertIn('logical_parent_id', self.doc.metadata)
        self.assertEqual('doc_002', self.doc.metadata['logical_parent_id'])
        self.assertIn('doc_id', self.doc.metadata)
        self.assertEqual('doc_002', self.doc.metadata['doc_id'])

    def test_add_logical_parent_id_to_metadata(self):
        parent_doc = LogicalStructureDoc(
            doc_id='doc_002',
            doc_type='journal',
            metadata={'publisher': 'New York Times'},
            lines=[],
            text_regions=[]
        )
        self.doc.logical_parent = parent_doc
        self.doc.add_logical_parent_id_to_metadata()
        self.assertIn('logical_parent_type', self.doc.metadata)
        self.assertEqual('doc', self.doc.metadata['logical_parent_type'])
        self.assertIn('logical_parent_id', self.doc.metadata)
        self.assertEqual('doc_002', self.doc.metadata['logical_parent_id'])
        self.assertIn('doc_id', self.doc.metadata)
        self.assertEqual('doc_002', self.doc.metadata['doc_id'])


if __name__ == '__main__':
    unittest.main()
