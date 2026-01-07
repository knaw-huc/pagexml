from unittest import TestCase

import pagexml.model.physical_document_model as pdm


class TestPhysicalStructureDoc(TestCase):

    def setUp(self) -> None:
        self.scan_id = "INST_ID_3209_0002"
        self.coords = pdm.Coords([(10, 10), (20, 10), (20, 20), (10, 20)])
        self.region = pdm.PageXMLRegion(doc_id='r1', coords=self.coords)

    def test_set_derived_id_from_scan(self):
        self.region.set_derived_id(self.scan_id)
        self.assertEqual(f"{self.scan_id}-region-{self.coords.box_string}", self.region.id)

    def test_set_derived_id_from_region(self):
        parent_id = f"{self.scan_id}-page-0-0-400-400"
        self.region.set_derived_id(parent_id)
        self.assertEqual(f"{self.scan_id}-region-{self.coords.box_string}", self.region.id)
