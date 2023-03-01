from typing import Tuple

import pagexml.model.physical_document_model as pdm


def is_above_point(page_doc: pdm.PageXMLDoc, point: Tuple[int, int]):
    if page_doc.coords.bottom < point[1]:
        return True
    elif page_doc.coords.top > point[1]:
        return False
    elif isinstance(page_doc, pdm.PageXMLTextRegion) and page_doc.num_lines > 0:
        first_line = page_doc.lines[0]
        last_line = page_doc.lines[-1]
        if first_line.baseline is not None and first_line.baseline.top > point[1]:
            return False
        if last_line.baseline is not None and last_line.baseline.bottom < point[1]:
            return True


def split_horizontally(page_doc: pdm.PageXMLTextRegion, point: Tuple[int, int]):
    above = []
    below = []
    for tr in page_doc.text_regions:
        if tr.coords.bottom < point[1]:
            above.append(tr)
        elif tr.coords.top > point[1]:
            below.append(tr)
