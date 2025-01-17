# pagexml-tools

[![GitHub Actions](https://github.com/knaw-huc/pagexml/workflows/tests/badge.svg)](https://github.com/knaw-huc/pagexml/actions)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation Status](https://readthedocs.org/projects/pagexml/badge/?version=latest)](https://pagexml.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/pagexml-tools)](https://pypi.org/project/pagexml-tools/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pagexml-tools)](https://pypi.org/project/pagexml-tools/)

Utility functions for reading [PageXML](https://www.primaresearch.org/tools/PAGELibraries) files

## installing

### using poetry

```commandline
poetry add pagexml-tools
```

### using pip

```commandline
pip install pagexml-tools
```

## Using

PageXML-tools contains functions for parsing and for a range of analysis tasks.

### Parsing PageXML files and the Physical Document model

There is a tutorial that demonstrates the [physical document model API](./notebooks/Demo-understanding-the-document-model.ipynb)

PageXML-tools contains basic functionality for parsing a PageXML file that returns
a document model representing the content of the file. The HTR/OCR process that generates
PageXML, recognises text in an image of a physical document.

```python
from pagexml.parser import parse_pagexml_file

pagexml_file = "path/to/pagexml_file.xml"

page_doc = parse_pagexml_file(pagexml_file)

# a page document has an ID
print(page_doc.id)

# print descriptive statistics
print(page_doc.stats)

# iterative over text regions and lines
for tr in page_doc.text_regions:
    # a text_region has an ID and a bounding box derived from its coordinates
    print(tr.id, tr.coords.box)
    # a text_region can have sub-text_regions and lines
    for line in tr.lines:
        # a line has an ID, coordinates and text
        print(line.id, line.coords.box, line.text)
```

###

In addition to the basic parsing and handling of PageXML output, there is
functionality to support a range of tasks:

- reading sets of PageXML files from a archive (tar, zip) file ([tutorial](./notebooks/Demo-reading-pagexml-files-from-archive.ipynb)),
- searching in text ([keyword in context](./notebooks/Demo-text-search-simple.ipynb), [keywords or fuzzy search](./notebooks/Demo-text-search-in-pagexml-archive.ipynb))
- reading and working with tables ([table processing](./notebooks/Demo-table-processing.ipynb))
- classifying physical document types in a large set of PageXML documents ([tutorial](./notebooks/Demo-analysing-scan-characteristics.ipynb)),
- checking the quality of the HTR/OCR process ([tutorial](./notebooks/Demo-analysing-scan-characteristics-checking-quality.ipynb)),
- comparing subsets ([tutorial](./notebooks/Demo-analysing-scan-characteristics-comparing-subsets.ipynb)),
- identifying document sections in sequences of PageXML documents ([tutorial](./notebooks/Demo-analysing-scan-characteristics-book-sections.ipynb)),
- turning text lines into running text ([tutorial](./notebooks/Demo-from-lines-to-running-text.ipynb)),
- supporting different reading orders ([tutorial](./notebooks/Demo-sorting.ipynb)),
- reinterpreting and restructuring text regions and lines ([tutorial](./notebooks/Demo-restructuring-documents.ipynb)),
- turning physical structure into logical structure,

----

[USAGE](https://pagexml.readthedocs.io/en/latest/) |
[CONTRIBUTING](CONTRIBUTING.md) |
[LICENSE](LICENSE)
