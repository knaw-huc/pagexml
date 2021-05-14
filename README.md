# pagexml
Utility functions for reading [PageXML](https://www.primaresearch.org/tools/PAGELibraries) files

## Installing

```commandline
pip install git+https://github.com/knaw-huc/pagexml
```

## Usage

```python
from pagexml.parser import parse_pagexml_file
from pagexml.model.physical_document_model import pretty_print_textregion

filepath = 'example.xml'
scan = parse_pagexml_file(filepath)
pretty_print_textregion(scan, print_stats=True)
```
