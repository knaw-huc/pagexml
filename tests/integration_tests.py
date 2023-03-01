import glob
import traceback

from icecream import ic

from pagexml.helper.pagexml_helper import pretty_print_textregion
from pagexml.parser import parse_pagexml_file


def main():
    pagexml_basedir = "../golden-agents/pagexml/"
    test_dirs = glob.glob(pagexml_basedir + '[0-9AN]*')
    # test_dirs = glob.glob(pagexml_basedir + '10025*')
    assert len(test_dirs) > 0
    for d in test_dirs:
        files = glob.glob(d + '/*.xml')
        assert len(files) > 0
        for file_name in files:
            ic(file_name)
            print(file_name)
            try:
                scan_doc = parse_pagexml_file(file_name)
                pretty_print_textregion(scan_doc, print_stats=True)
            except Exception:
                print(traceback.format_exc())


if __name__ == '__main__':
    main()
