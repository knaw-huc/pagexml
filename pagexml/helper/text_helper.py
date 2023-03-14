import gzip
import re
from typing import Dict, Generator, Iterable, List, Set, Tuple, Union

from fuzzy_search.similarity import SkipgramSimilarity

import pagexml.model.physical_document_model as pdm
import pagexml.parser as parser


def read_lines_from_line_files(pagexml_line_files: Union[str, List[str]]) -> Generator[str, None, None]:
    if isinstance(pagexml_line_files, str):
        pagexml_line_files = [pagexml_line_files]
    for line_file in pagexml_line_files:
        with gzip.open(line_file, 'rt') as fh:
            for line in fh:
                yield line


def get_bbox(doc: pdm.PageXMLDoc):
    if doc.coords.points is None:
        return None
    return f"{doc.coords.x},{doc.coords.y},{doc.coords.w},{doc.coords.h}"


def get_line_format_json(page_doc: pdm.PageXMLTextRegion,
                         use_outer_textregions: bool = False,
                         add_bounding_box: bool = False) -> Generator[Dict[str, any], None, None]:
    if page_doc.num_text_regions == 0 and page_doc.num_lines > 0:
        trs = [page_doc]
    elif use_outer_textregions is True:
        trs = page_doc.text_regions
    else:
        trs = page_doc.get_inner_text_regions()
    for tr in trs:
        for line in tr.get_lines():
            json_doc = {
                'doc_id': page_doc.id,
                'textregion_id': tr.id,
                'line_id': line.id,
                'text': line.text
            }
            if add_bounding_box is True:
                json_doc['doc_box'] = get_bbox(page_doc)
                json_doc['textregion_box'] = get_bbox(tr)
                json_doc['line_box'] = get_bbox(line)
            yield json_doc
    return None


def get_line_format_tsv(page_doc: pdm.PageXMLTextRegion,
                        headers: List[str],
                        use_outer_textregions: bool = False,
                        add_bounding_box: bool = False) -> Generator[List[str], None, None]:
    for line_json in get_line_format_json(page_doc, use_outer_textregions=use_outer_textregions,
                                          add_bounding_box=add_bounding_box):
        line_list = [line_json[header] for header in headers]
        yield [val if val is not None else '' for val in line_list]


def make_list(var) -> list:
    return var if isinstance(var, list) else [var]


class LineReader(Iterable):

    def __init__(self, pagexml_files: Union[str, List[str]] = None,
                 pagexml_docs: Union[pdm.PageXMLDoc, List[pdm.PageXMLDoc]] = None,
                 pagexml_line_files: Union[str, List[str]] = None,
                 line_file_headers: List[str] = None,
                 has_headers: bool = True,
                 use_outer_textregions: bool = False,
                 add_bounding_box: bool = False,
                 groupby: str = None):
        """A Line Reader class that turns a list of PageXML files, PageXML objects,
        or a PageXML line file into an iterable over the lines.

        :param pagexml_files: an optional list of PageXML filenames
        :type pagexml_files: List[str]
        :param pagexml_docs: an optional list of PageXMLDoc objects
        :type pagexml_docs: List[PageXMLDoc]
        :param pagexml_line_files: an optional list of PageXML line files
        :type pagexml_line_files: List[str]
        :param line_file_headers: an optional list of column headers to use for headerless line files
        :type line_file_headers: List[str]
        :param has_headers: whether the pagexml_line_files have a header line
        :type has_headers: bool
        :param use_outer_textregions: use ID of outer text regions (when True) otherwise ID of inner
            text regions
        :type use_outer_textregions: bool
        :param add_bounding_box: whether the line format output should include bounding boxes of each element
        :type add_bounding_box: bool
        :param groupby: group lines by 'doc_id' or 'textregion_id'
        :type groupby: str
        """
        self.pagexml_files = []
        self.pagexml_docs = []
        self.pagexml_line_files = []
        self.line_file_headers = line_file_headers
        if line_file_headers is not None:
            self.has_header = False
        else:
            self.has_headers = has_headers
        self.use_outer_textregions = use_outer_textregions
        self.add_bounding_box = add_bounding_box
        self.groupby = groupby
        if pagexml_files is None and pagexml_docs is None and pagexml_line_files is None:
            raise TypeError(f"MUST use one of the following optional arguments: "
                            f"'pagexml_files', 'pagexml_docs' or 'pagexml_line_file'.")
        if pagexml_line_files:
            self.pagexml_line_files = make_list(pagexml_line_files)
        if pagexml_files:
            self.pagexml_files = make_list(pagexml_files)
        if pagexml_docs:
            self.pagexml_docs = make_list(pagexml_docs)

    def __iter__(self) -> Generator[Dict[str, str], None, None]:
        if self.groupby is None:
            for line in self._iter():
                yield line
        else:
            lines = []
            prev_id = None
            for line in self._iter():
                if line[self.groupby] != prev_id:
                    if len(lines) > 0:
                        yield lines
                    lines = []
                lines.append(line)
                prev_id = line[self.groupby]
            if len(lines) > 0:
                yield lines

    def _iter(self) -> Generator[Dict[str, any], None, None]:
        if self.pagexml_line_files:
            for line in self._iter_from_line_file():
                yield line
        if len(self.pagexml_files) > 0:
            pagexml_doc_iterator = parser.parse_pagexml_files(self.pagexml_files)
            for line in self._iter_from_pagexml_docs(pagexml_doc_iterator):
                yield line
        if len(self.pagexml_docs) > 0:
            self._iter_from_pagexml_docs(self.pagexml_docs)

    def _iter_from_pagexml_docs(self, pagexml_doc_iterator) -> Generator[Dict[str, any], None, None]:
        for pagexml_doc in pagexml_doc_iterator:
            for line in get_line_format_json(pagexml_doc, use_outer_textregions=self.use_outer_textregions,
                                             add_bounding_box=self.add_bounding_box):
                yield line

    def _iter_from_line_file(self) -> Generator[Dict[str, any], None, None]:
        line_iterator = read_lines_from_line_files(self.pagexml_line_files)
        if self.has_headers is True:
            header_line = next(line_iterator)
            self.line_file_headers = header_line.strip().split('\t')
        elif self.line_file_headers is None:
            self.line_file_headers = [
                'doc_id', 'textregion_id', 'line_id', 'text'
            ]
            if self.add_bounding_box is True:
                self.line_file_headers.extend(['doc_box', 'textregion_box', 'line_box'])
        for li, line in enumerate(line_iterator):
            try:
                cols = line.strip('\r\n').split('\t')
                yield {header: cols[hi] for hi, header in enumerate(self.line_file_headers)}
            except (IndexError, ValueError):
                print(f"line {li} in file {self.pagexml_line_files}:")
                line = line.strip('\n')
                print(f'#{line}#')
                raise


def transform_box_to_coords(box_string: str) -> pdm.Coords:
    x, y, w, h = [int(part) for part in box_string.split(',')]
    points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    return pdm.Coords(points)


def read_pagexml_docs_from_line_file(line_files: Union[str, List[str]], has_headers: bool = True,
                                     headers: List[str] = None,
                                     add_bounding_box: bool = False) -> Generator[pdm.PageXMLTextRegion, None, None]:
    """Read lines from one or more PageXML line format files and return them
    as PageXMLTextLine objects, grouped by their PageXML document."""
    line_iterator = LineReader(pagexml_line_files=line_files, line_file_headers=headers,
                               has_headers=has_headers, add_bounding_box=add_bounding_box)
    curr_doc = None
    curr_tr = None
    for li, line_dict in enumerate(line_iterator):
        doc_coords, tr_coords, line_coords = None, None, None
        if add_bounding_box is True:
            doc_coords = transform_box_to_coords(line_dict['doc_box'])
            tr_coords = transform_box_to_coords(line_dict['textregion_box'])
            line_coords = transform_box_to_coords(line_dict['line_box'])
        if curr_doc is None or curr_doc.id != line_dict['doc_id']:
            if curr_doc is not None:
                yield curr_doc
            curr_doc = pdm.PageXMLScan(doc_id=line_dict['doc_id'], coords=doc_coords)
            curr_tr = None
        if curr_tr is None or curr_tr.id != line_dict['textregion_id']:
            curr_tr = pdm.PageXMLTextRegion(doc_id=line_dict['textregion_id'], coords=tr_coords)
            curr_doc.add_child(curr_tr)
            # print(f'creating tr with id {curr_tr.id} and appending to doc with id {curr_doc.id}')
        line = pdm.PageXMLTextLine(doc_id=line_dict['line_id'],
                                   text=line_dict['text'], coords=line_coords)
        curr_tr.add_child(line)
        # print('curr_doc:', curr_doc.id, '\tline doc_id:', line_dict['doc_id'])
        # print('curr_tr:', curr_tr.id, '\tline textregion_id:', line_dict['textregion_id'])
    if curr_doc is not None:
        yield curr_doc


def make_line_format_file(page_docs: Iterable[pdm.PageXMLTextRegion],
                          line_format_file: str, headers: List[str] = None,
                          use_outer_textregions: bool = False, add_bounding_box: bool = False):
    """Create a line format file for a list of PageXMLDoc objects."""
    if headers is None:
        headers = [
            'doc_id', 'textregion_id', 'line_id', 'text',
            'doc_box', 'textregion_box', 'line_box'
        ]
    with gzip.open(line_format_file, 'wt') as fh:
        header_string = '\t'.join(headers)
        fh.write(f'{header_string}\n')
        for page_doc in page_docs:
            for line_tsv in get_line_format_tsv(page_doc, headers,
                                                use_outer_textregions=use_outer_textregions,
                                                add_bounding_box=add_bounding_box):
                line_string = '\t'.join(line_tsv)
                fh.write(f'{line_string}\n')


# SPLIT_PATTERN = r'[ \.,\!\?\(\)\[\]\{\}"\':;]+'
# def get_line_words(line, split_pattern: str = SPLIT_PATTERN) -> List[str]:
#     return [word for word in re.split(split_pattern, line) if word != '']


def get_line_words(line: Union[pdm.PageXMLTextLine, str], word_break_chars: Union[str, Set[str]] = '-') -> List[str]:
    """Return a list of the words for a given line.

    :param line: a line of text (string or PageXMLTextline)
    :type line: Union[str, PageXMLTextline]
    :param word_break_chars: a string of one or more line break characters
    :type word_break_chars: str
    :return: a list of words
    :rtype: List[str]
    """
    new_terms = []
    if line is None or line == '':
        return new_terms
    if line[-1] in word_break_chars and len(line) >= 2:
        if line[-2] in word_break_chars:
            line = line[:-1]
        elif line[-2] == ' ':
            line = line[:-2] + line[-1]
    # if line.endswith(f'{word_break_chars}{word_break_chars}'):
    #     line = line[:-1]
    # elif line.endswith(f' {word_break_chars}'):
    #     line = line[:-2] + word_break_chars
    terms = [term for term in re.split(r'\b', line) if term != '']
    for ti, term in enumerate(terms):
        if ti == 0:
            new_terms.append(term)
        else:
            prev_term = terms[ti - 1]
            # if term[0] == '-' and prev_term[0].isalpha():
            #     new_terms[-1] = new_terms[-1] + term.strip()
            # elif term[0].isalpha() and prev_term[-1] == '-':
            #     new_terms[-1] = new_terms[-1] + term
            if term[0] in word_break_chars and prev_term[0].isalpha():
                new_terms[-1] = new_terms[-1] + term.strip()
            elif term[0].isalpha() and prev_term[-1] in word_break_chars:
                new_terms[-1] = new_terms[-1] + term
            elif term == ' ':
                continue
            else:
                new_terms.append(term.strip())
    return new_terms


def get_page_lines_words(page: pdm.PageXMLPage, word_break_chars='-') -> Generator[List[str], None, None]:
    """Return a generator object yielding lists of words per line of a PageXML Page.

    :param page: a PageXML page object
    :type page: PageXMLPage
    :param word_break_chars: a string of one or more line break characters
    :type word_break_chars: str
    :return: a generator object yielding a list of words per page line
    :rtype: Generator[List[str], None, None]
    """
    for line in page.get_lines():
        if line.text is None:
            continue
        try:
            words = get_line_words(line.text, word_break_chars=word_break_chars)
        except TypeError:
            print(line.text)
            raise
        yield words


def split_line_words(words: List[str]) -> Tuple[List[str], List[str], List[str]]:
    start_words, mid_words, end_words = [], [], []
    if len(words) >= 1:
        start_words = [words[0]]
        end_words = [words[-1]]
    if len(words) >= 2:
        mid_words = words[1:-1]
    return start_words, mid_words, end_words


def remove_word_break_chars(end_word: str, start_word: str, word_break_chars='-=:') -> str:
    if end_word[-1] in word_break_chars:
        if len(end_word) >= 2 and end_word[-2] in word_break_chars:
            end_word = end_word[:-2]
        else:
            end_word = end_word[:-1]
    if start_word[0] in word_break_chars:
        start_word = start_word[1:]
    return end_word + start_word


def remove_hyphen(word: str) -> str:
    if word[-1] in {'-', '=', ':', }:
        if len(word) >= 2 and word[-2:] == '--':
            return word[:-2]
        return word[:-1]
    return word


def find_term_in_context(term: str,
                         line_reader: LineReader,
                         max_hits: int = -1,
                         context_size: int = 3,
                         ignorecase: bool = True) -> Union[Generator[str, None, None], None]:
    """Find a term and its context in text lines from a line reader iterable.
    The term can include wildcard symbol at either the start or end of the term, or both.

    :param term: a term to find in a list of lines
    :type: str
    :param line_reader: an iterable for a list of lines
    :type line_reader: LineReader
    :param max_hits: the maximum number of term matches to return
    :type max_hits: int
    :param context_size: the number of words before and after each term to return as context
    :type context_size: int
    :param ignorecase: flag to indicate whether case should be ignored
    :type ignorecase: bool
    :return: a generator yield occurrences of the term with its context
    :type: Generator[str, None, None]
    """
    pre_regex = r'(\w+\W+){,' + f'{context_size}' + r'}\b('
    post_regex = r')\b(\W+\w+){,' + f'{context_size}' + '}'
    pre_width = context_size * 10
    num_contexts = 0
    match_term = term
    if term.startswith('*'):
        match_term = r'\w*' + match_term[1:]
    if term.endswith('*'):
        match_term = match_term[:-1] + r'\w*'
    for doc in line_reader:
        if 'text' not in doc or doc['text'] is None:
            continue
        if ignorecase:
            re_gen = re.finditer(pre_regex + match_term + post_regex, doc['text'], re.IGNORECASE)
        else:
            re_gen = re.finditer(pre_regex + match_term + post_regex, doc['text'])
        for match in re_gen:
            main = match.group(2)
            pre, post = match.group(0).split(main, 1)
            context = {
                'term': term,
                'term_match': main,
                'match_offset': match.start,
                'pre': pre,
                'post': post,
                'context': f"{pre: >{pre_width}}{main}{post}",
                'doc_id': doc['doc_id']
            }
            num_contexts += 1
            yield context
            if num_contexts == max_hits:
                return None
    return None


def make_skipgram_similarity_dict(line_reader: LineReader, ngram_length: int = 2,
                                  skip_length: int = 1) -> SkipgramSimilarity:
    skip_sim = SkipgramSimilarity(ngram_length=ngram_length, skip_length=skip_length)
    for line in line_reader:
        if line['text'] is None:
            continue
        words = [word for word in re.split(r'\W+', line['text']) if word != '']
        skip_sim.index_terms(words, reset_index=False)
    return skip_sim
