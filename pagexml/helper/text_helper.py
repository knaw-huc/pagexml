from typing import Dict, Generator, Iterable, List, Set, Tuple, Union
import re
import gzip
import math
from collections import Counter, defaultdict
from itertools import combinations

import pagexml.parser as parser
import pagexml.helper.file_helper as file_helper
import pagexml.model.physical_document_model as pdm


def read_lines_from_line_files(pagexml_line_files: List[str]) -> Generator[str, None, None]:
    for line_file in pagexml_line_files:
        with gzip.open(line_file, 'rt') as fh:
            for line in fh:
                yield line


def get_line_format_json(page_doc: pdm.PageXMLTextRegion,
                         use_outer_textregions: bool = False) -> Generator[Dict[str, any], None, None]:
    if page_doc.num_text_regions == 0 and page_doc.num_lines > 0:
        trs = [page_doc]
    elif use_outer_textregions is True:
        trs = page_doc.text_regions
    else:
        trs = page_doc.get_inner_text_regions()
    for tr in trs:
        for line in tr.get_lines():
            yield {
                'page_doc_id': page_doc.id,
                'textregion_id': tr.id,
                'line_id': line.id,
                'text': line.text
            }
    return None


def get_line_format_tsv(page_doc: pdm.PageXMLTextRegion,
                        headers: List[str],
                        use_outer_textregions: bool = False) -> Generator[List[str], None, None]:
    for line_json in get_line_format_json(page_doc, use_outer_textregions=use_outer_textregions):
        line_list = [line_json[header] for header in headers]
        yield [val if val is not None else '' for val in line_list]


class SkipGram:
    """A skipgram object containing the skipgram string, its offset in the text and
    the length of the string in the text (including the skips)."""

    def __init__(self, skipgram_string: str, offset: int, skipgram_length: int):
        self.string = skipgram_string
        self.offset = offset
        self.length = skipgram_length


def insert_skips(window: str, skipgram_combinations: List[List[int]]):
    """For a given skip gram window, return all skip grams for a given configuration."""
    for combination in skipgram_combinations:
        skip_gram = window[0]
        try:
            for index in combination:
                skip_gram += window[index]
            yield skip_gram, combination[-1] + 1
        except IndexError:
            pass


def text2skipgrams(text: str, ngram_size: int = 2, skip_size: int = 2) -> Generator[SkipGram, None, None]:
    """Turn a text string into a list of skipgrams.

    :param text: an text string
    :type text: str
    :param ngram_size: an integer indicating the number of characters in the ngram
    :type ngram_size: int
    :param skip_size: an integer indicating how many skip characters in the ngrams
    :type skip_size: int
    :return: An iterator returning tuples of skip_gram and offset
    :rtype: Generator[tuple]"""
    if ngram_size <= 0 or skip_size < 0:
        raise ValueError('ngram_size must be a positive integer, skip_size must be a positive integer or zero')
    indexes = [i for i in range(0, ngram_size + skip_size)]
    skipgram_combinations = [combination for combination in combinations(indexes[1:], ngram_size - 1)]
    for offset in range(0, len(text) - 1):
        window = text[offset:offset + ngram_size + skip_size]
        for skipgram, skipgram_length in insert_skips(window, skipgram_combinations):
            yield SkipGram(skipgram, offset, skipgram_length)


def make_list(var) -> list:
    return var if isinstance(var, list) else [var]


class LineReader:

    def __init__(self, pagexml_files: Union[str, List[str]] = None,
                 pagexml_docs: Union[pdm.PageXMLDoc, List[pdm.PageXMLDoc]] = None,
                 pagexml_line_files: Union[str, List[str]] = None,
                 line_file_headers: List[str] = None,
                 has_header: bool = True,
                 use_outer_textregions: bool = False,
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
        :param has_header: whether the pagexml_line_files have a header line
        :type has_header: bool
        :param use_outer_textregions: use ID of outer text regions (when True) otherwise ID of inner
        text regions
        :type use_outer_textregions: bool
        :param groupby: group lines by 'page_doc_id' or 'textregion_id'
        :type groupby: str
        """
        self.pagexml_files = []
        self.pagexml_docs = []
        self.pagexml_line_files = []
        self.line_file_headers = line_file_headers
        self.has_header = has_header
        self.use_outer_textregions = use_outer_textregions
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

    def _iter(self):
        if self.pagexml_line_files:
            for line in self._iter_from_line_file():
                yield line
        if len(self.pagexml_files) > 0:
            pagexml_doc_iterator = parser.parse_pagexml_files(self.pagexml_files)
            self._iter_from_pagexml_docs(pagexml_doc_iterator)
        if len(self.pagexml_docs) > 0:
            self._iter_from_pagexml_docs(self.pagexml_docs)

    def _iter_from_pagexml_docs(self, pagexml_doc_iterator):
        for pagexml_doc in pagexml_doc_iterator:
            for line in get_line_format_json(pagexml_doc, use_outer_textregions=self.use_outer_textregions):
                yield line

    def _iter_from_line_file(self):
        line_iterator = read_lines_from_line_files(self.pagexml_line_files)
        if self.has_header is True:
            header_line = next(line_iterator)
            self.line_file_headers = header_line.strip().split('\t')
        elif self.line_file_headers is None:
            self.line_file_headers = [
                'doc_id', 'textregion_id', 'line_id', 'text'
            ]
        num_splits = len(self.line_file_headers) - 1
        for li, line in enumerate(line_iterator):
            try:
                cols = line.strip().split('\t', num_splits)
                yield {header: cols[hi] for hi, header in enumerate(self.line_file_headers)}
            except ValueError:
                print(f"line {li} in file {self.pagexml_line_files}:")
                print(line)
                raise


def make_page_extractor(archive_file: str,
                        show_progress: bool = False) -> Generator[pdm.PageXMLScan, None, None]:
    """Convenience function to return a generator that yield a PageXMLScan object per PageXML file
    in a zip/tar archive file."""
    for page_fileinfo, page_data in file_helper.read_page_archive_file(archive_file,
                                                                       show_progress=show_progress):
        scan = parser.parse_pagexml_file(pagexml_file=page_fileinfo['archived_filename'], pagexml_data=page_data)
        yield scan


def make_line_format_file(page_docs: Iterable[pdm.PageXMLTextRegion],
                          line_format_file: str):
    """Transform a list of PageXMLDoc objects"""
    headers = ['page_doc_id', 'textregion_id', 'line_id', 'text']
    with gzip.open(line_format_file, 'wt') as fh:
        header_string = '\t'.join(headers)
        fh.write(f'{header_string}\n')
        for page_doc in page_docs:
            for line_tsv in get_line_format_tsv(page_doc, headers):
                line_string = '\t'.join(line_tsv)
                fh.write(f'{line_string}\n')


# SPLIT_PATTERN = r'[ \.,\!\?\(\)\[\]\{\}"\':;]+'
# def get_line_words(line, split_pattern: str = SPLIT_PATTERN) -> List[str]:
#     return [word for word in re.split(split_pattern, line) if word != '']


def get_line_words(line: Union[pdm.PageXMLTextLine, str], line_break_chars: Union[str, Set[str]] = '-') -> List[str]:
    """Return a list of the words for a given line.

    :param line: a line of text (string or PageXMLTextline)
    :type line: Union[str, PageXMLTextline]
    :param line_break_chars: a string of one or more line break characters
    :type line_break_chars: str
    :return: a list of words
    :rtype: List[str]
    """
    new_terms = []
    if line is None or line == '':
        return new_terms
    if line[-1] in line_break_chars and len(line) >= 2:
        if line[-2] in line_break_chars:
            line = line[:-1]
        elif line[-2] == ' ':
            line = line[:-2] + line[-1]
    # if line.endswith(f'{line_break_chars}{line_break_chars}'):
    #     line = line[:-1]
    # elif line.endswith(f' {line_break_chars}'):
    #     line = line[:-2] + line_break_chars
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
            if term[0] in line_break_chars and prev_term[0].isalpha():
                new_terms[-1] = new_terms[-1] + term.strip()
            elif term[0].isalpha() and prev_term[-1] in line_break_chars:
                new_terms[-1] = new_terms[-1] + term
            elif term == ' ':
                continue
            else:
                new_terms.append(term.strip())
    return new_terms


def get_page_lines_words(page: pdm.PageXMLPage, line_break_chars='-') -> Generator[List[str], None, None]:
    """Return a generator object yielding lists of words per line of a PageXML Page.

    :param page: a PageXML page object
    :type page: PageXMLPage
    :param line_break_chars: a string of one or more line break characters
    :type line_break_chars: str
    :return: a generator object yielding a list of words per page line
    :rtype: Generator[List[str], None, None]
    """
    for line in page.get_lines():
        if line.text is None:
            continue
        try:
            words = get_line_words(line.text, line_break_chars=line_break_chars)
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


def remove_line_break_chars(end_word: str, start_word: str, line_break_chars='-=:') -> str:
    if end_word[-1] in line_break_chars:
        if len(end_word) >= 2 and end_word[-2] in line_break_chars:
            end_word = end_word[:-2]
        else:
            end_word = end_word[:-1]
    if start_word[0] in line_break_chars:
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


def vector_length(skipgram_freq):
    return math.sqrt(sum([skipgram_freq[skip] ** 2 for skip in skipgram_freq]))


class Vocabulary:

    def __init__(self):
        """A Vocabulary class to map terms to identifiers."""
        self.term_id = {}
        self.id_term = {}
        self.term_freq = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(vocabulary_size="{len(self.term_id)}")'

    def __len__(self):
        return len(self.term_id)

    def reset_index(self):
        self.term_id = {}
        self.id_term = {}
        self.term_freq = {}

    def add_terms(self, terms: List[str], reset_index: bool = True):
        """Add a list of terms to the vocabulary. Use 'reset_index=True' to reset
        the vocabulary before adding the terms.

        :param terms: a list of terms to add to the vocabulary
        :type terms: List[str]
        :param reset_index: a flag to indicate whether to empty the vocabulary before adding terms
        :type reset_index: bool
        """
        if reset_index is True:
            self.reset_index()
        for term in terms:
            if term in self.term_id:
                continue
            self._index_term(term)

    def _index_term(self, term: str):
        term_id = len(self.term_id)
        self.term_id[term] = term_id
        self.id_term[term_id] = term

    def term2id(self, term: str):
        """Return the term ID for a given term."""
        return self.term_id[term] if term in self.term_id else None

    def id2term(self, term_id: int):
        """Return the term for a given term ID."""
        return self.id_term[term_id] if term_id in self.id_term else None


def get_skip_coocs(seq_ids: List[str], skip_size: int = 0) -> Generator[Tuple[int, int], None, None]:
    for ci, curr_id in enumerate(seq_ids):
        for next_id in seq_ids[ci + 1: ci + 2 + skip_size]:
            yield curr_id, next_id


class SkipCooccurrence:

    def __init__(self, vocabulary: Vocabulary, skip_size: int = 1, sentences: Iterable[List[str]] = None):
        """A class to count the co-occurrence frequency of word skipgrams."""
        self.cooc_freq = defaultdict(int)
        self.vocabulary = vocabulary
        self.skip_size: int = skip_size
        if sentences is not None:
            self.calculate_skip_cooccurrences(sentences)

    def calculate_skip_cooccurrences(self, sentences: Iterable[List[str]], skip_size: int = 0):
        """Count the frequency of term (skip) co-occurrences for a given list of sentences.

        :param sentences: a list of sentences, where each sentence is itself a list of term tokens
        :type sentences: Iterable[List[str]
        :param skip_size: the maximum number of skips to allow between co-occurring terms
        :type skip_size: int
        """
        for sent in sentences:
            seq_ids = [self.vocabulary.term2id(t) for t in sent]
            self.cooc_freq.update(get_skip_coocs(seq_ids, skip_size=skip_size))

    def _cooc_ids2terms(self, cooc_ids: Tuple[int, int]) -> Tuple[str, str]:
        id1, id2 = cooc_ids
        return self.vocabulary.id2term(id1), self.vocabulary.id2term(id2)

    def get_term_coocs(self, term: str) -> Union[None, Generator[Tuple[str, str], None, None]]:
        term_id = self.vocabulary.term2id(term)
        if term_id is None:
            return None
        for cooc_ids in self.cooc_freq:
            if term_id in cooc_ids:
                yield self._cooc_ids2terms(cooc_ids), self.cooc_freq[cooc_ids]


class SkipgramSimilarity:

    def __init__(self, ngram_length: int = 3, skip_length: int = 0, terms: List[str] = None,
                 max_length_diff: int = 2):
        """A class to index terms by their character skipgrams and to find similar terms for a given
        input term based on the cosine similarity of their skipgram overlap.

        :param ngram_length: the number of characters per ngram
        :type ngram_length: int
        :param skip_length: the maximum number of characters to skip
        :type skip_length: int
        :param terms: a list of terms
        :type terms: List[str]
        :param max_length_diff: the maximum difference in length between a search term and a term in the
        index to be considered a match. This is an efficiency parameter to reduce the number of candidate
        similar terms to ones that are roughly similar in length to the search term.
        :type max_length_diff: int
        :
        """
        self.ngram_length = ngram_length
        self.skip_length = skip_length
        self.vocabulary = Vocabulary()
        self.vector_length = {}
        self.max_length_diff = max_length_diff
        self.skipgram_index = defaultdict(lambda: defaultdict(Counter))
        if terms is not None:
            self.index_terms(terms)

    def _reset_index(self):
        self.vocabulary.reset_index()
        self.vector_length = {}
        self.skipgram_index = defaultdict(lambda: defaultdict(Counter))

    def index_terms(self, terms: List[str], reset_index: bool = True):
        """Make a frequency index of the skip grams for a given list of terms.
        By default, indexing is cumulative, that is, everytime you call index_terms
        with a list of terms, they are added to the index. Use 'reset_index=True' to
        reset the index before indexing the given terms.

        :param terms: a list of term to index
        :type terms: List[str]
        :param reset_index: whether to reset the index before indexing or to keep the existing index
        :type reset_index: bool
        """
        if reset_index is True:
            self._reset_index()
        self.vocabulary.add_terms(terms)
        for term in terms:
            self._index_term_skips(term)

    def _term_to_skip(self, term):
        skip_gen = text2skipgrams(term, ngram_size=self.ngram_length, skip_size=self.skip_length)
        return Counter([skip.string for skip in skip_gen])

    def _index_term_skips(self, term: str):
        term_id = self.vocabulary.term_id[term]
        skipgram_freq = self._term_to_skip(term)
        self.vector_length[term_id] = vector_length(skipgram_freq)
        for skipgram in skipgram_freq:
            # print(skip.string)
            self.skipgram_index[skipgram][len(term)][term_id] = skipgram_freq[skipgram]

    def _get_term_vector_length(self, term, skipgram_freq):
        if term not in self.vocabulary.term_id:
            return vector_length(skipgram_freq)
        else:
            term_id = self.vocabulary.term_id[term]
            return self.vector_length[term_id]

    def _compute_dot_product(self, term):
        skipgram_freq = self._term_to_skip(term)
        term_vl = self._get_term_vector_length(term, skipgram_freq)
        # print(term, 'vl:', term_vl)
        dot_product = defaultdict(int)
        for skipgram in skipgram_freq:
            for term_length in range(len(term) - self.max_length_diff, len(term) + self.max_length_diff + 1):
                for term_id in self.skipgram_index[skipgram][term_length]:
                    dot_product[term_id] += skipgram_freq[skipgram] * self.skipgram_index[skipgram][term_length][
                        term_id]
                    # print(term_id, self.vocab_map[term_id], dot_product[term_id])
        for term_id in dot_product:
            dot_product[term_id] = dot_product[term_id] / (term_vl * self.vector_length[term_id])
        return dot_product

    def rank_similar(self, term: str, top_n: int = 10, score_cutoff: float = 0.5):
        """Return a ranked list of similar terms from the index for a given input term,
        based on their character skipgram cosine similarity.

        :param term: a term (any string) to match against the indexed terms
        :type term: str
        :param top_n: the number of highest ranked terms to return
        :type top_n: int (default 10)
        :param score_cutoff: the minimum similarity score after which to cutoff the ranking
        :type score_cutoff: float
        :return: a ranked list of terms and their similarity scores
        :rtype: List[Tuple[str, float]]
        """
        dot_product = self._compute_dot_product(term)
        top_terms = []
        for term_id in sorted(dot_product, key=lambda t: dot_product[t], reverse=True):
            if dot_product[term_id] < score_cutoff:
                break
            term = self.vocabulary.id_term[term_id]
            top_terms.append((term, dot_product[term_id]))
            if len(top_terms) == top_n:
                break
        return top_terms
