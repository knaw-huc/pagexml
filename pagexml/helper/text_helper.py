import gzip
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Union, Generator, Iterable

import pagexml.model.physical_document_model as pdm
import pagexml.parser as parser


class SkipGram:

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


class LineReader:

    def __init__(self, pagexml_files: Union[str, List[str]] = None,
                 pagexml_docs: Union[pdm.PageXMLDoc, List[pdm.PageXMLDoc]] = None,
                 pagexml_line_file: Union[str, List[str]] = None):
        self.pagexml_files = []
        self.pagexml_docs = []
        self.pagexml_line_file = []
        if pagexml_line_file:
            if isinstance(pagexml_line_file, list):
                self.pagexml_line_file = pagexml_line_file
            else:
                self.pagexml_line_file = [pagexml_line_file]
        if pagexml_files is None and pagexml_docs is None and pagexml_line_file is None:
            raise TypeError(f"MUST use one of the following optional arguments: "
                            f"'pagexml_files', 'pagexml_docs' or 'pagexml_line_file'.")
        if pagexml_files:
            self.pagexml_files = [pagexml_files] if isinstance(pagexml_files, str) else pagexml_files
        elif pagexml_docs:
            self.pagexml_docs = [pagexml_docs] if isinstance(pagexml_docs, pdm.PageXMLDoc) else pagexml_docs

    def __iter__(self) -> Generator[Dict[str, str], None, None]:
        if self.pagexml_line_file:
            for line_file in self.pagexml_line_file:
                with gzip.open(line_file, 'rt') as fh:
                    for li, line in enumerate(fh):
                        try:
                            doc_id, line_id, line_text = line.strip().split('\t', 2)
                            yield {"doc_id": doc_id, "line_id": line_id, "text": line_text}
                        except ValueError:
                            print(f"line {li} in file {self.pagexml_line_file}:")
                            print(line)
                            raise
        if len(self.pagexml_files) > 0:
            self.pagexml_docs = parser.parse_pagexml_files(self.pagexml_files)
        for pagexml_doc in self.pagexml_docs:
            for line in pagexml_doc.get_lines():
                yield {"id": line.id, "text": line.text}


# SPLIT_PATTERN = r'[ \.,\!\?\(\)\[\]\{\}"\':;]+'
# def get_line_words(line, split_pattern: str = SPLIT_PATTERN) -> List[str]:
#     return [word for word in re.split(split_pattern, line) if word != '']


def get_line_words(line, line_break_char: str = '-'):
    new_terms = []
    if line.endswith(f'{line_break_char}{line_break_char}'):
        line = line[:-1]
    elif line.endswith(f' {line_break_char}'):
        line = line[:-2] + line_break_char
    terms = [term for term in re.split(r'\b', line) if term != '']
    for ti, term in enumerate(terms):
        if ti == 0:
            new_terms.append(term)
        else:
            prev_term = terms[ti - 1]
            if term[0] == line_break_char and prev_term[0].isalpha():
                new_terms[-1] = new_terms[-1] + term.strip()
            elif term[0].isalpha() and prev_term[-1] == line_break_char:
                new_terms[-1] = new_terms[-1] + term
            elif term == ' ':
                continue
            else:
                new_terms.append(term.strip())
    return new_terms


def get_page_lines_words(page: pdm.PageXMLPage) -> Generator[List[str], None, None]:
    for line in page.get_lines():
        if line.text is None:
            continue
        try:
            words = get_line_words(line.text)
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
                         ignorecase: bool = True):
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
                'doc_id': doc["doc_id"]
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

    def index_terms(self, terms: List[str], reset_index: bool = True):
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
        return self.term_id[term] if term in self.term_id else None

    def id2term(self, term_id: int):
        return self.id_term[term_id] if term_id in self.id_term else None


def get_skip_coocs(seq_ids: List[str], skip_size: int = 1) -> Generator[Tuple[int, int], None, None]:
    for ci, curr_id in enumerate(seq_ids):
        for offset in range(1, skip_size + 1):
            if ci + offset >= len(seq_ids):
                break
            next_id = seq_ids[ci + offset]
            yield curr_id, next_id


class SkipCooccurrence:

    def __init__(self, vocabulary: Vocabulary, skip_size: int = 1):
        self.cooc_freq = defaultdict(int)
        self.vocabulary = vocabulary
        self.skip_size: int = skip_size

    def calculate_skip_cooccurrences(self, sentences: Iterable, skip_size: int = None):
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
        if reset_index is True:
            self._reset_index()
        self.vocabulary.index_terms(terms)
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

    def rank_similar(self, term: str, top_n: int = 10):
        dot_product = self._compute_dot_product(term)
        top_terms = []
        for term_id in sorted(dot_product, key=lambda t: dot_product[t], reverse=True):
            term = self.vocabulary.id_term[term_id]
            top_terms.append((term, dot_product[term_id]))
            if len(top_terms) == top_n:
                break
        return top_terms
