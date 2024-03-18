from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np

import pagexml.helper.text_helper as text_helper
import pagexml.model.physical_document_model as pdm

_SMALL = 1e-20

wpl_to_cat = {}
wpl_cat_min = {}
wpl_cat_max = {}

for wpl in range(0, 101):
    if wpl > 0:
        wpl_cat = (np.log(wpl) * 2).round(0) / 2
    else:
        wpl_cat = -np.inf
    wpl_to_cat[wpl] = wpl_cat
    if wpl_cat not in wpl_cat_min:
        wpl_cat_min[wpl_cat] = wpl
    wpl_cat_max[wpl_cat] = wpl

wpl_cat_range = {wpl_cat: f"{wpl_cat_min[wpl_cat]}-{wpl_cat_max[wpl_cat]}" for wpl_cat in wpl_cat_min}


def get_line_text(text_line: Union[str, Dict[str, any]]) -> Union[str, None]:
    """Convenience function to return the text string of a text line, regardless
    of whether text_line is a str, or a dictionary or a NoneType."""
    if isinstance(text_line, str):
        return text_line
    elif text_line is None:
        return None
    elif isinstance(text_line, pdm.PageXMLTextLine):
        return text_line.text
    elif isinstance(text_line, dict):
        if 'text' in text_line:
            return text_line['text']
        else:
            raise KeyError("text_line dict has no 'text' property.")
    else:
        raise TypeError("text_line must be a string or a dictionary with a 'text' property")


def compute_expected(observed: np.array) -> np.array:
    """Computes the contingency table of the expected values given a contingency table
    of the observed values."""
    expected = np.array([
        [
            observed[0, :].sum() * observed[:, 0].sum() / observed.sum(),
            observed[0, :].sum() * observed[:, 1].sum() / observed.sum()
        ],
        [
            observed[1, :].sum() * observed[:, 0].sum() / observed.sum(),
            observed[1, :].sum() * observed[:, 1].sum() / observed.sum()
        ]
    ])
    return expected


def get_observed(token: str, target_counter: Counter, target_total: int,
                 reference_counter: Counter, reference_total: int):
    """Computes the contingency table of the observed values given a target token, and
    target and reference analysers and counters."""
    # a: word in target corpus
    t_target = target_counter[token] if token in target_counter else 0
    # b: word in ref corpus
    t_ref = reference_counter[token] if token in reference_counter else 0
    # c: other words in target corpus
    nt_target = target_total - t_target
    # d: other words in ref corpus
    nt_ref = reference_total - t_ref

    observed = np.array([
        [t_target, t_ref],
        [nt_target, nt_ref]
    ])
    return observed


def compute_log_likelihood(token: str, target_counter: Counter, target_total: int,
                           reference_counter: Counter, reference_total: int) -> Tuple[float, str]:
    observed = get_observed(token, target_counter, target_total, reference_counter,
                            reference_total)
    """Computes the log likelihood ratio for given a target token, and target and 
    reference analysers and counters."""
    expected = compute_expected(observed)
    sum_likelihood = 0
    for i in [0, 1]:
        for j in [0, 1]:
            sum_likelihood += observed[i, j] * np.log((observed[i, j] + _SMALL) / (expected[i, j] + _SMALL))
    return 2 * sum_likelihood, 'more' if observed[0, 0] > expected[0, 0] else 'less'


def get_keyness_vocab(target_counter: Counter, reference_counter: Counter) -> Set[str]:
    return set(list(target_counter.keys()) + list(reference_counter.keys()))


def compute_keyness(target_counter: Counter, reference_counter: Counter,
                    vocab: Iterable[str] = None):
    """Compute the keyness score of each token in vocabulary for a given target
    counter and reference counter (available counters are 'all', 'start', 'mid'
    or 'end).

    The return value is a dictionary with two properties, 'less' and 'more', each
    with a Counter object. The 'less' counter contains the log likelihood ratio
    for tokens that are less common in the target counter than in the reference
    counter. The 'more' counter contains the log likelihood ratio for tokens that
    are more common in the target counter than in the reference counter.

    :param target_counter: the counter used for token frequencies of the target
        corpus (possible values: 'all', 'start', 'mid' or 'end')
    :type target_counter: str
    :param reference_counter: the counter used for token frequencies of the
        reference corpus (possible values: 'all', 'start', 'mid' or 'end')
    :param vocab: an optional vocabulary for which to compute keyness values.
    :type vocab: Iterable[str]
    """
    log_likelihood = {
        'less': Counter(),
        'more': Counter()
    }
    if vocab is None:
        vocab = set(list(target_counter.keys()) +
                    list(reference_counter.keys()))
    target_total = sum(target_counter.values())
    reference_total = sum(reference_counter.values())
    for token in vocab:
        ll, pref = compute_log_likelihood(token, target_counter, target_total,
                                          reference_counter, reference_total)
        log_likelihood[pref][token] = ll
    return log_likelihood


def compute_complement_keyness(target_analyser: LineAnalyser,
                               target_counter: str):
    """Compute the keyness score of each token in vocabulary for a given target
    counter and its complement as the reference counter (available counters are
    'all', 'start', 'mid' or 'end). The complement is the 'all' counter minus the
    target counter.

    The return value is a dictionary with two properties, 'less' and 'more', each
    with a Counter object. The 'less' counter contains the log likelihood ratio
    for tokens that are less common in the target counter than in the reference
    counter. The 'more' counter contains the log likelihood ratio for tokens that
    are more common in the target counter than in the reference counter.

    :param target_analyser: the target LineAnalyser
    :type target_analyser: LineAnalyser
    :param target_counter: the counter used for token frequencies of the target
        corpus (possible values: 'all', 'start', 'mid' or 'end')

    :type target_counter: str
    """
    target_counter = target_analyser.freq[target_counter]
    reference_counter = Counter()
    for token in target_analyser.freq['all']:
        reference_counter[token] = target_analyser.freq['all'][token] - target_counter[token]
    return compute_keyness(target_counter, reference_counter)


class LineAnalyser:

    def __init__(self, word_break_chars: Union[str, Set[str]] = '-',
                 ignorecase: bool = False, token_type: str = None):
        self.token_type = token_type
        self.word_break_chars = set(word_break_chars) if isinstance(word_break_chars, str) else word_break_chars
        self.ignorecase = ignorecase
        self.freq = {
            'all': Counter(),
            'start': Counter(),
            'mid': Counter(),
            'end': Counter(),
        }
        self.frac = {
            'all': Counter(),
            'start': Counter(),
            'mid': Counter(),
            'end': Counter(),
        }
        self.stats = {
            'total_all_tokens': 0,
            'total_mid_tokens': 0,
            'total_end_tokens': 0,
            'total_start_tokens': 0,
            'total_lines': 0
        }
        self.num_lines = 0

    def __add__(self, other: LineAnalyser):
        merge = LineAnalyser(word_break_chars=self.word_break_chars,
                             ignorecase=self.ignorecase,
                             token_type=self.token_type)
        merge.freq['mid'] = self.freq['mid'] + other.freq['mid']
        merge.freq['end'] = self.freq['end'] + other.freq['end']
        merge.freq['all'] = self.freq['all'] + other.freq['all']
        merge.freq['start'] = self.freq['start'] + other.freq['start']
        merge.set_stats()
        return merge

    def __repr__(self):
        token_stats = json.dumps(self.num_tokens())
        type_stats = json.dumps(self.num_types())
        return f"{self.__class__.__name__}(num_{self.token_type}_tokens={token_stats}, " \
               f"num_{self.token_type}_types={type_stats}, num_lines={self.num_lines})"

    def num_types(self):
        """Returns descriptive statistics of the number of types per counter."""
        return {
            'all': len(self.freq['all']),
            'start': len(self.freq['start']),
            'mid': len(self.freq['mid']),
            'end': len(self.freq['end'])
        }

    def num_tokens(self):
        """Returns descriptive statistics of the number of tokens per counter."""
        return {
            'all': sum(self.freq['all'].values()),
            'start': sum(self.freq['start'].values()),
            'mid': sum(self.freq['mid'].values()),
            'end': sum(self.freq['end'].values())
        }

    def reset_counters(self):
        """Reset all the counters."""
        self.freq = {
            'all': Counter(),
            'start': Counter(),
            'mid': Counter(),
            'end': Counter(),
        }
        self.frac = {
            'all': Counter(),
            'start': Counter(),
            'mid': Counter(),
            'end': Counter(),
        }
        self.stats = {
            'total_all_tokens': 0,
            'total_mid_tokens': 0,
            'total_end_tokens': 0,
            'total_start_tokens': 0,
            'total_lines': 0
        }
        self.num_lines = 0

    def set_stats(self):
        self.stats['total_all_tokens'] = sum(self.freq['all'].values())
        self.stats['total_end_tokens'] = sum(self.freq['end'].values())
        self.stats['total_mid_tokens'] = sum(self.freq['mid'].values())
        self.stats['total_start_tokens'] = sum(self.freq['start'].values())
        all_total = sum(self.freq['all'].values())
        start_total = sum(self.freq['start'].values())
        mid_total = sum(self.freq['mid'].values())
        end_total = sum(self.freq['end'].values())
        for token_type, all_freq in self.freq['all'].most_common():
            self.frac['all'][token_type] = self.freq['all'][token_type] / all_total
            self.frac['start'][token_type] = self.freq['start'][token_type] / start_total
            self.frac['mid'][token_type] = self.freq['mid'][token_type] / mid_total
            self.frac['end'][token_type] = self.freq['end'][token_type] / end_total
        self.stats['total_lines'] = self.num_lines

    def _iter_lines(self, text_lines: Iterable[any]):
        for text_line in text_lines:
            line_text = get_line_text(text_line)
            if line_text is None or len(line_text) == 0:
                continue
            if self.ignorecase is True:
                line_text = line_text.lower()
            yield line_text

    def analyse_line_chars(self, text_lines: Iterable[any]):
        """Analyse the frequency of characters at the start, middle and end of a text line,
        for a given list of text lines."""
        # print('analysing line characters')
        for line_text in self._iter_lines(text_lines):
            self.freq['all'].update(line_text)
            first_char = line_text[0]
            self.freq['start'].update([first_char])
            if len(line_text) > 1:
                last_char = line_text[-1]
                self.freq['end'].update([last_char])
                mid_chars = line_text[1:-1]
                self.freq['mid'].update(mid_chars)
        self.set_stats()

    def analyse_line_words(self, text_lines: Iterable[any]):
        """Gather corpus statistics for a list of text lines on words at the start, middle and end
        of a text line.

        :param text_lines: an iterable for text lines (either strings or dictionaries with a 'text' property
        :type text_lines: Iterable[any]
        """
        for line_text in self._iter_lines(text_lines):
            words = text_helper.get_line_words(line_text, word_break_chars=self.word_break_chars)
            start_words, mid_words, end_words = text_helper.split_line_words(words)
            self.freq['mid'].update(mid_words)
            self.freq['start'].update(start_words)
            self.freq['end'].update(end_words)
            self.freq['all'].update(words)
            self.num_lines += 1
        self.set_stats()

    def get_stats(self):
        """Return statistics on the frequency of characters occuring at the start, middle and end of
        a text line."""
        stats = defaultdict(list)
        for token_type, all_freq in self.freq['all'].most_common():
            try:
                stats['token_type'].append(token_type)
                stats['all_freq'].append(self.freq['all'][token_type])
                stats['all_frac'].append(self.frac['all'][token_type])
                stats['start_freq'].append(self.freq['start'][token_type])
                stats['start_frac'].append(self.frac['start'][token_type])
                stats['start_rel_frac'].append(self.frac['start'][token_type] / self.frac['all'][token_type])
                stats['mid_freq'].append(self.freq['mid'][token_type])
                stats['mid_frac'].append(self.frac['mid'][token_type])
                stats['mid_rel_frac'].append(self.frac['mid'][token_type] / self.frac['all'][token_type])
                stats['end_freq'].append(self.freq['end'][token_type])
                stats['end_frac'].append(self.frac['end'][token_type])
                stats['end_rel_frac'].append(self.frac['end'][token_type] / self.frac['all'][token_type])
            except ZeroDivisionError:
                print(token_type, all_freq)
                raise
        return stats


class LineCharAnalyser(LineAnalyser):

    def __init__(self, text_lines: Iterable[any] = None, word_break_chars: Union[str, Set[str]] = '-',
                 ignorecase: bool = False):
        """A character frequency analyser of a list of text lines. Four frequencies are calculated:
        - all_freq: the overall frequency of a character
        - start_freq: the frequency of a character as the first character in a text line
        - mid_freq: the frequency of a character in the middle of in a text line (so neither as the first nor
        last character of a line)
        - end_freq: the frequency of a character as the last character in a text line

        """
        super().__init__(word_break_chars=word_break_chars, ignorecase=ignorecase, token_type='char')
        if text_lines is not None:
            self.analyse_line_chars(text_lines)


class LineWordAnalyser(LineAnalyser):

    def __init__(self, text_lines: Iterable[any] = None, word_break_chars: Union[str, Set[str]] = '-',
                 ignorecase: bool = False):
        """A line word analyser class for building PageXML word-based corpus statistics.

        :param word_break_chars: a list of characters that can occur as word breaks.
        :type word_break_chars: str
        """
        super().__init__(word_break_chars=word_break_chars, ignorecase=ignorecase, token_type='word')
        if text_lines is not None:
            self.analyse_line_words(text_lines)

    def analyse_line_word_categories(self, text_lines: Iterable[str, pdm.PageXMLTextLine, Dict[str, any]],
                                     **kwargs) -> Dict[str, Counter]:
        """Collect counts on the frequency of different word types, e.g. numbers, title words, stopwords, etc.
        To get counts on stopwords, a stopword list must be passed. For information on what keyword arguments
        can be passed, see pagexml.analysis.text_stats.get_word_cat_stats.

        :param text_lines: an iterable with text lines
        :type text_lines: Iterable[str, PageXMLTextLine, Dict[str, any]
        """
        cat_stats = defaultdict(Counter)
        for line_text in self._iter_lines(text_lines):
            words = text_helper.get_line_words(line_text, word_break_chars=self.word_break_chars)
            word_stats = get_word_cat_stats(words, **kwargs)
            for word_cat in word_stats:
                cat_stats[word_cat] += word_stats[word_cat]
        return cat_stats


def make_line_analyser(token_type: str, word_break_chars, ignorecase: bool = False):
    if token_type == 'char':
        return LineCharAnalyser(word_break_chars=word_break_chars, ignorecase=ignorecase)
    elif token_type == 'word':
        return LineWordAnalyser(word_break_chars=word_break_chars, ignorecase=ignorecase)
    else:
        raise ValueError(f"invalid token type: {token_type}, must be 'char' or 'word'")


def merge_analysers(line_analysers: List[LineAnalyser]) -> LineAnalyser:
    """Merge a list of LineAnalyser objects into a new, single LineAnalyser."""
    token_types = set([la.token_type for la in line_analysers])
    ignorecases = set([la.ignorecase for la in line_analysers])
    word_break_chars = set([lbc for la in line_analysers for lbc in la.word_break_chars])
    if len(token_types) > 1:
        raise TypeError(f"Cannot merge LineAnalysers of different token types: {token_types}")
    if len(ignorecases) > 1:
        raise TypeError(f"Cannot merge LineAnalysers with different ignorecases: {ignorecases}")
    la = line_analysers[0]
    merged_analyser = make_line_analyser(token_type=la.token_type, word_break_chars=word_break_chars)
    for analyser in line_analysers:
        for token in analyser.freq['all']:
            merged_analyser.freq['all'][token] += analyser.freq['all'][token]
            if token in analyser.freq['start']:
                merged_analyser.freq['start'][token] += analyser.freq['start'][token]
            if token in analyser.freq['mid']:
                merged_analyser.freq['mid'][token] += analyser.freq['mid'][token]
            if token in analyser.freq['end']:
                merged_analyser.freq['end'][token] += analyser.freq['end'][token]
    merged_analyser.num_lines = sum([la.num_lines for la in line_analysers])
    merged_analyser.set_stats()
    return merged_analyser


class WordBreakDetector(LineWordAnalyser):

    def __init__(self, min_bigram_word_freq: int = 5, word_break_chars: Union[str, Set[str]] = '-',
                 ignorecase: bool = False, lines: Iterable = None):
        """A line break detector class that uses corpus statistics and a configurable list
        of line break characters to determine, for two subsequent lines, whether the first line ends
        with a line break (a word broken off mid-word on the first line and continued on the second line.)

        :param min_bigram_word_freq: the minimum frequency of word bigrams to be considered common bigrams
        :type min_bigram_word_freq: int
        :param word_break_chars: a list of characters that can occur as line breaks.
        :type word_break_chars: str
        :param ignorecase: whether to ignore differences in case
        :type ignorecase: bool
        :param lines: an iterable for text lines (either strings or dictionaries with a 'text' property
        :type lines: Iterable[any]
        """
        super().__init__(word_break_chars=word_break_chars, ignorecase=ignorecase)
        self.end_with_wbd_freq = Counter()
        self.start_with_wbd_freq = Counter()
        self.mid_bigram_freq = Counter()
        self.typical_start_merged_with = defaultdict(Counter)
        self.typical_end_merged_with = defaultdict(Counter)
        self.common_start_merged_with = defaultdict(Counter)
        self.common_end_merged_with = defaultdict(Counter)
        self.typical_merge_starts = set()
        self.typical_non_merge_starts = set()
        self.typical_merge_ends = set()
        self.typical_non_merge_ends = set()
        self.common_merge_starts = set()
        self.common_non_merge_starts = set()
        self.common_merge_ends = set()
        self.common_non_merge_ends = set()
        self.min_bigram_word_freq = min_bigram_word_freq
        if lines is not None:
            self.set_counters(lines)
            self.set_stats()

    def reset_counters(self):
        """Reset all the counters."""
        super().reset_counters()
        self.end_with_wbd_freq = Counter()
        self.start_with_wbd_freq = Counter()
        self.mid_bigram_freq = Counter()
        self.typical_start_merged_with = defaultdict(Counter)
        self.typical_end_merged_with = defaultdict(Counter)
        self.common_start_merged_with = defaultdict(Counter)
        self.common_end_merged_with = defaultdict(Counter)
        self.typical_merge_starts = set()
        self.typical_non_merge_starts = set()
        self.typical_merge_ends = set()
        self.typical_non_merge_ends = set()
        self.common_merge_starts = set()
        self.common_non_merge_starts = set()
        self.common_merge_ends = set()
        self.common_non_merge_ends = set()

    def print_counter_stats(self):
        """Print overall statistics on the vocabulary derived from the analysed text lines."""
        line_count = sum(self.freq["start"].values())
        print("number of lines:", line_count)
        # print("number of non-empty lines:", sum(self.start_freq.values()))
        # print("number of empty lines:", line_count - sum(self.start_freq.values()))
        print("number of words per line:", sum(self.freq["all"].values()) / line_count)
        print(f'{"all:": <12}{len(self.freq["all"]): >10} types\t{sum(self.freq["all"].values()): >10} tokens')
        print(f'{"start:": <12}{len(self.freq["start"]): >10} types\t{sum(self.freq["start"].values()): >10} tokens')
        print(f'{"mid:": <12}{len(self.freq["mid"]): >10} types\t{sum(self.freq["mid"].values()): >10} tokens')
        print(f'{"end:": <12}{len(self.freq["end"]): >10} types\t{sum(self.freq["end"].values()): >10} tokens')
        print(f'{"mid bigrams:": <12}{len(self.mid_bigram_freq): >10} types'
              f'\t{sum(self.mid_bigram_freq.values()): >10} tokens')
        print(f'Number of typical merge line ends: {len(self.typical_merge_ends)}')
        print(f'Number of typical merge line starts: {len(self.typical_merge_starts)}')
        print(f'Number of common merge line ends: {len(self.common_merge_ends)}')
        print(f'Number of common merge line starts: {len(self.common_merge_starts)}')

    def set_counters(self, lines: Iterable[any]):
        """Gather corpus statistics for a list of text lines on words at the start, middle and end
        of a text line, and on word bigrams in the middle of a line.

        :param lines: an iterable for text lines (either strings or dictionaries with a 'text' property
        :type lines: Iterable[any]
        """
        print('Step 1: setting unigram counters')
        self._set_unigram_counters(lines)
        print('Step 2: setting bigram counter')
        self._set_bigram_counter(lines)
        print('Step 3: setting common merge counters')
        self._set_merged_with(lines)
        print('Step 4: setting typical line ends and starts')
        self._set_typical_start_ends()
        print('Step 5: setting common line ends and starts')
        self._set_common_start_ends()

    def _set_unigram_counters(self, lines: Iterable[Union[str, Dict[str, str]]]):
        li = 0
        for li, line in enumerate(lines):
            if line["text"] is None:
                continue
            words = text_helper.get_line_words(line["text"], word_break_chars=self.word_break_chars)
            start_words, mid_words, end_words = text_helper.split_line_words(words)
            start_with_wbd_words = [w for w in start_words if w[0] in self.word_break_chars]
            start_with_wbd_words = ['<LBC>' + w[:1] for w in start_with_wbd_words]
            end_with_wbd_words = [w for w in end_words if w[-1] in self.word_break_chars]
            end_with_wbd_words += [w[:-1] + '<LBC>' for w in end_with_wbd_words]
            self.freq['mid'].update(mid_words)
            self.freq['start'].update(start_words)
            self.freq['end'].update(end_words)
            self.start_with_wbd_freq.update(start_with_wbd_words)
            self.end_with_wbd_freq.update(end_with_wbd_words)
            self.freq['all'].update(words)
        print(li + 1,
              f'lines processed'
              f'\tall: {len(self.freq["all"]): >8} types'
              f'\t{sum(self.freq["all"].values()): >8} tokens')

    def _set_bigram_counter(self, lines: Iterable[Union[str, Dict[str, str]]]):
        li = 0
        for li, line in enumerate(lines):
            if line["text"] is None:
                continue
            words = text_helper.get_line_words(line["text"], word_break_chars=self.word_break_chars)
            start_words, mid_words, end_words = text_helper.split_line_words(words)
            for i in range(0, len(mid_words) - 2):
                if self.freq['mid'][mid_words[i]] < self.min_bigram_word_freq or \
                        self.freq['mid'][mid_words[i + 1]] < self.min_bigram_word_freq:
                    continue
                self.mid_bigram_freq.update([(mid_words[i], mid_words[i + 1])])
        print(li + 1, f'lines processed\tall: {len(self.mid_bigram_freq)} bigrams')

    def _set_merged_with(self, lines: Iterable[Union[str, Dict[str, str]]],
                         min_common_freq: int = 1000) -> None:
        prev_words = []
        typical_start_words, typical_end_words = get_typical_start_end_words(self)
        for li, line in enumerate(lines):
            if line["text"] is None:
                continue
            words = text_helper.get_line_words(line["text"], word_break_chars=self.word_break_chars)
            if len(prev_words) == 0 or len(words) == 0:
                pass
            else:
                end_word = prev_words[-1]
                start_word = words[0]
                merge_word = end_word + start_word
                reduce_word = text_helper.remove_word_break_chars(end_word, start_word, self.word_break_chars)
                merge_word = merge_word if self.freq['mid'][merge_word] > self.freq['mid'][reduce_word] else reduce_word
                if len(merge_word) > 0 and merge_word[-1] in self.word_break_chars:
                    # when the line start word ends with a hyphen, e.g. 'geval-' + 'len-' -> 'gevallen'
                    if self.freq['mid'][merge_word[-1]] > self.freq['mid'][merge_word]:
                        merge_word = merge_word[:-1]
                if end_word not in self.word_break_chars and self.freq['mid'][merge_word] > 1:
                    if start_word in typical_start_words:
                        self.typical_start_merged_with[start_word].update([(end_word, merge_word)])
                    if end_word in typical_end_words:
                        self.typical_end_merged_with[end_word].update([(start_word, merge_word)])
                    if self.freq['start'][start_word] >= min_common_freq:
                        self.common_start_merged_with[start_word].update([(end_word, merge_word)])
                    if self.freq['end'][end_word] >= min_common_freq:
                        self.common_end_merged_with[end_word].update([(start_word, merge_word)])
            prev_words = words

    def _set_typical_start_ends(self):
        typical_start_words, typical_end_words = get_typical_start_end_words(self)
        for end_word in sorted(typical_end_words, key=lambda w: sum(self.typical_end_merged_with[w].values())):
            merge_exist_frac = sum(self.typical_end_merged_with[end_word].values()) / self.freq['end'][end_word]
            if merge_exist_frac > 0.5:
                self.typical_merge_ends.add(end_word)
            elif merge_exist_frac < 0.05:
                self.typical_non_merge_ends.add(end_word)
            # merge_freq = sum([self.freq['mid'][merged_word]
            #                   for start_word, merged_word in self.typical_end_merged_with[end_word]])
            # freqs = f"{self.freq['end'][end_word]: >8}{self.freq['mid'][end_word]: >8}{self.freq['all'][end_word]: >8}"
            # print(f"{end_word: <20}{freqs}{merge_exist_frac: >8.2f}{merge_freq: >8}")
        for start_word in sorted(typical_start_words, key=lambda w: sum(self.typical_start_merged_with[w].values())):
            merge_exist_frac = sum(self.typical_start_merged_with[start_word].values()) / self.freq['start'][start_word]
            if merge_exist_frac > 0.5:
                self.typical_merge_starts.add(start_word)
            elif merge_exist_frac < 0.05 and start_word.isupper() is False:
                self.typical_non_merge_starts.add(start_word)
            # merge_freq = sum([self.freq['mid'][merged_word]
            #                   for end_word, merged_word in self.typical_start_merged_with[start_word]])
            # freqs = f"{self.freq['start'][start_word]: >8}{self.freq['mid'][start_word]: >8}" \
            #         f"{self.freq['all'][start_word]: >8}"
            # print(f"{start_word: <20}{freqs}{merge_exist_frac: >8.2f}{merge_freq: >8}")

    def _set_common_start_ends(self):
        for start_word in sorted(self.common_start_merged_with,
                                 key=lambda t: self.freq['mid'][t] / self.freq['start'][t]):
            merge_exist_frac = sum(self.common_start_merged_with[start_word].values()) / self.freq['start'][start_word]
            merge_freq = sum([self.freq['mid'][merged_word]
                              for end_word, merged_word in self.common_start_merged_with[start_word]])
            if start_word in self.typical_merge_starts or start_word in self.typical_non_merge_starts:
                continue
            if merge_exist_frac < 0.02:
                self.common_non_merge_starts.add(start_word)
            if merge_exist_frac < 0.2 or merge_freq < 100:
                continue
            self.common_merge_starts.add(start_word)
        for end_word in sorted(self.common_end_merged_with, key=lambda t: self.freq['mid'][t] / self.freq['end'][t]):
            merge_exist_frac = sum(self.common_end_merged_with[end_word].values()) / self.freq['end'][end_word]
            merge_freq = sum([self.freq['mid'][merged_word]
                              for start_word, merged_word in self.common_end_merged_with[end_word]])
            if end_word in self.typical_merge_ends or end_word in self.typical_non_merge_ends:
                continue
            if merge_exist_frac < 0.02:
                self.common_non_merge_ends.add(end_word)
            if merge_exist_frac < 0.2 or merge_freq < 100:
                continue
            self.common_merge_ends.add(end_word)


def show_word_break_context(wbd: WordBreakDetector, end_word: str, start_word: str, merge_word: str,
                            match: str = None):
    last = f'{end_word: <15}{wbd.freq["mid"][end_word]: >8}{wbd.freq["end"][end_word]: >8}'
    first = f'\t{start_word: <15}{wbd.freq["start"][start_word]: >8}{wbd.freq["mid"][start_word]: >8}' \
            f'{wbd.freq["all"][start_word]: >8}'
    merge = f'\t{merge_word: <15}{wbd.freq["all"][merge_word]: >8}'
    if match:
        print(f'{last}{first}{merge}\t{match}')
    else:
        print(f'{last}{first}{merge}')


def determine_word_break(curr_words: List[str], prev_words: List[str],
                         wbd: WordBreakDetector = None,
                         word_break_chars: Union[str, Set[str]] = '-',
                         debug: bool = False) -> Tuple[bool, Union[str, None]]:
    """Determine for a current line and previous line (as lists of words) whether the first line
    ends with a line break.

    :param curr_words: a list of words for the current line to be merged with the previous line
    :type curr_words: List[str]
    :param prev_words: a list of words for the previous line to be merged with the current line
    :type prev_words: List[str]
    :return: a flag whether the previous line ends in a line break and the merged word composed of
        the previous line's last word and current line's first word (or None if the words should not be merged)
    :param wbd: a line break detector object
    :type wbd: WordBreakDetector
    :param word_break_chars: a list of characters that can occur as word breaks.
    :type word_break_chars: str
    :rtype: Union[str, None]
    :param debug: print debugging information
    """
    if wbd is not None and wbd.word_break_chars is not None:
        word_break_chars = set([char for char in wbd.word_break_chars])
    if len(prev_words) == 0 or len(curr_words) == 0:
        # print('includes non-word')
        return False, None
    end_word = prev_words[-1]
    start_word = curr_words[0]
    merge_word = end_word + start_word
    reduce_word = text_helper.remove_word_break_chars(end_word, start_word, word_break_chars)
    if wbd is None:
        return (True, reduce_word) if end_word[-1] in word_break_chars else (False, None)
    merge_word = merge_word if wbd.freq['all'][merge_word] > wbd.freq['all'][reduce_word] else reduce_word
    if debug:
        print(f"end: #{end_word}#\tstart: #{start_word}#")
        print('reduce_word', reduce_word)
        print('merge_word', merge_word)
    bigram_freq = wbd.mid_bigram_freq[(end_word, start_word)]
    if end_word[-1] in word_break_chars:
        bigram_freq = wbd.mid_bigram_freq[(end_word[:-1], start_word)]
        # print(end_word, wbd.freq['mid'][end_word], start_word, wbd.freq['mid'][start_word],
        #       (end_word[:-1], start_word),
        #       merge_word, wbd.freq['mid'][merge_word], 'bigram_freq:', bigram_freq)
    if has_non_merge_word(wbd, end_word, start_word):
        if debug:
            print('has_none_merge_word', end_word, start_word)
        return False, None
    if end_start_are_bigram(wbd, merge_word, bigram_freq, factor=5):
        if debug:
            print('end_start_are_bigram', end_word, start_word)
        return False, None
    elif start_is_titleword(start_word):
        if end_start_are_hyphenated_compound(wbd, end_word, start_word, merge_word):
            merge_word = end_word + start_word
            # print('end_start_are_hyphenated_compound', end_word, start_word)
            if debug:
                print('end_start_are_hyphenated_compound', end_word, wbd.freq['mid'][end_word],
                      start_word, wbd.freq['mid'][start_word],
                      merge_word, wbd.freq['mid'][merge_word])
            return True, merge_word
        elif start_word_has_incorrect_titlecase(wbd, end_word, start_word, factor=10):
            # print('start_word_has_incorrect_titlecase', end_word, start_word)
            if debug:
                print('start_word_has_incorrect_titlecase', end_word, wbd.freq['mid'][end_word],
                      start_word, wbd.freq['mid'][start_word],
                      merge_word, wbd.freq['mid'][merge_word])
            return True, merge_word
        else:
            if debug:
                print('start_word_is_titleword', end_word, start_word)
            return False, None
    elif has_common_merge_end(wbd, end_word, start_word):
        # print('has_common_merge_end', end_word, start_word, merge_word)
        if debug:
            print('has_common_merge_end', end_word, wbd.freq['mid'][end_word], start_word,
                  wbd.freq['mid'][start_word],
                  merge_word, wbd.freq['mid'][merge_word])
        return True, merge_word
    elif has_word_break_symbol(wbd, end_word, start_word, merge_word):
        # print('has_word_break_symbol', end_word, start_word, merge_word)
        if debug:
            print('has_word_break_symbol', end_word, wbd.freq['mid'][end_word], start_word,
                  wbd.freq['mid'][start_word],
                  merge_word, wbd.freq['mid'][merge_word])
        return True, merge_word
    if end_start_are_bigram(wbd, merge_word, bigram_freq, factor=2):
        if debug:
            print('end_start_are_bigram', end_word, start_word)
        return False, None
    if end_is_common_word(wbd, end_word, common_freq=1000):
        if debug:
            print('end_is_common_word', end_word, start_word)
        return False, None
    elif merge_is_more_common(wbd, end_word, start_word, merge_word):
        # print('merge_is_more_common', end_word, start_word)
        if debug:
            print('merge_is_more_common', end_word, wbd.freq['mid'][end_word], start_word,
                  wbd.freq['mid'][start_word],
                  merge_word, wbd.freq['mid'][merge_word])
        return True, merge_word
    elif end_word[-1] in wbd.word_break_chars:
        # print('merge_word_break', end_word, start_word, merge_word)
        if debug:
            print('merge line break', end_word, wbd.freq['mid'][end_word], start_word,
                  wbd.freq['mid'][start_word],
                  merge_word, wbd.freq['mid'][merge_word])
        return True, merge_word
        # show_word_break_context(wbd, end_word, start_word, merge_word)
    else:
        if debug:
            print('OTHER', end_word, wbd.freq['mid'][end_word], start_word,
                  wbd.freq['mid'][start_word], merge_word, wbd.freq['mid'][merge_word])
        return False, None


def merge_is_more_common(wbd, end_word, start_word, merge_word):
    if wbd.freq['all'][merge_word] > wbd.freq['mid'][end_word] and \
            wbd.freq['all'][merge_word] > wbd.freq['mid'][start_word]:
        return True
    elif is_non_mid_word(wbd, end_word, factor=5) and \
            wbd.freq['all'][merge_word] > wbd.freq['mid'][start_word]:
        return True
    elif is_non_mid_word(wbd, start_word, factor=5) and \
            wbd.freq['all'][merge_word] > wbd.freq['mid'][end_word]:
        return True
    elif wbd.freq['all'][merge_word] > 0:
        return True
    else:
        return False


def end_is_common_word(wbd: WordBreakDetector, end_word: str,
                       common_freq: int = 100,
                       debug: bool = False) -> bool:
    # return wbd.freq['all'][end_word] >= common_freq or wbd.freq['all'][start_word] >= common_freq
    if debug:
        print(end_word, wbd.freq['mid'][end_word], common_freq)
    return wbd.freq['mid'][end_word] >= common_freq


def has_word_break_symbol(wbd, end_word, start_word, merge_word):
    if end_word[-1] != '-':
        return False
    if wbd.freq['all'][merge_word] > wbd.freq['all'][end_word]:
        return True
    elif wbd.freq['all'][merge_word] > 0:
        return True
    elif wbd.freq['mid'][start_word] > wbd.freq['start'][start_word]:
        return False
    elif start_word.isdigit():
        return False
    else:
        return True


def has_common_merge_end(wbd: WordBreakDetector, end_word: str,
                         start_word: str) -> bool:
    if end_word in wbd.typical_merge_ends:
        return True
    elif start_word in wbd.typical_merge_starts:
        return True
    if start_word in wbd.common_non_merge_starts:
        return False
    else:
        return False


def is_non_mid_word(wbd: WordBreakDetector, word: str, factor: int = 5) -> bool:
    if wbd.freq['end'][word] > factor * wbd.freq['mid'][word]:
        return True
    if wbd.freq['start'][word] > factor * wbd.freq['mid'][word]:
        return True
    return False


def start_word_has_incorrect_titlecase(wbd: WordBreakDetector, end_word: str,
                                       start_word: str, factor: int = 10) -> bool:
    if start_word in wbd.common_non_merge_starts:
        return False
    if start_word.isupper():
        return False
    if wbd.freq['all'][start_word] < factor and wbd.freq['all'][end_word] < factor:
        return False
    return is_non_mid_word(wbd, start_word) and is_non_mid_word(wbd, end_word)


def end_start_are_hyphenated_compound(wbd: WordBreakDetector, end_word: str,
                                      start_word: str, merge_word: str) -> bool:
    if start_word.isupper():
        # entire start word is upper case, so no part of compound
        return False
    if end_word[0].isupper() and end_word[-1] == '-' and start_word[0].isupper():
        # end word and start word are both in title case, and end word
        # ends with hyphen, so they're probably a hyphenated compound
        if wbd.freq['mid'][start_word] == 0 and wbd.freq['all'][merge_word] == 0 and \
                wbd.freq['all'][end_word + start_word] == 0:
            # start_word is never observed in the middle of a line, so is likely
            # a broken off word, and is incorrectly title cased or
            # its sentence is not the correct one following end_word
            return False
        if wbd.freq['mid'][end_word] == 0 and wbd.freq['all'][merge_word] == 0 and \
                wbd.freq['all'][end_word + start_word] == 0:
            # end_word is never observed in the middle of a line, so is likely
            # a broken off word, and start_word is incorrectly title cased or
            # its sentence is not the correct one following end_word
            return False
        else:
            return True
    else:
        return False


def start_is_titleword(start_word: str) -> bool:
    return start_word[0].isupper()


def end_start_are_bigram(wbd: WordBreakDetector, merge_word: str, bigram_freq: int,
                         factor: int = 5) -> bool:
    # the bigram of end word and start word is much more frequent than their
    # merge, so treat as bigram
    # if frequency of merge_word is zero, bigram_freq should be at least 5
    return bigram_freq > factor and bigram_freq > factor * wbd.freq['all'][merge_word]


def determine_word_break_typical_merge_end(wbd: WordBreakDetector, end_word: str,
                                           start_word: str, merge_word: str) -> bool:
    if end_word in wbd.typical_merge_ends:
        if wbd.freq['all'][merge_word] >= 10:
            return True
        elif wbd.freq['mid'][start_word] > 100 and end_word.endswith('-'):
            if wbd.freq['mid'][end_word[:-1]] > 100 and wbd.freq['mid'][end_word[:-1]] > 10 * wbd.freq['mid'][end_word]:
                return False
            elif wbd.freq['all'][merge_word] > 0:
                return True
            else:
                return False
        else:
            return True


def has_non_merge_word(wbd: WordBreakDetector, end_word: str,
                       start_word: str, debug: bool = False) -> bool:
    if not re.search(r'\w', end_word):
        # end word is just punctuation, so don't merge
        if debug:
            print(f'end_word is punctuation: #{end_word}#')
        return True
    if not re.search(r'\w', start_word):
        # start word is just punctuation, so don't merge
        if debug:
            print(f'start_word is punctuation: #{start_word}#')
        return True
    if end_word == '-':
        # end word is just hyphen, so don't merge
        if debug:
            print(f'end_word is hyphen: #{end_word}#')
        return True
    if start_word == '-':
        # start word is just hyphen, so don't merge
        if debug:
            print(f'start_word is hyphen: #{start_word}#')
        return True
    elif end_word in wbd.typical_non_merge_ends:
        # start word is a non-merge word so don't merge
        if debug:
            print(f'end_word is non_merge_end: #{end_word}#')
        return True
    elif start_word in wbd.typical_non_merge_starts:
        # start word is a non-merge word so don't merge
        if debug:
            print(f'start_word is non_merge_start: #{start_word}#')
        return True
    else:
        return False


def get_typical_start_end_words(wbd: WordBreakDetector,
                                threshold: float = 0.5) -> Tuple[Set[str], Set[str]]:
    typical_start_words = set()
    for end_word in wbd.freq['start']:
        if wbd.freq['start'][end_word] > 100 and \
                wbd.freq['start'][end_word] / wbd.freq['all'][end_word] > threshold:
            typical_start_words.add(end_word)
    typical_end_words = set()
    for end_word in wbd.freq['end']:
        if wbd.freq['end'][end_word] > 100 and \
                wbd.freq['end'][end_word] / wbd.freq['all'][end_word] > threshold:
            typical_end_words.add(end_word)
    return typical_start_words, typical_end_words


def get_words_per_line(lines: List[pdm.PageXMLTextLine], use_re_word_boundaries: bool = False,
                       alpha_words_only: bool = False):
    """Return a Counter of the number of words per line of a PageXML pagexml_doc object.

    :param lines: a list of PageXMLTextLine objects
    :type lines: List[PageXMLTextLine]
    :param use_re_word_boundaries: whether to split words of a line using RegEx word boundaries
    :type use_re_word_boundaries: bool
    :param alpha_words_only: whether to only count words consisting of alpha characters (e.g. no numbers)
    :type alpha_words_only: bool
    :return: a counter of the number of words per line of a pagexml_doc
    :rtype: Counter
    """
    words_per_line = Counter()
    if isinstance(lines, pdm.PageXMLTextRegion):
        lines = lines.get_lines()
    for line in lines:
        if line.text is None or line.text == '':
            words = []
        elif use_re_word_boundaries:
            words = [w.replace(' ', '') for w in re.split(r'\b', line.text)]
        else:
            words = [w for w in line.text.split(' ')]
        words = [w for w in words if w != ' ' and w != '']
        if alpha_words_only is True:
            words = [w for w in words if w.isalpha()]
        # words_per_line.update([len(words)])
        if len(words) in wpl_to_cat:
            wpl_cat = wpl_to_cat[len(words)]
        else:
            wpl_cat = max(wpl_cat_range.keys())
        words_per_line.update([wpl_cat_range[wpl_cat]])
    return words_per_line


def get_doc_words(pagexml_doc: pdm.PageXMLTextRegion, use_re_word_boundaries: bool = False) -> List[str]:
    """Return a list of words that are part of a PageXML pagexml_doc object.

    :param pagexml_doc: a PageXML document object
    :type pagexml_doc: PageXMLTextRegion
    :param use_re_word_boundaries: whether to split words of a line using RegEx word boundaries
    :type use_re_word_boundaries: bool
    :return: a list of all words on a pagexml_doc
    :rtype: List[str]
    """
    lines = [line for line in pagexml_doc.get_lines() if line.text is not None]
    if use_re_word_boundaries:
        return [w.replace(' ', '') for line in lines for w in re.split(r'\b', line.text) if w != ' ' and w != '']
    else:
        return [w for line in lines for w in line.text.split(' ')]


def get_word_cat_stats(words, stop_words=None, max_word_length: int = 30,
                       word_length_bin_size: int = 5):
    """Calculate word type statistics for the word of a given PageXML scan.

    :param words: a list of words on a scan
    :type words: List[str]
    :param stop_words: a list of stopwords
    :type stop_words: List[str]
    :param max_word_length: the maximum length of words to be considered a regular word
    :type max_word_length: int (default 30 characters)
    :param word_length_bin_size: bin size for grouping words within a character length interval
    :type word_length_bin_size: int (default per 5 characters)
    """
    puncs = set(string.punctuation)
    num_oversized_words = len([w for w in words if len(w) > max_word_length])
    word_length_freq = Counter([len(w) for w in words if len(w) <= max_word_length])
    word_cat_stats = {
        'num_words': len(words),
        'num_alpha_words': len([w for w in words if w.isalpha()]),
        'num_number_words': len([w for w in words if w.isdigit()]),
        'num_title_words': len([w for w in words if w.istitle()]),
        'num_non_title_words': len([w for w in words if w.istitle() is False]),
        'num_stop_words': len([w for w in words if w in stop_words]) if stop_words is not None else None,
        'num_punctuation_words': len([w for w in words if all(j in puncs for j in w)]),
        'num_oversized_words': num_oversized_words
    }
    word_length_bin = word_length_bin_size
    word_cat_stats[f'num_words_length_{word_length_bin}'] = 0
    for wl in range(1, max_word_length + 1):
        if wl > word_length_bin:
            word_length_bin += word_length_bin_size
            word_cat_stats[f'num_words_length_{word_length_bin}'] = 0
        word_cat_stats[f'num_words_length_{word_length_bin}'] += word_length_freq[wl]
    return word_cat_stats
