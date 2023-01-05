from __future__ import annotations
from typing import Dict, Iterable, List, Set, Tuple, Union
from collections import Counter
from collections import defaultdict
import re
import json
import string

import pagexml.model.physical_document_model as pdm
import pagexml.helper.text_helper as text_helper


def get_line_text(text_line: Union[str, Dict[str, any]]) -> Union[str, None]:
    """Convenience function to return the text str of a text line, regardless of whether text_line
    is a str or a dictionary or a NoneType."""
    if isinstance(text_line, str):
        return text_line
    elif text_line is None:
        return None
    elif isinstance(text_line, dict):
        if 'text' in text_line:
            return text_line['text']
        else:
            raise KeyError("text_line dict has no 'text' property.")
    else:
        raise TypeError("text_line must be a string or a dictionary with a 'text' property")


class LineAnalyser:

    def __init__(self, line_break_chars: Union[str, Set[str]] = '-',
                 ignorecase: bool = False, token_type: str = None):
        self.token_type = token_type
        self.line_break_chars = set(line_break_chars) if isinstance(line_break_chars, str) else line_break_chars
        self.ignorecase = ignorecase
        self.all_freq = Counter()
        self.mid_freq = Counter()
        self.start_freq = Counter()
        self.end_freq = Counter()
        self.all_frac = Counter()
        self.mid_frac = Counter()
        self.start_frac = Counter()
        self.end_frac = Counter()
        self.stats = {
            'total_all_tokens': 0,
            'total_mid_tokens': 0,
            'total_end_tokens': 0,
            'total_start_tokens': 0,
            'total_lines': 0
        }
        self.num_lines = 0

    def __add__(self, other: LineAnalyser):
        merge = LineAnalyser(line_break_chars=self.line_break_chars,
                             ignorecase=self.ignorecase,
                             token_type=self.token_type)
        merge.mid_freq = self.mid_freq + other.mid_freq
        merge.end_freq = self.end_freq + other.end_freq
        merge.all_freq = self.all_freq + other.all_freq
        merge.start_freq = self.start_freq + other.start_freq
        merge._set_stats()
        return merge

    def __repr__(self):
        token_stats = json.dumps(self.num_tokens())
        type_stats = json.dumps(self.num_types())
        return f"{self.__class__.__name__}(num_{self.token_type}_tokens={token_stats}, " \
               f"num_{self.token_type}_types={type_stats}, num_lines={self.num_lines})"

    def num_types(self):
        """Returns descriptive statistics of the number of types per counter."""
        return {
            'all': len(self.all_freq),
            'start': len(self.start_freq),
            'mid': len(self.mid_freq),
            'end': len(self.end_freq)
        }

    def num_tokens(self):
        """Returns descriptive statistics of the number of tokens per counter."""
        return {
            'all': sum(self.all_freq.values()),
            'start': sum(self.start_freq.values()),
            'mid': sum(self.mid_freq.values()),
            'end': sum(self.end_freq.values())
        }

    def reset_counters(self):
        """Reset all the counters."""
        self.mid_freq = Counter()
        self.end_freq = Counter()
        self.start_freq = Counter()
        self.all_freq = Counter()
        self.all_frac = Counter()
        self.mid_frac = Counter()
        self.start_frac = Counter()
        self.end_frac = Counter()
        self.stats = {
            'total_all_tokens': 0,
            'total_mid_tokens': 0,
            'total_end_tokens': 0,
            'total_start_tokens': 0,
            'total_lines': 0
        }
        self.num_lines = 0

    def _set_stats(self):
        self.stats['total_all_tokens'] = sum(self.all_freq.values())
        self.stats['total_end_tokens'] = sum(self.end_freq.values())
        self.stats['total_mid_tokens'] = sum(self.mid_freq.values())
        self.stats['total_start_tokens'] = sum(self.start_freq.values())
        all_total = sum(self.all_freq.values())
        start_total = sum(self.start_freq.values())
        mid_total = sum(self.mid_freq.values())
        end_total = sum(self.end_freq.values())
        for token_type, all_freq in self.all_freq.most_common():
            self.all_frac[token_type] = self.all_freq[token_type] / all_total
            self.start_frac[token_type] = self.start_freq[token_type] / start_total
            self.mid_frac[token_type] = self.mid_freq[token_type] / mid_total
            self.end_frac[token_type] = self.end_freq[token_type] / end_total
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
            self.all_freq.update(line_text)
            first_char = line_text[0]
            self.start_freq.update([first_char])
            if len(line_text) > 1:
                last_char = line_text[-1]
                self.end_freq.update([last_char])
                mid_chars = line_text[1:-1]
                self.mid_freq.update(mid_chars)
        self._set_stats()

    def analyse_line_words(self, text_lines: Iterable[any]):
        """Gather corpus statistics for a list of text lines on words at the start, middle and end
        of a text line.

        :param text_lines: an iterable for text lines (either strings or dictionaries with a 'text' property
        :type text_lines: Iterable[any]
        """
        for line_text in self._iter_lines(text_lines):
            words = text_helper.get_line_words(line_text, line_break_chars=self.line_break_chars)
            start_words, mid_words, end_words = text_helper.split_line_words(words)
            self.mid_freq.update(mid_words)
            self.start_freq.update(start_words)
            self.end_freq.update(end_words)
            self.all_freq.update(words)
            self.num_lines += 1

    def get_stats(self):
        """Return statistics on the frequency of characters occuring at the start, middle and end of
        a text line."""
        stats = defaultdict(list)
        for token_type, all_freq in self.all_freq.most_common():
            stats['token_type'].append(token_type)
            stats['all_freq'].append(self.all_freq[token_type])
            stats['all_frac'].append(self.all_frac[token_type])
            stats['start_freq'].append(self.start_freq[token_type])
            stats['start_frac'].append(self.start_frac[token_type])
            stats['start_rel_frac'].append(self.start_frac[token_type] / self.all_frac[token_type])
            stats['mid_freq'].append(self.mid_freq[token_type])
            stats['mid_frac'].append(self.mid_frac[token_type])
            stats['mid_rel_frac'].append(self.mid_frac[token_type] / self.all_frac[token_type])
            stats['end_freq'].append(self.end_freq[token_type])
            stats['end_frac'].append(self.end_frac[token_type])
            stats['end_rel_frac'].append(self.end_frac[token_type] / self.all_frac[token_type])
        return stats


class LineCharAnalyser(LineAnalyser):

    def __init__(self, text_lines: Iterable[any] = None, line_break_chars: Union[str, Set[str]] = '-',
                 ignorecase: bool = False):
        """A character frequency analyser of a list of text lines. Four frequencies are calculated:
        - all_freq: the overall frequency of a character
        - start_freq: the frequency of a character as the first character in a text line
        - mid_freq: the frequency of a character in the middle of in a text line (so neither as the first or
        last character of a line)
        - end_freq: the frequency of a character as the last character in a text line

        """
        super().__init__(line_break_chars=line_break_chars, ignorecase=ignorecase, token_type='char')
        if text_lines is not None:
            self.analyse_line_chars(text_lines)


class LineWordAnalyser(LineAnalyser):

    def __init__(self, text_lines: Iterable[any] = None, line_break_chars: Union[str, Set[str]] = '-',
                 ignorecase: bool = False):
        """A line word analyser class for building PageXML word-based corpus statistics.

        :param line_break_chars: a list of characters that can occur as line breaks.
        :type line_break_chars: str
        """
        super().__init__(line_break_chars=line_break_chars, ignorecase=ignorecase, token_type='word')
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
            words = text_helper.get_line_words(line_text, line_break_chars=self.line_break_chars)
            word_stats = get_word_cat_stats(words, **kwargs)
            for word_cat in word_stats:
                cat_stats[word_cat] += word_stats[word_cat]
        return cat_stats


class LineBreakDetector(LineWordAnalyser):

    def __init__(self, min_bigram_word_freq: int = 5, line_break_chars: Union[str, Set[str]] = '-',
                 ignorecase: bool = False):
        """A line break detector class that uses corpus statistics and a configurable list
        of line break characters to determine, for two subsequent lines, whether the first line ends
        with a line break (a word break off mid-word on the first line and continued on the second line.

        :param min_bigram_word_freq: the minimum frequency of word bigrams to be considered common bigrams
        :type min_bigram_word_freq: int
        :param line_break_chars: a list of characters that can occur as line breaks.
        :type line_break_chars: str
        """
        super().__init__(line_break_chars=line_break_chars, ignorecase=ignorecase)
        self.end_with_lbd_freq = Counter()
        self.start_with_lbd_freq = Counter()
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

    def reset_counters(self):
        """Reset all the counters."""
        self.mid_freq = Counter()
        self.end_freq = Counter()
        self.end_with_lbd_freq = Counter()
        self.start_freq = Counter()
        self.start_with_lbd_freq = Counter()
        self.all_freq = Counter()
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
        line_count = sum(self.start_freq.values())
        print("number of lines:", line_count)
        # print("number of non-empty lines:", sum(self.start_freq.values()))
        # print("number of empty lines:", line_count - sum(self.start_freq.values()))
        print("number of words per line:", sum(self.all_freq.values()) / line_count)
        print(f'{"all:": <12}{len(self.all_freq): >10} types\t{sum(self.all_freq.values()): >10} tokens')
        print(f'{"start:": <12}{len(self.start_freq): >10} types\t{sum(self.start_freq.values()): >10} tokens')
        print(f'{"mid:": <12}{len(self.mid_freq): >10} types\t{sum(self.mid_freq.values()): >10} tokens')
        print(f'{"end:": <12}{len(self.end_freq): >10} types\t{sum(self.end_freq.values()): >10} tokens')
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
            words = text_helper.get_line_words(line["text"], line_break_chars=self.line_break_chars)
            start_words, mid_words, end_words = text_helper.split_line_words(words)
            start_with_lbd_words = [w for w in start_words if w[0] in self.line_break_chars]
            start_with_lbd_words = ['<LBC>' + w[:1] for w in start_with_lbd_words]
            end_with_lbd_words = [w for w in end_words if w[-1] in self.line_break_chars]
            end_with_lbd_words += [w[:-1] + '<LBC>' for w in end_with_lbd_words]
            self.mid_freq.update(mid_words)
            self.start_freq.update(start_words)
            self.end_freq.update(end_words)
            self.start_with_lbd_freq.update(start_with_lbd_words)
            self.end_with_lbd_freq.update(end_with_lbd_words)
            self.all_freq.update(words)
        print(li + 1,
              f'lines processed'
              f'\tall: {len(self.all_freq): >8} types'
              f'\t{sum(self.all_freq.values()): >8} tokens')

    def _set_bigram_counter(self, lines: Iterable[Union[str, Dict[str, str]]]):
        li = 0
        for li, line in enumerate(lines):
            if line["text"] is None:
                continue
            words = text_helper.get_line_words(line["text"], line_break_chars=self.line_break_chars)
            start_words, mid_words, end_words = text_helper.split_line_words(words)
            for i in range(0, len(mid_words) - 2):
                if self.mid_freq[mid_words[i]] < self.min_bigram_word_freq or \
                        self.mid_freq[mid_words[i + 1]] < self.min_bigram_word_freq:
                    continue
                self.mid_bigram_freq.update([(mid_words[i], mid_words[i + 1])])
        print(li + 1, f'lines processed\tall: {len(self.mid_bigram_freq)} bigrams')

    def _set_merged_with(self, lines: Iterable[Union[str, Dict[str, str]]],
                         min_common_freq: int = 1000) -> None:
        prev_words = []
        typical_start_words, typical_end_words = get_typical_start_end_words(self)
        li = 0
        for li, line in enumerate(lines):
            if line["text"] is None:
                continue
            words = text_helper.get_line_words(line["text"], line_break_chars=self.line_break_chars)
            if len(prev_words) == 0 or len(words) == 0:
                pass
            else:
                end_word = prev_words[-1]
                start_word = words[0]
                merge_word = end_word + start_word
                reduce_word = text_helper.remove_line_break_chars(end_word, start_word, self.line_break_chars)
                merge_word = merge_word if self.mid_freq[merge_word] > self.mid_freq[reduce_word] else reduce_word
                if len(merge_word) > 0 and merge_word[-1] in self.line_break_chars:
                    # when the line start word ends with a hyphen, e.g. 'geval-' + 'len-' -> 'gevallen'
                    if self.mid_freq[merge_word[-1]] > self.mid_freq[merge_word]:
                        merge_word = merge_word[:-1]
                if end_word not in self.line_break_chars and self.mid_freq[merge_word] > 1:
                    if start_word in typical_start_words:
                        self.typical_start_merged_with[start_word].update([(end_word, merge_word)])
                    if end_word in typical_end_words:
                        self.typical_end_merged_with[end_word].update([(start_word, merge_word)])
                    if self.start_freq[start_word] >= min_common_freq:
                        self.common_start_merged_with[start_word].update([(end_word, merge_word)])
                    if self.end_freq[end_word] >= min_common_freq:
                        self.common_end_merged_with[end_word].update([(start_word, merge_word)])
            prev_words = words
        print(li + 1, 'lines processed')
        print('finished!')

    def _set_typical_start_ends(self):
        typical_start_words, typical_end_words = get_typical_start_end_words(self)
        for end_word in sorted(typical_end_words, key=lambda w: sum(self.typical_end_merged_with[w].values())):
            merge_exist_frac = sum(self.typical_end_merged_with[end_word].values()) / self.end_freq[end_word]
            if merge_exist_frac > 0.5:
                self.typical_merge_ends.add(end_word)
            elif merge_exist_frac < 0.05:
                self.typical_non_merge_ends.add(end_word)
            # merge_freq = sum([self.mid_freq[merged_word]
            #                   for start_word, merged_word in self.typical_end_merged_with[end_word]])
            # freqs = f"{self.end_freq[end_word]: >8}{self.mid_freq[end_word]: >8}{self.all_freq[end_word]: >8}"
            # print(f"{end_word: <20}{freqs}{merge_exist_frac: >8.2f}{merge_freq: >8}")
        for start_word in sorted(typical_start_words, key=lambda w: sum(self.typical_start_merged_with[w].values())):
            merge_exist_frac = sum(self.typical_start_merged_with[start_word].values()) / self.start_freq[start_word]
            if merge_exist_frac > 0.5:
                self.typical_merge_starts.add(start_word)
            elif merge_exist_frac < 0.05 and start_word.isupper() is False:
                self.typical_non_merge_starts.add(start_word)
            # merge_freq = sum([self.mid_freq[merged_word]
            #                   for end_word, merged_word in self.typical_start_merged_with[start_word]])
            # freqs = f"{self.start_freq[start_word]: >8}{self.mid_freq[start_word]: >8}{self.all_freq[start_word]: >8}"
            # print(f"{start_word: <20}{freqs}{merge_exist_frac: >8.2f}{merge_freq: >8}")

    def _set_common_start_ends(self):
        for start_word in sorted(self.common_start_merged_with, key=lambda t: self.mid_freq[t] / self.start_freq[t]):
            merge_exist_frac = sum(self.common_start_merged_with[start_word].values()) / self.start_freq[start_word]
            merge_freq = sum([self.mid_freq[merged_word]
                              for end_word, merged_word in self.common_start_merged_with[start_word]])
            if start_word in self.typical_merge_starts or start_word in self.typical_non_merge_starts:
                continue
            if merge_exist_frac < 0.02:
                self.common_non_merge_starts.add(start_word)
            if merge_exist_frac < 0.2 or merge_freq < 100:
                continue
            self.common_merge_starts.add(start_word)
        for end_word in sorted(self.common_end_merged_with, key=lambda t: self.mid_freq[t] / self.end_freq[t]):
            merge_exist_frac = sum(self.common_end_merged_with[end_word].values()) / self.end_freq[end_word]
            merge_freq = sum([self.mid_freq[merged_word]
                              for start_word, merged_word in self.common_end_merged_with[end_word]])
            if end_word in self.typical_merge_ends or end_word in self.typical_non_merge_ends:
                continue
            if merge_exist_frac < 0.02:
                self.common_non_merge_ends.add(end_word)
            if merge_exist_frac < 0.2 or merge_freq < 100:
                continue
            self.common_merge_ends.add(end_word)


def show_line_break_context(lbd: LineBreakDetector, end_word: str, start_word: str, merge_word: str,
                            match: str = None):
    last = f'{end_word: <15}{lbd.mid_freq[end_word]: >8}{lbd.end_freq[end_word]: >8}'
    first = f'\t{start_word: <15}{lbd.start_freq[start_word]: >8}{lbd.mid_freq[start_word]: >8}' \
            f'{lbd.all_freq[start_word]: >8}'
    merge = f'\t{merge_word: <15}{lbd.all_freq[merge_word]: >8}'
    if match:
        print(f'{last}{first}{merge}\t{match}')
    else:
        print(f'{last}{first}{merge}')


def determine_line_break(lbd: LineBreakDetector, curr_words: List[str],
                         prev_words: List[str], debug: bool = False) -> Tuple[bool, Union[str, None]]:
    """Determine for a current line and previous line (as lists of words) whether the first line ends with a line break.

    :param lbd: a line break detector object
    :type lbd: LineBreakDetector
    :param curr_words: a list of words for the current line to be merged with the previous line
    :type curr_words: List[str]
    :param prev_words: a list of words for the previous line to be merged with the current line
    :type prev_words: List[str]
    :return: a flag whether the previous line ends in a line break and the merged word composed of
    the previous line's last word and current line's first word (or None if the words should not be merged)
    :rtype: Union[str, None]
    """
    if len(prev_words) == 0 or len(curr_words) == 0:
        # print('includes non-word')
        return False, None
    end_word = prev_words[-1]
    start_word = curr_words[0]
    merge_word = end_word + start_word
    reduce_word = text_helper.remove_line_break_chars(end_word, start_word, lbd.line_break_chars)
    merge_word = merge_word if lbd.all_freq[merge_word] > lbd.all_freq[reduce_word] else reduce_word
    if debug:
        print(f"end: #{end_word}#\tstart: #{start_word}#")
        print('reduce_word', reduce_word)
        print('merge_word', merge_word)
    bigram_freq = lbd.mid_bigram_freq[(end_word, start_word)]
    if end_word[-1] in lbd.line_break_chars:
        bigram_freq = lbd.mid_bigram_freq[(end_word[:-1], start_word)]
        # print(end_word, lbd.mid_freq[end_word], start_word, lbd.mid_freq[start_word], (end_word[:-1], start_word),
        #       merge_word, lbd.mid_freq[merge_word], 'bigram_freq:', bigram_freq)
    if has_non_merge_word(lbd, end_word, start_word):
        if debug:
            print('has_none_merge_word', end_word, start_word)
        return False, None
    if end_start_are_bigram(lbd, merge_word, bigram_freq, factor=5):
        if debug:
            print('end_start_are_bigram', end_word, start_word)
        return False, None
    elif start_is_titleword(start_word):
        if end_start_are_hyphenated_compound(lbd, end_word, start_word, merge_word):
            merge_word = end_word + start_word
            # print('end_start_are_hyphenated_compound', end_word, start_word)
            if debug:
                print('end_start_are_hyphenated_compound', end_word, lbd.mid_freq[end_word], start_word, lbd.mid_freq[start_word],
                  merge_word, lbd.mid_freq[merge_word])
            return True, merge_word
        elif start_word_has_incorrect_titlecase(lbd, end_word, start_word, factor=10):
            # print('start_word_has_incorrect_titlecase', end_word, start_word)
            if debug:
                print('start_word_has_incorrect_titlecase', end_word, lbd.mid_freq[end_word], start_word, lbd.mid_freq[start_word],
                  merge_word, lbd.mid_freq[merge_word])
            return True, merge_word
        else:
            if debug:
                print('start_word_is_titleword', end_word, start_word)
            return False, None
    elif has_common_merge_end(lbd, end_word, start_word):
        # print('has_common_merge_end', end_word, start_word, merge_word)
        if debug:
            print('has_common_merge_end', end_word, lbd.mid_freq[end_word], start_word, lbd.mid_freq[start_word],
                  merge_word, lbd.mid_freq[merge_word])
        return True, merge_word
    elif has_line_break_symbol(lbd, end_word, start_word, merge_word):
        # print('has_line_break_symbol', end_word, start_word, merge_word)
        if debug:
            print('has_line_break_symbol', end_word, lbd.mid_freq[end_word], start_word, lbd.mid_freq[start_word],
                  merge_word, lbd.mid_freq[merge_word])
        return True, merge_word
    if end_start_are_bigram(lbd, merge_word, bigram_freq, factor=2):
        if debug:
            print('end_start_are_bigram', end_word, start_word)
        return False, None
    if end_is_common_word(lbd, end_word, start_word, common_freq=1000):
        if debug:
            print('end_is_common_word', end_word, start_word)
        return False, None
    elif merge_is_more_common(lbd, end_word, start_word, merge_word):
        # print('merge_is_more_common', end_word, start_word)
        if debug:
            print('merge_is_more_common', end_word, lbd.mid_freq[end_word], start_word, lbd.mid_freq[start_word],
                  merge_word, lbd.mid_freq[merge_word])
        return True, merge_word
    elif end_word[-1] in lbd.line_break_chars:
        # print('merge_line_break', end_word, start_word, merge_word)
        if debug:
            print('merge line break', end_word, lbd.mid_freq[end_word], start_word, lbd.mid_freq[start_word],
                  merge_word, lbd.mid_freq[merge_word])
        return True, merge_word
        # show_line_break_context(lbd, end_word, start_word, merge_word)
    else:
        if debug:
            print('OTHER', end_word, lbd.mid_freq[end_word], start_word, lbd.mid_freq[start_word], merge_word, lbd.mid_freq[merge_word])
        return False, None


def merge_is_more_common(lbd, end_word, start_word, merge_word):
    if lbd.all_freq[merge_word] > lbd.mid_freq[end_word] and \
            lbd.all_freq[merge_word] > lbd.mid_freq[start_word]:
        return True
    elif is_non_mid_word(lbd, end_word, factor=5) and \
            lbd.all_freq[merge_word] > lbd.mid_freq[start_word]:
        return True
    elif is_non_mid_word(lbd, start_word, factor=5) and \
            lbd.all_freq[merge_word] > lbd.mid_freq[end_word]:
        return True
    elif lbd.all_freq[merge_word] > 0:
        return True
    else:
        return False


def end_is_common_word(lbd: LineBreakDetector, end_word: str,
                       start_word: str, common_freq: int = 100,
                       debug: bool = False) -> bool:
    # return lbd.all_freq[end_word] >= common_freq or lbd.all_freq[start_word] >= common_freq
    if debug:
        print(end_word, lbd.mid_freq[end_word], common_freq)
    return lbd.mid_freq[end_word] >= common_freq


def has_line_break_symbol(lbd, end_word, start_word, merge_word):
    if end_word[-1] != '-':
        return False
    if lbd.all_freq[merge_word] > lbd.all_freq[end_word]:
        return True
    elif lbd.all_freq[merge_word] > 0:
        return True
    elif lbd.mid_freq[start_word] > lbd.start_freq[start_word]:
        return False
    elif start_word.isdigit():
        return False
    else:
        return True


def has_common_merge_end(lbd: LineBreakDetector, end_word: str,
                         start_word: str) -> bool:
    if end_word in lbd.typical_merge_ends:
        return True
    elif start_word in lbd.typical_merge_starts:
        return True
    if start_word in lbd.common_non_merge_starts:
        return False
    else:
        return False


def is_non_mid_word(lbd: LineBreakDetector, word: str, factor: int = 5) -> bool:
    if lbd.end_freq[word] > factor * lbd.mid_freq[word]:
        return True
    if lbd.start_freq[word] > factor * lbd.mid_freq[word]:
        return True
    return False


def start_word_has_incorrect_titlecase(lbd: LineBreakDetector, end_word: str,
                                       start_word: str, factor: int = 10) -> bool:
    if start_word in lbd.common_non_merge_starts:
        return False
    if start_word.isupper():
        return False
    if lbd.all_freq[start_word] < factor and lbd.all_freq[end_word] < factor:
        return False
    return is_non_mid_word(lbd, start_word) and is_non_mid_word(lbd, end_word)


def end_start_are_hyphenated_compound(lbd: LineBreakDetector, end_word: str,
                                      start_word: str, merge_word: str) -> bool:
    if start_word.isupper():
        # entire start word is upper case, so no part of compound
        return False
    if end_word[0].isupper() and end_word[-1] == '-' and start_word[0].isupper():
        # end word and start word are both in title case, and end word
        # ends with hyphen, so they're probably a hyphenated compound
        if lbd.mid_freq[start_word] == 0 and lbd.all_freq[merge_word] == 0 and \
                lbd.all_freq[end_word + start_word] == 0:
            # start_word is never observed in the middle of a line, so is likely
            # a broken off word, and is incorrectly title cased or
            # its sentence is not the correct one following end_word
            return False
        if lbd.mid_freq[end_word] == 0 and lbd.all_freq[merge_word] == 0 and \
                lbd.all_freq[end_word + start_word] == 0:
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


def end_start_are_bigram(lbd: LineBreakDetector, merge_word: str, bigram_freq: int,
                         factor: int = 5) -> bool:
    # the bigram of end word and start word is much more frequent than their
    # merge, so treat as bigram
    # if frequency of merge_word is zero, bigram_freq should be at least 5
    return bigram_freq > factor and bigram_freq > factor * lbd.all_freq[merge_word]


def determine_line_break_typical_merge_end(lbd: LineBreakDetector, end_word: str,
                                           start_word: str, merge_word: str) -> bool:
    if end_word in lbd.typical_merge_ends:
        if lbd.all_freq[merge_word] >= 10:
            return True
        elif lbd.mid_freq[start_word] > 100 and end_word.endswith('-'):
            if lbd.mid_freq[end_word[:-1]] > 100 and lbd.mid_freq[end_word[:-1]] > 10 * lbd.mid_freq[end_word]:
                return False
            elif lbd.all_freq[merge_word] > 0:
                return True
            else:
                return False
        else:
            return True


def has_non_merge_word(lbd: LineBreakDetector, end_word: str,
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
    elif end_word in lbd.typical_non_merge_ends:
        # start word is a non-merge word so don't merge
        if debug:
            print(f'end_word is non_merge_end: #{end_word}#')
        return True
    elif start_word in lbd.typical_non_merge_starts:
        # start word is a non-merge word so don't merge
        if debug:
            print(f'start_word is non_merge_start: #{start_word}#')
        return True
    else:
        return False


def get_typical_start_end_words(lbd: LineBreakDetector,
                                threshold: float = 0.5) -> Tuple[Set[str], Set[str]]:
    typical_start_words = set()
    for end_word in lbd.start_freq:
        if lbd.start_freq[end_word] > 100 and \
                lbd.start_freq[end_word] / lbd.all_freq[end_word] > threshold:
            typical_start_words.add(end_word)
    typical_end_words = set()
    for end_word in lbd.end_freq:
        if lbd.end_freq[end_word] > 100 and \
                lbd.end_freq[end_word] / lbd.all_freq[end_word] > threshold:
            typical_end_words.add(end_word)
    return typical_start_words, typical_end_words


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
        return [w for line in lines for w in re.split(r'\b', line.text) if w != ' ' and w != '']
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
        'num_number_words': len([w for w in words if w.isdigit()]),
        'num_title_words': len([w for w in words if w.istitle()]),
        'num_non_title_words': len([w for w in words if w.istitle() is False]),
        'num_stop_words': len([w for w in words if w in stop_words]) if stop_words is not None else None,
        'num_punctuation_words': len([w for w in words if all(j in puncs for j in w)]),
        'num_oversized_words': num_oversized_words
    }
    word_length_bin = word_length_bin_size
    word_cat_stats[f'num_words_length_{word_length_bin}'] = 0
    for wl in range(1, max_word_length+1):
        if wl == word_length_bin:
            word_length_bin += word_length_bin_size
            word_cat_stats[f'num_words_length_{word_length_bin}'] = 0
        word_cat_stats[f'num_words_length_{word_length_bin}'] += word_length_freq[wl]
    return word_cat_stats
