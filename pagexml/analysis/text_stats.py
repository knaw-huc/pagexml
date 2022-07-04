from typing import List, Set, Tuple, Union
from collections import Counter
from collections import defaultdict
import re

import pagexml.helper.text_helper as text_helper


class LineBreakDetector:

    def __init__(self, min_bigram_word_freq: int = 5, line_break_char: str = '-'):
        self.mid_freq = Counter()
        self.end_freq = Counter()
        self.start_freq = Counter()
        self.all_freq = Counter()
        self.mid_bigram_freq = Counter()
        self.line_break_char = line_break_char
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
        self.mid_freq = Counter()
        self.end_freq = Counter()
        self.start_freq = Counter()
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

    def set_counters(self, line_reader: text_helper.LineReader):
        print('Step 1: setting unigram counters')
        self._set_unigram_counters(line_reader)
        print('Step 2: setting bigram counter')
        self._set_bigram_counter(line_reader)
        print('Step 3: setting common merge counters')
        self._set_merged_with(line_reader)
        print('Step 4: setting typical line ends and starts')
        self._set_typical_start_ends()
        print('Step 5: setting common line ends and starts')
        self._set_common_start_ends()

    def _set_unigram_counters(self, line_reader: text_helper.LineReader):
        li = 0
        for li, line in enumerate(line_reader):
            if line["text"] is None:
                continue
            words = text_helper.get_line_words(line["text"], line_break_char=self.line_break_char)
            start_words, mid_words, end_words = text_helper.split_line_words(words)
            self.mid_freq.update(mid_words)
            self.start_freq.update(start_words)
            self.end_freq.update(end_words)
            self.all_freq.update(words)
        print(li + 1,
              f'lines processed'
              f'\tall: {len(self.all_freq): >8} types'
              f'\t{sum(self.all_freq.values()): >8} tokens')

    def _set_bigram_counter(self, line_reader: text_helper.LineReader):
        li = 0
        for li, line in enumerate(line_reader):
            if line["text"] is None:
                continue
            words = text_helper.get_line_words(line["text"], line_break_char=self.line_break_char)
            start_words, mid_words, end_words = text_helper.split_line_words(words)
            for i in range(0, len(mid_words) - 2):
                if self.mid_freq[mid_words[i]] < self.min_bigram_word_freq or \
                        self.mid_freq[mid_words[i + 1]] < self.min_bigram_word_freq:
                    continue
                self.mid_bigram_freq.update([(mid_words[i], mid_words[i + 1])])
        print(li + 1, f'lines processed\tall: {len(self.mid_bigram_freq)} bigrams')

    def _set_merged_with(self, line_reader: text_helper.LineReader, min_common_freq: int = 1000) -> None:
        prev_words = []
        typical_start_words, typical_end_words = get_typical_start_end_words(self)
        li = 0
        for li, line in enumerate(line_reader):
            if line["text"] is None:
                continue
            words = text_helper.get_line_words(line["text"], line_break_char=self.line_break_char)
            if len(prev_words) == 0 or len(words) == 0:
                pass
            else:
                end_word = prev_words[-1]
                start_word = words[0]
                merge_word = end_word + start_word
                reduce_word = text_helper.remove_hyphen(end_word) + start_word
                merge_word = merge_word if self.mid_freq[merge_word] > self.mid_freq[reduce_word] else reduce_word
                if merge_word[-1] == '-':
                    # when the line start word ends with a hyphen, e.g. 'geval-' + 'len-' -> 'gevallen'
                    if self.mid_freq[merge_word[-1]] > self.mid_freq[merge_word]:
                        merge_word = merge_word[:-1]
                if end_word != '-' and self.mid_freq[merge_word] > 1:
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
                         prev_words: List[str]) -> Tuple[bool, Union[str, None]]:
    if len(prev_words) == 0 or len(curr_words) == 0:
        return False, None
    end_word = prev_words[-1]
    start_word = curr_words[0]
    merge_word = end_word + start_word
    reduce_word = text_helper.remove_hyphen(end_word) + start_word
    merge_word = merge_word if lbd.all_freq[merge_word] > lbd.all_freq[reduce_word] else reduce_word
    bigram_freq = lbd.mid_bigram_freq[(end_word, start_word)]
    if end_word.endswith('-'):
        bigram_freq = lbd.mid_bigram_freq[(end_word[:-1], start_word)]
    if has_non_merge_word(lbd, end_word, start_word):
        return False, None
    if end_start_are_bigram(lbd, merge_word, bigram_freq, factor=5):
        return False, None
    elif start_is_titleword(start_word):
        if end_start_are_hyphenated_compound(lbd, end_word, start_word, merge_word):
            merge_word = end_word + start_word
            return True, merge_word
        elif start_word_has_incorrect_titlecase(lbd, end_word, start_word, factor=10):
            return True, merge_word
        else:
            return False, None
    elif has_common_merge_end(lbd, end_word, start_word):
        return True, merge_word
    elif has_line_break_symbol(lbd, end_word, start_word, merge_word):
        return True, merge_word
    if end_start_are_bigram(lbd, merge_word, bigram_freq, factor=2):
        return False, None
    if one_is_common_word(lbd, end_word, start_word, common_freq=1000):
        return False, None
    elif merge_is_more_common(lbd, end_word, start_word, merge_word):
        return True, merge_word
    else:
        # show_line_break_context(lbd, end_word, start_word, merge_word)
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


def one_is_common_word(lbd: LineBreakDetector, end_word: str,
                       start_word: str, common_freq: int = 100) -> bool:
    return lbd.all_freq[end_word] >= common_freq or lbd.all_freq[start_word] >= common_freq


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


def has_non_merge_word(lbd: LineBreakDetector, end_word: str, start_word: str) -> bool:
    if not re.search(r'\w', end_word) or not re.search(r'\w', start_word):
        # one of the words is just punctuation, so don't merge
        return True
    if end_word == '-' or start_word == '-':
        # one of the words is just hyphen, so don't merge
        return True
    elif end_word in lbd.typical_non_merge_ends or start_word in lbd.typical_non_merge_starts:
        # one of the words is a non-merge word so don't merge
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
