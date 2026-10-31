"""Locate a value printed inside another value within that value's words."""

from __future__ import annotations

import dataclasses
import re
from typing import Mapping, Sequence

import Levenshtein

from ..io_alto import ALTOPage
from ..models import AlignmentRegion
from .alto_text_index import ALTOTextIndex
from .candidate import AlignmentCandidate
from .path_pattern import JSONPathPattern

InsideRule = tuple[JSONPathPattern, tuple[JSONPathPattern, ...]]

# How a value's text must be in its container's text to be located inside:
# "exact" a substring of the input text as given, "normalized" a substring
# of the normalized text, "words" whole consecutive words of the normalized
# text, punctuation aside.
INSIDE_MATCHES = ("exact", "normalized", "words")

# What the words a container matched read when its inside values are aligned
# on them: "json" the container's text, "alto" their OCR text.
INSIDE_LOCATIONS = ("json", "alto")


def parse_inside_rules(inside_paths: Mapping[str, str | Sequence[str]]) -> tuple[InsideRule, ...]:
    """The value pattern with its container patterns, in the given order."""

    rules = []
    for value, containers in inside_paths.items():
        names = (containers,) if isinstance(containers, str) else tuple(containers)
        if not names:
            raise ValueError(f"inside_paths {value!r} names no container")
        rules.append((JSONPathPattern.parse(value), tuple(JSONPathPattern.parse(name) for name in names)))
    return tuple(rules)


def inside_parents(
    regions: Sequence[AlignmentRegion],
    rules: Sequence[InsideRule],
    match: str = "exact",
) -> dict[int, AlignmentRegion]:
    """Per value printed inside another one, the value it was taken out of.

    A word goes to one value only, so a value printed in another one's words
    - a part number in its designation ("3" of "Kapitola 3"), an affiliation
    in its person's description - would lose them to it, or take them from
    it. A value under a rule's pattern whose text is in the text of a value
    under one of the rule's container patterns (as `match` tells), with the
    same bound indices, is aligned on that value's words only
    (container_page), one level below it. The rule says a text was taken out
    of another one, so by default it must be a substring of it as given.
    The first container holding it, in the order of the patterns and then of
    the input, is its parent; the first rule a value matches decides.
    """

    if match not in INSIDE_MATCHES:
        raise ValueError(f"inside match {match!r} is not one of {', '.join(INSIDE_MATCHES)}")
    parents: dict[int, AlignmentRegion] = {}
    for region in regions:
        if region.json_text_path is None or not _text(region, match):
            continue
        for value_pattern, container_patterns in rules:
            bound = value_pattern.match(region.json_text_path)
            if bound is None:
                continue
            parent = next(
                (
                    container
                    for container_pattern in container_patterns
                    for container in regions
                    if container is not region
                    and container.json_text_path is not None
                    and _text(container, match)
                    and _agree(bound, container_pattern.match(container.json_text_path))
                    and _contains(_text(container, match), _text(region, match), match)
                ),
                None,
            )
            if parent is not None:
                parents[region.region_id] = parent
            break
    return parents


def container_page(
    container: AlignmentRegion,
    span: AlignmentCandidate,
    alto_page: ALTOPage,
    alto_index: ALTOTextIndex,
    location: str = "json",
) -> ALTOPage:
    """The words `container` matched, as a page of their own to align its inside values on.

    With `location` "alto" the words keep their OCR text. By default ("json")
    each word reads the part of the container's normalized text aligned to
    it by a character alignment of the container's text with the words'
    text, so a value is aligned against the text the container gives rather
    than against the OCR's reading of it. The words keep their boxes.
    """

    if location not in INSIDE_LOCATIONS:
        raise ValueError(f"inside location {location!r} is not one of {', '.join(INSIDE_LOCATIONS)}")
    words = alto_page.words[span.start_word:span.end_word + 1]
    texts = [word.text for word in words]
    interval = alto_index.char_interval_for_word_interval(span.start_word, span.end_word)
    if location == "json" and interval is not None:
        container_text = container.input_text_normalized or ""
        page_text = alto_index.normalized_text[interval[0]:interval[1]]
        # The container character aligned to each character of the words'
        # text; a word reads from its first character's up to the next word's,
        # the first word from the start and the last one to the end, so every
        # character of the container's text goes to one word.
        to_container = [len(container_text)] * (len(page_text) + 1)
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(container_text, page_text):
            for j in range(j1, j2):
                to_container[j] = i1 if tag == "insert" else i1 + (j - j1) * (i2 - i1) // (j2 - j1)
        starts = {
            word_span.word_index - span.start_word: word_span.char_start - interval[0]
            for word_span in alto_index.word_spans
            if span.start_word <= word_span.word_index <= span.end_word
        }
        bounds = sorted(starts.items())
        texts = [""] * len(words)
        for number, (word, char) in enumerate(bounds):
            begin = 0 if number == 0 else to_container[char]
            finish = to_container[bounds[number + 1][1]] if number + 1 < len(bounds) else len(container_text)
            texts[word] = container_text[begin:finish].strip()
    return ALTOPage(
        source_path=alto_page.source_path,
        words=tuple(
            dataclasses.replace(word, index=number, text=text)
            for number, (word, text) in enumerate(zip(words, texts))
        ),
    )


def on_page(
    candidate: AlignmentCandidate,
    span: AlignmentCandidate,
    alto_index: ALTOTextIndex,
) -> AlignmentCandidate:
    """A candidate on a container's page (container_page) as one on the whole page."""

    start_word, end_word = candidate.start_word + span.start_word, candidate.end_word + span.start_word
    start_char, end_char = alto_index.char_interval_for_word_interval(start_word, end_word) or (
        candidate.start_char, candidate.end_char)
    return dataclasses.replace(
        candidate,
        start_word=start_word,
        end_word=end_word,
        start_char=start_char,
        end_char=end_char,
        matched_text=alto_index.text_for_word_interval(start_word, end_word),
        normalized_matched_text=alto_index.normalized_text_for_word_interval(start_word, end_word),
    )


def _agree(bound: tuple[tuple[str, int], ...], container_bound: tuple[tuple[str, int], ...] | None) -> bool:
    if container_bound is None:
        return False
    container = dict(container_bound)
    return all(container.get(name, index) == index for name, index in bound)


def _tokens(text: str) -> list[str]:
    return [token for token in (re.sub(r"\W", "", part) for part in text.split()) if token]


def _text(region: AlignmentRegion, match: str) -> str:
    if match == "exact":
        return "" if region.input_text is None else str(region.input_text)
    return region.input_text_normalized or ""


def _contains(container_text: str, text: str, match: str) -> bool:
    """Whether `text` is in `container_text` as `match` tells."""

    if match != "words":
        return text in container_text
    tokens, words = _tokens(text), _tokens(container_text)
    return bool(tokens) and any(
        words[start:start + len(tokens)] == tokens for start in range(len(words) - len(tokens) + 1)
    )
