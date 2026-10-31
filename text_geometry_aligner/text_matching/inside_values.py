"""Locate a value printed inside another value within that value's words."""

from __future__ import annotations

import dataclasses
import re
from typing import Mapping, Sequence

from ..io_alto import ALTOPage
from ..models import AlignmentRegion
from ..normalization import TextNormalizer
from .alto_text_index import ALTOTextIndex
from .candidate import AlignmentCandidate
from .candidate_generators.base import CandidateGenerator
from .path_pattern import JSONPathPattern

InsideRule = tuple[JSONPathPattern, tuple[JSONPathPattern, ...]]

# How a value's text must be in its container's text to be located inside:
# "exact" a substring of the input text as given, "normalized" a substring
# of the normalized text, "words" whole consecutive words of the normalized
# text, punctuation aside.
INSIDE_MATCHES = ("exact", "normalized", "words")


def parse_inside_rules(inside_paths: Mapping[str, str | Sequence[str]]) -> tuple[InsideRule, ...]:
    """The value pattern with its container patterns, in the given order."""

    rules = []
    for value, containers in inside_paths.items():
        names = (containers,) if isinstance(containers, str) else tuple(containers)
        if not names:
            raise ValueError(f"inside_paths {value!r} names no container")
        rules.append((JSONPathPattern.parse(value), tuple(JSONPathPattern.parse(name) for name in names)))
    return tuple(rules)


def inside_containers(
    regions: Sequence[AlignmentRegion],
    rules: Sequence[InsideRule],
    match: str = "exact",
) -> dict[int, list[AlignmentRegion]]:
    """Per value printed inside another one, the values it may be inside, in order.

    A word goes to one value only, so a value printed in another one's words
    - a part number in its designation ("3" of "Kapitola 3"), an affiliation
    in its person's description - would lose them to it, or take them from
    it. A value under a rule's pattern whose text is in the text of a value
    under one of the rule's container patterns (as `match` tells), with the
    same bound indices, is left out of the alignment and located afterwards
    within its container's words (locate_inside). The rule says a text was
    taken out of another one, so by default it must be a substring of it as
    given. The containers come in the order of their patterns, then of the
    input; the first rule a value matches decides.
    """

    if match not in INSIDE_MATCHES:
        raise ValueError(f"inside match {match!r} is not one of {', '.join(INSIDE_MATCHES)}")
    by_region: dict[int, list[AlignmentRegion]] = {}
    for region in regions:
        if region.json_text_path is None or not _text(region, match):
            continue
        for value_pattern, container_patterns in rules:
            bound = value_pattern.match(region.json_text_path)
            if bound is None:
                continue
            containers = [
                container
                for container_pattern in container_patterns
                for container in regions
                if container is not region
                and container.json_text_path is not None
                and _text(container, match)
                and _agree(bound, container_pattern.match(container.json_text_path))
                and _contains(_text(container, match), _text(region, match), match)
            ]
            if containers:
                by_region[region.region_id] = containers
            break
    return by_region


def locate_inside(
    region: AlignmentRegion,
    containers: Sequence[AlignmentRegion],
    selected: Mapping[int, AlignmentCandidate],
    alto_page: ALTOPage,
    alto_index: ALTOTextIndex,
    candidate_generator: CandidateGenerator,
    normalizer: TextNormalizer,
    minimum_similarity_int: int = 0,
) -> tuple[AlignmentCandidate, AlignmentRegion] | None:
    """The best match of `region` within the words of its first container that has one.

    The candidate generator looks for the value among the container's
    matched words only, so a word misread by the OCR is found as in the
    alignment. A container without a match of its own gives nothing.
    """

    for container in containers:
        span = selected.get(container.region_id)
        if span is None:
            continue
        words = alto_page.words[span.start_word:span.end_word + 1]
        sub_page = ALTOPage(
            source_path=alto_page.source_path,
            words=tuple(dataclasses.replace(word, index=number) for number, word in enumerate(words)),
        )
        sub_index = ALTOTextIndex(sub_page, normalizer)
        candidates = [
            candidate
            for candidate in candidate_generator.generate([region], sub_index)
            if candidate.region_id == region.region_id
            and candidate.similarity_int >= minimum_similarity_int
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda candidate: (
            candidate.quality_chars, candidate.exact, candidate.similarity_int, -candidate.start_word))
        # The container's words are joined into the same text in both indexes,
        # so character offsets differ by where its first word starts.
        full_spans = {word_span.word_index: word_span for word_span in alto_index.word_spans}
        char_offset = next(
            (full_spans[span.start_word + sub_span.word_index].char_start - sub_span.char_start
             for sub_span in sub_index.word_spans),
            0,
        )
        return dataclasses.replace(
            best,
            start_word=best.start_word + span.start_word,
            end_word=best.end_word + span.start_word,
            start_char=best.start_char + char_offset,
            end_char=best.end_char + char_offset,
        ), container
    return None


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
