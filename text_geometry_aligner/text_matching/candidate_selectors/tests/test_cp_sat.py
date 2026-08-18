"""Tests for the CP-SAT candidate selector."""

import importlib.util
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

import text_geometry_aligner as alignment
from text_geometry_aligner.text_matching.candidate_generators import (
    anchored_fuzzy as candidate_generation,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None,
    reason="OR-Tools is not installed",
)


def _page(*texts: str, block_indexes: tuple[int, ...] | None = None) -> alignment.ALTOPage:
    if block_indexes is None:
        block_indexes = (0,) * len(texts)
    words = tuple(
        alignment.ALTOWord(
            index=index,
            text=text,
            bbox=alignment.BoundingBox(index * 10, 0, 9, 10),
            line_index=0,
            block_index=block_indexes[index],
        )
        for index, text in enumerate(texts)
    )
    return alignment.ALTOPage(source_path=Path("test.xml"), words=words)


def _value(value_id: int, text: str) -> alignment.AlignmentRegion:
    return alignment.AlignmentRegion(
        region_id=value_id,
        label=f"value_{value_id}",
        input_text=text,
        input_text_normalized=text.casefold(),
        json_text_path=(f"value_{value_id}",),
    )


def _distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _combined_generator(
    fuzzy_config: alignment.FuzzyCandidateConfig | None = None,
) -> alignment.CompositeCandidateGenerator:
    return alignment.CompositeCandidateGenerator(
        (
            alignment.ExactTextCandidateGenerator(),
            alignment.AnchoredFuzzyTextCandidateGenerator(fuzzy_config),
        )
    )


def test_equal_matches_prefer_earlier_alto_start_independently_of_id(
    lowercase_normalizer,
) -> None:
    values = (_value(0, "Rome"),)
    generated = alignment.ExactTextCandidateGenerator().generate(
        values,
        alignment.ALTOTextIndex(
            _page("Rome", "other", "Rome"),
            lowercase_normalizer,
        ),
    )
    earlier, later = generated
    candidates = (
        replace(earlier, candidate_id=1),
        replace(later, candidate_id=0),
    )

    selected = alignment.CPSATCandidateSelector().select(
        candidates,
        values,
    )

    assert len(selected) == 1
    assert selected[0].start_word == 0
    assert selected[0].candidate_id == 1


def test_long_near_exact_phrase_beats_short_exact_substring(
    lowercase_normalizer,
) -> None:
    regions = (
        _value(0, "Rome"),
        _value(1, "Library of Reme"),
    )
    index = alignment.ALTOTextIndex(
        _page("Library", "of", "Rome"),
        lowercase_normalizer,
    )

    with mock.patch.object(
        candidate_generation,
        "_load_levenshtein_distance",
        return_value=_distance,
    ):
        candidates = _combined_generator().generate(
            regions,
            index,
        )
    selected = alignment.CPSATCandidateSelector().select(
        candidates,
        regions,
    )

    assert len(selected) == 1
    assert selected[0].region_id == 1
    assert (selected[0].start_word, selected[0].end_word) == (0, 2)
    assert not selected[0].exact
