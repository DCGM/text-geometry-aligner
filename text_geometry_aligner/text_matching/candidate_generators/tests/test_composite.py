"""Tests for the composite candidate generator."""

from dataclasses import replace
from pathlib import Path

import pytest

import text_geometry_aligner as alignment


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


class _StaticCandidateGenerator(alignment.CandidateGenerator):
    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        self.calls = []

    def generate(self, regions, alto_index):
        self.calls.append((regions, alto_index))
        return self.candidates


def test_requires_at_least_one_generator() -> None:
    with pytest.raises(ValueError, match="at least one generator"):
        alignment.CompositeCandidateGenerator(())


def test_earlier_generator_wins_duplicate_span_and_ids_are_reassigned(
    lowercase_normalizer,
) -> None:
    index = alignment.ALTOTextIndex(_page("Rome"), lowercase_normalizer)
    values = (_value(0, "rome"),)
    exact_candidate = alignment.ExactTextCandidateGenerator().generate(
        values,
        index,
    )[0]
    later_duplicate = replace(
        exact_candidate,
        candidate_id=17,
        exact=False,
        edit_distance=1,
        source="later",
    )
    first = _StaticCandidateGenerator((exact_candidate,))
    second = _StaticCandidateGenerator((later_duplicate,))

    candidates = alignment.CompositeCandidateGenerator(
        (first, second)
    ).generate(values, index)

    assert len(candidates) == 1
    assert candidates[0].exact
    assert candidates[0].candidate_id == 0
    assert len(first.calls) == 1
    assert len(second.calls) == 1
