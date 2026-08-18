"""Tests for the pass-through candidate selector."""

from pathlib import Path

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


def test_returns_candidates_unchanged_and_in_the_same_order(lowercase_normalizer) -> None:
    values = (_value(0, "rome"),)
    candidates = alignment.ExactTextCandidateGenerator().generate(
        values,
        alignment.ALTOTextIndex(_page("Rome", "Rome"), lowercase_normalizer),
    )

    selected = alignment.PassThroughCandidateSelector().select(
        candidates,
        values,
    )

    assert selected is candidates
