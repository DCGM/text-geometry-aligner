"""Tests for text builders."""

import pytest

import text_geometry_aligner as alignment


def _word(
    index: int,
    text: str,
    x: float,
    y: float = 0,
    width: float = 10,
    height: float = 10,
) -> alignment.ALTOWord:
    return alignment.ALTOWord(
        index=index,
        text=text,
        bbox=alignment.BoundingBox(x, y, width, height),
        line_index=0,
        block_index=0,
    )


def test_base_builder_requires_a_build_implementation() -> None:
    with pytest.raises(TypeError):
        alignment.TextBuilder()


def test_space_separated_builder_joins_words_in_input_order() -> None:
    builder = alignment.SpaceSeparatedTextBuilder()

    assert builder.build((_word(1, "ONE", 0),)) == "ONE"
    assert (
        builder.build((_word(1, "ONE", 0), _word(2, "TWO", 10)))
        == "ONE TWO"
    )
    assert builder.build(()) is None
