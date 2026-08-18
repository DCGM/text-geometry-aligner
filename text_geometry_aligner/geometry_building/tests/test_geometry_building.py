"""Tests for the orthogonal polygon geometry builder."""

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


def test_base_builder_requires_a_build_implementation() -> None:
    with pytest.raises(TypeError):
        alignment.GeometryBuilder()


def test_single_word_polygon_is_a_closed_rectangle() -> None:
    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(
        _page("WORD").words
    )

    assert polygon.points == ((0, 0), (9, 0), (9, 10), (0, 10), (0, 0))
    assert polygon.to_json() == [[0, 0], [9, 0], [9, 10], [0, 10], [0, 0]]
    assert polygon.bounds == alignment.BoundingBox(0, 0, 9, 10)


def test_multiline_polygon_breaks_at_bottom_of_upper_line() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="FULL",
            bbox=alignment.BoundingBox(0, 0, 100, 10),
            line_index=0,
        ),
        alignment.ALTOWord(
            index=1,
            text="HALF",
            bbox=alignment.BoundingBox(0, 12, 50, 10),
            line_index=1,
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == (
        (0, 0),
        (100, 0),
        (100, 10),
        (50, 10),
        (50, 22),
        (0, 22),
        (0, 0),
    )


def test_multiline_polygon_steps_on_both_sides() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="UPPER",
            bbox=alignment.BoundingBox(20, 0, 80, 10),
            line_index=0,
        ),
        alignment.ALTOWord(
            index=1,
            text="LOWER",
            bbox=alignment.BoundingBox(0, 12, 50, 10),
            line_index=1,
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == (
        (20, 0),
        (100, 0),
        (100, 10),
        (50, 10),
        (50, 22),
        (0, 22),
        (0, 12),
        (20, 12),
        (20, 0),
    )


def test_each_edge_uses_its_own_inward_or_outward_break() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="UPPER",
            bbox=alignment.BoundingBox(0, 0, 100, 10),
            line_index=0,
        ),
        alignment.ALTOWord(
            index=1,
            text="LOWER",
            bbox=alignment.BoundingBox(20, 12, 100, 10),
            line_index=1,
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == (
        (0, 0),
        (100, 0),
        (100, 12),
        (120, 12),
        (120, 22),
        (20, 22),
        (20, 10),
        (0, 10),
        (0, 0),
    )


def test_both_outward_edges_break_at_lower_line_top() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="UPPER",
            bbox=alignment.BoundingBox(20, 0, 80, 10),
            line_index=0,
        ),
        alignment.ALTOWord(
            index=1,
            text="LOWER",
            bbox=alignment.BoundingBox(0, 12, 120, 10),
            line_index=1,
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == (
        (20, 0),
        (100, 0),
        (100, 12),
        (120, 12),
        (120, 22),
        (0, 22),
        (0, 12),
        (20, 12),
        (20, 0),
    )


def test_words_in_one_alto_line_become_one_clean_rectangle() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="FIRST",
            bbox=alignment.BoundingBox(0, 1, 20, 9),
            line_index=0,
        ),
        alignment.ALTOWord(
            index=1,
            text="SECOND",
            bbox=alignment.BoundingBox(25, 0, 30, 10),
            line_index=0,
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == ((0, 0), (55, 0), (55, 10), (0, 10), (0, 0))


def test_words_without_line_ids_retain_word_box_bands() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="FIRST",
            bbox=alignment.BoundingBox(0, 1, 20, 9),
            line_index=None,
            block_index=0,
        ),
        alignment.ALTOWord(
            index=1,
            text="SECOND",
            bbox=alignment.BoundingBox(25, 0, 30, 10),
            line_index=None,
            block_index=0,
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == (
        (25, 0),
        (55, 0),
        (55, 10),
        (0, 10),
        (0, 1),
        (25, 1),
        (25, 0),
    )


def test_equal_line_indexes_in_different_blocks_are_not_merged() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="UPPER",
            bbox=alignment.BoundingBox(0, 0, 100, 10),
            line_index=0,
            block_index=0,
        ),
        alignment.ALTOWord(
            index=1,
            text="LOWER",
            bbox=alignment.BoundingBox(0, 12, 50, 10),
            line_index=0,
            block_index=1,
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == (
        (0, 0),
        (100, 0),
        (100, 10),
        (50, 10),
        (50, 22),
        (0, 22),
        (0, 0),
    )


def test_ocr_line_splitting_does_not_split_a_visual_band() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="FIRST",
            bbox=alignment.BoundingBox(0, 0, 20, 10),
            line_index=4,
            block_index=0,
        ),
        alignment.ALTOWord(
            index=1,
            text="SECOND",
            bbox=alignment.BoundingBox(25, 0, 30, 10),
            line_index=9,
            block_index=3,
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == ((0, 0), (55, 0), (55, 10), (0, 10), (0, 0))


def test_word_input_order_does_not_affect_polygon() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="UPPER",
            bbox=alignment.BoundingBox(0, 0, 100, 10),
            line_index=8,
        ),
        alignment.ALTOWord(
            index=1,
            text="LOWER",
            bbox=alignment.BoundingBox(20, 12, 60, 10),
            line_index=2,
        ),
    )
    builder = alignment.OrthogonalPolygonGeometryBuilder()

    forward = builder.build(words)
    reversed_order = builder.build(tuple(reversed(words)))

    assert forward == reversed_order


def test_wide_narrow_wide_layout_retains_middle_concavity() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="WIDE",
            bbox=alignment.BoundingBox(0, 0, 100, 10),
        ),
        alignment.ALTOWord(
            index=1,
            text="NARROW",
            bbox=alignment.BoundingBox(0, 12, 50, 10),
        ),
        alignment.ALTOWord(
            index=2,
            text="WIDE",
            bbox=alignment.BoundingBox(0, 24, 100, 10),
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == (
        (0, 0),
        (100, 0),
        (100, 10),
        (50, 10),
        (50, 24),
        (100, 24),
        (100, 34),
        (0, 34),
        (0, 0),
    )


def test_nonoverlapping_bands_are_connected_without_dropping_words() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="UPPER",
            bbox=alignment.BoundingBox(0, 0, 20, 10),
        ),
        alignment.ALTOWord(
            index=1,
            text="LOWER",
            bbox=alignment.BoundingBox(40, 12, 20, 10),
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == (
        (0, 0),
        (20, 0),
        (20, 10),
        (60, 10),
        (60, 22),
        (40, 22),
        (40, 12),
        (0, 12),
        (0, 0),
    )
    assert polygon.bounds == alignment.BoundingBox(0, 0, 60, 22)


def test_touching_disjoint_bands_fall_back_without_dropping_words() -> None:
    words = (
        alignment.ALTOWord(
            index=0,
            text="UPPER",
            bbox=alignment.BoundingBox(0, 0, 20, 10),
        ),
        alignment.ALTOWord(
            index=1,
            text="LOWER",
            bbox=alignment.BoundingBox(40, 10, 20, 10),
        ),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

    assert polygon.points == ((0, 0), (60, 0), (60, 20), (0, 20), (0, 0))


def test_degenerate_word_box_still_produces_closed_output() -> None:
    word = alignment.ALTOWord(
        index=0,
        text="POINT",
        bbox=alignment.BoundingBox(5, 7, 0, 0),
    )

    polygon = alignment.OrthogonalPolygonGeometryBuilder().build((word,))

    assert polygon.points == (
        (5, 7),
        (5, 7),
        (5, 7),
        (5, 7),
        (5, 7),
    )
    assert polygon.bounds == alignment.BoundingBox(5, 7, 0, 0)
