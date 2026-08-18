"""Tests for word-to-region assignment strategies."""

from pathlib import Path
from unittest import mock

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


def _page(*words: alignment.ALTOWord) -> alignment.ALTOPage:
    return alignment.ALTOPage(
        source_path=Path("page.xml"),
        words=tuple(words),
        width=100,
        height=100,
    )


def _bbox(
    x: float,
    y: float = 0,
    width: float = 10,
    height: float = 10,
) -> dict[str, float]:
    return {"x": x, "y": y, "width": width, "height": height}


@pytest.fixture
def words():
    return (
        _word(0, "FIRST", 0),
        _word(1, "SECOND", 10),
    )


@pytest.fixture
def regions():
    return (
        alignment.AlignmentRegion(
            region_id=0,
            label="first",
            json_geometry_path=("first_bbox",),
            json_text_path=("first",),
            input_geometry=alignment.BoundingBox(0, 0, 20, 10),
        ),
        alignment.AlignmentRegion(
            region_id=1,
            label="second",
            json_geometry_path=("second_bbox",),
            json_text_path=("second",),
            input_geometry=alignment.BoundingBox(0, 0, 20, 10),
        ),
    )


def test_greatest_coverage_wins_and_ties_use_region_order(words, regions) -> None:
    overlaps = (
        alignment.GeometryWordOverlap(0, 0, 0.7, 0.5, 0.7),
        alignment.GeometryWordOverlap(1, 0, 0.8, 0.4, 0.8),
        alignment.GeometryWordOverlap(0, 1, 0.9, 0.3, 0.9),
        alignment.GeometryWordOverlap(1, 1, 0.9, 0.3, 0.9),
    )

    assigned = alignment.GreatestCoverageWordAssigner().assign(
        regions,
        words,
        overlaps,
    )

    assert tuple(item.word_index for item in assigned[0].overlaps) == (1,)
    assert tuple(item.overlap_score for item in assigned[0].overlaps) == (0.9,)
    assert tuple(item.word_index for item in assigned[1].overlaps) == (0,)
    assert tuple(item.overlap_score for item in assigned[1].overlaps) == (0.8,)


def test_equal_scores_prefer_directional_coverages_then_region(words, regions) -> None:
    overlaps = (
        alignment.GeometryWordOverlap(0, 0, 0.4, 1.0, 1.0),
        alignment.GeometryWordOverlap(1, 0, 1.0, 0.4, 1.0),
        alignment.GeometryWordOverlap(0, 1, 1.0, 0.6, 1.0),
        alignment.GeometryWordOverlap(1, 1, 1.0, 0.6, 1.0),
    )

    assigned = alignment.GreatestCoverageWordAssigner().assign(
        regions,
        words,
        overlaps,
    )

    assert tuple(item.word_index for item in assigned[0].overlaps) == (1,)
    assert tuple(item.word_index for item in assigned[1].overlaps) == (0,)


def test_all_over_threshold_retains_shared_words(words, regions) -> None:
    overlaps = (
        alignment.GeometryWordOverlap(0, 0, 0.7, 0.5, 0.7),
        alignment.GeometryWordOverlap(1, 0, 0.8, 0.4, 0.8),
    )

    assigned = alignment.AllOverThresholdWordAssigner().assign(
        regions,
        words,
        overlaps,
    )

    assert tuple(item.word_index for item in assigned[0].overlaps) == (0,)
    assert tuple(item.word_index for item in assigned[1].overlaps) == (0,)


def test_aligner_uses_builders_after_assignment(words, regions) -> None:
    text_builder = mock.create_autospec(
        alignment.TextBuilder,
        instance=True,
    )
    text_builder.build.return_value = "CUSTOM TEXT"
    geometry_builder = mock.create_autospec(
        alignment.GeometryBuilder,
        instance=True,
    )
    geometry_builder.build.return_value = alignment.BoundingBox(0, 0, 20, 10)
    aligner = alignment.GeometryAligner(
        text_builder=text_builder,
        geometry_builder=geometry_builder,
    )
    result = aligner.align_data(
        _page(*words),
        {"title_bbox": _bbox(2, width=17)},
    )
    region = result.pages[0].regions[0]

    text_builder.build.assert_called_once_with(words)
    geometry_builder.build.assert_called_once_with(words)
    assert region.alto_text == "CUSTOM TEXT"
    assert region.alto_geometry == alignment.BoundingBox(0, 0, 20, 10)
    assert tuple(word.word_coverage for word in region.words or ()) == (0.8, 0.9)
    assert tuple(word.overlap_score for word in region.words or ()) == (0.8, 0.9)
    assert region.alignment_score == pytest.approx(0.85)
