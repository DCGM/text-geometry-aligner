"""Tests for bounding-box and polygon overlap calculation."""

from unittest import mock

import pytest

import text_geometry_aligner as alignment
from text_geometry_aligner.geometry_matching import overlap as overlap_module


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


@pytest.fixture
def calculator():
    return alignment.BoundingBoxOverlapCalculator()


@pytest.fixture
def word():
    return _word(0, "WORD", 0)


def _overlap(
    calculator,
    word,
    region_bbox: alignment.BoundingBox,
    threshold: float,
    strategy: alignment.GeometryOverlapStrategy = (
        alignment.GeometryOverlapStrategy.BIDIRECTIONAL_CONTAINMENT
    ),
) -> tuple[alignment.GeometryWordOverlap, ...]:
    region = alignment.AlignmentRegion(
        region_id=0,
        label="title",
        json_geometry_path=("title_bbox",),
        json_text_path=("title",),
        input_geometry=region_bbox,
    )
    return calculator.calculate(
        (region,),
        (word,),
        threshold,
        strategy,
    )


def test_threshold_is_inclusive_and_uses_word_area(calculator, word) -> None:
    at_threshold = _overlap(
        calculator,
        word,
        alignment.BoundingBox(0, 0, 6.5, 10),
        0.65,
        alignment.GeometryOverlapStrategy.WORD_COVERAGE,
    )
    below_threshold = _overlap(
        calculator,
        word,
        alignment.BoundingBox(0, 0, 6.49, 10),
        0.65,
        alignment.GeometryOverlapStrategy.WORD_COVERAGE,
    )

    assert at_threshold[0].word_coverage == pytest.approx(0.65)
    assert at_threshold[0].input_geometry_coverage == pytest.approx(1.0)
    assert at_threshold[0].overlap_score == pytest.approx(0.65)
    assert below_threshold == ()


def test_bidirectional_containment_accepts_tight_detection(calculator, word) -> None:
    overlaps = _overlap(
        calculator,
        word,
        alignment.BoundingBox(2, 2, 4, 4),
        0.65,
    )

    assert len(overlaps) == 1
    assert overlaps[0].word_coverage == pytest.approx(0.16)
    assert overlaps[0].input_geometry_coverage == pytest.approx(1.0)
    assert overlaps[0].overlap_score == pytest.approx(1.0)


def test_word_coverage_can_reject_tight_detection(calculator, word) -> None:
    assert (
        _overlap(
            calculator,
            word,
            alignment.BoundingBox(2, 2, 4, 4),
            0.65,
            alignment.GeometryOverlapStrategy.WORD_COVERAGE,
        )
        == ()
    )


def test_bidirectional_containment_accepts_word_inside_region(calculator, word) -> None:
    overlaps = _overlap(
        calculator,
        word,
        alignment.BoundingBox(-5, -5, 20, 20),
        0.65,
    )

    assert len(overlaps) == 1
    assert overlaps[0].word_coverage == pytest.approx(1.0)
    assert overlaps[0].input_geometry_coverage == pytest.approx(0.25)
    assert overlaps[0].overlap_score == pytest.approx(1.0)


def test_insufficient_overlap_in_both_directions_is_rejected(calculator, word) -> None:
    assert (
        _overlap(
            calculator,
            word,
            alignment.BoundingBox(8, 8, 10, 10),
            0.65,
        )
        == ()
    )


def test_zero_area_input_geometry_has_no_overlap(calculator, word) -> None:
    assert (
        _overlap(
            calculator,
            word,
            alignment.BoundingBox(0, 0, 0, 10),
            0.0,
        )
        == ()
    )


def test_boundary_contact_has_no_coverage_even_at_zero_threshold(calculator, word) -> None:
    assert (
        _overlap(
            calculator,
            word,
            alignment.BoundingBox(10, 0, 5, 10),
            0.0,
        )
        == ()
    )


def test_polygon_requires_optional_shapely_dependency(calculator, word) -> None:
    region = alignment.AlignmentRegion(
        region_id=0,
        label="title",
        json_geometry_path=("title_polygon",),
        json_text_path=("title",),
        input_geometry=alignment.Polygon(
            ((0, 0), (10, 0), (10, 10), (0, 10), (0, 0))
        ),
    )

    with pytest.raises(RuntimeError, match="requires Shapely"):
        calculator.calculate((region,), (word,), 0.65)


def test_factory_falls_back_to_bbox_when_shapely_is_unavailable() -> None:
    region = alignment.AlignmentRegion(
        region_id=0,
        label="title",
        json_geometry_path=("title_bbox",),
        json_text_path=("title",),
        input_geometry=alignment.BoundingBox(0, 0, 10, 10),
    )
    with mock.patch.object(
        overlap_module,
        "_load_shapely_box_factory",
        side_effect=ImportError,
    ):
        calculator = alignment.create_overlap_calculator((region,))

    assert isinstance(calculator, alignment.BoundingBoxOverlapCalculator)


def test_factory_reports_missing_shapely_for_polygon() -> None:
    region = alignment.AlignmentRegion(
        region_id=0,
        label="title",
        json_geometry_path=("title_polygon",),
        json_text_path=("title",),
        input_geometry=alignment.Polygon(
            ((0, 0), (10, 0), (10, 10), (0, 10), (0, 0))
        ),
    )
    with (
        mock.patch.object(
            overlap_module,
            "_load_shapely_box_factory",
            side_effect=ImportError,
        ),
        pytest.raises(RuntimeError, match="requires Shapely"),
    ):
        alignment.create_overlap_calculator((region,))


def test_polygon_uses_bidirectional_containment(word) -> None:
    class FakeShape:
        def __init__(self, points):
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            self.bounds = (min(xs), min(ys), max(xs), max(ys))
            self.area = (self.bounds[2] - self.bounds[0]) * (
                self.bounds[3] - self.bounds[1]
            )
            self.is_empty = False
            self.is_valid = True

        def intersection(self, other):
            x_min = max(self.bounds[0], other.bounds[0])
            y_min = max(self.bounds[1], other.bounds[1])
            x_max = min(self.bounds[2], other.bounds[2])
            y_max = min(self.bounds[3], other.bounds[3])
            area = max(0, x_max - x_min) * max(0, y_max - y_min)
            return type("Intersection", (), {"area": area})()

    def fake_box(x_min, y_min, x_max, y_max):
        return FakeShape(
            (
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            )
        )

    region = alignment.AlignmentRegion(
        region_id=0,
        label="title",
        input_geometry=alignment.Polygon(
            ((2, 2), (6, 2), (6, 6), (2, 6), (2, 2))
        ),
    )

    overlaps = alignment.ShapelyOverlapCalculator(
        geometry_factory=fake_box,
        polygon_class=FakeShape,
    ).calculate(
        (region,),
        (word,),
        0.65,
    )

    assert len(overlaps) == 1
    assert overlaps[0].word_coverage == pytest.approx(0.16)
    assert overlaps[0].input_geometry_coverage == pytest.approx(1.0)
    assert overlaps[0].overlap_score == pytest.approx(1.0)


def test_shapely_and_fallback_backends_agree_for_bboxes(word) -> None:
    class FakeShape:
        def __init__(self, x_min, y_min, x_max, y_max):
            self.bounds = (x_min, y_min, x_max, y_max)

        def intersection(self, other):
            x_min = max(self.bounds[0], other.bounds[0])
            y_min = max(self.bounds[1], other.bounds[1])
            x_max = min(self.bounds[2], other.bounds[2])
            y_max = min(self.bounds[3], other.bounds[3])
            area = max(0, x_max - x_min) * max(0, y_max - y_min)
            return type("Intersection", (), {"area": area})()

    def fake_box(x_min, y_min, x_max, y_max):
        return FakeShape(x_min, y_min, x_max, y_max)

    region = alignment.AlignmentRegion(
        region_id=0,
        label="title",
        json_geometry_path=("title_bbox",),
        json_text_path=("title",),
        input_geometry=alignment.BoundingBox(0, 0, 6.5, 10),
    )
    fallback = alignment.BoundingBoxOverlapCalculator().calculate(
        (region,),
        (word,),
        0.65,
    )
    shapely = alignment.ShapelyOverlapCalculator(
        geometry_factory=fake_box,
        polygon_class=object(),
    ).calculate(
        (region,),
        (word,),
        0.65,
    )

    assert fallback == shapely
