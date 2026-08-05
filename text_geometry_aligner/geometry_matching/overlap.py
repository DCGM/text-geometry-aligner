from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

from ..io_alto import ALTOWord
from ..models import (
    AlignmentRegion,
    BoundingBox,
    Polygon,
    _format_json_path,
)


class GeometryOverlapStrategy(str, Enum):
    """Score used to accept an input-geometry/ALTO-word overlap."""

    BIDIRECTIONAL_CONTAINMENT = "bidirectional-containment"
    WORD_COVERAGE = "word-coverage"


@dataclass(frozen=True)
class GeometryWordOverlap:
    """Directional coverages and selected score for one overlap."""

    region_id: int
    word_index: int
    word_coverage: float
    input_geometry_coverage: float
    overlap_score: float


class GeometryOverlapCalculator(ABC):
    """Calculate eligible input-geometry/ALTO-word overlaps."""

    @abstractmethod
    def calculate(
        self,
        regions: Sequence[AlignmentRegion],
        words: Sequence[ALTOWord],
        minimum_overlap_coverage: float,
        overlap_strategy: GeometryOverlapStrategy | str = (
            GeometryOverlapStrategy.BIDIRECTIONAL_CONTAINMENT
        ),
    ) -> tuple[GeometryWordOverlap, ...]:
        raise NotImplementedError


class BoundingBoxOverlapCalculator(GeometryOverlapCalculator):
    """Dependency-free rectangle overlap calculator."""

    def calculate(
        self,
        regions: Sequence[AlignmentRegion],
        words: Sequence[ALTOWord],
        minimum_overlap_coverage: float,
        overlap_strategy: GeometryOverlapStrategy | str = (
            GeometryOverlapStrategy.BIDIRECTIONAL_CONTAINMENT
        ),
    ) -> tuple[GeometryWordOverlap, ...]:
        _validate_threshold(minimum_overlap_coverage)
        parsed_strategy = GeometryOverlapStrategy(overlap_strategy)
        if any(isinstance(region.input_geometry, Polygon) for region in regions):
            raise RuntimeError(
                "Polygon geometry alignment requires Shapely. Install it "
                "with: python -m pip install Shapely"
            )

        overlaps: list[GeometryWordOverlap] = []
        for region in regions:
            geometry = region.input_geometry
            if not isinstance(geometry, BoundingBox):
                continue
            input_geometry_area = geometry.width * geometry.height
            if input_geometry_area <= 0:
                continue
            for word in words:
                if word.bbox.width <= 0 or word.bbox.height <= 0:
                    continue
                word_area = word.bbox.width * word.bbox.height
                intersection_area = _bbox_intersection_area(
                    geometry,
                    word.bbox,
                )
                if intersection_area <= 0:
                    continue
                overlap = _build_overlap(
                    region_id=region.region_id,
                    word_index=word.index,
                    intersection_area=intersection_area,
                    word_area=word_area,
                    input_geometry_area=input_geometry_area,
                    overlap_strategy=parsed_strategy,
                )
                if (
                    overlap.overlap_score + 1e-12
                    >= minimum_overlap_coverage
                ):
                    overlaps.append(overlap)
        return tuple(overlaps)


class ShapelyOverlapCalculator(GeometryOverlapCalculator):
    """Shapely-backed bbox and polygon overlap calculator."""

    def __init__(
        self,
        geometry_factory: Callable[..., Any] | None = None,
        polygon_class: Any = None,
    ):
        self.geometry_factory = (
            geometry_factory or _load_shapely_box_factory()
        )
        self.polygon_class = (
            polygon_class or _load_shapely_polygon_class()
        )

    def calculate(
        self,
        regions: Sequence[AlignmentRegion],
        words: Sequence[ALTOWord],
        minimum_overlap_coverage: float,
        overlap_strategy: GeometryOverlapStrategy | str = (
            GeometryOverlapStrategy.BIDIRECTIONAL_CONTAINMENT
        ),
    ) -> tuple[GeometryWordOverlap, ...]:
        _validate_threshold(minimum_overlap_coverage)
        parsed_strategy = GeometryOverlapStrategy(overlap_strategy)
        region_shapes = {
            region.region_id: self._region_shape(region)
            for region in regions
        }
        word_shapes = {
            word.index: self.geometry_factory(
                word.bbox.x,
                word.bbox.y,
                word.bbox.x_max,
                word.bbox.y_max,
            )
            for word in words
            if word.bbox.width > 0 and word.bbox.height > 0
        }

        overlaps: list[GeometryWordOverlap] = []
        for region in regions:
            region_shape = region_shapes[region.region_id]
            geometry = region.input_geometry
            if geometry is None:
                continue
            input_geometry_area = _input_geometry_area(
                geometry,
                region_shape,
            )
            if input_geometry_area <= 0:
                continue
            region_bounds = geometry.bounds
            for word in words:
                word_shape = word_shapes.get(word.index)
                if word_shape is None:
                    continue
                if _bbox_intersection_area(region_bounds, word.bbox) <= 0:
                    continue
                intersection_area = float(
                    region_shape.intersection(word_shape).area
                )
                if intersection_area <= 0:
                    continue
                word_area = word.bbox.width * word.bbox.height
                overlap = _build_overlap(
                    region_id=region.region_id,
                    word_index=word.index,
                    intersection_area=intersection_area,
                    word_area=word_area,
                    input_geometry_area=input_geometry_area,
                    overlap_strategy=parsed_strategy,
                )
                if (
                    overlap.overlap_score + 1e-12
                    >= minimum_overlap_coverage
                ):
                    overlaps.append(
                        overlap
                    )
        return tuple(overlaps)

    def _region_shape(self, region: AlignmentRegion) -> Any:
        geometry = region.input_geometry
        if geometry is None:
            raise ValueError(f"Region {region.region_id} has no input geometry")
        if isinstance(geometry, BoundingBox):
            return self.geometry_factory(
                geometry.x,
                geometry.y,
                geometry.x_max,
                geometry.y_max,
            )

        shape = self.polygon_class(geometry.points[:-1])
        if shape.is_empty or shape.area <= 0 or not shape.is_valid:
            raise ValueError(
                "Invalid polygon at "
                f"{_format_json_path(region.json_geometry_path or ())}"
            )
        return shape


def create_overlap_calculator(
    regions: Sequence[AlignmentRegion],
) -> GeometryOverlapCalculator:
    try:
        box_factory = _load_shapely_box_factory()
    except ImportError as exc:
        if any(
            isinstance(region.input_geometry, Polygon)
            for region in regions
        ):
            raise RuntimeError(
                "Polygon geometry alignment requires Shapely. Install it "
                "with: python -m pip install Shapely"
            ) from exc
        return BoundingBoxOverlapCalculator()
    return ShapelyOverlapCalculator(box_factory)


def _load_shapely_box_factory() -> Callable[..., Any]:
    from shapely.geometry import box

    return box


def _load_shapely_polygon_class() -> Any:
    from shapely.geometry import Polygon as ShapelyPolygon

    return ShapelyPolygon


def _bbox_intersection_area(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    width = min(first.x_max, second.x_max) - max(first.x, second.x)
    height = min(first.y_max, second.y_max) - max(first.y, second.y)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _build_overlap(
    *,
    region_id: int,
    word_index: int,
    intersection_area: float,
    word_area: float,
    input_geometry_area: float,
    overlap_strategy: GeometryOverlapStrategy,
) -> GeometryWordOverlap:
    word_coverage = _bounded_ratio(intersection_area, word_area)
    input_geometry_coverage = _bounded_ratio(
        intersection_area,
        input_geometry_area,
    )
    overlap_score = (
        word_coverage
        if overlap_strategy is GeometryOverlapStrategy.WORD_COVERAGE
        else max(word_coverage, input_geometry_coverage)
    )
    return GeometryWordOverlap(
        region_id=region_id,
        word_index=word_index,
        word_coverage=word_coverage,
        input_geometry_coverage=input_geometry_coverage,
        overlap_score=overlap_score,
    )


def _bounded_ratio(numerator: float, denominator: float) -> float:
    return min(1.0, max(0.0, numerator / denominator))


def _input_geometry_area(
    geometry: BoundingBox | Polygon,
    shape: Any,
) -> float:
    if isinstance(geometry, BoundingBox):
        return geometry.width * geometry.height
    return float(shape.area)


def _validate_threshold(minimum_overlap_coverage: float) -> None:
    if not 0.0 <= minimum_overlap_coverage <= 1.0:
        raise ValueError(
            "minimum_overlap_coverage must be within [0, 1]"
        )
