from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .text_matching.candidate import AlignmentCandidate

JSONPathPart = str | int
JSONPath = tuple[JSONPathPart, ...]
ScalarText = str | int | float

_SUSPICIOUS_INPUT_GEOMETRY_MIN_WIDTH = 5.0
_SUSPICIOUS_INPUT_GEOMETRY_MIN_HEIGHT = 5.0
_SUSPICIOUS_INPUT_GEOMETRY_MIN_AREA = 100.0

logger = logging.getLogger(__name__)


def _format_json_path(path: JSONPath) -> str:
    if not path:
        return "$"

    output = "$"
    for component in path:
        if isinstance(component, int):
            output += f"[{component}]"
        elif component.isidentifier():
            output += f".{component}"
        else:
            output += f"[{json.dumps(component, ensure_ascii=False)}]"
    return output


class AlignmentMode(str, Enum):
    """Information used to select ALTO words."""

    TEXT = "text"
    GEOMETRY = "geometry"


class InputFormat(str, Enum):
    """Supported sources from which alignment regions can be created."""

    JSON = "json"
    LABEL_STUDIO = "label_studio"
    YOLO = "yolo"


class OutputTextSource(str, Enum):
    """Source text written to output and rendered labels."""

    JSON = "json"
    ALTO = "alto"


class OutputGeometrySource(str, Enum):
    """Source geometry written to output and rendered."""

    INPUT = "input"
    ALTO = "alto"


class OutputGeometryFormat(str, Enum):
    """Supported geometry representations in aligned output."""

    BBOX = "bbox"
    POLYGON = "polygon"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box using top-left coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x_max(self) -> float:
        return self.x + self.width

    @property
    def y_max(self) -> float:
        return self.y + self.height

    @property
    def bounds(self) -> BoundingBox:
        return self

    def to_json(self) -> dict[str, int | float]:
        return {
            "x": _clean_number(self.x),
            "y": _clean_number(self.y),
            "width": _clean_number(self.width),
            "height": _clean_number(self.height),
        }


Point = tuple[float, float]


@dataclass(frozen=True)
class Polygon:
    """Closed polygon in the ALTO coordinate system."""

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 4:
            raise ValueError(
                "A closed polygon requires at least three vertices"
            )
        if self.points[0] != self.points[-1]:
            raise ValueError(
                "Polygon points must be closed by repeating the first point"
            )

    @property
    def bounds(self) -> BoundingBox:
        vertices = self.points[:-1]
        x_min = min(point[0] for point in vertices)
        y_min = min(point[1] for point in vertices)
        x_max = max(point[0] for point in vertices)
        y_max = max(point[1] for point in vertices)
        return BoundingBox(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
        )

    def to_json(self) -> list[list[int | float]]:
        return [
            [_clean_number(x), _clean_number(y)]
            for x, y in self.points
        ]


OutputGeometry = BoundingBox | Polygon


@dataclass
class AlignmentWord:
    """One ALTO word assigned to an alignment region."""

    word_index: int
    text: str
    bbox: BoundingBox
    text_normalized: str | None = None
    word_coverage: float | None = None
    input_geometry_coverage: float | None = None
    overlap_score: float | None = None
    line_index: int | None = None
    block_index: int | None = None
    element_id: str | None = None


@dataclass
class AlignmentRegion:
    """One input value or geometry enriched with matched ALTO data."""

    region_id: int
    label: str
    input_text: ScalarText | None = None
    input_text_normalized: str | None = None
    input_geometry: OutputGeometry | None = None
    text_alignment_candidate: AlignmentCandidate | None = None
    alto_text: str | None = None
    alto_text_normalized: str | None = None
    alto_geometry: OutputGeometry | None = None
    words: list[AlignmentWord] | None = None
    category_id: int | None = None
    input_geometry_confidence: float | None = None
    json_text_path: JSONPath | None = None
    json_geometry_path: JSONPath | None = None
    alignment_score: float | None = None

    def __post_init__(self) -> None:
        if self.region_id < 0:
            raise ValueError("region_id must not be negative")
        if not self.label:
            raise ValueError("label must not be empty")
        if self.category_id is not None and self.category_id < 0:
            raise ValueError("category_id must not be negative")
        if (
            self.input_geometry_confidence is not None
            and not 0.0 <= self.input_geometry_confidence <= 1.0
        ):
            raise ValueError(
                "input_geometry_confidence must be within [0, 1]"
            )
        if (
            self.alignment_score is not None
            and not 0.0 <= self.alignment_score <= 1.0
        ):
            raise ValueError("alignment_score must be within [0, 1]")
        self._warn_about_suspicious_input_geometry()

    def _warn_about_suspicious_input_geometry(self) -> None:
        geometry = self.input_geometry
        if geometry is None:
            return

        reasons: list[str] = []
        coordinate_values = _geometry_coordinate_values(geometry)
        non_finite_values = [
            f"{name}={value!r}"
            for name, value in coordinate_values
            if not math.isfinite(value)
        ]
        if non_finite_values:
            reasons.append(
                "non-finite values: " + ", ".join(non_finite_values)
            )

        negative_coordinates = [
            f"{name}={value:g}"
            for name, value in coordinate_values
            if math.isfinite(value)
            and _is_coordinate_name(name)
            and value < 0
        ]
        if negative_coordinates:
            reasons.append(
                "negative coordinates: " + ", ".join(negative_coordinates)
            )

        bounds = geometry.bounds
        if (
            math.isfinite(bounds.width)
            and bounds.width < _SUSPICIOUS_INPUT_GEOMETRY_MIN_WIDTH
        ):
            reasons.append(
                f"width={bounds.width:g} is below "
                f"{_SUSPICIOUS_INPUT_GEOMETRY_MIN_WIDTH:g}"
            )
        if (
            math.isfinite(bounds.height)
            and bounds.height < _SUSPICIOUS_INPUT_GEOMETRY_MIN_HEIGHT
        ):
            reasons.append(
                f"height={bounds.height:g} is below "
                f"{_SUSPICIOUS_INPUT_GEOMETRY_MIN_HEIGHT:g}"
            )

        area = _geometry_area(geometry)
        if (
            math.isfinite(area)
            and area < _SUSPICIOUS_INPUT_GEOMETRY_MIN_AREA
        ):
            reasons.append(
                f"area={area:g} is below "
                f"{_SUSPICIOUS_INPUT_GEOMETRY_MIN_AREA:g}"
            )

        if not reasons:
            return

        json_geometry_path = (
            None
            if self.json_geometry_path is None
            else _format_json_path(self.json_geometry_path)
        )
        json_text_path = (
            None
            if self.json_text_path is None
            else _format_json_path(self.json_text_path)
        )
        logger.warning(
            "Suspicious input geometry for region_id=%d label=%r "
            "category_id=%r json_geometry_path=%r json_text_path=%r: %r; "
            "reasons: %s",
            self.region_id,
            self.label,
            self.category_id,
            json_geometry_path,
            json_text_path,
            geometry,
            "; ".join(reasons),
        )

    @property
    def matched(self) -> bool:
        return self.words is not None


def _geometry_coordinate_values(
    geometry: OutputGeometry,
) -> tuple[tuple[str, float], ...]:
    if isinstance(geometry, BoundingBox):
        return (
            ("x", geometry.x),
            ("y", geometry.y),
            ("width", geometry.width),
            ("height", geometry.height),
        )
    return tuple(
        (f"points[{index}].{axis}", value)
        for index, point in enumerate(geometry.points)
        for axis, value in zip(("x", "y"), point)
    )


def _is_coordinate_name(name: str) -> bool:
    return name in {"x", "y"} or name.endswith((".x", ".y"))


def _geometry_area(geometry: OutputGeometry) -> float:
    if isinstance(geometry, BoundingBox):
        return abs(geometry.width * geometry.height)
    signed_double_area = sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(geometry.points, geometry.points[1:])
    )
    return abs(signed_double_area) / 2


@dataclass
class AlignmentPage:
    """One input page and its ALTO enrichment."""

    page_key: str
    input_format: InputFormat
    regions: list[AlignmentRegion]
    input_file_path: Path | None = None
    alto_file_path: Path | None = None
    json_source_data: dict[str, Any] | None = field(
        default=None,
        repr=False,
    )
    alto_page_id: str | None = None
    alto_width: float | None = None
    alto_height: float | None = None

    def __post_init__(self) -> None:
        if not self.page_key:
            raise ValueError("page_key must not be empty")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region IDs must be unique within a page")

    @property
    def matched_count(self) -> int:
        return sum(region.matched for region in self.regions)

    @property
    def unmatched_count(self) -> int:
        return len(self.regions) - self.matched_count


@dataclass
class AlignmentDocument:
    """Complete multi-page alignment container."""

    alignment_mode: AlignmentMode
    pages: list[AlignmentPage]
    input_path: Path | None = None
    alto_path: Path | None = None

    @property
    def matched_count(self) -> int:
        return sum(page.matched_count for page in self.pages)

    @property
    def unmatched_count(self) -> int:
        return sum(page.unmatched_count for page in self.pages)


def _clean_number(value: float) -> int | float:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return int(rounded)
    return value
