"""Interface for constructing output geometry from matched ALTO words."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..alto_io import ALTOWord
from ..models import (
    BoundingBox,
    OutputGeometry,
    OutputGeometryFormat,
    Polygon,
)


class GeometryBuilder(ABC):
    """Convert matched ALTO words into output geometry."""

    @abstractmethod
    def build(self, words: Sequence[ALTOWord]) -> OutputGeometry:
        raise NotImplementedError


def validate_geometry_format(
    geometry: OutputGeometry,
    expected_format: OutputGeometryFormat,
) -> None:
    """Validate that a geometry builder returned its configured format."""

    expected_type = (
        BoundingBox
        if expected_format is OutputGeometryFormat.BBOX
        else Polygon
    )
    if not isinstance(geometry, expected_type):
        raise TypeError(
            f"Geometry builder returned {type(geometry).__name__}, "
            f"but output ALTO geometry format {expected_format.value!r} "
            f"requires {expected_type.__name__}"
        )
