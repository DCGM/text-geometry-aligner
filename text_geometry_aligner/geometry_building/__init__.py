"""Output-geometry building interfaces and implementations."""

from .base import GeometryBuilder, validate_geometry_format
from .orthogonal_polygon import OrthogonalPolygonGeometryBuilder
from .union_bounding_box import UnionBoundingBoxGeometryBuilder

__all__ = [
    "GeometryBuilder",
    "OrthogonalPolygonGeometryBuilder",
    "UnionBoundingBoxGeometryBuilder",
    "validate_geometry_format",
]
