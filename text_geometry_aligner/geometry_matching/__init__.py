"""Geometry-to-ALTO overlap calculation and word assignment."""

from .assignment import (
    AllOverThresholdWordAssigner,
    GeometryWordAssigner,
    GreatestCoverageWordAssigner,
    RegionWordAssignment,
    WordAssignmentStrategy,
    create_word_assigner,
)
from .overlap import (
    BoundingBoxOverlapCalculator,
    GeometryOverlapCalculator,
    GeometryOverlapStrategy,
    GeometryWordOverlap,
    ShapelyOverlapCalculator,
    create_overlap_calculator,
)

__all__ = [
    "AllOverThresholdWordAssigner",
    "BoundingBoxOverlapCalculator",
    "GeometryOverlapCalculator",
    "GeometryOverlapStrategy",
    "GeometryWordOverlap",
    "GeometryWordAssigner",
    "GreatestCoverageWordAssigner",
    "RegionWordAssignment",
    "ShapelyOverlapCalculator",
    "WordAssignmentStrategy",
    "create_overlap_calculator",
    "create_word_assigner",
]
