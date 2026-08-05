from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from ..io_alto import ALTOWord
from ..models import AlignmentRegion
from .overlap import GeometryWordOverlap


class WordAssignmentStrategy(str, Enum):
    GREATEST_COVERAGE = "greatest-coverage"
    ALL_OVER_THRESHOLD = "all-over-threshold"


@dataclass(frozen=True)
class RegionWordAssignment:
    """Selected ALTO word overlaps for one region."""

    region_id: int
    overlaps: tuple[GeometryWordOverlap, ...]


class GeometryWordAssigner(ABC):
    """Resolve eligible overlaps into final region assignments."""

    @abstractmethod
    def assign(
        self,
        regions: Sequence[AlignmentRegion],
        words: Sequence[ALTOWord],
        overlaps: Sequence[GeometryWordOverlap],
    ) -> tuple[RegionWordAssignment, ...]:
        raise NotImplementedError

    @staticmethod
    def _build_assignments(
        regions: Sequence[AlignmentRegion],
        retained: Sequence[GeometryWordOverlap],
    ) -> tuple[RegionWordAssignment, ...]:
        overlaps_by_region: dict[int, list[GeometryWordOverlap]] = defaultdict(
            list
        )
        for overlap in retained:
            overlaps_by_region[overlap.region_id].append(overlap)

        assignments: list[RegionWordAssignment] = []
        for region in regions:
            region_overlaps = sorted(
                overlaps_by_region.get(region.region_id, ()),
                key=lambda item: item.word_index,
            )
            assignments.append(
                RegionWordAssignment(
                    region_id=region.region_id,
                    overlaps=tuple(region_overlaps),
                )
            )
        return tuple(assignments)


class GreatestCoverageWordAssigner(GeometryWordAssigner):
    """Assign each word to its greatest-overlap region."""

    def assign(
        self,
        regions: Sequence[AlignmentRegion],
        words: Sequence[ALTOWord],
        overlaps: Sequence[GeometryWordOverlap],
    ) -> tuple[RegionWordAssignment, ...]:
        del words
        best_by_word: dict[int, GeometryWordOverlap] = {}
        for overlap in overlaps:
            current = best_by_word.get(overlap.word_index)
            if current is None or _preference(overlap) > _preference(current):
                best_by_word[overlap.word_index] = overlap
        return self._build_assignments(
            regions,
            tuple(best_by_word.values()),
        )


class AllOverThresholdWordAssigner(GeometryWordAssigner):
    """Retain every eligible region-to-word overlap."""

    def assign(
        self,
        regions: Sequence[AlignmentRegion],
        words: Sequence[ALTOWord],
        overlaps: Sequence[GeometryWordOverlap],
    ) -> tuple[RegionWordAssignment, ...]:
        del words
        return self._build_assignments(regions, overlaps)


def _preference(
    overlap: GeometryWordOverlap,
) -> tuple[float, float, float, int]:
    return (
        overlap.overlap_score,
        overlap.word_coverage,
        overlap.input_geometry_coverage,
        -overlap.region_id,
    )


def create_word_assigner(
    strategy: WordAssignmentStrategy | str,
) -> GeometryWordAssigner:
    parsed_strategy = WordAssignmentStrategy(strategy)
    if parsed_strategy is WordAssignmentStrategy.GREATEST_COVERAGE:
        return GreatestCoverageWordAssigner()
    return AllOverThresholdWordAssigner()
