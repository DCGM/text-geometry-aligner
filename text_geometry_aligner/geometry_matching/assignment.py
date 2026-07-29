from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from ..alto_io import ALTOWord
from ..models import AlignmentRegion
from .overlap import WordCoverage


class WordAssignmentStrategy(str, Enum):
    GREATEST_COVERAGE = "greatest-coverage"
    ALL_OVER_THRESHOLD = "all-over-threshold"


@dataclass(frozen=True)
class RegionWordAssignment:
    """Selected ALTO word indexes and coverages for one region."""

    region_id: int
    word_indexes: tuple[int, ...]
    word_coverages: tuple[float, ...]


class GeometryWordAssigner(ABC):
    """Resolve eligible word coverage into final region assignments."""

    @abstractmethod
    def assign(
        self,
        regions: Sequence[AlignmentRegion],
        words: Sequence[ALTOWord],
        coverages: Sequence[WordCoverage],
    ) -> tuple[RegionWordAssignment, ...]:
        raise NotImplementedError

    @staticmethod
    def _build_assignments(
        regions: Sequence[AlignmentRegion],
        retained: Sequence[WordCoverage],
    ) -> tuple[RegionWordAssignment, ...]:
        coverage_by_region: dict[int, list[WordCoverage]] = defaultdict(list)
        for coverage in retained:
            coverage_by_region[coverage.region_id].append(coverage)

        assignments: list[RegionWordAssignment] = []
        for region in regions:
            region_coverages = sorted(
                coverage_by_region.get(region.region_id, ()),
                key=lambda item: item.word_index,
            )
            assignments.append(
                RegionWordAssignment(
                    region_id=region.region_id,
                    word_indexes=tuple(
                        coverage.word_index
                        for coverage in region_coverages
                    ),
                    word_coverages=tuple(
                        coverage.coverage
                        for coverage in region_coverages
                    ),
                )
            )
        return tuple(assignments)


class GreatestCoverageWordAssigner(GeometryWordAssigner):
    """Assign each word to its greatest-coverage region."""

    def assign(
        self,
        regions: Sequence[AlignmentRegion],
        words: Sequence[ALTOWord],
        coverages: Sequence[WordCoverage],
    ) -> tuple[RegionWordAssignment, ...]:
        del words
        best_by_word: dict[int, WordCoverage] = {}
        for coverage in coverages:
            current = best_by_word.get(coverage.word_index)
            if current is None or (
                coverage.coverage > current.coverage
                or (
                    coverage.coverage == current.coverage
                    and coverage.region_id < current.region_id
                )
            ):
                best_by_word[coverage.word_index] = coverage
        return self._build_assignments(
            regions,
            tuple(best_by_word.values()),
        )


class AllOverThresholdWordAssigner(GeometryWordAssigner):
    """Retain every region-to-word coverage produced by the calculator."""

    def assign(
        self,
        regions: Sequence[AlignmentRegion],
        words: Sequence[ALTOWord],
        coverages: Sequence[WordCoverage],
    ) -> tuple[RegionWordAssignment, ...]:
        del words
        return self._build_assignments(regions, coverages)


def create_word_assigner(
    strategy: WordAssignmentStrategy | str,
) -> GeometryWordAssigner:
    parsed_strategy = WordAssignmentStrategy(strategy)
    if parsed_strategy is WordAssignmentStrategy.GREATEST_COVERAGE:
        return GreatestCoverageWordAssigner()
    return AllOverThresholdWordAssigner()
