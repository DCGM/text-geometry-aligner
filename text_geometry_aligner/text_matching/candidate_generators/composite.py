"""Composition of multiple text candidate generators."""

from __future__ import annotations

from typing import Sequence

from ..alto_text_index import ALTOTextIndex
from ...models import AlignmentRegion
from ..candidate import AlignmentCandidate
from .base import CandidateGenerator
from .utils import candidate_sort_key, replace_candidate_id


class CompositeCandidateGenerator(CandidateGenerator):
    """Combine ordered generators with first-generator duplicate precedence."""

    def __init__(self, generators: Sequence[CandidateGenerator]):
        if not generators:
            raise ValueError("CompositeCandidateGenerator requires at least one generator")
        self.generators = tuple(generators)

    def generate(
        self,
        regions: Sequence[AlignmentRegion],
        alto_index: ALTOTextIndex,
    ) -> tuple[AlignmentCandidate, ...]:
        by_value_and_span: dict[tuple[int, int, int], AlignmentCandidate] = {}
        for generator in self.generators:
            for candidate in generator.generate(regions, alto_index):
                key = (
                    candidate.region_id,
                    candidate.start_word,
                    candidate.end_word,
                )
                by_value_and_span.setdefault(key, candidate)

        candidates = sorted(
            by_value_and_span.values(),
            key=candidate_sort_key,
        )
        return tuple(
            replace_candidate_id(candidate, candidate_id)
            for candidate_id, candidate in enumerate(candidates)
        )
