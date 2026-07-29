"""Exact text candidate generation."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ...alto_processing import ALTOTextIndex
from ...models import AlignmentRegion
from ..candidate import SIMILARITY_SCALE, AlignmentCandidate
from .base import CandidateGenerator
from .utils import (
    input_text,
    normalized_input_text,
    normalized_query_length,
    replace_candidate_id,
)


class ExactTextCandidateGenerator(CandidateGenerator):
    """Generate every strict normalized exact whole-word occurrence."""

    EXACT_SIMILARITY = SIMILARITY_SCALE

    def generate(
        self,
        regions: Sequence[AlignmentRegion],
        alto_index: ALTOTextIndex,
    ) -> tuple[AlignmentCandidate, ...]:
        regions_by_query: dict[str, list[AlignmentRegion]] = defaultdict(list)
        for region in regions:
            normalized_query = normalized_input_text(region)
            if normalized_query:
                regions_by_query[normalized_query].append(region)

        candidates: list[AlignmentCandidate] = []
        for normalized_query, query_regions in regions_by_query.items():
            occurrences = alto_index.find_exact_occurrences(normalized_query)
            for region in query_regions:
                for start_char, end_char, start_word, end_word in occurrences:
                    candidates.append(
                        AlignmentCandidate(
                            candidate_id=len(candidates),
                            region_id=region.region_id,
                            json_text_path=region.json_text_path,
                            start_word=start_word,
                            end_word=end_word,
                            start_char=start_char,
                            end_char=end_char,
                            query_text=input_text(region),
                            matched_text=alto_index.text_for_word_interval(
                                start_word,
                                end_word,
                            ),
                            normalized_query_text=normalized_query,
                            normalized_matched_text=(
                                alto_index.normalized_text_for_word_interval(
                                    start_word,
                                    end_word,
                                )
                            ),
                            exact=True,
                            edit_distance=0,
                            cer_int=0,
                            similarity_int=self.EXACT_SIMILARITY,
                            query_length=normalized_query_length(region),
                            quality_chars=len(normalized_query),
                            source="exact",
                        )
                    )

        candidates.sort(
            key=lambda candidate: (
                candidate.region_id,
                candidate.start_word,
                candidate.end_word,
                candidate.start_char,
                candidate.end_char,
            )
        )
        return tuple(
            replace_candidate_id(candidate, candidate_id)
            for candidate_id, candidate in enumerate(candidates)
        )
