"""Shared helpers for text candidate generators."""

from __future__ import annotations

from typing import Any

from ...models import AlignmentRegion
from ..candidate import AlignmentCandidate


def input_text(region: AlignmentRegion) -> str:
    """Return one region's raw input text as a matching string."""

    return "" if region.input_text is None else str(region.input_text)


def normalized_input_text(region: AlignmentRegion) -> str:
    """Return one region's normalized input text for matching."""

    return region.input_text_normalized or ""


def normalized_query_length(region: AlignmentRegion) -> int:
    """Count non-whitespace characters in normalized input text."""

    return sum(
        not character.isspace()
        for character in normalized_input_text(region)
    )


def replace_candidate_id(
    candidate: AlignmentCandidate,
    candidate_id: int,
) -> AlignmentCandidate:
    return AlignmentCandidate(
        candidate_id=candidate_id,
        region_id=candidate.region_id,
        json_text_path=candidate.json_text_path,
        start_word=candidate.start_word,
        end_word=candidate.end_word,
        start_char=candidate.start_char,
        end_char=candidate.end_char,
        query_text=candidate.query_text,
        matched_text=candidate.matched_text,
        normalized_query_text=candidate.normalized_query_text,
        normalized_matched_text=candidate.normalized_matched_text,
        exact=candidate.exact,
        edit_distance=candidate.edit_distance,
        cer_int=candidate.cer_int,
        similarity_int=candidate.similarity_int,
        query_length=candidate.query_length,
        quality_chars=candidate.quality_chars,
        source=candidate.source,
    )


def candidate_sort_key(candidate: AlignmentCandidate) -> tuple[Any, ...]:
    return (
        candidate.region_id,
        candidate.start_word,
        candidate.end_word,
        -int(candidate.exact),
        candidate.edit_distance,
        candidate.source,
    )
