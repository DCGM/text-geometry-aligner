"""Diagnostics for candidate-based text matching."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

from ..models import AlignmentRegion, _format_json_path
from .candidate import AlignmentCandidate

logger = logging.getLogger(__name__)


def _find_ambiguous_region_ids(
    candidates: Sequence[AlignmentCandidate],
) -> tuple[int, ...]:
    by_region: dict[int, list[AlignmentCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_region[candidate.region_id].append(candidate)

    ambiguous: list[int] = []
    for region_id, region_candidates in by_region.items():
        best_key = max(
            (
                candidate.quality_chars,
                int(candidate.exact),
                candidate.similarity_int,
            )
            for candidate in region_candidates
        )
        best_spans = {
            (candidate.start_word, candidate.end_word)
            for candidate in region_candidates
            if (
                candidate.quality_chars,
                int(candidate.exact),
                candidate.similarity_int,
            )
            == best_key
        }
        if len(best_spans) > 1:
            ambiguous.append(region_id)
            logger.warning(
                "Region %d has %d equally ranked candidate spans",
                region_id,
                len(best_spans),
            )
    return tuple(sorted(ambiguous))


def _find_conflicted_region_ids(
    candidates: Sequence[AlignmentCandidate],
    selected_candidates: Sequence[AlignmentCandidate],
    regions: Sequence[AlignmentRegion],
) -> tuple[int, ...]:
    selected_region_ids = {
        candidate.region_id for candidate in selected_candidates
    }
    selected_words = {
        word_index
        for candidate in selected_candidates
        for word_index in candidate.word_indexes
    }
    by_region: dict[int, list[AlignmentCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_region[candidate.region_id].append(candidate)

    conflicted: list[int] = []
    for region in regions:
        if region.region_id in selected_region_ids:
            continue
        region_candidates = by_region.get(region.region_id, ())
        if region_candidates and all(
            any(
                word_index in selected_words
                for word_index in candidate.word_indexes
            )
            for candidate in region_candidates
        ):
            conflicted.append(region.region_id)
            logger.warning(
                "All candidates for %s conflict with selected ALTO words",
                _format_json_path(region.json_text_path or ()),
            )
    return tuple(conflicted)
