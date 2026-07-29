"""Reading-order text alignment candidate generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ...alto_processing import ALTOTextIndex
from ...models import AlignmentRegion
from ..candidate import (
    CER_SCALE,
    SIMILARITY_SCALE,
    AlignmentCandidate,
)
from .base import CandidateGenerator
from .utils import (
    input_text,
    normalized_input_text,
    normalized_query_length,
)

@dataclass(frozen=True)
class OrderedAlignmentCandidateConfig:
    """Per-value acceptance limits after one ordered global alignment."""

    query_length_boundary: int = 6
    max_cer_at_or_above_boundary: float = 0.20
    max_edit_distance_below_boundary: int = 1

    def __post_init__(self) -> None:
        if self.query_length_boundary < 0:
            raise ValueError("query_length_boundary must not be negative")
        if not 0.0 <= self.max_cer_at_or_above_boundary <= 1.0:
            raise ValueError("max_cer_at_or_above_boundary must be within [0, 1]")
        if self.max_edit_distance_below_boundary < 0:
            raise ValueError(
                "max_edit_distance_below_boundary must not be negative"
            )

    def accepts(
        self,
        edit_distance: int,
        query_reference_length: int,
        boundary_length: int,
    ) -> bool:
        if boundary_length >= self.query_length_boundary:
            return (
                edit_distance / query_reference_length
                <= self.max_cer_at_or_above_boundary + 1e-12
            )
        return edit_distance <= self.max_edit_distance_below_boundary


class OrderedAlignmentCandidateGenerator(CandidateGenerator):
    """Derive final per-value spans from one JSON-to-ALTO global alignment."""

    SOURCE = "ordered-global-alignment"

    def __init__(
        self,
        config: Optional[OrderedAlignmentCandidateConfig] = None,
    ):
        self.config = config or OrderedAlignmentCandidateConfig()

    def generate(
        self,
        regions: Sequence[AlignmentRegion],
        alto_index: ALTOTextIndex,
    ) -> tuple[AlignmentCandidate, ...]:
        source_text, source_owners = _combine_regions(regions)
        target_text = alto_index.normalized_text
        if not source_text or not target_text:
            return ()

        distance_function, editops_function = _load_levenshtein_functions()
        source_to_target = _source_to_target_mapping(
            source_text,
            target_text,
            editops_function(source_text, target_text),
        )
        words_by_region = _assign_words_to_values(
            source_text,
            target_text,
            source_owners,
            source_to_target,
            alto_index,
        )
        regions_by_id = {
            region.region_id: region
            for region in regions
        }

        candidates: list[AlignmentCandidate] = []
        for region_id in sorted(words_by_region):
            region = regions_by_id[region_id]
            normalized_query = normalized_input_text(region)
            if not normalized_query:
                continue

            owned_words = words_by_region[region_id]
            start_word = min(owned_words)
            end_word = max(owned_words)
            start_word, end_word, normalized_matched_text, edit_distance = (
                _trim_misaligned_edge_words(
                    normalized_query,
                    start_word,
                    end_word,
                    alto_index,
                    distance_function,
                )
            )
            char_interval = alto_index.char_interval_for_word_interval(
                start_word,
                end_word,
            )
            if char_interval is None:
                continue

            if not self.config.accepts(
                edit_distance,
                len(normalized_query),
                normalized_query_length(region),
            ):
                continue

            cer = edit_distance / len(normalized_query)
            start_char, end_char = char_interval
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
                    normalized_matched_text=normalized_matched_text,
                    exact=edit_distance == 0,
                    edit_distance=edit_distance,
                    cer_int=round(cer * CER_SCALE),
                    similarity_int=round(
                        max(0.0, 1.0 - cer) * SIMILARITY_SCALE
                    ),
                    query_length=normalized_query_length(region),
                    quality_chars=max(
                        0,
                        len(normalized_query) - edit_distance,
                    ),
                    source=self.SOURCE,
                )
            )

        _validate_final_candidates(candidates)
        return tuple(candidates)


def _combine_regions(
    regions: Sequence[AlignmentRegion],
) -> tuple[str, tuple[Optional[int], ...]]:
    parts: list[str] = []
    owners: list[Optional[int]] = []
    for region in sorted(regions, key=lambda item: item.region_id):
        normalized_text = normalized_input_text(region)
        if not normalized_text:
            continue
        if parts:
            parts.append(" ")
            owners.append(None)
        parts.append(normalized_text)
        owners.extend([region.region_id] * len(normalized_text))
    return "".join(parts), tuple(owners)


def _source_to_target_mapping(
    source: str,
    target: str,
    edit_operations: Sequence[Any],
) -> tuple[Optional[int], ...]:
    mapping: list[Optional[int]] = [None] * len(source)
    source_index = 0
    target_index = 0

    for operation in edit_operations:
        tag, operation_source, operation_target = _unpack_edit_operation(
            operation
        )
        while (
            source_index < operation_source
            and target_index < operation_target
        ):
            mapping[source_index] = target_index
            source_index += 1
            target_index += 1

        if tag == "replace":
            mapping[operation_source] = operation_target
            source_index = operation_source + 1
            target_index = operation_target + 1
        elif tag == "delete":
            source_index = operation_source + 1
            target_index = operation_target
        elif tag == "insert":
            source_index = operation_source
            target_index = operation_target + 1
        else:
            raise ValueError(f"Unsupported Levenshtein edit operation: {tag}")

    while source_index < len(source) and target_index < len(target):
        mapping[source_index] = target_index
        source_index += 1
        target_index += 1
    return tuple(mapping)


def _unpack_edit_operation(operation: Any) -> tuple[str, int, int]:
    if hasattr(operation, "tag"):
        return operation.tag, operation.src_pos, operation.dest_pos
    tag, source_position, target_position = operation
    return str(tag), int(source_position), int(target_position)


def _assign_words_to_values(
    source_text: str,
    target_text: str,
    source_owners: Sequence[Optional[int]],
    source_to_target: Sequence[Optional[int]],
    alto_index: ALTOTextIndex,
) -> dict[int, tuple[int, ...]]:
    votes_by_word: dict[int, Counter[int]] = defaultdict(Counter)
    for source_index, target_index in enumerate(source_to_target):
        region_id = source_owners[source_index]
        if region_id is None or target_index is None:
            continue
        word_index = alto_index.word_index_for_char(target_index)
        if word_index is None:
            continue
        weight = 2 if source_text[source_index] == target_text[target_index] else 1
        votes_by_word[word_index][region_id] += weight

    words_by_region: dict[int, list[int]] = defaultdict(list)
    for word_index, votes in sorted(votes_by_word.items()):
        region_id = min(
            votes,
            key=lambda candidate_region_id: (
                -votes[candidate_region_id],
                candidate_region_id,
            ),
        )
        words_by_region[region_id].append(word_index)
    return {
        region_id: tuple(word_indexes)
        for region_id, word_indexes in words_by_region.items()
    }


def _trim_misaligned_edge_words(
    normalized_query: str,
    start_word: int,
    end_word: int,
    alto_index: ALTOTextIndex,
    distance_function: Any,
) -> tuple[int, int, str, int]:
    """Remove edge words consumed only because the global edit path is tied.

    A minimum-edit character alignment is not unique. For example, when a
    complete ALTO word is inserted after a JSON value, an equally optimal edit
    path may align the final characters of that value into the inserted word.
    Greedily removing an edge is safe only when it strictly improves that
    value's own edit distance.
    """

    def score(
        interval_start: int,
        interval_end: int,
    ) -> tuple[str, int]:
        matched_text = alto_index.normalized_text_for_word_interval(
            interval_start,
            interval_end,
        )
        return (
            matched_text,
            int(distance_function(normalized_query, matched_text)),
        )

    matched_text, edit_distance = score(start_word, end_word)
    while start_word < end_word:
        left_text, left_distance = score(start_word + 1, end_word)
        right_text, right_distance = score(start_word, end_word - 1)
        best_distance, side = min(
            (left_distance, "left"),
            (right_distance, "right"),
        )
        if best_distance >= edit_distance:
            break
        if side == "left":
            start_word += 1
            matched_text = left_text
        else:
            end_word -= 1
            matched_text = right_text
        edit_distance = best_distance

    return start_word, end_word, matched_text, edit_distance


def _validate_final_candidates(
    candidates: Sequence[AlignmentCandidate],
) -> None:
    region_ids: set[int] = set()
    occupied_words: set[int] = set()
    for candidate in candidates:
        if candidate.region_id in region_ids:
            raise RuntimeError(
                f"Ordered alignment produced multiple candidates for value "
                f"{candidate.region_id}"
            )
        region_ids.add(candidate.region_id)

        overlapping_words = occupied_words.intersection(candidate.word_indexes)
        if overlapping_words:
            raise RuntimeError(
                "Ordered alignment produced overlapping candidate word spans: "
                f"{sorted(overlapping_words)}"
            )
        occupied_words.update(candidate.word_indexes)


def _load_levenshtein_functions() -> tuple[Any, Any]:
    try:
        from Levenshtein import distance, editops
    except ImportError as exc:
        raise RuntimeError(
            "Levenshtein is required for ordered alignment generation. "
            "Install it with: python -m pip install Levenshtein"
        ) from exc
    return distance, editops
