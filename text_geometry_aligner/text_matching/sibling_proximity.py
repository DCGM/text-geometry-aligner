"""Re-pick equally good text candidates by their distance to sibling values."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

from ..io_alto import ALTOWord
from ..models import BoundingBox, JSONPath, _format_json_path
from .candidate import AlignmentCandidate

logger = logging.getLogger(__name__)


def refine_by_sibling_proximity(
    candidates: Sequence[AlignmentCandidate],
    selected_candidates: Sequence[AlignmentCandidate],
    words: Sequence[ALTOWord],
) -> tuple[AlignmentCandidate, ...]:
    """Move each selected value to its equally good candidate nearest its siblings.

    A value printed more than once ("V PRAZE", a year) has several equally
    good candidates, and selectors break such ties by position, not by
    context. Siblings are the other selected values sharing the longest
    object path with the value, at least one object deep: the place, the
    publisher and the date of ``("publication", 0)``. A value without such
    siblings keeps its candidate.

    Alternatives never match worse than the selected candidate and never use
    a word of another selected value; a span holding an equally good span plus
    stray words is not one. Among them the one nearest to the siblings wins,
    then the earliest. Moves repeat until none happens, so a word one value
    lets go of can go to another.
    """

    chosen = {candidate.region_id: candidate for candidate in selected_candidates}
    owners = {
        word_index: candidate.region_id
        for candidate in selected_candidates
        for word_index in candidate.word_indexes
    }
    by_region: dict[int, list[AlignmentCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_region[candidate.region_id].append(candidate)

    for _ in range(len(chosen)):
        moved = False
        for region_id in sorted(chosen, key=lambda item: len(by_region[item])):
            current = chosen[region_id]
            alternatives = [
                candidate
                for candidate in by_region[region_id]
                if candidate.similarity_int >= current.similarity_int
                and (candidate.exact or not current.exact)
                and all(
                    owners.get(word_index, region_id) == region_id
                    for word_index in candidate.word_indexes
                )
            ]
            alternatives = [
                candidate
                for candidate in alternatives
                if not any(
                    set(other.word_indexes) < set(candidate.word_indexes)
                    for other in alternatives
                )
            ]
            if len(alternatives) < 2 and current in alternatives:
                continue
            anchors = _sibling_bounds(region_id, chosen, words)
            best = min(
                alternatives,
                key=lambda candidate: (
                    min(
                        (
                            _gap(_bounds(candidate, words), anchor)
                            for anchor in anchors
                        ),
                        default=0.0,
                    ),
                    candidate.start_word,
                ),
            )
            if best is current:
                continue
            for word_index in current.word_indexes:
                del owners[word_index]
            owners.update({word_index: region_id for word_index in best.word_indexes})
            chosen[region_id] = best
            moved = True
            logger.debug(
                "Moved %s from words %d-%d to %d-%d next to its siblings",
                _format_json_path(current.json_text_path or ()),
                current.start_word,
                current.end_word,
                best.start_word,
                best.end_word,
            )
        if not moved:
            break

    return tuple(
        sorted(chosen.values(), key=lambda candidate: candidate.candidate_id)
    )


def _object_path(path: JSONPath | None) -> JSONPath:
    """``("publication", 0)`` for ``("publication", 0, "placeTerm", 0)``."""

    path = tuple(path or ())
    return path[:-2] if path and isinstance(path[-1], int) else path[:-1]


def _common_prefix_length(left: JSONPath, right: JSONPath) -> int:
    length = 0
    for left_component, right_component in zip(left, right):
        if left_component != right_component:
            break
        length += 1
    return length


def _sibling_bounds(
    region_id: int,
    chosen: dict[int, AlignmentCandidate],
    words: Sequence[ALTOWord],
) -> list[BoundingBox]:
    path = _object_path(chosen[region_id].json_text_path)
    prefixes = {
        other_id: _common_prefix_length(path, _object_path(other.json_text_path))
        for other_id, other in chosen.items()
        if other_id != region_id
    }
    depth = max(prefixes.values(), default=0)
    if depth < 2:
        return []
    return [
        _bounds(chosen[other_id], words)
        for other_id, prefix in prefixes.items()
        if prefix == depth
    ]


def _bounds(
    candidate: AlignmentCandidate,
    words: Sequence[ALTOWord],
) -> BoundingBox:
    boxes = [words[word_index].bbox for word_index in candidate.word_indexes]
    x = min(box.x for box in boxes)
    y = min(box.y for box in boxes)
    return BoundingBox(
        x,
        y,
        max(box.x_max for box in boxes) - x,
        max(box.y_max for box in boxes) - y,
    )


def _gap(left: BoundingBox, right: BoundingBox) -> float:
    dx = max(0.0, right.x - left.x_max, left.x - right.x_max)
    dy = max(0.0, right.y - left.y_max, left.y - right.y_max)
    return (dx * dx + dy * dy) ** 0.5
