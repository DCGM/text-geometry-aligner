"""Give a value the match of an identical value printed once for both."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from ..models import AlignmentRegion, _format_json_path
from .candidate import AlignmentCandidate
from .path_pattern import JSONPathPattern

logger = logging.getLogger(__name__)


def copy_identical_matches(
    regions: Sequence[AlignmentRegion],
    selected: Mapping[int, AlignmentCandidate],
    patterns: Sequence[JSONPathPattern],
) -> dict[int, AlignmentRegion]:
    """The value each unmatched value takes its match from.

    Words go to one value only, so a text the input repeats on purpose - a
    description printed once under several names, one person in two roles -
    is found for one value and missed for the others. An unmatched value
    under a pattern takes the match of a value under the same pattern with
    the same normalized text, the same bound indices and an exact match of
    its own. A value matched approximately keeps its match: the text may be
    printed twice, once misread by the OCR. When the identical values match
    different words, which one the text stands for cannot be told and
    nothing is copied.
    """

    copies: dict[int, AlignmentRegion] = {}
    for pattern in patterns:
        matched = []
        for region in regions:
            bound = None if region.json_text_path is None else pattern.match(region.json_text_path)
            if bound is not None and region.input_text_normalized:
                matched.append((region, (bound, region.input_text_normalized)))
        donors: dict[tuple, list[AlignmentRegion]] = {}
        for region, key in matched:
            candidate = selected.get(region.region_id)
            if candidate is not None and candidate.exact:
                donors.setdefault(key, []).append(region)
        for region, key in matched:
            candidate = selected.get(region.region_id)
            if region.region_id in copies or candidate is not None or key not in donors:
                continue
            spans = {
                (selected[donor.region_id].start_word, selected[donor.region_id].end_word)
                for donor in donors[key]
            }
            if len(spans) > 1:
                logger.info(
                    "Not copying a match to %s: identical values match %d different spans",
                    _format_json_path(region.json_text_path or ()),
                    len(spans),
                )
                continue
            copies[region.region_id] = donors[key][0]
    return copies
