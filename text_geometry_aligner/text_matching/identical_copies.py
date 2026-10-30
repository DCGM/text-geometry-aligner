"""Give a value the match of an identical value printed once for both."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..models import AlignmentRegion, JSONPath, _format_json_path
from .candidate import AlignmentCandidate

logger = logging.getLogger(__name__)

_COMPONENT = re.compile(r"\.?([A-Za-z_][A-Za-z0-9_]*)|\[(\*|[A-Za-z_][A-Za-z0-9_]*|\d+)\]")


@dataclass(frozen=True)
class CopyPathPattern:
    """A JSON path pattern whose values may take the match of an identical value.

    Written as a path with list indices in brackets: ``agents[*].description[*]``.
    ``[*]`` matches any index, ``[0]`` that index only, and a name such as
    ``[i]`` any index the donor's path has there too, so
    ``entries[i].descriptions[*]`` copies only within one entry. A leading
    ``$`` is allowed.
    """

    text: str
    components: tuple[str | int, ...]

    @classmethod
    def parse(cls, text: str) -> "CopyPathPattern":
        body = text[1:] if text.startswith("$") else text
        components: list[str | int] = []
        position = 0
        while position < len(body):
            match = _COMPONENT.match(body, position)
            if match is None or (match.group(1) is not None and components and not body.startswith(".", position)):
                raise ValueError(f"Invalid copy path pattern {text!r} at {position}")
            key, index = match.groups()
            if key is not None:
                components.append(key)
            elif index.isdigit():
                components.append(int(index))
            else:
                # An index wildcard or a bound name, told apart from a key
                # by being written in brackets.
                components.append(f"[{index}]")
            position = match.end()
        if not components:
            raise ValueError("Copy path pattern must not be empty")
        return cls(text=text, components=tuple(components))

    def match(self, path: JSONPath) -> tuple[tuple[str, int], ...] | None:
        """The indices bound by name when `path` matches, else None."""

        if len(path) != len(self.components):
            return None
        bound: dict[str, int] = {}
        for component, part in zip(self.components, path):
            if isinstance(component, int) or not component.startswith("["):
                if component != part:
                    return None
            elif not isinstance(part, int):
                return None
            elif component != "[*]":
                name = component[1:-1]
                if bound.setdefault(name, part) != part:
                    return None
        return tuple(sorted(bound.items()))


def copy_identical_matches(
    regions: Sequence[AlignmentRegion],
    selected: Mapping[int, AlignmentCandidate],
    patterns: Sequence[CopyPathPattern],
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
