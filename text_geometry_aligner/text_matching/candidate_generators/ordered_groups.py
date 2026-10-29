"""Order one anchor per JSON group; locate other fields independently nearby."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ...models import AlignmentRegion
from ..alto_text_index import ALTOTextIndex
from ..candidate import AlignmentCandidate
from .base import CandidateGenerator
from .ordered_alignment import OrderedAlignmentCandidateGenerator
from .utils import replace_candidate_id
from ..column_order import ColumnReadingOrder


class OrderedGroupsCandidateGenerator(CandidateGenerator):
    """A lightweight sequential mode for lists of records with unordered fields.

    Pick the longest field with a normal text candidate as each record's anchor.
    Run the existing fast global ordered alignment only on those anchors. Other
    fields retain their best text quality and take the candidate geometrically
    closest to their own anchor. Unanchored groups stay unmatched. An optional
    reading-order helper can handle clear columns; no application field names
    are interpreted. This produces review hints,
    not a guarantee of correct alignment for complex layouts.
    """

    def __init__(self, field_generator: CandidateGenerator, *, group_depth: int = 2,
                 anchor_generator: CandidateGenerator | None = None,
                 reading_order: ColumnReadingOrder | None = None):
        if type(group_depth) is not int or group_depth < 1:
            raise ValueError("group_depth must be a positive integer")
        self.field_generator = field_generator
        self.group_depth = group_depth
        self.anchor_generator = anchor_generator or OrderedAlignmentCandidateGenerator()
        self.reading_order = reading_order

    def prepare_page(self, page):
        return self.reading_order.reorder(page) if self.reading_order else page

    def generate(self, regions: Sequence[AlignmentRegion], alto_index: ALTOTextIndex
                 ) -> tuple[AlignmentCandidate, ...]:
        groups = defaultdict(list)
        for region in regions:
            if not region.json_text_path:
                raise ValueError("Ordered groups require JSON text paths")
            groups[tuple(region.json_text_path[:self.group_depth])].append(region)
        by_region = defaultdict(list)
        for candidate in self.field_generator.generate(regions, alto_index):
            by_region[candidate.region_id].append(candidate)
        anchors = []
        for group in groups.values():
            available = [region for region in group if by_region[region.region_id]]
            if available:
                anchors.append(max(available, key=lambda region: (
                    max(c.query_length for c in by_region[region.region_id]), -region.region_id)))
        located = {candidate.region_id: candidate for candidate in
                   self.anchor_generator.generate(anchors, alto_index)}
        output = []
        anchor_by_group = {tuple(r.json_text_path[:self.group_depth]): located[r.region_id]
                           for r in anchors if r.region_id in located}
        for key, group in groups.items():
            anchor = anchor_by_group.get(key)
            if anchor is None:
                continue
            anchor_box = _bounds(anchor, alto_index)
            for region in group:
                if region.region_id == anchor.region_id:
                    output.append(anchor)
                    continue
                candidates = by_region[region.region_id]
                if not candidates:
                    continue
                # Geometry disambiguates equally good readings, never chooses a
                # worse textual match just because it happens to be nearby.
                quality = max((c.quality_chars, c.exact) for c in candidates)
                best = [c for c in candidates if (c.quality_chars, c.exact) == quality]
                output.append(min(best, key=lambda candidate: (
                    _distance(_bounds(candidate, alto_index), anchor_box),
                    abs(candidate.start_word - anchor.start_word), candidate.start_word)))
        return tuple(replace_candidate_id(candidate, i) for i, candidate in enumerate(output))


def _bounds(candidate, index):
    boxes = [index.page.words[i].bbox for i in candidate.word_indexes]
    return (min(b.x for b in boxes), min(b.y for b in boxes),
            max(b.x_max for b in boxes), max(b.y_max for b in boxes))


def _distance(left, right):
    # Same-row context is useful even when OCR puts a page-number column in a
    # separate block. No assumption about JSON field order is needed.
    dx = max(0, right[0] - left[2], left[0] - right[2])
    dy = max(0, right[1] - left[3], left[1] - right[3])
    return dy, dx
