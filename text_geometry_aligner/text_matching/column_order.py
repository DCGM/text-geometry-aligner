"""Conservative two-column reading order for a private ALTO matching view."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from statistics import median
from typing import Sequence
import re

from ..io_alto import ALTOPage
from ..models import BoundingBox

_ROMAN = re.compile(r'M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})')


class ColumnReadingOrder:
    """Read clear left/right columns separately, preserving every word's geometry.

    Optional regions describe separate pages in a stitched image, in page order.
    Without them the entire ALTO page is one region. A split requires a substantial
    gutter, at least six alphabetic lines per column, overlapping vertical spans,
    and few crossing lines. Ambiguous regions retain their existing word order.
    Spanning headings separate vertical bands and are not forced into a column.
    """

    def __init__(self, regions: Sequence[BoundingBox] = ()):
        self.regions = tuple(regions)
        if any(r.width <= 0 or r.height <= 0 for r in self.regions):
            raise ValueError("Reading-order regions must have positive dimensions")
        self.detected_regions = []

    def reorder(self, page: ALTOPage) -> ALTOPage:
        self.detected_regions = []
        if not page.words:
            return page
        # ALTO readers supply line indexes. Without line identity, leave the
        # input alone rather than invent text lines from isolated word boxes.
        if any(w.line_index is None and w.alto_line_id is None for w in page.words):
            return page
        regions = self.regions or (BoundingBox(
            0, 0, page.width or max(w.bbox.x_max for w in page.words),
            page.height or max(w.bbox.y_max for w in page.words)),)
        grouped = defaultdict(list)
        for word in page.words:
            grouped[(word.line_index, word.alto_line_id)].append(word)
        scoped = [[] for _ in regions]
        for words in grouped.values():
            box = _bounds(words)
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            i = min(range(len(regions)), key=lambda j: _point_gap(x, y, regions[j]))
            scoped[i].append(words)
        ordered = []
        for i, (region, lines) in enumerate(zip(regions, scoped)):
            gutter = _gutter(lines, region)
            if gutter is None:
                # Preserve exact original order, including unusual interleaved
                # lines, in regions where no confident split was detected.
                ordered.extend(sorted((w for line in lines for w in line), key=lambda w: w.index))
                continue
            self.detected_regions.append({'region': i, 'columns': 2, 'gutter': list(gutter)})
            offset = len(ordered)
            middle = sum(gutter) / 2
            spanning = [line for line in lines if _bounds(line)[0] < gutter[0]
                        and _bounds(line)[2] > gutter[1]]
            remaining = [line for line in lines if line not in spanning]
            for divider in sorted(spanning, key=_line_key):
                above = [line for line in remaining if _line_key(line)[0] < _line_key(divider)[0]]
                ordered.extend(_columns(above, middle))
                remaining = [line for line in remaining if line not in above]
                ordered.extend(sorted(divider, key=lambda w: w.bbox.x))
            ordered.extend(_columns(remaining, middle))
            self.detected_regions[-1]['reordered'] = (
                [w.index for w in ordered[offset:]] != sorted(w.index for line in lines for w in line))
        if not self.detected_regions:
            return page
        assert len(ordered) == len(page.words)
        assert {w.index for w in ordered} == {w.index for w in page.words}
        if all(a is b for a, b in zip(ordered, page.words)):
            return page
        return replace(page, words=tuple(replace(word, index=i) for i, word in enumerate(ordered)))


def _bounds(words):
    return (min(w.bbox.x for w in words), min(w.bbox.y for w in words),
            max(w.bbox.x_max for w in words), max(w.bbox.y_max for w in words))


def _line_key(words):
    return median(w.bbox.y + w.bbox.height / 2 for w in words), min(w.bbox.x for w in words)


def _columns(lines, middle):
    text_lines = sorted((line for line in lines if _letter_count(line) >= 3),
                        key=lambda line: min(w.index for w in line))
    sides = [(_bounds(line)[0] + _bounds(line)[2]) / 2 >= middle for line in text_lines]
    if not any(left and not right for left, right in zip(sides, sides[1:])):
        # Existing OCR already finishes the left column before the right.
        # Preserve its line/word order, including separately stored numbers.
        return sorted((w for line in lines for w in line), key=lambda w: w.index)
    ordered = []
    for right in [False, True]:
        selected = [line for line in lines if ((_bounds(line)[0] + _bounds(line)[2]) / 2 >= middle) == right]
        for line in sorted(selected, key=_line_key):
            ordered.extend(sorted(line, key=lambda w: w.bbox.x))
    return ordered


def _gutter(lines, region):
    boxes = [_bounds(line) for line in lines if _letter_count(line) >= 3]
    if len(boxes) < 12:
        return None
    edges = sorted({b[j] for b in boxes for j in [0, 2]})
    choices = []
    for left_edge, right_edge in zip(edges, edges[1:]):
        cut = (left_edge + right_edge) / 2
        if not region.x + .2 * region.width < cut < region.x + .8 * region.width:
            continue
        left = [b for b in boxes if b[2] <= cut]
        right = [b for b in boxes if b[0] >= cut]
        crossing = len(boxes) - len(left) - len(right)
        if min(len(left), len(right)) < 6 or crossing > max(1, .08 * len(boxes)):
            continue
        gutter = max(b[2] for b in left), min(b[0] for b in right)
        if gutter[1] - gutter[0] < max(12, .025 * region.width):
            continue
        ly0, ly1 = min(b[1] for b in left), max(b[3] for b in left)
        ry0, ry1 = min(b[1] for b in right), max(b[3] for b in right)
        if min(ly1, ry1) - max(ly0, ry0) < .5 * min(ly1 - ly0, ry1 - ry0):
            continue
        choices.append((crossing, -(gutter[1] - gutter[0]), abs(cut - region.x - region.width / 2), gutter))
    return min(choices)[-1] if choices else None


def _point_gap(x, y, region):
    dx = max(0, region.x - x, x - region.x_max)
    dy = max(0, region.y - y, y - region.y_max)
    return dx * dx + dy * dy


def _letter_count(line):
    total = 0
    for word in line:
        token = word.text.strip('.()[]').upper()
        if token and _ROMAN.fullmatch(token):
            continue
        total += sum(c.isalpha() for c in word.text)
    return total
