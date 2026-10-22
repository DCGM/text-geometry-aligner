#!/usr/bin/env python3
"""Add ALTO bboxes to nested bibliographic metadata JSON (e.g. the VLM ground truth) and render them.

Reads every <stem>.json in INPUT_DIR, pairs it with ALTO_DIR/<stem>.xml and writes <stem>.json into
OUTPUT_DIR: the input unchanged plus a "<key>_bbox" next to every transcribed value, as the text
aligner writes it (null when the value was not found in the OCR). With --images-dir and --render-dir
each page is also drawn with its boxes, labelled "<field>: <value> [similarity]".

Metadata repeats page text on purpose - an agent's name is part of its statement of responsibility, a
firm can be both publisher and printer - while the aligner gives every ALTO word to one value only.
Each top-level key is therefore aligned on its own: values of one key compete for words, values of
different keys may share them. A value printed more than once ("V PRAZE", a year) takes, among its
equally good matches, the one nearest to the other values of its object (the place nearest to its
publisher and date, a title of address nearest to the name) instead of the first one on the page.

Coded values (--skip-key: language codes, roles, levels) are not transcriptions and get no bbox.
Publishers with "publisherInferred": true are not printed as publishers either; their bbox is null.
So is the bbox of a match below --min-similarity.

A TSV with every value, its similarity and the matched OCR text goes to OUTPUT_DIR/alignment.tsv.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
from pathlib import Path

from text_geometry_aligner import CPSATCandidateSelector, TextAligner
from text_geometry_aligner.io_alto import ALTOPage, ALTOReader
from text_geometry_aligner.models import BoundingBox
from text_geometry_aligner.normalization import TextNormalizationPipeline
from text_geometry_aligner.rendering import PillowAlignmentRenderer
from text_geometry_aligner.text_matching.candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
)

SKIP_KEYS = ("lang", "role", "language", "originalLanguage", "level")
INFERRED_FLAG = "publisherInferred"
INFERRED_KEY = "publisher"
GEOMETRY_SUFFIX = "_bbox"


def masked(node, skip_keys: set[str]):
    """Copy of node with skipped and inferred values set to None, so list indexes and paths stay the same."""
    if isinstance(node, dict):
        result = {}
        for key, value in node.items():
            if key in skip_keys or (key == INFERRED_KEY and node.get(INFERRED_FLAG) is True):
                result[key] = [None] * len(value) if isinstance(value, list) else None
            else:
                result[key] = masked(value, skip_keys)
        return result
    if isinstance(node, list):
        return [masked(value, skip_keys) for value in node]
    return node


def null_inferred_publishers(node) -> None:
    if isinstance(node, dict):
        if node.get(INFERRED_FLAG) is True and isinstance(node.get(INFERRED_KEY), list):
            node[INFERRED_KEY + GEOMETRY_SUFFIX] = [None] * len(node[INFERRED_KEY])
        for value in node.values():
            null_inferred_publishers(value)
    elif isinstance(node, list):
        for value in node:
            null_inferred_publishers(value)


def object_path(path) -> tuple:
    """Path of the object holding the value: ("publication", 0) for ("publication", 0, "placeTerm", 0)."""
    path = tuple(path)
    return path[:-2] if path and isinstance(path[-1], int) else path[:-1]


def common_prefix_length(a: tuple, b: tuple) -> int:
    length = 0
    for x, y in zip(a, b):
        if x != y:
            break
        length += 1
    return length


def gap(a: BoundingBox, b: BoundingBox) -> float:
    dx = max(0.0, b.x - a.x_max, a.x - b.x_max)
    dy = max(0.0, b.y - a.y_max, a.y - b.y_max)
    return (dx * dx + dy * dy) ** 0.5


class ProximityCPSATCandidateSelector(CPSATCandidateSelector):
    """CP-SAT selection, then each value with several equally good matches takes the one nearest its siblings.

    CP-SAT breaks ties by the earliest ALTO position, which also favours a fuzzy span with a stray word in
    front ("1934. RSKÉHO …" for "PROFESORSKÉHO …"). Siblings are the selected values sharing the longest
    object path with the value, at least one object deep (both in ("publication", 0)). Among equally good
    matches a span that holds another one plus stray words drops out, then the nearest to the siblings
    wins. A move never takes a word from another selected value; moves repeat until none happens, so a
    word one value lets go can go to another.
    """

    alto_page: ALTOPage | None = None

    def select(self, candidates, regions):
        selected = super().select(candidates, regions)
        if self.alto_page is None:
            return selected
        chosen = {candidate.region_id: candidate for candidate in selected}
        owner = {word: candidate.region_id for candidate in selected for word in candidate.word_indexes}
        by_region: dict[int, list] = {}
        for candidate in candidates:
            by_region.setdefault(candidate.region_id, []).append(candidate)

        for _ in range(len(chosen)):
            moved = False
            for region_id in sorted(chosen, key=lambda region_id: len(by_region[region_id])):
                current = chosen[region_id]
                alternatives = [
                    candidate for candidate in by_region[region_id]
                    if candidate.similarity_int >= current.similarity_int
                    and (candidate.exact or not current.exact)
                    and all(owner.get(word, region_id) == region_id for word in candidate.word_indexes)
                ]
                # A span holding an equally good span plus stray words is not an alternative.
                alternatives = [
                    candidate for candidate in alternatives
                    if not any(set(other.word_indexes) < set(candidate.word_indexes) for other in alternatives)
                ]
                if len(alternatives) < 2 and current in alternatives:
                    continue
                anchors = self._sibling_boxes(region_id, chosen)
                best = min(alternatives, key=lambda candidate: (
                    min((gap(self._box(candidate), anchor) for anchor in anchors), default=0.0),
                    candidate.start_word))
                if best is not current:
                    for word in current.word_indexes:
                        del owner[word]
                    owner.update({word: region_id for word in best.word_indexes})
                    chosen[region_id] = best
                    moved = True
            if not moved:
                break
        return tuple(chosen.values())

    def _sibling_boxes(self, region_id: int, chosen: dict) -> list[BoundingBox]:
        path = object_path(chosen[region_id].json_text_path)
        prefixes = {other_id: common_prefix_length(path, object_path(other.json_text_path))
                    for other_id, other in chosen.items() if other_id != region_id}
        depth = max(prefixes.values(), default=0)
        if depth < 2:
            return []
        return [self._box(chosen[other_id]) for other_id, prefix in prefixes.items() if prefix == depth]

    def _box(self, candidate) -> BoundingBox:
        boxes = [self.alto_page.words[index].bbox for index in candidate.word_indexes]
        x, y = min(box.x for box in boxes), min(box.y for box in boxes)
        return BoundingBox(x, y, max(box.x_max for box in boxes) - x, max(box.y_max for box in boxes) - y)


def geometry_after_values(node):
    """Same data with every "<key>_bbox" placed right after its "<key>"."""
    if isinstance(node, list):
        return [geometry_after_values(value) for value in node]
    if not isinstance(node, dict):
        return node
    result = {}
    for key, value in node.items():
        if key.endswith(GEOMETRY_SUFFIX) and key[:-len(GEOMETRY_SUFFIX)] in node:
            continue
        result[key] = geometry_after_values(value)
        if key + GEOMETRY_SUFFIX in node:
            result[key + GEOMETRY_SUFFIX] = node[key + GEOMETRY_SUFFIX]
    return result


def field_name(path) -> str:
    return ".".join(part for part in path if isinstance(part, str))


def build_aligner(args: argparse.Namespace) -> TextAligner:
    fuzzy_config = FuzzyCandidateConfig(
        query_length_boundary=6,
        max_cer_at_or_above_boundary=args.max_cer,
        max_edit_distance_below_boundary=1,
        max_candidates_per_value=5,
    )
    return TextAligner(
        candidate_generator=CompositeCandidateGenerator(
            (ExactTextCandidateGenerator(), AnchoredFuzzyTextCandidateGenerator(fuzzy_config))
        ),
        candidate_selector=ProximityCPSATCandidateSelector(require_optimal=True),
        normalizer=TextNormalizationPipeline.from_optional_names(["lowercase", "strip-punctuation"]),
        renderer=PillowAlignmentRenderer(font_size=args.font_size),
    )


def align_page(aligner: TextAligner, alto_reader: ALTOReader, input_path: Path, alto_path: Path,
               skip_keys: set[str], min_similarity: float):
    data = json.loads(input_path.read_text(encoding="utf-8"))
    alto_page = alto_reader.read(alto_path)
    aligner.candidate_selector.alto_page = alto_page
    regions = []
    for key, value in masked(data, skip_keys).items():
        group = aligner.json_reader.from_data({key: value}, page_key=input_path.stem)
        if not group.regions:
            continue
        aligner._attach_alto(group, alto_page, alto_path)
        aligner.align_page(alto_page, group)
        for region in group.regions:
            if region.alignment_score is not None and region.alignment_score < min_similarity:
                # A short value within the edit-distance limit can still be a different word ("I" -> "IV.").
                region.text_alignment_candidate = region.alto_text = region.alto_text_normalized = None
                region.alto_geometry = region.words = region.alignment_score = None
        regions.extend(group.regions)

    page = aligner.json_reader.from_data(data, page_key=input_path.stem, input_file_path=input_path)
    aligner._attach_alto(page, alto_page, alto_path)
    for region_id, region in enumerate(regions):
        region.region_id = region_id
    page.regions = regions
    output = aligner.json_writer.to_data(page)
    null_inferred_publishers(output)
    return page, geometry_after_values(output)


def render(aligner: TextAligner, page, image_path: Path, render_path: Path) -> None:
    labelled = copy.copy(page)
    labelled.regions = []
    for region in page.regions:
        region = copy.copy(region)
        region.input_text = f"{field_name(region.json_text_path)}: {region.input_text}"
        labelled.regions.append(region)
    aligner.renderer.render(image_path=image_path, output_path=render_path, page=labelled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("alto_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--skip-key", action="append", help=f"key that gets no bbox (default: {', '.join(SKIP_KEYS)})")
    parser.add_argument("--max-cer", type=float, default=0.20, help="fuzzy match limit for values of 6+ characters")
    parser.add_argument("--min-similarity", type=float, default=0.5,
                        help="matches below it get a null bbox (default: 0.5)")
    parser.add_argument("--font-size", type=int, default=22, help="label font size of the renders (default: 22)")
    args = parser.parse_args()
    if (args.images_dir is None) != (args.render_dir is None):
        parser.error("--images-dir and --render-dir must be given together")
    logging.basicConfig(level=logging.ERROR)

    skip_keys = set(args.skip_key or SKIP_KEYS)
    aligner = build_aligner(args)
    alto_reader = ALTOReader()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for input_path in sorted(args.input_dir.glob("*.json")):
        alto_path = args.alto_dir / f"{input_path.stem}.xml"
        if not alto_path.exists():
            print(f"no ALTO for {input_path.name}", file=sys.stderr)
            continue
        page, output = align_page(aligner, alto_reader, input_path, alto_path, skip_keys,
                                  args.min_similarity)
        (args.output_dir / input_path.name).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
        for region in page.regions:
            rows.append((input_path.stem, field_name(region.json_text_path), region.input_text,
                         "" if region.alignment_score is None else f"{region.alignment_score:.3f}",
                         region.alto_text or ""))
        if args.render_dir is not None:
            render(aligner, page, args.images_dir / f"{input_path.stem}.jpg", args.render_dir / f"{input_path.stem}.jpg")

    with (args.output_dir / "alignment.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("page", "field", "value", "similarity", "alto_text"))
        writer.writerows(rows)
    matched = sum(row[3] != "" for row in rows)
    print(f"{len({row[0] for row in rows})} pages, {len(rows)} values, {matched} matched, "
          f"{len(rows) - matched} without a bbox", file=sys.stderr)


if __name__ == "__main__":
    main()
