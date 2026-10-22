#!/usr/bin/env python3
"""Add ALTO bboxes to nested bibliographic metadata JSON (e.g. the VLM ground truth) and render them.

Reads every <stem>.json in INPUT_DIR, pairs it with ALTO_DIR/<stem>.xml and writes <stem>.json into
OUTPUT_DIR: the input plus a "<key>_bbox" right after every transcribed value (null when the value was
not found in the OCR). With --images-dir and --render-dir each page is also drawn with its boxes,
labelled "<field>: <value> [similarity]".

This is the text aligner with the options the metadata needs:
- values under different top-level keys may share words (an agent's name is part of its statement of
  responsibility, a firm can be both publisher and printer);
- a value printed more than once takes the equally good match nearest to its siblings (the imprint's
  "V PRAZE", not the first one on the page);
- coded values (--skip-key: language codes, roles, levels) get no bbox;
- matches below --min-similarity get none ("I" -> "IV.").
Publishers with "publisherInferred": true are not printed as publishers; they are not aligned and keep
a null bbox.

A TSV with every aligned value, its similarity and the matched OCR text goes to OUTPUT_DIR/alignment.tsv.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from text_geometry_aligner import CPSATCandidateSelector, PillowAlignmentRenderer, TextAligner
from text_geometry_aligner.normalization import TextNormalizationPipeline
from text_geometry_aligner.text_matching.candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
)

SKIP_KEYS = ("lang", "role", "language", "originalLanguage", "level")
INFERRED_FLAG = "publisherInferred"
INFERRED_KEY = "publisher"


def is_inferred_publisher(data, path) -> bool:
    if INFERRED_KEY not in path:
        return False
    node = data
    for component in path[:path.index(INFERRED_KEY)]:
        node = node[component]
    return isinstance(node, dict) and node.get(INFERRED_FLAG) is True


class MetadataTextAligner(TextAligner):
    """Text aligner leaving inferred publishers out of the matching, so their geometry stays null."""

    def align_page(self, alto_page, page):
        regions = page.regions
        page.regions = [region for region in regions
                        if not is_inferred_publisher(page.json_source_data, region.json_text_path)]
        try:
            return super().align_page(alto_page, page)
        finally:
            page.regions = regions


def field_name(path) -> str:
    return ".".join(part for part in path if isinstance(part, str))


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

    aligner = MetadataTextAligner(
        candidate_generator=CompositeCandidateGenerator((
            ExactTextCandidateGenerator(),
            AnchoredFuzzyTextCandidateGenerator(FuzzyCandidateConfig(max_cer_at_or_above_boundary=args.max_cer)),
        )),
        candidate_selector=CPSATCandidateSelector(),
        normalizer=TextNormalizationPipeline.from_optional_names(["lowercase", "strip-punctuation"]),
        renderer=PillowAlignmentRenderer(font_size=args.font_size, show_field_names=True),
        skip_keys=args.skip_key or SKIP_KEYS,
        min_similarity=args.min_similarity,
        share_words_between_top_level_keys=True,
        prefer_sibling_proximity=True,
    )
    document = aligner.process_directories(
        alto_input_dir=args.alto_dir,
        input_dir=args.input_dir,
        json_output_dir=args.output_dir,
        images_input_dir=args.images_dir,
        render_output_dir=args.render_dir,
    )

    rows = [(page.page_key, field_name(region.json_text_path), region.input_text,
             "" if region.alignment_score is None else f"{region.alignment_score:.3f}", region.alto_text or "")
            for page in document.pages for region in page.regions
            if not is_inferred_publisher(page.json_source_data, region.json_text_path)]
    with (args.output_dir / "alignment.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("page", "field", "value", "similarity", "alto_text"))
        writer.writerows(rows)
    matched = sum(row[3] != "" for row in rows)
    print(f"{len(document.pages)} pages, {len(rows)} values, {matched} matched, {len(rows) - matched} without a bbox",
          file=sys.stderr)


if __name__ == "__main__":
    main()
