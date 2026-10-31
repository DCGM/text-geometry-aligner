#!/usr/bin/env python3
"""Text-to-geometry alignment orchestration and CLI."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Any, Sequence

import text_geometry_aligner.base_aligner as base_aligner_module
from text_geometry_aligner.io_alto import ALTOPage, ALTOReader, ALTOWord
from text_geometry_aligner.base_aligner import (
    AlignmentPageWriter,
    BaseAligner,
    add_common_cli_arguments,
    validate_common_cli_arguments,
)
from text_geometry_aligner.geometry_building import (
    GeometryBuilder,
    validate_geometry_format,
)
from text_geometry_aligner.io_json import (
    AlignmentJSONWriter,
    JSONTextReader,
)
from text_geometry_aligner.io_label_studio import LabelStudioWriter
from text_geometry_aligner.label_mapping import LabelMapper
from text_geometry_aligner.models import (
    AlignmentDocument,
    AlignmentMode,
    AlignmentPage,
    AlignmentRegion,
    AlignmentWord,
    InputFormat,
    OutputGeometryFormat,
    OutputGeometrySource,
    OutputTextSource,
    _format_json_path,
)
from text_geometry_aligner.normalization import (
    TextNormalizationPipeline,
    TextNormalizer,
)
from text_geometry_aligner.rendering import AlignmentRenderer
from text_geometry_aligner.text_building import (
    TextBuilder,
)
from text_geometry_aligner.text_matching import (
    ALTOTextIndex,
    AlignmentCandidate,
    CER_SCALE,
    SIMILARITY_SCALE,
)
from text_geometry_aligner.text_matching.candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
    OrderedAlignmentCandidateConfig,
    OrderedAlignmentCandidateGenerator,
    OrderedGroupsCandidateGenerator,
)
from text_geometry_aligner.text_matching.candidate_selectors import (
    CPSATCandidateSelector,
    CandidateSelector,
    PassThroughCandidateSelector,
)
from text_geometry_aligner.text_matching.diagnostics import (
    _find_ambiguous_region_ids,
    _find_conflicted_region_ids,
)
from text_geometry_aligner.text_matching.identical_copies import (
    copy_identical_matches,
)
from text_geometry_aligner.text_matching.path_pattern import JSONPathPattern
from text_geometry_aligner.text_matching.sibling_proximity import (
    refine_by_sibling_proximity,
)
from text_geometry_aligner.text_matching.column_order import ColumnReadingOrder
from text_geometry_aligner.utils import _parse_logging_level

logger = logging.getLogger(__name__)


class TextAligner(BaseAligner):
    """Align JSON text regions to ALTO and enrich their shared hierarchy."""

    alignment_mode = AlignmentMode.TEXT
    supported_input_formats = (InputFormat.JSON,)

    def __init__(
        self,
        *,
        candidate_generator: CandidateGenerator,
        candidate_selector: CandidateSelector,
        geometry_suffix: str | None = None,
        output_geometry_format: OutputGeometryFormat | str = (
            OutputGeometryFormat.BBOX
        ),
        normalizer: TextNormalizer | None = None,
        alto_reader: ALTOReader | None = None,
        json_reader: JSONTextReader | None = None,
        json_writer: AlignmentPageWriter | None = None,
        geometry_builder: GeometryBuilder | None = None,
        text_builder: TextBuilder | None = None,
        renderer: AlignmentRenderer | None = None,
        overwrite_existing_geometry: bool = False,
        output_text_source: OutputTextSource | str = OutputTextSource.JSON,
        label_mapper: LabelMapper | None = None,
        skip_keys: Sequence[str] = (),
        min_similarity: float = 0.0,
        share_words_between_top_level_keys: bool = False,
        prefer_sibling_proximity: bool = False,
        share_words_depth: int | None = None,
        share_words_keys: Sequence[str] = (),
        copy_identical_paths: Sequence[str] = (),
        word_group_paths: Sequence[str] = (),
    ):
        if geometry_suffix == "":
            raise ValueError("geometry_suffix must not be empty")
        if label_mapper is not None and json_reader is not None:
            raise ValueError(
                "label_mapper cannot be combined with a custom json_reader; "
                "configure the reader with the mapper instead"
            )
        if skip_keys and json_reader is not None:
            raise ValueError(
                "skip_keys cannot be combined with a custom json_reader; "
                "configure the reader with the keys instead"
            )
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be within [0, 1]")
        if share_words_depth is not None and share_words_depth < 1:
            raise ValueError("share_words_depth must be at least 1")
        if (
            share_words_between_top_level_keys
            and share_words_depth not in (None, 1)
        ):
            raise ValueError(
                "share_words_between_top_level_keys is share_words_depth=1; "
                "do not combine it with another depth"
            )
        parsed_word_group_paths = tuple(
            JSONPathPattern.parse(pattern) for pattern in word_group_paths
        )
        if parsed_word_group_paths and (
            share_words_between_top_level_keys
            or share_words_depth is not None
            or share_words_keys
        ):
            raise ValueError(
                "word_group_paths replace share_words_depth and "
                "share_words_keys; do not combine them"
            )
        for position, first in enumerate(parsed_word_group_paths):
            for second in parsed_word_group_paths[position + 1:]:
                if first.overlaps(second):
                    raise ValueError(
                        f"word_group_paths {first.text!r} and {second.text!r} "
                        "match the same values at the same depth"
                    )
        parsed_output_geometry_format = OutputGeometryFormat(
            output_geometry_format
        )
        resolved_geometry_suffix = geometry_suffix or (
            f"_{parsed_output_geometry_format.value}"
        )
        resolved_json_writer = json_writer or AlignmentJSONWriter(
            alignment_mode=self.alignment_mode,
            geometry_suffix=resolved_geometry_suffix,
            output_geometry_format=parsed_output_geometry_format,
            output_text_source=output_text_source,
            output_geometry_source=OutputGeometrySource.ALTO,
        )
        super().__init__(
            alto_reader=alto_reader,
            json_writer=resolved_json_writer,
            output_geometry_format=output_geometry_format,
            geometry_builder=geometry_builder,
            text_builder=text_builder,
            renderer=renderer,
        )
        self.geometry_suffix = resolved_geometry_suffix
        self.output_text_source = OutputTextSource(output_text_source)
        self.overwrite_existing_geometry = overwrite_existing_geometry
        self.normalizer = (
            normalizer
            or TextNormalizationPipeline.from_optional_names()
        )
        self.candidate_generator = candidate_generator
        self.candidate_selector = candidate_selector
        self.min_similarity = min_similarity
        self.share_words_between_top_level_keys = (
            share_words_between_top_level_keys
        )
        self.share_words_depth = (
            1 if share_words_between_top_level_keys else share_words_depth
        )
        self.share_words_keys = frozenset(share_words_keys)
        self.copy_identical_paths = tuple(
            JSONPathPattern.parse(pattern) for pattern in copy_identical_paths
        )
        self.word_group_paths = parsed_word_group_paths
        self.prefer_sibling_proximity = prefer_sibling_proximity
        self.json_reader = json_reader or JSONTextReader(
            geometry_suffix=self.geometry_suffix,
            overwrite_existing_geometry=overwrite_existing_geometry,
            label_mapper=label_mapper,
            skip_keys=skip_keys,
        )

    @property
    def render_text_source(self) -> OutputTextSource:
        return self.output_text_source

    def read_input_page(
        self,
        input_file: Path,
        input_format: InputFormat,
        page_key: str,
    ) -> AlignmentPage:
        if input_format is not InputFormat.JSON:
            raise ValueError("Text alignment supports JSON input only")
        return self.json_reader.read(
            input_file,
            page_key=page_key,
        )

    def align_data(
        self,
        alto_page: ALTOPage,
        input_data: Any,
    ) -> AlignmentDocument:
        """Align loaded JSON and return a one-page document."""

        page = self.json_reader.from_data(
            input_data,
            page_key="page",
        )
        page.alto_file_path = alto_page.source_path
        page.alto_page_id = alto_page.page_id
        page.alto_width = alto_page.width
        page.alto_height = alto_page.height
        self.align_page(alto_page, page)
        return AlignmentDocument(
            alignment_mode=self.alignment_mode,
            pages=[page],
        )

    def align_pages(
        self,
        alto_pages: Sequence[ALTOPage],
        input_data: Any,
    ) -> AlignmentDocument:
        """Align loaded JSON to several ALTO pages read as one text, in their order.

        A multi-page input - a table of contents over several pages - is one
        text whose values may run over a page break. Each page is prepared for
        matching on its own (the candidate generator's reading order), then
        the pages are stacked one below the other into one page and aligned
        in one pass: values compete for the words of all pages, and a value
        printed across a page break matches the end of one page and the start
        of the next. Every matched word keeps its own page's coordinates and
        ALTO IDs and names that page in `page_index`. A region's `page_index`
        is the page of its first word, and its geometry covers its words on
        that page only. Returns a one-page document without ALTO page data.
        """

        prepare_page = getattr(self.candidate_generator, "prepare_page", None)
        prepared = [
            alto_page if prepare_page is None else prepare_page(alto_page)
            for alto_page in alto_pages
        ]
        words: list[ALTOWord] = []
        origins: list[tuple[int, ALTOWord]] = []
        top = 0.0
        line_offset = block_offset = 0
        for page_index, alto_page in enumerate(prepared):
            for word in alto_page.words:
                origins.append((page_index, word))
                words.append(dataclasses.replace(
                    word,
                    index=len(words),
                    bbox=dataclasses.replace(word.bbox, y=word.bbox.y + top),
                    line_index=None if word.line_index is None else word.line_index + line_offset,
                    block_index=None if word.block_index is None else word.block_index + block_offset,
                ))
            top += alto_page.height or max((word.bbox.y_max for word in alto_page.words), default=0.0)
            line_offset += 1 + max((word.line_index for word in alto_page.words
                                    if word.line_index is not None), default=-1)
            block_offset += 1 + max((word.block_index for word in alto_page.words
                                     if word.block_index is not None), default=-1)
        stacked = ALTOPage(
            source_path=prepared[0].source_path if prepared else Path("."),
            words=tuple(words),
            width=max((alto_page.width or 0.0 for alto_page in prepared), default=None) or None,
            height=top or None,
        )

        page = self.json_reader.from_data(input_data, page_key="page")
        self._align_prepared_page(stacked, page)
        for region in page.regions:
            if not region.words:
                continue
            region.words = [
                dataclasses.replace(
                    word,
                    word_index=origins[word.word_index][1].index,
                    bbox=origins[word.word_index][1].bbox,
                    page_index=origins[word.word_index][0],
                )
                for word in region.words
            ]
            region.page_index = region.words[0].page_index
            region.alto_geometry = self.geometry_builder.build([
                original
                for page_index, original in (origins[index] for index in range(
                    region.text_alignment_candidate.start_word,
                    region.text_alignment_candidate.end_word + 1,
                ))
                if page_index == region.page_index
            ])
        return AlignmentDocument(alignment_mode=self.alignment_mode, pages=[page])

    def align_page(
        self,
        alto_page: ALTOPage,
        page: AlignmentPage,
    ) -> AlignmentPage:
        prepare_page = getattr(self.candidate_generator, "prepare_page", None)
        if prepare_page is not None:
            alto_page = prepare_page(alto_page)
        return self._align_prepared_page(alto_page, page)

    def _align_prepared_page(
        self,
        alto_page: ALTOPage,
        page: AlignmentPage,
    ) -> AlignmentPage:
        for region in page.regions:
            region.input_text_normalized = (
                None
                if region.input_text is None
                else self.normalizer.normalize(str(region.input_text))
            )

        alto_index = ALTOTextIndex(alto_page, self.normalizer)
        candidates = self.candidate_generator.generate(
            page.regions,
            alto_index,
        )
        if self.min_similarity > 0.0:
            # A short value within the edit-distance limit can still be a
            # different word ("I" -> "IV."); such candidates never compete.
            minimum_similarity_int = round(
                self.min_similarity * SIMILARITY_SCALE
            )
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.similarity_int >= minimum_similarity_int
            )
        ambiguous_region_ids = _find_ambiguous_region_ids(candidates)
        selected_candidates: list[AlignmentCandidate] = []
        conflicted_region_ids: list[int] = []
        for partition_regions, partition_candidates in self._word_partitions(
            page.regions,
            candidates,
        ):
            partition_selected = self.candidate_selector.select(
                partition_candidates,
                partition_regions,
            )
            if self.prefer_sibling_proximity:
                partition_selected = refine_by_sibling_proximity(
                    partition_candidates,
                    partition_selected,
                    alto_page.words,
                )
            selected_candidates.extend(partition_selected)
            conflicted_region_ids.extend(
                _find_conflicted_region_ids(
                    partition_candidates,
                    partition_selected,
                    partition_regions,
                )
            )
        selected_by_region = {
            candidate.region_id: candidate
            for candidate in selected_candidates
        }
        copied_from = {}
        if self.copy_identical_paths:
            for region_id, donor in copy_identical_matches(
                page.regions,
                selected_by_region,
                self.copy_identical_paths,
            ).items():
                selected_by_region[region_id] = selected_by_region[
                    donor.region_id
                ]
                copied_from[region_id] = donor.json_text_path

        for region in page.regions:
            candidate = selected_by_region.get(region.region_id)
            if candidate is None:
                logger.warning(
                    "No alignment for %s: %r (normalized=%r)",
                    _format_json_path(region.json_text_path or ()),
                    region.input_text,
                    region.input_text_normalized,
                )
                continue

            region.text_alignment_candidate = candidate
            region.alignment_copied_from = copied_from.get(region.region_id)
            region.alto_text_normalized = (
                alto_index.normalized_text_for_word_interval(
                    candidate.start_word,
                    candidate.end_word,
                )
            )
            matched_words = alto_page.words[
                candidate.start_word : candidate.end_word + 1
            ]
            if not matched_words:
                continue
            alto_text = self.text_builder.build(matched_words)
            if alto_text is None:
                continue

            region.words = [
                AlignmentWord(
                    word_index=word.index,
                    text=word.text,
                    bbox=word.bbox,
                    text_normalized=(
                        alto_index.normalized_word_for_index(word.index)
                    ),
                    line_index=word.line_index,
                    block_index=word.block_index,
                    alto_word_id=word.alto_word_id,
                    alto_line_id=word.alto_line_id,
                    alto_block_id=word.alto_block_id,
                )
                for word in matched_words
            ]
            region.alto_text = alto_text
            region.alto_geometry = self.geometry_builder.build(matched_words)
            region.alignment_score = (
                candidate.similarity_int / SIMILARITY_SCALE
            )
            validate_geometry_format(
                region.alto_geometry,
                self.output_geometry_format,
            )
            logger.info(
                "Matched %s: %r -> words %d-%d (%r), source=%s, "
                "edit_distance=%d, CER=%.4f%s",
                _format_json_path(region.json_text_path or ()),
                region.input_text,
                candidate.start_word,
                candidate.end_word,
                candidate.matched_text,
                candidate.source,
                candidate.edit_distance,
                candidate.cer_int / CER_SCALE,
                ""
                if region.alignment_copied_from is None
                else ", copied from "
                + _format_json_path(region.alignment_copied_from),
            )

        logger.info(
            "Page alignment summary: regions=%d candidates=%d matched=%d "
            "unmatched=%d ambiguous=%d conflicted=%d",
            len(page.regions),
            len(candidates),
            page.matched_count,
            page.unmatched_count,
            len(ambiguous_region_ids),
            len(conflicted_region_ids),
        )
        return page

    def _word_partitions(
        self,
        regions: Sequence[AlignmentRegion],
        candidates: Sequence[AlignmentCandidate],
    ) -> list[tuple[list[AlignmentRegion], list[AlignmentCandidate]]]:
        """Groups of regions that compete for ALTO words, with their candidates.

        Every region competes with every other one by default. With a share
        depth N, only regions whose JSON paths agree on their first N
        components (keys and list indices) compete: depth 1 lets values under
        different top-level keys share words (an agent's name inside its
        statement of responsibility), depth 2 on {"items": [...]} makes every
        list element its own group. Regions under a share key (at any depth)
        form a group of their own within their partition, so a whole statement
        can share words with the values that split it.

        Word group paths name the groups instead: the values under a pattern
        form its group, those under a deeper pattern inside it a group of
        their own, and the values no pattern covers one more group. Under
        ``[*]`` all list elements are one group, under a name such as ``[i]``
        each element is its own.
        """

        if (
            self.share_words_depth is None
            and not self.share_words_keys
            and not self.word_group_paths
        ):
            return [(list(regions), list(candidates))]

        def key(path: Any) -> Any:
            parts = tuple(path or ())
            if self.word_group_paths:
                return _word_group(self.word_group_paths, parts)
            depth = self.share_words_depth
            group = parts[:depth] if depth else ()
            shared = next(
                (
                    part
                    for part in parts
                    if isinstance(part, str) and part in self.share_words_keys
                ),
                None,
            )
            if shared is None:
                return group
            return (*group, ("share-key", shared))

        partitions: dict[
            Any,
            tuple[list[AlignmentRegion], list[AlignmentCandidate]],
        ] = {}
        for region in regions:
            partitions.setdefault(
                key(region.json_text_path),
                ([], []),
            )[0].append(region)
        for candidate in candidates:
            partitions.setdefault(
                key(candidate.json_text_path),
                ([], []),
            )[1].append(candidate)
        return list(partitions.values())

def _word_group(patterns: Sequence[JSONPathPattern], path: tuple) -> Any:
    """The group of the deepest pattern over `path`; () when none is."""

    deepest = None
    for pattern in patterns:
        bound = pattern.match(path, prefix=True)
        if bound is not None and (
            deepest is None
            or len(pattern.components) > len(deepest[0].components)
        ):
            deepest = (pattern, bound)
    if deepest is None:
        return ()
    pattern, bound = deepest
    return ("word-group", pattern.text, bound)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align scalar JSON text values to ALTO words and add geometry."
        )
    )
    add_common_cli_arguments(
        parser,
        input_formats=TextAligner.supported_input_formats,
    )
    parser.add_argument(
        "--geometry-suffix",
        default=None,
        help="Override the generated _bbox or _polygon suffix",
    )
    parser.add_argument(
        "--output-text-source",
        choices=tuple(item.value for item in OutputTextSource),
        default=OutputTextSource.JSON.value,
        help="Use input JSON or matched ALTO text (default: json)",
    )
    parser.add_argument(
        "--text-normalizer",
        action="append",
        choices=("lowercase", "strip-diacritics", "strip-punctuation"),
        default=None,
        help="Optional comparison normalization; repeat to compose",
    )
    parser.add_argument(
        "--candidate-generator",
        choices=("exact", "combined", "ordered-alignment", "ordered-groups"),
        default="combined",
        help="Candidate generation strategy (default: combined)",
    )
    parser.add_argument(
        "--ordered-group-depth", type=int, default=2,
        help="For ordered-groups: JSON path depth defining records (default: 2, "
             "elements of a top-level list). Anchors follow record order; fields do not",
    )
    parser.add_argument(
        "--ordered-group-columns", action="store_true",
        help="For ordered-groups: conservatively restore two-column OCR order",
    )
    parser.add_argument(
        "--candidate-selector",
        choices=("cp-sat", "pass-through"),
        default="cp-sat",
        help="Candidate selection strategy (default: cp-sat)",
    )
    parser.add_argument(
        "--fuzzy-query-length-boundary",
        type=int,
        default=6,
        help="Length boundary between edit-distance and CER rules",
    )
    parser.add_argument(
        "--fuzzy-max-cer-at-or-above-boundary",
        type=float,
        default=0.20,
        help="Maximum CER at or above the boundary",
    )
    parser.add_argument(
        "--fuzzy-max-edit-distance-below-boundary",
        type=int,
        default=1,
        help="Maximum edit distance below the boundary",
    )
    parser.add_argument(
        "--fuzzy-max-candidates-per-value",
        type=int,
        default=5,
        help="Maximum fuzzy candidates retained per region",
    )
    parser.add_argument(
        "--solver-time-limit-seconds",
        type=float,
        default=None,
        help="Optional CP-SAT time limit",
    )
    parser.add_argument(
        "--overwrite-existing-geometry",
        action="store_true",
        help="Replace existing selected-format geometry",
    )
    parser.add_argument(
        "--skip-key",
        action="append",
        default=None,
        help="JSON key whose values (nested ones too) are not aligned; repeat "
        "for more keys",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Discard candidates below this similarity (default: 0)",
    )
    parser.add_argument(
        "--share-words-between-top-level-keys",
        action="store_true",
        help="Let values under different top-level JSON keys match the same "
        "ALTO words (the same as --share-words-depth 1)",
    )
    parser.add_argument(
        "--share-words-depth",
        type=int,
        default=None,
        help="Values compete for ALTO words only when their JSON paths agree "
        "on this many first components (keys and list indices); 2 makes each "
        "element of a top-level list its own group",
    )
    parser.add_argument(
        "--share-words-key",
        action="append",
        default=None,
        help="JSON key whose values (nested ones too) form their own group, so "
        "they may match the words of the other values; repeat for more keys",
    )
    parser.add_argument(
        "--word-group-path",
        action="append",
        default=None,
        help="JSON path pattern (titleInfo, agents[*], entries[i]) whose "
        "values compete for ALTO words only with each other; a deeper pattern "
        "makes a group of its own inside it, values no pattern covers form one "
        "more group; replaces --share-words-depth and --share-words-key; "
        "repeat for more patterns",
    )
    parser.add_argument(
        "--copy-identical-path",
        action="append",
        default=None,
        help="JSON path pattern (agents[*].description[*]; [i] binds an index "
        "the donor shares) whose unmatched values take the match of an "
        "exactly matched value with the same text under the pattern; repeat "
        "for more patterns",
    )
    parser.add_argument(
        "--prefer-sibling-proximity",
        action="store_true",
        help="Among equally good matches take the one nearest to the other "
        "values of the same JSON object",
    )
    parser.add_argument(
        "--logging-level",
        type=_parse_logging_level,
        default=logging.INFO,
        help="Logging level (default: INFO)",
    )
    return parser


def _build_candidate_generator(
    args: argparse.Namespace,
) -> CandidateGenerator:
    if args.ordered_group_columns and args.candidate_generator != "ordered-groups":
        raise ValueError("--ordered-group-columns requires --candidate-generator ordered-groups")
    if args.candidate_generator == "exact":
        return ExactTextCandidateGenerator()
    if args.candidate_generator == "ordered-alignment":
        return OrderedAlignmentCandidateGenerator(
            OrderedAlignmentCandidateConfig(
                query_length_boundary=args.fuzzy_query_length_boundary,
                max_cer_at_or_above_boundary=(
                    args.fuzzy_max_cer_at_or_above_boundary
                ),
                max_edit_distance_below_boundary=(
                    args.fuzzy_max_edit_distance_below_boundary
                ),
            )
        )
    fuzzy_config = FuzzyCandidateConfig(
        query_length_boundary=args.fuzzy_query_length_boundary,
        max_cer_at_or_above_boundary=(
            args.fuzzy_max_cer_at_or_above_boundary
        ),
        max_edit_distance_below_boundary=(
            args.fuzzy_max_edit_distance_below_boundary
        ),
        max_candidates_per_value=args.fuzzy_max_candidates_per_value,
    )
    field_generator = CompositeCandidateGenerator(
        (
            ExactTextCandidateGenerator(),
            AnchoredFuzzyTextCandidateGenerator(fuzzy_config),
        )
    )
    if args.candidate_generator == "ordered-groups":
        return OrderedGroupsCandidateGenerator(field_generator, group_depth=args.ordered_group_depth,
            reading_order=ColumnReadingOrder() if args.ordered_group_columns else None,
            anchor_generator=OrderedAlignmentCandidateGenerator(OrderedAlignmentCandidateConfig(
                query_length_boundary=args.fuzzy_query_length_boundary,
                max_cer_at_or_above_boundary=args.fuzzy_max_cer_at_or_above_boundary,
                max_edit_distance_below_boundary=args.fuzzy_max_edit_distance_below_boundary)))
    return field_generator


def _build_candidate_selector(
    args: argparse.Namespace,
) -> CandidateSelector:
    if args.candidate_selector == "pass-through":
        return PassThroughCandidateSelector()
    return CPSATCandidateSelector(
        time_limit_seconds=args.solver_time_limit_seconds,
        require_optimal=True,
    )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_common_cli_arguments(parser, args)
    logging.basicConfig(
        level=args.logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        normalizer = TextNormalizationPipeline.from_optional_names(
            args.text_normalizer
        )
        if not 0.0 <= args.min_similarity <= 1.0:
            raise ValueError("--min-similarity must be within [0, 1]")
        candidate_generator = _build_candidate_generator(args)
        candidate_selector = _build_candidate_selector(args)
        label_mapper = (
            None
            if args.class_mapping_file is None
            else LabelMapper.from_file(args.class_mapping_file)
        )
        json_writer = (
            LabelStudioWriter(
                alignment_mode=AlignmentMode.TEXT,
                image_prefix=args.label_studio_image_prefix,
                output_text_source=args.output_text_source,
                output_geometry_source=OutputGeometrySource.ALTO,
            )
            if args.output_json_format == "label-studio"
            else None
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    aligner = TextAligner(
        candidate_generator=candidate_generator,
        candidate_selector=candidate_selector,
        geometry_suffix=args.geometry_suffix,
        output_geometry_format=args.output_alto_geometry_format,
        normalizer=normalizer,
        geometry_builder=base_aligner_module._build_geometry_builder(
            OutputGeometryFormat(args.output_alto_geometry_format)
        ),
        text_builder=base_aligner_module._build_text_builder(
            args.output_alto_text_format
        ),
        json_writer=json_writer,
        renderer=base_aligner_module._build_renderer(args),
        overwrite_existing_geometry=args.overwrite_existing_geometry,
        output_text_source=args.output_text_source,
        label_mapper=label_mapper,
        skip_keys=args.skip_key or (),
        min_similarity=args.min_similarity,
        share_words_between_top_level_keys=(
            args.share_words_between_top_level_keys
        ),
        share_words_depth=args.share_words_depth,
        share_words_keys=args.share_words_key or (),
        copy_identical_paths=args.copy_identical_path or (),
        word_group_paths=args.word_group_path or (),
        prefer_sibling_proximity=args.prefer_sibling_proximity,
    )
    aligner.process_directories(
        alto_input_dir=args.alto_dir,
        input_dir=args.input_dir,
        json_output_dir=args.json_output_dir,
        input_format=args.input_format,
        images_input_dir=args.images_dir,
        render_output_dir=args.render_dir,
        fail_on_missing_alto=args.fail_on_missing_alto,
    )


if __name__ == "__main__":
    main()
