#!/usr/bin/env python3
"""Text-to-geometry alignment orchestration and CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import text_geometry_aligner.base_aligner as base_aligner_module
from text_geometry_aligner.alto_io import ALTOPage, ALTOReader
from text_geometry_aligner.base_aligner import (
    BaseAligner,
    add_common_cli_arguments,
    validate_common_cli_arguments,
)
from text_geometry_aligner.geometry_building import (
    GeometryBuilder,
    validate_geometry_format,
)
from text_geometry_aligner.json_io import JSONReader, JSONWriter
from text_geometry_aligner.json_processing import (
    AlignmentJSONExporter,
    JSONTextExtractor,
)
from text_geometry_aligner.models import (
    AlignmentDocument,
    AlignmentMode,
    AlignmentPage,
    AlignmentWord,
    InputFormat,
    OutputGeometryFormat,
    OutputGeometrySource,
    OutputTextSource,
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
from text_geometry_aligner.utils import (
    _format_json_path,
    _parse_logging_level,
)

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
        json_reader: JSONReader | None = None,
        json_writer: JSONWriter | None = None,
        geometry_builder: GeometryBuilder | None = None,
        text_builder: TextBuilder | None = None,
        renderer: AlignmentRenderer | None = None,
        overwrite_existing_geometry: bool = False,
        output_text_source: OutputTextSource | str = OutputTextSource.JSON,
    ):
        if geometry_suffix == "":
            raise ValueError("geometry_suffix must not be empty")
        super().__init__(
            alto_reader=alto_reader,
            json_writer=json_writer,
            output_geometry_format=output_geometry_format,
            geometry_builder=geometry_builder,
            text_builder=text_builder,
            renderer=renderer,
        )
        self.json_reader = json_reader or JSONReader()
        self.geometry_suffix = geometry_suffix or (
            f"_{self.output_geometry_format.value}"
        )
        self.output_text_source = OutputTextSource(output_text_source)
        self.overwrite_existing_geometry = overwrite_existing_geometry
        self.normalizer = (
            normalizer
            or TextNormalizationPipeline.from_optional_names()
        )
        self.candidate_generator = candidate_generator
        self.candidate_selector = candidate_selector
        self.extractor = JSONTextExtractor(
            geometry_suffix=self.geometry_suffix,
            overwrite_existing_geometry=overwrite_existing_geometry,
        )
        self.json_exporter = AlignmentJSONExporter(
            alignment_mode=self.alignment_mode,
            geometry_suffix=self.geometry_suffix,
            output_geometry_format=self.output_geometry_format,
            output_text_source=self.output_text_source,
            output_geometry_source=OutputGeometrySource.ALTO,
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
        return self.extractor.extract_alignment_page(
            self.json_reader.read(input_file),
            page_key=page_key,
            input_file_path=input_file,
        )

    def align_data(
        self,
        alto_page: ALTOPage,
        input_data: Any,
    ) -> AlignmentDocument:
        """Align loaded JSON and return a one-page document."""

        page = self.extractor.extract_alignment_page(
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

    def align_page(
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
        ambiguous_region_ids = _find_ambiguous_region_ids(candidates)
        selected_candidates = self.candidate_selector.select(
            candidates,
            page.regions,
        )
        conflicted_region_ids = _find_conflicted_region_ids(
            candidates,
            selected_candidates,
            page.regions,
        )
        selected_by_region = {
            candidate.region_id: candidate
            for candidate in selected_candidates
        }

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
                    element_id=word.element_id,
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
                "edit_distance=%d, CER=%.4f",
                _format_json_path(region.json_text_path or ()),
                region.input_text,
                candidate.start_word,
                candidate.end_word,
                candidate.matched_text,
                candidate.source,
                candidate.edit_distance,
                candidate.cer_int / CER_SCALE,
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

    def export_page(self, page: AlignmentPage) -> dict[str, object]:
        return self.json_exporter.export(page)

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
        choices=("exact", "combined", "ordered-alignment"),
        default="combined",
        help="Candidate generation strategy (default: combined)",
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
        "--logging-level",
        type=_parse_logging_level,
        default=logging.INFO,
        help="Logging level (default: INFO)",
    )
    return parser


def _build_candidate_generator(
    args: argparse.Namespace,
) -> CandidateGenerator:
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
    return CompositeCandidateGenerator(
        (
            ExactTextCandidateGenerator(),
            AnchoredFuzzyTextCandidateGenerator(fuzzy_config),
        )
    )


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
        candidate_generator = _build_candidate_generator(args)
        candidate_selector = _build_candidate_selector(args)
    except ValueError as exc:
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
        overwrite_existing_geometry=args.overwrite_existing_geometry,
        output_text_source=args.output_text_source,
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
