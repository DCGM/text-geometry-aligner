#!/usr/bin/env python3
"""Geometry-to-text alignment orchestration and CLI."""

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
from text_geometry_aligner.geometry_matching import (
    GeometryOverlapCalculator,
    GeometryWordAssigner,
    WordAssignmentStrategy,
    create_overlap_calculator,
    create_word_assigner,
)
from text_geometry_aligner.json_io import JSONReader, JSONWriter
from text_geometry_aligner.json_processing import (
    AlignmentJSONExporter,
    JSONGeometryExtractor,
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
from text_geometry_aligner.rendering import AlignmentRenderer
from text_geometry_aligner.text_building import (
    TextBuilder,
)
from text_geometry_aligner.utils import (
    _format_json_path,
    _parse_logging_level,
)
from text_geometry_aligner.yolo_processing import YOLOGeometryExtractor

logger = logging.getLogger(__name__)

class GeometryAligner(BaseAligner):
    """Align input geometry to ALTO and enrich the shared hierarchy."""

    alignment_mode = AlignmentMode.GEOMETRY
    supported_input_formats = (InputFormat.JSON, InputFormat.YOLO)

    def __init__(
        self,
        *,
        geometry_suffix: str = "_bbox",
        minimum_word_coverage: float = 0.65,
        word_assignment_strategy: WordAssignmentStrategy | str = (
            WordAssignmentStrategy.GREATEST_COVERAGE
        ),
        overwrite_existing_text: bool = False,
        output_geometry_source: OutputGeometrySource | str = (
            OutputGeometrySource.INPUT
        ),
        output_geometry_format: OutputGeometryFormat | str = (
            OutputGeometryFormat.BBOX
        ),
        overlap_calculator: GeometryOverlapCalculator | None = None,
        word_assigner: GeometryWordAssigner | None = None,
        text_builder: TextBuilder | None = None,
        geometry_builder: GeometryBuilder | None = None,
        alto_reader: ALTOReader | None = None,
        json_reader: JSONReader | None = None,
        json_writer: JSONWriter | None = None,
        renderer: AlignmentRenderer | None = None,
        yolo_extractor: YOLOGeometryExtractor | None = None,
    ):
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")
        if not 0.0 <= minimum_word_coverage <= 1.0:
            raise ValueError(
                "minimum_word_coverage must be within [0, 1]"
            )
        super().__init__(
            alto_reader=alto_reader,
            json_writer=json_writer,
            output_geometry_format=output_geometry_format,
            geometry_builder=geometry_builder,
            text_builder=text_builder,
            renderer=renderer,
        )
        self.json_reader = json_reader or JSONReader()
        self.geometry_suffix = geometry_suffix
        self.minimum_word_coverage = minimum_word_coverage
        self.overwrite_existing_text = overwrite_existing_text
        self.output_geometry_source = OutputGeometrySource(
            output_geometry_source
        )
        self.overlap_calculator = overlap_calculator
        self.word_assignment_strategy = WordAssignmentStrategy(
            word_assignment_strategy
        )
        self.word_assigner = word_assigner or create_word_assigner(
            self.word_assignment_strategy
        )
        self.json_extractor = JSONGeometryExtractor(
            geometry_suffix=geometry_suffix,
            overwrite_existing_text=overwrite_existing_text,
        )
        self.yolo_extractor = yolo_extractor or YOLOGeometryExtractor()
        self.json_exporter = AlignmentJSONExporter(
            alignment_mode=self.alignment_mode,
            geometry_suffix=geometry_suffix,
            output_geometry_format=self.output_geometry_format,
            output_text_source=OutputTextSource.ALTO,
            output_geometry_source=self.output_geometry_source,
        )

    @property
    def render_text_source(self) -> OutputTextSource:
        return OutputTextSource.ALTO

    @property
    def render_geometry_source(self) -> OutputGeometrySource:
        return self.output_geometry_source

    def read_input_page(
        self,
        input_file: Path,
        input_format: InputFormat,
        page_key: str,
    ) -> AlignmentPage:
        if input_format is InputFormat.YOLO:
            return self.yolo_extractor.extract_alignment_page(
                input_file,
                page_key=page_key,
            )
        return self.json_extractor.extract_alignment_page(
            self.json_reader.read(input_file),
            page_key=page_key,
            input_file_path=input_file,
        )

    def align_data(
        self,
        alto_page: ALTOPage,
        input_data: Any,
    ) -> AlignmentDocument:
        """Align loaded geometry JSON and return a one-page document."""

        page = self.json_extractor.extract_alignment_page(
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
        regions = page.regions
        if not regions:
            return page
        calculator = self.overlap_calculator or create_overlap_calculator(
            regions
        )
        coverages = calculator.calculate(
            regions,
            alto_page.words,
            self.minimum_word_coverage,
        )
        assignments = self.word_assigner.assign(
            regions,
            alto_page.words,
            coverages,
        )
        regions_by_id = {
            region.region_id: region for region in regions
        }
        words_by_index = {
            word.index: word for word in alto_page.words
        }

        for assignment in assignments:
            region = regions_by_id[assignment.region_id]
            words = tuple(
                words_by_index[index]
                for index in assignment.word_indexes
            )
            if words:
                if len(assignment.word_coverages) != len(words):
                    raise ValueError(
                        "word coverages and assigned words must have the "
                        "same length"
                    )

                alto_text = self.text_builder.build(words)
                if alto_text is not None:
                    region.words = [
                        AlignmentWord(
                            word_index=word.index,
                            text=word.text,
                            bbox=word.bbox,
                            coverage=assignment.word_coverages[index],
                            line_index=word.line_index,
                            block_index=word.block_index,
                            element_id=word.element_id,
                        )
                        for index, word in enumerate(words)
                    ]
                    region.alto_text = alto_text
                    alto_geometry = self.geometry_builder.build(words)
                    validate_geometry_format(
                        alto_geometry,
                        self.output_geometry_format,
                    )
                    region.alto_geometry = alto_geometry
                    region.alignment_score = (
                        sum(assignment.word_coverages)
                        / len(assignment.word_coverages)
                    )
            if not region.matched:
                logger.warning(
                    "No ALTO words matched geometry at %s",
                    _format_json_path(region.json_geometry_path or ()),
                )
                continue
            logger.info(
                "Matched geometry %s to %d words (%r), average "
                "coverage=%.4f",
                _format_json_path(region.json_geometry_path or ()),
                len(region.words or ()),
                region.alto_text,
                region.alignment_score or 0.0,
            )

        logger.info(
            "Geometry alignment summary: regions=%d matched=%d unmatched=%d",
            len(regions),
            page.matched_count,
            page.unmatched_count,
        )
        return page

    def export_page(self, page: AlignmentPage) -> dict[str, object]:
        return self.json_exporter.export(page)

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract ALTO text using JSON or YOLO geometry."
    )
    add_common_cli_arguments(
        parser,
        input_formats=GeometryAligner.supported_input_formats,
    )
    parser.add_argument(
        "--geometry-suffix",
        default="_bbox",
        help="Suffix identifying geometry keys in JSON input",
    )
    parser.add_argument(
        "--minimum-word-coverage",
        type=float,
        default=0.65,
        help="Minimum covered fraction of an ALTO word (default: 0.65)",
    )
    parser.add_argument(
        "--word-assignment-strategy",
        choices=tuple(item.value for item in WordAssignmentStrategy),
        default=WordAssignmentStrategy.GREATEST_COVERAGE.value,
        help="Shared-word resolution strategy",
    )
    parser.add_argument(
        "--output-geometry-source",
        choices=tuple(item.value for item in OutputGeometrySource),
        default=OutputGeometrySource.INPUT.value,
        help="Use input or ALTO-derived geometry (default: input)",
    )
    parser.add_argument(
        "--overwrite-existing-text",
        action="store_true",
        help="Process and replace existing JSON text destinations",
    )
    parser.add_argument(
        "--logging-level",
        type=_parse_logging_level,
        default=logging.INFO,
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_common_cli_arguments(parser, args)
    logging.basicConfig(
        level=args.logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        aligner = GeometryAligner(
            geometry_suffix=args.geometry_suffix,
            minimum_word_coverage=args.minimum_word_coverage,
            word_assignment_strategy=args.word_assignment_strategy,
            overwrite_existing_text=args.overwrite_existing_text,
            output_geometry_source=args.output_geometry_source,
            output_geometry_format=args.output_alto_geometry_format,
            text_builder=base_aligner_module._build_text_builder(
                args.output_alto_text_format
            ),
            geometry_builder=base_aligner_module._build_geometry_builder(
                OutputGeometryFormat(args.output_alto_geometry_format)
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))

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
