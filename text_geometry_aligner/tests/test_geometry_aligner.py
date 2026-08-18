"""Tests for geometry-to-text alignment."""

import contextlib
import io
from pathlib import Path
from unittest import mock

import pytest

import text_geometry_aligner as alignment
from text_geometry_aligner import (
    base_aligner as base_aligner_module,
    geometry_aligner as geometry_aligner_module,
)


def _word(
    index: int,
    text: str,
    x: float,
    y: float = 0,
    width: float = 10,
    height: float = 10,
) -> alignment.ALTOWord:
    return alignment.ALTOWord(
        index=index,
        text=text,
        bbox=alignment.BoundingBox(x, y, width, height),
        line_index=0,
        block_index=0,
    )


def _page(*words: alignment.ALTOWord) -> alignment.ALTOPage:
    return alignment.ALTOPage(
        source_path=Path("page.xml"),
        words=tuple(words),
        width=100,
        height=100,
    )


def _bbox(
    x: float,
    y: float = 0,
    width: float = 10,
    height: float = 10,
) -> dict[str, float]:
    return {"x": x, "y": y, "width": width, "height": height}


def test_extracts_noncontiguous_words_in_alto_order(alignment_output) -> None:
    page = _page(
        _word(0, "FIRST", 0),
        _word(1, "MIDDLE", 10),
        _word(2, "LAST", 20),
    )
    aligner = alignment.GeometryAligner()
    result = aligner.align_data(
        page,
        {
            "outer_bbox": _bbox(0, width=30, height=8),
            "middle_bbox": _bbox(10),
        },
    )

    output = alignment_output(aligner, result)
    outer, middle = result.pages[0].regions
    assert output["outer"] == "FIRST LAST"
    assert output["middle"] == "MIDDLE"
    assert tuple(word.word_index for word in outer.words or ()) == (0, 2)
    assert outer.alignment_score == pytest.approx(0.8)
    assert middle.alto_text == "MIDDLE"


def test_all_over_threshold_can_retain_shared_word(alignment_output) -> None:
    aligner = alignment.GeometryAligner(
        word_assignment_strategy="all-over-threshold"
    )
    result = aligner.align_data(
        _page(_word(0, "WORD", 0)),
        {
            "first_bbox": _bbox(0),
            "second_bbox": _bbox(0),
        },
    )

    output = alignment_output(aligner, result)
    assert output["first"] == "WORD"
    assert output["second"] == "WORD"


def test_custom_text_builder_is_used_by_default_assigner(alignment_output) -> None:
    text_builder = mock.create_autospec(
        alignment.TextBuilder,
        instance=True,
    )
    text_builder.build.return_value = "BUILT"

    aligner = alignment.GeometryAligner(
        text_builder=text_builder,
    )
    result = aligner.align_data(
        _page(_word(0, "WORD", 0)),
        {"title_bbox": _bbox(0)},
    )

    assert alignment_output(aligner, result)["title"] == "BUILT"


def test_custom_assigner_and_text_builder_are_independent_components() -> None:
    assigner = mock.create_autospec(
        alignment.GeometryWordAssigner,
        instance=True,
    )
    text_builder = mock.create_autospec(
        alignment.TextBuilder,
        instance=True,
    )
    aligner = alignment.GeometryAligner(
        word_assigner=assigner,
        text_builder=text_builder,
    )

    assert aligner.word_assigner is assigner
    assert aligner.text_builder is text_builder


def test_unmatched_region_creates_null_and_render_score_zero(alignment_output) -> None:
    aligner = alignment.GeometryAligner()
    result = aligner.align_data(
        _page(_word(0, "WORD", 0)),
        {"missing_bbox": _bbox(50)},
    )

    page = result.pages[0]
    assert alignment_output(aligner, result)["missing"] is None
    assert page.unmatched_count == 1
    rendered = alignment.PillowAlignmentRenderer._render_alignments(
        page,
        alignment.OutputTextSource.ALTO,
        alignment.OutputGeometrySource.INPUT,
        alignment.OutputGeometryFormat.BBOX,
    )
    assert rendered[0].text == "null"
    assert rendered[0].score == 0.0


def test_existing_destination_is_skipped_before_overlap_calculation(
    alignment_output,
) -> None:
    calculator = mock.create_autospec(
        alignment.GeometryOverlapCalculator,
        instance=True,
    )
    aligner = alignment.GeometryAligner(
        overlap_calculator=calculator,
    )
    result = aligner.align_data(
        _page(_word(0, "WORD", 0)),
        {"title": "", "title_bbox": _bbox(0)},
    )

    calculator.calculate.assert_not_called()
    assert result.pages[0].regions == []
    assert alignment_output(aligner, result)["title"] == ""


def test_overwrite_processes_existing_destination(alignment_output) -> None:
    aligner = alignment.GeometryAligner(overwrite_existing_text=True)
    result = aligner.align_data(
        _page(_word(0, "ALTO", 0)),
        {"title": "JSON", "title_bbox": _bbox(0)},
    )

    assert alignment_output(aligner, result)["title"] == "ALTO"


def test_overwrite_replaces_existing_text_with_null_when_unmatched(
    alignment_output,
) -> None:
    aligner = alignment.GeometryAligner(overwrite_existing_text=True)
    result = aligner.align_data(
        _page(_word(0, "ALTO", 0)),
        {"title": "JSON", "title_bbox": _bbox(50)},
    )

    assert alignment_output(aligner, result)["title"] is None


def test_geometry_render_label_uses_average_overlap_score() -> None:
    aligner = alignment.GeometryAligner()
    result = aligner.align_data(
        _page(_word(0, "WORD", 0)),
        {"title_bbox": _bbox(0, width=8, height=20)},
    )

    rendered = alignment.PillowAlignmentRenderer._render_alignments(
        result.pages[0],
        alignment.OutputTextSource.ALTO,
        alignment.OutputGeometrySource.INPUT,
        alignment.OutputGeometryFormat.BBOX,
    )
    assert (
        alignment.PillowAlignmentRenderer()._build_label(rendered[0])
        == "WORD [0.80]"
    )


def test_defaults_and_strategies_are_exposed() -> None:
    parser = geometry_aligner_module.build_argument_parser()
    required = [
        "--alto-dir",
        "alto",
        "--input-dir",
        "json",
        "--json-output-dir",
        "output",
    ]

    defaults = parser.parse_args(required)
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    all_matches = parser.parse_args(
        [
            *required,
            "--geometry-suffix",
            "_polygon",
            "--minimum-overlap-coverage",
            "0.4",
            "--overlap-strategy",
            "word-coverage",
            "--word-assignment-strategy",
            "all-over-threshold",
            "--output-alto-text-format",
            "space-separated",
            "--output-alto-geometry-format",
            "polygon",
            "--overwrite-existing-text",
        ]
    )

    assert defaults.geometry_suffix == "_bbox"
    assert defaults.minimum_overlap_coverage == 0.65
    assert defaults.overlap_strategy == "bidirectional-containment"
    assert defaults.word_assignment_strategy == "greatest-coverage"
    assert defaults.output_alto_text_format == "space-separated"
    assert defaults.output_alto_geometry_format == "bbox"
    assert "--text-builder" not in option_strings
    assert "--output-geometry-format" not in option_strings
    assert not defaults.overwrite_existing_text
    assert all_matches.geometry_suffix == "_polygon"
    assert all_matches.minimum_overlap_coverage == 0.4
    assert all_matches.overlap_strategy == "word-coverage"
    assert all_matches.word_assignment_strategy == "all-over-threshold"
    assert all_matches.output_alto_text_format == "space-separated"
    assert all_matches.output_alto_geometry_format == "polygon"
    assert all_matches.overwrite_existing_text


def test_invalid_assignment_strategy_is_rejected() -> None:
    parser = geometry_aligner_module.build_argument_parser()
    with (
        pytest.raises(SystemExit),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        parser.parse_args(
            [
                "--alto-dir",
                "alto",
                "--input-dir",
                "json",
                "--json-output-dir",
                "output",
                "--word-assignment-strategy",
                "unknown",
            ]
        )


def test_invalid_overlap_strategy_is_rejected() -> None:
    parser = geometry_aligner_module.build_argument_parser()
    with (
        pytest.raises(SystemExit),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        parser.parse_args(
            [
                "--alto-dir",
                "alto",
                "--input-dir",
                "json",
                "--json-output-dir",
                "output",
                "--overlap-strategy",
                "unknown",
            ]
        )


def test_invalid_alto_text_format_is_rejected() -> None:
    parser = geometry_aligner_module.build_argument_parser()
    with (
        pytest.raises(SystemExit),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        parser.parse_args(
            [
                "--alto-dir",
                "alto",
                "--input-dir",
                "json",
                "--json-output-dir",
                "output",
                "--output-alto-text-format",
                "unknown",
            ]
        )


def test_cli_alto_builder_resolution() -> None:
    builder = base_aligner_module._build_text_builder("space-separated")

    assert isinstance(builder, alignment.SpaceSeparatedTextBuilder)
    assert isinstance(
        base_aligner_module._build_geometry_builder(
            alignment.OutputGeometryFormat.POLYGON
        ),
        alignment.OrthogonalPolygonGeometryBuilder,
    )
    with pytest.raises(
        ValueError,
        match="Unsupported output ALTO text format",
    ):
        base_aligner_module._build_text_builder("unknown")


def test_geometry_aligner_rejects_mapper_with_custom_reader() -> None:
    with pytest.raises(ValueError, match="custom input readers"):
        alignment.GeometryAligner(
            yolo_reader=alignment.YOLOReader(),
            label_mapper=alignment.LabelMapper.from_data({"Title": "heading"}),
        )


def test_geometry_aligner_passes_mapper_to_both_default_readers() -> None:
    mapper = alignment.LabelMapper.from_data({"Title": "heading"})
    aligner = alignment.GeometryAligner(label_mapper=mapper)

    assert aligner.json_reader.label_mapper is mapper
    assert aligner.yolo_reader.label_mapper is mapper
