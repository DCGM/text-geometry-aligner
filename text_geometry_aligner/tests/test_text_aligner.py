"""Tests for text-to-geometry alignment via the TextAligner orchestrator."""

import contextlib
import io
from pathlib import Path

import pytest

import text_geometry_aligner as alignment
from text_geometry_aligner import (
    base_aligner as base_aligner_module,
    text_aligner as aligner_module,
)


def _page(*texts: str, block_indexes: tuple[int, ...] | None = None) -> alignment.ALTOPage:
    if block_indexes is None:
        block_indexes = (0,) * len(texts)
    words = tuple(
        alignment.ALTOWord(
            index=index,
            text=text,
            bbox=alignment.BoundingBox(index * 10, 0, 9, 10),
            line_index=0,
            block_index=block_indexes[index],
        )
        for index, text in enumerate(texts)
    )
    return alignment.ALTOPage(source_path=Path("test.xml"), words=words)


class _FirstCandidateSelector:
    def select(self, candidates, regions):
        return tuple(candidates[:1])


class _AllCandidateSelector:
    def select(self, candidates, regions):
        return tuple(candidates)


def _align_polygon(
    lowercase_normalizer,
    geometry_suffix: str | None = None,
):
    aligner = alignment.TextAligner(
        geometry_suffix=geometry_suffix,
        output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_AllCandidateSelector(),
        normalizer=lowercase_normalizer,
    )
    return (
        aligner,
        aligner.align_data(
            _page("FIRST", "SECOND"),
            {"title": "First Second"},
        ),
    )


def test_polygon_format_selects_builder_suffix_and_result_metadata(
    lowercase_normalizer, alignment_output
) -> None:
    aligner, result = _align_polygon(lowercase_normalizer)
    output = alignment_output(aligner, result)
    region = result.pages[0].regions[0]

    assert output["title_polygon"] == [
        [0, 0],
        [19, 0],
        [19, 10],
        [0, 10],
        [0, 0],
    ]
    assert "title_bbox" not in output
    assert aligner.output_geometry_format is alignment.OutputGeometryFormat.POLYGON
    assert isinstance(region.alto_geometry, alignment.Polygon)


def test_explicit_geometry_suffix_overrides_format_default(
    lowercase_normalizer, alignment_output
) -> None:
    aligner, result = _align_polygon(lowercase_normalizer, "_shape")
    output = alignment_output(aligner, result)

    assert "title_shape" in output
    assert "title_polygon" not in output


def test_extractor_ignores_geometry_from_both_formats() -> None:
    values = alignment.JSONTextReader(
        geometry_suffix="_polygon",
        overwrite_existing_geometry=True,
    ).from_data(
        {
            "title": "Rome",
            "title_bbox": {
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
            },
            "title_polygon": [[1, 2], [4, 2], [1, 2]],
        }
    ).regions

    assert [value.json_text_path for value in values] == [("title",)]


def test_existing_bbox_is_retained_when_polygon_is_added(
    lowercase_normalizer, alignment_output
) -> None:
    existing_bbox = {"x": 1, "y": 2, "width": 3, "height": 4}
    aligner = alignment.TextAligner(
        output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_AllCandidateSelector(),
        normalizer=lowercase_normalizer,
    )

    result = aligner.align_data(
        _page("ROME"),
        {"title": "Rome", "title_bbox": existing_bbox},
    )
    output = alignment_output(aligner, result)

    assert output["title_bbox"] == existing_bbox
    assert "title_polygon" in output


def test_existing_geometry_is_preserved_by_default(alignment_output) -> None:
    existing_polygon = [[1, 2], [4, 2], [1, 2]]
    aligner = alignment.TextAligner(
        output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_AllCandidateSelector(),
    )

    result = aligner.align_data(
        _page("ROME"),
        {"title": "Rome", "title_polygon": existing_polygon},
    )

    assert alignment_output(aligner, result)["title_polygon"] == existing_polygon
    assert result.matched_count == 0


def test_overwrite_existing_geometry_realigns_selected_format(
    alignment_output,
) -> None:
    aligner = alignment.TextAligner(
        output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
        overwrite_existing_geometry=True,
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_AllCandidateSelector(),
    )

    result = aligner.align_data(
        _page("Rome"),
        {
            "title": "Rome",
            "title_polygon": [[1, 2], [4, 2], [1, 2]],
        },
    )

    assert alignment_output(aligner, result)["title_polygon"] == [
        [0, 0],
        [9, 0],
        [9, 10],
        [0, 10],
        [0, 0],
    ]
    assert result.matched_count == 1


def _align_text_source(
    lowercase_normalizer,
    output_text_source: alignment.OutputTextSource | str | None = None,
):
    kwargs = {}
    if output_text_source is not None:
        kwargs["output_text_source"] = output_text_source
    aligner = alignment.TextAligner(
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_FirstCandidateSelector(),
        normalizer=lowercase_normalizer,
        **kwargs,
    )
    return (
        aligner,
        aligner.align_data(_page("ROME"), {"title": "Rome"}),
    )


def test_json_is_the_default_output_and_rendering_text_source(
    lowercase_normalizer, alignment_output
) -> None:
    aligner, result = _align_text_source(lowercase_normalizer)
    page = result.pages[0]
    region = page.regions[0]
    rendered = alignment.PillowAlignmentRenderer._render_alignments(
        page,
        aligner.render_text_source,
        aligner.render_geometry_source,
        aligner.render_geometry_format,
    )

    assert alignment_output(aligner, result)["title"] == "Rome"
    assert aligner.output_text_source is alignment.OutputTextSource.JSON
    assert region.input_text == "Rome"
    assert region.alto_text == "ROME"
    assert (
        alignment.PillowAlignmentRenderer()._build_label(rendered[0])
        == "Rome [1.00]"
    )


def test_alto_source_updates_output_and_rendering_text(
    lowercase_normalizer, alignment_output
) -> None:
    aligner, result = _align_text_source(
        lowercase_normalizer, alignment.OutputTextSource.ALTO
    )
    page = result.pages[0]
    rendered = alignment.PillowAlignmentRenderer._render_alignments(
        page,
        aligner.render_text_source,
        aligner.render_geometry_source,
        aligner.render_geometry_format,
    )

    assert alignment_output(aligner, result)["title"] == "ROME"
    assert aligner.output_text_source is alignment.OutputTextSource.ALTO
    assert page.regions[0].alto_text == "ROME"
    assert (
        alignment.PillowAlignmentRenderer()._build_label(rendered[0])
        == "ROME [1.00]"
    )


def test_public_orchestrator_uses_new_name_and_module() -> None:
    assert alignment.TextAligner.__name__ == "TextAligner"
    assert (
        alignment.TextAligner.__module__ == "text_geometry_aligner.text_aligner"
    )


def test_specific_models_live_with_their_processing_domain() -> None:
    assert alignment.ALTOWord.__module__ == "text_geometry_aligner.io_alto.reader"
    assert alignment.ALTOPage.__module__ == "text_geometry_aligner.io_alto.reader"
    assert not hasattr(alignment, "OCRWord")
    assert (
        alignment.AlignmentCandidate.__module__
        == "text_geometry_aligner.text_matching.candidate"
    )
    assert (
        alignment.ALTOWordSpan.__module__
        == "text_geometry_aligner.text_matching.alto_text_index"
    )
    assert not hasattr(alignment, "OCRWordSpan")


def test_io_apis_have_no_compatibility_aliases() -> None:
    assert alignment.ALTOReader.__module__ == "text_geometry_aligner.io_alto.reader"
    assert (
        alignment.ALTOTextIndex.__module__
        == "text_geometry_aligner.text_matching.alto_text_index"
    )
    assert (
        alignment.JSONTextReader.__module__
        == "text_geometry_aligner.io_json.text_reader"
    )
    assert (
        alignment.JSONGeometryReader.__module__
        == "text_geometry_aligner.io_json.geometry_reader"
    )
    assert (
        alignment.AlignmentJSONWriter.__module__
        == "text_geometry_aligner.io_json.writer"
    )
    assert not hasattr(alignment, "JSONReader")
    assert not hasattr(alignment, "JSONWriter")
    assert not hasattr(alignment, "JSONTextExtractor")
    assert not hasattr(alignment, "JSONGeometryExtractor")
    assert not hasattr(alignment, "AlignmentJSONExporter")
    assert not hasattr(alignment, "YOLOGeometryExtractor")
    assert not hasattr(alignment, "JSONGeometryMerger")
    assert not hasattr(alignment, "JSONTextMerger")
    assert not hasattr(alignment, "ALTOParser")
    assert not hasattr(alignment, "JSONValueWriter")
    assert not hasattr(alignment, "AlignmentInputNormalizer")
    assert not hasattr(alignment, "PreparedTextRegion")
    assert not hasattr(alignment, "StrictTextNormalizer")
    assert not hasattr(alignment, "JSONValueExtractor")
    assert not hasattr(alignment, "JSONAlignmentMerger")
    assert (
        alignment.GeometryBuilder.__module__
        == "text_geometry_aligner.geometry_building.base"
    )
    assert (
        alignment.TextBuilder.__module__ == "text_geometry_aligner.text_building.base"
    )
    assert alignment.UnionBoundingBoxGeometryBuilder.__module__ == (
        "text_geometry_aligner.geometry_building.union_bounding_box"
    )
    assert alignment.OrthogonalPolygonGeometryBuilder.__module__ == (
        "text_geometry_aligner.geometry_building.orthogonal_polygon"
    )
    assert alignment.SpaceSeparatedTextBuilder.__module__ == (
        "text_geometry_aligner.text_building.space_separated"
    )
    assert ".text_matching." in alignment.ExactTextCandidateGenerator.__module__
    assert not hasattr(alignment, "create_geometry_builder")
    assert not hasattr(alignment, "CandidateGenerationStrategy")
    assert not hasattr(alignment, "HybridTextCandidateGenerator")


def test_no_non_cp_sat_fallback_selectors_remain() -> None:
    assert not hasattr(alignment, "AutoCandidateSelector")
    assert not hasattr(alignment, "BranchAndBoundCandidateSelector")


def test_matching_strategy_arguments_are_exposed() -> None:
    parser = aligner_module.build_argument_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--text-normalizer" in option_strings
    assert "--output-text-source" in option_strings
    assert "--output-alto-text-format" in option_strings
    assert "--output-alto-geometry-format" in option_strings
    assert "--output-geometry-format" not in option_strings
    assert "--candidate-generator" in option_strings
    assert "--candidate-selector" in option_strings
    assert "--overwrite-existing-geometry" in option_strings
    assert "--fuzzy-query-length-boundary" in option_strings
    assert "--fuzzy-max-cer-at-or-above-boundary" in option_strings
    assert "--fuzzy-max-edit-distance-below-boundary" in option_strings

    required_arguments = [
        "--alto-dir",
        "alto",
        "--input-dir",
        "json",
        "--json-output-dir",
        "output",
    ]
    default_args = parser.parse_args(required_arguments)
    alto_args = parser.parse_args(
        [*required_arguments, "--output-text-source", "alto"]
    )
    polygon_args = parser.parse_args(
        [
            *required_arguments,
            "--output-alto-geometry-format",
            "polygon",
        ]
    )
    exact_args = parser.parse_args(
        [*required_arguments, "--candidate-generator", "exact"]
    )
    ordered_args = parser.parse_args(
        [
            *required_arguments,
            "--candidate-generator",
            "ordered-alignment",
            "--candidate-selector",
            "pass-through",
        ]
    )
    stacked_normalizer_args = parser.parse_args(
        [
            *required_arguments,
            "--text-normalizer",
            "strip-diacritics",
            "--text-normalizer",
            "lowercase",
        ]
    )
    overwrite_args = parser.parse_args(
        [*required_arguments, "--overwrite-existing-geometry"]
    )
    assert default_args.output_text_source == "json"
    assert default_args.output_alto_text_format == "space-separated"
    assert default_args.output_alto_geometry_format == "bbox"
    assert default_args.candidate_generator == "combined"
    assert default_args.candidate_selector == "cp-sat"
    assert default_args.text_normalizer is None
    assert default_args.geometry_suffix is None
    assert not default_args.overwrite_existing_geometry
    assert alto_args.output_text_source == "alto"
    assert polygon_args.output_alto_geometry_format == "polygon"
    assert exact_args.candidate_generator == "exact"
    assert ordered_args.candidate_generator == "ordered-alignment"
    assert ordered_args.candidate_selector == "pass-through"
    assert stacked_normalizer_args.text_normalizer == [
        "strip-diacritics",
        "lowercase",
    ]
    assert overwrite_args.overwrite_existing_geometry

    with (
        pytest.raises(SystemExit),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        parser.parse_args(
            [*required_arguments, "--candidate-generator", "fuzzy"]
        )


def test_cli_factories_build_selected_matching_strategies() -> None:
    parser = aligner_module.build_argument_parser()
    required_arguments = [
        "--alto-dir",
        "alto",
        "--input-dir",
        "json",
        "--json-output-dir",
        "output",
    ]
    exact_args = parser.parse_args(
        [
            *required_arguments,
            "--candidate-generator",
            "exact",
            "--fuzzy-query-length-boundary",
            "-1",
        ]
    )
    combined_args = parser.parse_args(required_arguments)
    ordered_args = parser.parse_args(
        [
            *required_arguments,
            "--candidate-generator",
            "ordered-alignment",
            "--candidate-selector",
            "pass-through",
        ]
    )

    exact = aligner_module._build_candidate_generator(exact_args)
    combined = aligner_module._build_candidate_generator(combined_args)
    ordered = aligner_module._build_candidate_generator(ordered_args)
    cp_sat = aligner_module._build_candidate_selector(combined_args)
    pass_through = aligner_module._build_candidate_selector(ordered_args)
    text_builder = base_aligner_module._build_text_builder(
        combined_args.output_alto_text_format
    )
    geometry_builder = base_aligner_module._build_geometry_builder(
        alignment.OutputGeometryFormat(combined_args.output_alto_geometry_format)
    )

    assert isinstance(exact, alignment.ExactTextCandidateGenerator)
    assert isinstance(combined, alignment.CompositeCandidateGenerator)
    assert isinstance(ordered, alignment.OrderedAlignmentCandidateGenerator)
    assert isinstance(cp_sat, alignment.CPSATCandidateSelector)
    assert isinstance(pass_through, alignment.PassThroughCandidateSelector)
    assert isinstance(text_builder, alignment.SpaceSeparatedTextBuilder)
    assert isinstance(
        geometry_builder, alignment.UnionBoundingBoxGeometryBuilder
    )
    assert [type(generator) for generator in combined.generators] == [
        alignment.ExactTextCandidateGenerator,
        alignment.AnchoredFuzzyTextCandidateGenerator,
    ]


def test_text_aligner_rejects_mapper_with_custom_reader() -> None:
    with pytest.raises(ValueError, match="custom json_reader"):
        alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            json_reader=alignment.JSONTextReader(),
            label_mapper=alignment.LabelMapper.from_data({"Title": "heading"}),
        )


def test_text_aligner_passes_mapper_to_default_reader() -> None:
    mapper = alignment.LabelMapper.from_data({"Title": "heading"})
    aligner = alignment.TextAligner(
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=alignment.PassThroughCandidateSelector(),
        label_mapper=mapper,
    )

    assert aligner.json_reader.label_mapper is mapper
