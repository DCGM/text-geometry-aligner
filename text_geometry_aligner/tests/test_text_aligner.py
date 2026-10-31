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


def _cp_sat_aligner(lowercase_normalizer, **options) -> alignment.TextAligner:
    return alignment.TextAligner(
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=alignment.CPSATCandidateSelector(),
        normalizer=lowercase_normalizer,
        **options,
    )


def test_top_level_keys_can_share_words(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"statement": ["by Jan Novak"], "agents": [{"name": "Jan Novak"}]}
    page = _page("by", "Jan", "Novak")

    exclusive = _cp_sat_aligner(lowercase_normalizer)
    shared = _cp_sat_aligner(
        lowercase_normalizer,
        share_words_between_top_level_keys=True,
    )
    exclusive_output = alignment_output(exclusive, exclusive.align_data(page, data))
    shared_output = alignment_output(shared, shared.align_data(page, data))

    assert exclusive_output["statement_bbox"][0] is not None
    assert exclusive_output["agents"][0]["name_bbox"] is None
    assert shared_output["statement_bbox"][0] is not None
    assert shared_output["agents"][0]["name_bbox"] == {
        "x": 10,
        "y": 0,
        "width": 19,
        "height": 10,
    }


def test_share_depth_two_makes_each_list_element_its_own_group(
    lowercase_normalizer, alignment_output
) -> None:
    data = {
        "items": [
            {"section": "Recenze", "title": "Prvni"},
            {"section": "Recenze", "title": "Druha"},
        ]
    }
    page = _page("Recenze", "Prvni", "Druha")

    top_level = _cp_sat_aligner(lowercase_normalizer, share_words_depth=1)
    per_item = _cp_sat_aligner(lowercase_normalizer, share_words_depth=2)
    top_level_output = alignment_output(top_level, top_level.align_data(page, data))
    per_item_output = alignment_output(per_item, per_item.align_data(page, data))

    top_level_sections = [item["section_bbox"] for item in top_level_output["items"]]
    assert sum(box is not None for box in top_level_sections) == 1
    assert [item["section_bbox"]["x"] for item in per_item_output["items"]] == [0, 0]
    assert per_item_output["items"][1]["title_bbox"]["x"] == 20


def test_share_key_lets_a_whole_statement_share_words_with_its_parts(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"works": [{"citation": "Jan Novak Kniha", "title": "Kniha"}]}
    page = _page("Jan", "Novak", "Kniha")

    exclusive = _cp_sat_aligner(lowercase_normalizer)
    shared = _cp_sat_aligner(lowercase_normalizer, share_words_keys=["citation"])
    exclusive_output = alignment_output(exclusive, exclusive.align_data(page, data))
    shared_output = alignment_output(shared, shared.align_data(page, data))

    exclusive_work = exclusive_output["works"][0]
    assert (exclusive_work["citation_bbox"] is None) != (exclusive_work["title_bbox"] is None)
    assert shared_output["works"][0]["citation_bbox"]["x"] == 0
    assert shared_output["works"][0]["title_bbox"]["x"] == 20


def test_values_no_word_group_path_covers_compete_in_one_group(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"publisher": "Otto", "printer": "Otto", "title": "Kniha"}
    page = _page("Otto", "Kniha")

    sink = _cp_sat_aligner(lowercase_normalizer, word_group_paths=["title"])
    grouped = _cp_sat_aligner(lowercase_normalizer, word_group_paths=["publisher"])
    sink_output = alignment_output(sink, sink.align_data(page, data))
    grouped_output = alignment_output(grouped, grouped.align_data(page, data))

    assert (sink_output["publisher_bbox"] is None) != (sink_output["printer_bbox"] is None)
    assert grouped_output["publisher_bbox"]["x"] == grouped_output["printer_bbox"]["x"] == 0


def test_a_deeper_word_group_path_takes_its_values_out_of_the_outer_group(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"work": {"citation": "Jan Novak Kniha", "title": "Kniha", "subTitle": "Kniha"}}
    page = _page("Jan", "Novak", "Kniha")
    aligner = _cp_sat_aligner(lowercase_normalizer, word_group_paths=["work", "work.citation"])

    work = alignment_output(aligner, aligner.align_data(page, data))["work"]

    assert work["citation_bbox"]["x"] == 0
    # The title and the subtitle stay in the outer group and still compete.
    assert (work["title_bbox"] is None) != (work["subTitle_bbox"] is None)


def test_a_word_group_path_with_a_named_index_makes_each_element_its_own_group(
    lowercase_normalizer, alignment_output
) -> None:
    data = {
        "items": [
            {"section": "Recenze", "title": "Prvni"},
            {"section": "Recenze", "title": "Druha"},
        ]
    }
    page = _page("Recenze", "Prvni", "Druha")

    together = _cp_sat_aligner(lowercase_normalizer, word_group_paths=["items[*]"])
    apart = _cp_sat_aligner(lowercase_normalizer, word_group_paths=["items[i]"])
    together_output = alignment_output(together, together.align_data(page, data))
    apart_output = alignment_output(apart, apart.align_data(page, data))

    assert sum(item["section_bbox"] is not None for item in together_output["items"]) == 1
    assert [item["section_bbox"]["x"] for item in apart_output["items"]] == [0, 0]


@pytest.mark.parametrize(
    "patterns",
    [["agents[*]", "agents[0]"], ["agents[i]", "agents[*]"], ["title", "$title"]],
)
def test_word_group_paths_matching_the_same_values_at_the_same_depth_are_rejected(patterns) -> None:
    with pytest.raises(ValueError, match="same depth"):
        alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            word_group_paths=patterns,
        )


@pytest.mark.parametrize(
    "patterns",
    [["agents[0]", "agents[1]"], ["agents", "agents[*]", "agents[*].name"], ["agents[*].name", "series[*].name"]],
)
def test_word_group_paths_at_different_depths_or_on_different_values_are_accepted(patterns) -> None:
    aligner = alignment.TextAligner(
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=alignment.PassThroughCandidateSelector(),
        word_group_paths=patterns,
    )

    assert [pattern.text for pattern in aligner.word_group_paths] == patterns


@pytest.mark.parametrize(
    "options",
    [
        {"share_words_depth": 1},
        {"share_words_between_top_level_keys": True},
        {"share_words_keys": ["citation"]},
    ],
)
def test_word_group_paths_are_not_combined_with_share_options(options) -> None:
    with pytest.raises(ValueError, match="word_group_paths"):
        alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            word_group_paths=["agents"],
            **options,
        )


def _regions_by_path(document) -> dict:
    return {region.json_text_path: region for region in document.pages[0].regions}


def test_a_value_printed_inside_another_is_located_within_its_words(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"entries": [{"designation": "Kapitola 3", "partNumber": "3"}]}
    page = _page("Kapitola", "3")

    exclusive = _cp_sat_aligner(lowercase_normalizer)
    inside = _cp_sat_aligner(
        lowercase_normalizer, inside_paths={"entries[i].partNumber": "entries[i].designation"})
    exclusive_entry = alignment_output(exclusive, exclusive.align_data(page, data))["entries"][0]
    document = inside.align_data(page, data)
    entry = alignment_output(inside, document)["entries"][0]

    assert (exclusive_entry["designation_bbox"] is None) != (exclusive_entry["partNumber_bbox"] is None)
    assert entry["designation_bbox"] == {"x": 0, "y": 0, "width": 19, "height": 10}
    assert entry["partNumber_bbox"] == {"x": 10, "y": 0, "width": 9, "height": 10}
    assert _regions_by_path(document)["entries", 0, "partNumber"].alignment_located_in == (
        "entries", 0, "designation")


def test_a_value_not_printed_inside_its_container_is_aligned_on_its_own(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"agents": [{"description": ["profesor v Brne"], "affiliation": ["Praha"]}]}
    page = _page("profesor", "v", "Brne", "Praha")
    aligner = _cp_sat_aligner(
        lowercase_normalizer, inside_paths={"agents[i].affiliation[*]": "agents[i].description[*]"})

    document = aligner.align_data(page, data)

    assert alignment_output(aligner, document)["agents"][0]["affiliation_bbox"][0]["x"] == 30
    assert _regions_by_path(document)["agents", 0, "affiliation", 0].alignment_located_in is None


def _fuzzy_aligner(lowercase_normalizer, **options) -> alignment.TextAligner:
    return alignment.TextAligner(
        candidate_generator=alignment.CompositeCandidateGenerator((
            alignment.ExactTextCandidateGenerator(),
            alignment.AnchoredFuzzyTextCandidateGenerator(alignment.FuzzyCandidateConfig()),
        )),
        candidate_selector=alignment.CPSATCandidateSelector(),
        normalizer=lowercase_normalizer,
        **options,
    )


@pytest.mark.parametrize("inside_location", ["json", "alto"])
def test_a_value_inside_its_container_finds_a_word_the_ocr_misread(
    lowercase_normalizer, alignment_output, inside_location
) -> None:
    data = {"agents": [{"description": ["profesor Komenskeho university"],
                        "affiliation": ["Komenskeho university"]}]}
    page = _page("profesor", "Komenskcho", "university")
    aligner = _fuzzy_aligner(
        lowercase_normalizer,
        inside_paths={"agents[i].affiliation[*]": "agents[i].description[*]"},
        inside_location=inside_location,
    )

    agent = alignment_output(aligner, aligner.align_data(page, data))["agents"][0]

    assert agent["description_bbox"][0]["x"] == 0
    assert agent["affiliation_bbox"][0] == {"x": 10, "y": 0, "width": 19, "height": 10}


@pytest.mark.parametrize(("inside_location", "word"), [("json", 4), ("alto", 0)])
def test_a_value_inside_by_default_lands_where_its_container_prints_it(
    lowercase_normalizer, inside_location, word
) -> None:
    data = {"agents": [{"description": ["Praga Society, profesor v Praha"], "affiliation": ["Praha"]}]}
    page = _page("Praga", "Society,", "profesor", "v", "Prnna")
    options = {} if inside_location == "json" else {"inside_location": inside_location}
    aligner = _fuzzy_aligner(
        lowercase_normalizer, inside_paths={"agents[i].affiliation[*]": "agents[i].description[*]"}, **options)

    region = _regions_by_path(aligner.align_data(page, data))["agents", 0, "affiliation", 0]

    assert [word.word_index for word in region.words] == [word]


def test_values_inside_one_container_compete_for_its_words(lowercase_normalizer) -> None:
    data = {"agents": [{"description": ["Praha a Praha"], "affiliation": ["Praha", "Praha"]}]}
    aligner = _cp_sat_aligner(
        lowercase_normalizer, inside_paths={"agents[i].affiliation[*]": "agents[i].description[*]"})

    regions = _regions_by_path(aligner.align_data(_page("Praha", "a", "Praha"), data))

    assert sorted(
        [word.word_index for word in regions["agents", 0, "affiliation", number].words] for number in (0, 1)
    ) == [[0], [2]]


def test_a_value_inside_reads_its_container_text_on_words_the_ocr_ran_together(lowercase_normalizer) -> None:
    data = {"agents": [{"description": ["profesor v Praze"], "affiliation": ["v Praze"]}]}
    aligner = _fuzzy_aligner(
        lowercase_normalizer, inside_paths={"agents[i].affiliation[*]": "agents[i].description[*]"})

    region = _regions_by_path(aligner.align_data(_page("profesor", "vPraze"), data))["agents", 0, "affiliation", 0]

    assert [word.word_index for word in region.words] == [1]
    assert region.text_alignment_candidate.matched_text == "vPraze"


def test_a_value_is_located_in_the_first_of_its_containers_that_holds_it(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"entries": [{"title": "Kytice", "subTitle": "Erbenova sbirka", "agents": ["Erbenova"]}]}
    page = _page("Kytice", "Erbenova", "sbirka")
    aligner = _cp_sat_aligner(
        lowercase_normalizer,
        inside_paths={"entries[i].agents[*]": ["entries[i].title", "entries[i].subTitle"]},
    )

    document = aligner.align_data(page, data)

    entry = alignment_output(aligner, document)["entries"][0]
    assert entry["subTitle_bbox"]["x"] == 10
    assert entry["agents_bbox"][0]["x"] == 10
    assert _regions_by_path(document)["entries", 0, "agents", 0].alignment_located_in == (
        "entries", 0, "subTitle")


def test_a_value_inside_a_container_of_another_element_is_aligned_on_its_own(
    lowercase_normalizer,
) -> None:
    data = {"entries": [{"designation": "Kapitola 3"}, {"designation": "Kapitola 4", "partNumber": "3"}]}
    page = _page("Kapitola", "3", "Kapitola", "4", "3")
    aligner = _cp_sat_aligner(
        lowercase_normalizer, inside_paths={"entries[i].partNumber": "entries[i].designation"})

    region = _regions_by_path(aligner.align_data(page, data))["entries", 1, "partNumber"]

    assert region.alignment_located_in is None
    assert [word.word_index for word in region.words] == [4]


def test_a_value_whose_container_is_not_found_stays_unmatched(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"entries": [{"designation": "Oddil 3", "partNumber": "3"}]}
    page = _page("Kapitola", "3")
    aligner = _cp_sat_aligner(
        lowercase_normalizer, inside_paths={"entries[i].partNumber": "entries[i].designation"})

    entry = alignment_output(aligner, aligner.align_data(page, data))["entries"][0]

    assert entry["designation_bbox"] is None
    assert entry["partNumber_bbox"] is None


@pytest.mark.parametrize(
    ("inside_match", "located_in", "x"),
    [
        ("exact", None, 20),
        ("normalized", ("agents", 0, "description", 0), 10),
    ],
)
def test_a_value_is_inside_its_container_by_default_only_as_a_substring_of_its_input_text(
    lowercase_normalizer, alignment_output, inside_match, located_in, x
) -> None:
    data = {"agents": [{"description": ["Profesor, Praha"], "affiliation": ["praha"]}]}
    page = _page("Profesor,", "Praha", "praha")
    options = {} if inside_match == "exact" else {"inside_match": inside_match}
    aligner = _cp_sat_aligner(
        lowercase_normalizer, inside_paths={"agents[i].affiliation[*]": "agents[i].description[*]"}, **options)

    document = aligner.align_data(page, data)

    assert alignment_output(aligner, document)["agents"][0]["affiliation_bbox"][0]["x"] == x
    assert _regions_by_path(document)["agents", 0, "affiliation", 0].alignment_located_in == located_in


def test_a_value_inside_by_words_is_not_part_of_a_word_of_its_container(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"entries": [{"designation": "Kapitola 13", "partNumber": "3"}]}
    page = _page("Kapitola", "13", "3")
    aligner = _cp_sat_aligner(
        lowercase_normalizer, inside_paths={"entries[i].partNumber": "entries[i].designation"},
        inside_match="words")

    document = aligner.align_data(page, data)

    assert alignment_output(aligner, document)["entries"][0]["partNumber_bbox"]["x"] == 20
    assert _regions_by_path(document)["entries", 0, "partNumber"].alignment_located_in is None


@pytest.mark.parametrize("option", ["inside_match", "inside_location"])
def test_inside_options_must_be_known(option) -> None:
    with pytest.raises(ValueError, match=option):
        alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            **{option: "fuzzy"},
        )


def test_inside_paths_need_a_container() -> None:
    with pytest.raises(ValueError, match="names no container"):
        alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            inside_paths={"entries[i].partNumber": []},
        )


def test_a_value_printed_once_for_several_takes_the_match_of_its_identical_value(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"agents": [{"name": "Jan", "description": ["professor"]},
                       {"name": "Petr", "description": ["professor"]}]}
    page = _page("Jan", "Petr", "professor")

    plain = _cp_sat_aligner(lowercase_normalizer)
    copying = _cp_sat_aligner(lowercase_normalizer, copy_identical_paths=["agents[*].description[*]"])
    plain_output = alignment_output(plain, plain.align_data(page, data))
    document = copying.align_data(page, data)
    copying_output = alignment_output(copying, document)

    assert [agent["description_bbox"][0] is None for agent in plain_output["agents"]] == [False, True]
    assert [agent["description_bbox"][0]["x"] for agent in copying_output["agents"]] == [20, 20]
    copied = [region.alignment_copied_from for region in document.pages[0].regions
              if region.json_text_path == ("agents", 1, "description", 0)]
    assert copied == [("agents", 0, "description", 0)]


@pytest.mark.parametrize("pattern, copied", [
    ("entries[i].descriptions[*]", False),
    ("entries[*].descriptions[*]", True),
    ("$.entries[*].descriptions[*]", True),
    ("entries[*].titles[*]", False),
])
def test_a_bound_index_copies_only_within_its_list_element(
    lowercase_normalizer, alignment_output, pattern, copied
) -> None:
    data = {"entries": [{"descriptions": ["prof"]}, {"descriptions": ["prof"]}]}
    page = _page("prof")

    aligner = _cp_sat_aligner(lowercase_normalizer, copy_identical_paths=[pattern])
    output = alignment_output(aligner, aligner.align_data(page, data))

    assert (output["entries"][1]["descriptions_bbox"][0] is not None) is copied


def test_a_bound_index_copies_within_its_list_element(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"entries": [{"descriptions": ["prof", "prof"]}]}
    page = _page("prof")

    aligner = _cp_sat_aligner(lowercase_normalizer, copy_identical_paths=["entries[i].descriptions[*]"])
    output = alignment_output(aligner, aligner.align_data(page, data))

    assert [box["x"] for box in output["entries"][0]["descriptions_bbox"]] == [0, 0]


def test_a_value_matched_approximately_keeps_its_own_match(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"agents": [{"description": ["professor"]}, {"description": ["professor"]}]}
    page = _page("professor", "profesor")
    aligner = alignment.TextAligner(
        candidate_generator=alignment.CompositeCandidateGenerator((
            alignment.ExactTextCandidateGenerator(),
            alignment.AnchoredFuzzyTextCandidateGenerator(alignment.FuzzyCandidateConfig()),
        )),
        candidate_selector=alignment.CPSATCandidateSelector(),
        normalizer=lowercase_normalizer,
        copy_identical_paths=["agents[*].description[*]"],
    )

    output = alignment_output(aligner, aligner.align_data(page, data))

    # Printed twice, once misread by the OCR.
    assert sorted(agent["description_bbox"][0]["x"] for agent in output["agents"]) == [0, 10]


def test_identical_values_matching_different_words_give_no_copy(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"agents": [{"description": ["prof"]}, {"description": ["prof"]}, {"description": ["prof"]}]}
    page = _page("prof", "prof")

    aligner = _cp_sat_aligner(lowercase_normalizer, copy_identical_paths=["agents[*].description[*]"])
    output = alignment_output(aligner, aligner.align_data(page, data))

    assert sum(agent["description_bbox"][0] is not None for agent in output["agents"]) == 2


@pytest.mark.parametrize("pattern", ["", "agents..name", "agents[*", "agents[-1]", "[*]name"])
def test_an_invalid_copy_path_pattern_is_rejected(pattern) -> None:
    with pytest.raises(ValueError):
        alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            copy_identical_paths=[pattern],
        )


def test_pages_aligned_at_once_compete_and_a_value_runs_over_the_page_break(
    lowercase_normalizer,
) -> None:
    first = _page("Les", "a", "voda", "Jan")
    second = _page("Novak", "Pole")
    data = {"entries": [{"title": "Les a voda", "author": "Jan Novak"}, {"title": "Pole"}]}
    aligner = _cp_sat_aligner(lowercase_normalizer)

    document = aligner.align_pages([first, second], data)

    regions = {region.json_text_path: region for region in document.pages[0].regions}
    author = regions["entries", 0, "author"]
    assert [(word.text, word.page_index, word.word_index) for word in author.words] == [
        ("Jan", 0, 3), ("Novak", 1, 0)]
    # Its geometry is its words on the page it starts on, in that page's coordinates.
    assert author.page_index == 0 and author.alto_geometry.x == 30 and author.alto_geometry.width == 9
    title = regions["entries", 1, "title"]
    assert (title.page_index, title.words[0].bbox.y, title.alto_geometry.x) == (1, 0, 10)


def test_pages_aligned_at_once_are_each_put_in_reading_order_on_their_own() -> None:
    calls = []

    class Recording(alignment.ExactTextCandidateGenerator):
        def prepare_page(self, page):
            calls.append(len(page.words))
            return page

    aligner = alignment.TextAligner(candidate_generator=Recording(),
                                    candidate_selector=alignment.PassThroughCandidateSelector())
    aligner.align_pages([_page("a", "b"), _page("c")], {"x": "c"})

    assert calls == [2, 1]


def test_share_words_between_top_level_keys_is_depth_one() -> None:
    aligner = alignment.TextAligner(
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=alignment.PassThroughCandidateSelector(),
        share_words_between_top_level_keys=True,
    )

    assert aligner.share_words_depth == 1


@pytest.mark.parametrize(
    "options",
    [
        {"share_words_depth": 0},
        {"share_words_between_top_level_keys": True, "share_words_depth": 2},
    ],
)
def test_text_aligner_rejects_invalid_share_depth(options) -> None:
    with pytest.raises(ValueError, match="share_words"):
        alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            **options,
        )


def test_sibling_proximity_picks_the_repeat_next_to_its_siblings(
    lowercase_normalizer, alignment_output
) -> None:
    data = {"publication": [{"placeTerm": "Praha", "publisher": "Otto"}]}
    page = _page("Praha", "a", "b", "c", "d", "Otto", "Praha")

    earliest = _cp_sat_aligner(lowercase_normalizer)
    nearest = _cp_sat_aligner(lowercase_normalizer, prefer_sibling_proximity=True)
    earliest_output = alignment_output(earliest, earliest.align_data(page, data))
    nearest_output = alignment_output(nearest, nearest.align_data(page, data))

    assert earliest_output["publication"][0]["placeTerm_bbox"]["x"] == 0
    assert nearest_output["publication"][0]["placeTerm_bbox"]["x"] == 60
    assert nearest_output["publication"][0]["publisher_bbox"]["x"] == 50


def test_min_similarity_rejects_a_different_short_word(
    lowercase_normalizer, alignment_output
) -> None:
    def aligner(**options):
        return alignment.TextAligner(
            candidate_generator=aligner_module._build_candidate_generator(
                aligner_module.build_argument_parser().parse_args(
                    ["--alto-dir", "a", "--input-dir", "j", "--json-output-dir", "o"]
                )
            ),
            candidate_selector=alignment.CPSATCandidateSelector(),
            normalizer=lowercase_normalizer,
            **options,
        )

    data = {"partNumber": "I"}
    page = _page("IV")
    loose, strict = aligner(), aligner(min_similarity=0.5)

    assert alignment_output(loose, loose.align_data(page, data))["partNumber_bbox"]
    assert (
        alignment_output(strict, strict.align_data(page, data))["partNumber_bbox"]
        is None
    )


def test_text_aligner_validates_new_options() -> None:
    with pytest.raises(ValueError, match="min_similarity"):
        alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            min_similarity=1.5,
        )
    with pytest.raises(ValueError, match="custom json_reader"):
        alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            json_reader=alignment.JSONTextReader(),
            skip_keys=("lang",),
        )


def test_new_cli_options_reach_the_aligner_and_renderer() -> None:
    parser = aligner_module.build_argument_parser()
    required_arguments = [
        "--alto-dir",
        "alto",
        "--input-dir",
        "json",
        "--json-output-dir",
        "output",
    ]
    default_args = parser.parse_args(required_arguments)
    args = parser.parse_args(
        [
            *required_arguments,
            "--skip-key",
            "lang",
            "--skip-key",
            "role",
            "--min-similarity",
            "0.5",
            "--share-words-between-top-level-keys",
            "--prefer-sibling-proximity",
            "--render-font-size",
            "22",
            "--render-field-names",
        ]
    )

    assert default_args.skip_key is None
    assert default_args.min_similarity == 0.0
    assert not default_args.share_words_between_top_level_keys
    assert not default_args.prefer_sibling_proximity
    assert default_args.render_font_size == 36
    assert not default_args.render_field_names
    assert args.skip_key == ["lang", "role"]
    assert args.min_similarity == 0.5
    assert args.share_words_between_top_level_keys
    assert args.prefer_sibling_proximity

    renderer = base_aligner_module._build_renderer(args)
    assert renderer.font_size == 22
    assert renderer.show_field_names


def test_share_words_cli_options() -> None:
    parser = aligner_module.build_argument_parser()
    required_arguments = [
        "--alto-dir",
        "alto",
        "--input-dir",
        "json",
        "--json-output-dir",
        "output",
    ]
    default_args = parser.parse_args(required_arguments)
    args = parser.parse_args(
        [
            *required_arguments,
            "--share-words-depth",
            "2",
            "--share-words-key",
            "citation",
            "--share-words-key",
            "statement",
        ]
    )

    assert default_args.share_words_depth is None
    assert default_args.share_words_key is None
    assert args.share_words_depth == 2
    assert args.share_words_key == ["citation", "statement"]


def test_word_group_path_cli_option() -> None:
    parser = aligner_module.build_argument_parser()
    required_arguments = ["--alto-dir", "alto", "--input-dir", "json", "--json-output-dir", "output"]

    default_args = parser.parse_args(required_arguments)
    args = parser.parse_args([*required_arguments, "--word-group-path", "titleInfo",
                              "--word-group-path", "titleInfo[*].statement"])

    assert default_args.word_group_path is None
    assert args.word_group_path == ["titleInfo", "titleInfo[*].statement"]


def test_inside_path_cli_option_gives_the_containers_of_a_value_in_order() -> None:
    parser = aligner_module.build_argument_parser()
    required_arguments = ["--alto-dir", "alto", "--input-dir", "json", "--json-output-dir", "output"]

    args = parser.parse_args([*required_arguments,
                              "--inside-path", "entries[i].agents[*]=entries[i].title",
                              "--inside-path", "entries[i].agents[*]=entries[i].subTitle"])

    assert aligner_module._inside_paths(args.inside_path) == {
        "entries[i].agents[*]": ["entries[i].title", "entries[i].subTitle"]}
    assert args.inside_match == "exact"
    assert parser.parse_args([*required_arguments, "--inside-match", "words"]).inside_match == "words"
    assert args.inside_location == "json"
    assert parser.parse_args([*required_arguments, "--inside-location", "alto"]).inside_location == "alto"
    with pytest.raises(ValueError, match="VALUE=CONTAINER"):
        aligner_module._inside_paths(["entries[i].partNumber"])
