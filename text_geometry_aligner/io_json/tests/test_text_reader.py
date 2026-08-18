"""Tests for the JSON text reader."""

import json
from pathlib import Path

import text_geometry_aligner as alignment


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


def _normalize_regions(
    regions: tuple[alignment.AlignmentRegion, ...],
    normalizer: alignment.TextNormalizer,
) -> tuple[alignment.AlignmentRegion, ...]:
    for region in regions:
        region.input_text_normalized = (
            None
            if region.input_text is None
            else normalizer.normalize(str(region.input_text))
        )
    return regions


class _AllCandidateSelector:
    def select(self, candidates, regions):
        return tuple(candidates)


def test_extractor_exposes_each_scalar_list_element() -> None:
    values = alignment.JSONTextReader().from_data(
        {"publisher": ["First", "Second"]}
    ).regions

    assert [value.json_text_path for value in values] == [
        ("publisher", 0),
        ("publisher", 1),
    ]
    assert [value.json_geometry_path for value in values] == [
        ("publisher_bbox", 0),
        ("publisher_bbox", 1),
    ]


def test_each_list_value_gets_parallel_geometry_and_alto_text(
    lowercase_normalizer, alignment_output
) -> None:
    aligner = alignment.TextAligner(
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_AllCandidateSelector(),
        normalizer=lowercase_normalizer,
        output_text_source=alignment.OutputTextSource.ALTO,
    )
    result = aligner.align_data(
        _page("FIRST", "SECOND"),
        {"publisher": ["First", "Second"]},
    )
    output = alignment_output(aligner, result)

    assert output["publisher"] == ["FIRST", "SECOND"]
    assert output["publisher_bbox"] == [
        {"x": 0, "y": 0, "width": 9, "height": 10},
        {"x": 10, "y": 0, "width": 9, "height": 10},
    ]
    assert [
        region.json_text_path
        for region in result.pages[0].regions
        if region.matched
    ] == [("publisher", 0), ("publisher", 1)]


def test_publisher_list_phrase_participates_in_candidate_generation() -> None:
    page = _page(
        "ŠOLC",
        "a",
        "ŠIMÁČEK,",
        "společnost",
        "s",
        "r.",
        "o.",
        "v",
        "Praze.",
    )
    normalizer = alignment.TextNormalizationPipeline.from_optional_names()
    regions = _normalize_regions(
        alignment.JSONTextReader().from_data(
            {
                "partNumber": "I.",
                "placeTerm": "Praze.",
                "publisher": [
                    "ŠOLC a ŠIMÁČEK, společnost s r. o. v Praze."
                ],
            }
        ).regions,
        normalizer,
    )
    candidates = alignment.ExactTextCandidateGenerator().generate(
        regions,
        alignment.ALTOTextIndex(page, normalizer),
    )

    publisher_candidate = next(
        candidate
        for candidate in candidates
        if candidate.json_text_path == ("publisher", 0)
    )
    assert (
        publisher_candidate.start_word,
        publisher_candidate.end_word,
    ) == (0, 8)
    assert publisher_candidate.quality_chars > next(
        candidate.quality_chars
        for candidate in candidates
        if candidate.json_text_path == ("placeTerm",)
    )


def test_unmatched_list_elements_keep_a_null_geometry_slot(
    lowercase_normalizer, alignment_output
) -> None:
    aligner = alignment.TextAligner(
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_AllCandidateSelector(),
        normalizer=lowercase_normalizer,
    )
    result = aligner.align_data(
        _page("FIRST"),
        {"publisher": ["First", "Missing"]},
    )

    assert alignment_output(aligner, result)["publisher_bbox"] == [
        {"x": 0, "y": 0, "width": 9, "height": 10},
        None,
    ]


def test_polygon_output_preserves_list_shape(
    lowercase_normalizer, alignment_output
) -> None:
    aligner = alignment.TextAligner(
        output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_AllCandidateSelector(),
        normalizer=lowercase_normalizer,
    )
    result = aligner.align_data(
        _page("FIRST", "SECOND"),
        {"publisher": ["First", "Second"]},
    )
    output = alignment_output(aligner, result)

    assert output["publisher_polygon"] == [
        [[0, 0], [9, 0], [9, 10], [0, 10], [0, 0]],
        [[10, 0], [19, 0], [19, 10], [10, 10], [10, 0]],
    ]
    assert "publisher_bbox" not in output


def test_json_text_reader_supports_files_and_in_memory_data(tmp_path) -> None:
    data = {"title": "Rome"}
    memory_page = alignment.JSONTextReader().from_data(data)

    input_path = tmp_path / "custom.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")
    file_page = alignment.JSONTextReader().read(input_path)

    assert file_page.page_key == "custom"
    assert file_page.input_file_path == input_path
    assert file_page.regions == memory_page.regions
    assert file_page.json_source_data == data


def test_json_text_reader_maps_regions(label_mapper) -> None:
    region = (
        alignment.JSONTextReader(label_mapper=label_mapper)
        .from_data({"Title": "Text"})
        .regions[0]
    )

    assert (region.label, region.label_export) == ("Title", "heading")
