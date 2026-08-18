"""Tests for label mapping."""

import json

import pytest

import text_geometry_aligner as alignment
from text_geometry_aligner import geometry_aligner, text_aligner


def test_mapping_is_exact_single_step_and_allows_many_to_one() -> None:
    mapper = alignment.LabelMapper.from_data(
        {
            "Title": "heading",
            "Subtitle": "heading",
            "heading": "renamed-again",
        }
    )

    assert mapper.map("Title") == "heading"
    assert mapper.map("Subtitle") == "heading"
    assert mapper.map("heading") == "renamed-again"
    assert mapper.map("title") is None
    assert mapper.map("Unknown") is None


@pytest.mark.parametrize(
    "raw_data, exception, message",
    [
        ('["Title"]', TypeError, "root must be an object"),
        ('{"Title": 2}', TypeError, "names must be strings"),
        ('{"Title": "   "}', ValueError, "must not be empty"),
        (
            '{"Title": "one", "Title": "two"}',
            ValueError,
            "Duplicate class mapping key",
        ),
    ],
)
def test_mapping_file_is_utf8_and_validated(
    tmp_path, raw_data, exception, message
) -> None:
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps({"Místo vydání": "publication_place"}),
        encoding="utf-8",
    )
    mapper = alignment.LabelMapper.from_file(path)

    assert mapper.map("Místo vydání") == "publication_place"

    path.write_text(raw_data, encoding="utf-8")
    with pytest.raises(exception, match=message):
        alignment.LabelMapper.from_file(path)


@pytest.mark.parametrize(
    "build_parser",
    [text_aligner.build_argument_parser, geometry_aligner.build_argument_parser],
)
def test_cli_parsers_accept_mapping_file(build_parser) -> None:
    parser = build_parser()
    assert parser.get_default("class_mapping_file") is None
    namespace = parser.parse_args(
        [
            "--alto-dir",
            "alto",
            "--input-dir",
            "input",
            "--json-output-dir",
            "output",
            "--class-mapping-file",
            "mapping.json",
        ]
    )
    assert namespace.class_mapping_file == "mapping.json"
