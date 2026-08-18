"""Tests for the Label Studio writer."""

import contextlib
import io
import json
import logging

import pytest

import text_geometry_aligner as alignment
from text_geometry_aligner import base_aligner, geometry_aligner, text_aligner


def test_text_alignment_export_uses_json_text_polygon_bounds_and_scores() -> None:
    page = alignment.AlignmentPage(
        page_key="page-1",
        input_format=alignment.InputFormat.JSON,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="datum vydani",
                input_text=1728,
                alto_text="1728",
                alto_geometry=alignment.Polygon(
                    ((10, 20), (50, 20), (50, 40), (10, 40), (10, 20))
                ),
                input_geometry_confidence=0.6,
                alignment_score=0.8,
            ),
            alignment.AlignmentRegion(
                region_id=1,
                label="titulek",
                input_text="Žluťoučký kůň",
                alto_text="ŽLUŤOUČKÝ KŮŇ",
                alto_geometry=alignment.BoundingBox(100, 10, 80, 20),
                alignment_score=0.7,
            ),
        ],
        alto_width=200,
        alto_height=100,
    )
    writer = alignment.LabelStudioWriter(
        alignment_mode=alignment.AlignmentMode.TEXT,
        image_prefix="/data/local-files/?d=books/images/",
        output_text_source=alignment.OutputTextSource.JSON,
        output_geometry_source=alignment.OutputGeometrySource.ALTO,
    )

    output = writer.to_data(page)

    assert (
        output["data"]["image"]
        == "/data/local-files/?d=books/images/page-1.jpg"
    )
    prediction = output["predictions"][0]
    assert prediction["score"] == 0.6
    first, second = prediction["result"]
    assert first == {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": 5,
            "y": 20,
            "width": 20,
            "height": 20,
            "rectanglelabels": ["datum vydani"],
        },
        "meta": {
            "text": ["1728"],
            "input_geometry_confidence": 0.6,
            "alignment_score": 0.8,
        },
        "score": 0.6,
    }
    assert second["meta"]["text"] == ["Žluťoučký kůň"]
    assert "input_geometry_confidence" not in second["meta"]
    assert second["score"] == 0.7


def test_text_alignment_can_export_alto_text() -> None:
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.JSON,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="Title",
                input_text="Input",
                alto_text="ALTO",
                alto_geometry=alignment.BoundingBox(0, 0, 10, 10),
            )
        ],
        alto_width=100,
        alto_height=100,
    )

    output = alignment.LabelStudioWriter(
        alignment_mode=alignment.AlignmentMode.TEXT,
        image_prefix="images",
        output_text_source=alignment.OutputTextSource.ALTO,
    ).to_data(page)

    result = output["predictions"][0]["result"][0]
    assert result["meta"] == {"text": ["ALTO"]}
    assert "score" not in result
    assert "score" not in output["predictions"][0]


def test_geometry_alignment_uses_alto_text_and_selected_geometry() -> None:
    region = alignment.AlignmentRegion(
        region_id=0,
        label="Title",
        input_text="ignored input",
        input_geometry=alignment.BoundingBox(-10, 10, 20, 30),
        alto_text="Matched text",
        alto_geometry=alignment.BoundingBox(30, 20, 40, 10),
    )
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.YOLO,
        regions=[region],
        alto_width=200,
        alto_height=100,
    )

    input_output = alignment.LabelStudioWriter(
        alignment_mode=alignment.AlignmentMode.GEOMETRY,
        image_prefix="images",
        output_text_source=alignment.OutputTextSource.JSON,
        output_geometry_source=alignment.OutputGeometrySource.INPUT,
    ).to_data(page)
    alto_output = alignment.LabelStudioWriter(
        alignment_mode=alignment.AlignmentMode.GEOMETRY,
        image_prefix="images",
        output_text_source=alignment.OutputTextSource.JSON,
        output_geometry_source=alignment.OutputGeometrySource.ALTO,
    ).to_data(page)

    input_result = input_output["predictions"][0]["result"][0]
    alto_result = alto_output["predictions"][0]["result"][0]
    assert input_result["meta"]["text"] == ["Matched text"]
    assert input_result["value"] == {
        "x": -5,
        "y": 10,
        "width": 10,
        "height": 30,
        "rectanglelabels": ["Title"],
    }
    assert alto_result["value"] == {
        "x": 15,
        "y": 20,
        "width": 20,
        "height": 10,
        "rectanglelabels": ["Title"],
    }


def test_incomplete_regions_are_skipped_and_logged(caplog) -> None:
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.JSON,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="No geometry",
                input_text="Text",
            ),
            alignment.AlignmentRegion(
                region_id=1,
                label="No text",
                alto_geometry=alignment.BoundingBox(0, 0, 10, 10),
            ),
        ],
        alto_width=100,
        alto_height=100,
    )

    with caplog.at_level(
        logging.INFO, logger="text_geometry_aligner.io_label_studio.writer"
    ):
        output = alignment.LabelStudioWriter(
            alignment_mode=alignment.AlignmentMode.TEXT,
            image_prefix="images",
        ).to_data(page)

    assert output["predictions"] == [{"result": []}]
    assert "Skipped 2 incomplete" in caplog.records[0].getMessage()


@pytest.mark.parametrize("width, height", [(None, 100), (0, 100), (100, float("inf"))])
def test_invalid_dimensions_and_geometry_are_rejected(width, height) -> None:
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.JSON,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="Title",
                input_text="Text",
                alto_geometry=alignment.BoundingBox(0, 0, 10, 10),
            )
        ],
        alto_width=width,
        alto_height=height,
    )
    with pytest.raises(ValueError, match="ALTO dimensions"):
        alignment.LabelStudioWriter(
            alignment_mode=alignment.AlignmentMode.TEXT,
            image_prefix="images",
        ).to_data(page)

    invalid_page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.JSON,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="Title",
                input_text="Text",
                alto_geometry=alignment.BoundingBox(0, 0, 0, 10),
            )
        ],
        alto_width=100,
        alto_height=100,
    )
    with pytest.raises(ValueError, match="positive finite geometry"):
        alignment.LabelStudioWriter(
            alignment_mode=alignment.AlignmentMode.TEXT,
            image_prefix="images",
        ).to_data(invalid_page)


def test_write_is_utf8_and_atomic(tmp_path) -> None:
    page = alignment.AlignmentPage(
        page_key="stránka",
        input_format=alignment.InputFormat.JSON,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="název",
                input_text="Příliš žluťoučký kůň",
                alto_geometry=alignment.BoundingBox(0, 0, 10, 10),
            )
        ],
        alto_width=100,
        alto_height=100,
    )
    writer = alignment.LabelStudioWriter(
        alignment_mode=alignment.AlignmentMode.TEXT,
        image_prefix="images",
    )

    path = tmp_path / "nested" / "page.json"
    writer.write(page, path)
    output = json.loads(path.read_text(encoding="utf-8"))
    temporary_path = path.with_name(f".{path.name}.tmp")

    assert not temporary_path.exists()
    assert output["predictions"][0]["result"][0]["meta"]["text"] == [
        "Příliš žluťoučký kůň"
    ]


@pytest.mark.parametrize(
    "build_parser",
    [text_aligner.build_argument_parser, geometry_aligner.build_argument_parser],
)
def test_cli_exposes_format_and_requires_label_studio_prefix(build_parser) -> None:
    required = [
        "--alto-dir",
        "alto",
        "--input-dir",
        "input",
        "--json-output-dir",
        "output",
    ]
    parser = build_parser()
    defaults = parser.parse_args(required)
    assert defaults.output_json_format == "package"
    assert defaults.label_studio_image_prefix is None
    base_aligner.validate_common_cli_arguments(parser, defaults)

    missing_prefix = parser.parse_args(
        [*required, "--output-json-format", "label-studio"]
    )
    with (
        pytest.raises(SystemExit),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        base_aligner.validate_common_cli_arguments(
            parser,
            missing_prefix,
        )

    configured = parser.parse_args(
        [
            *required,
            "--output-json-format",
            "label-studio",
            "--label-studio-image-prefix",
            "/data/local-files/?d=books/images/",
        ]
    )
    base_aligner.validate_common_cli_arguments(
        parser,
        configured,
    )


def test_label_studio_writer_uses_export_label() -> None:
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.YOLO,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="Title",
                label_export="heading",
                input_geometry=alignment.BoundingBox(0, 0, 10, 10),
                alto_text="Text",
            )
        ],
        alto_width=100,
        alto_height=100,
    )

    output = alignment.LabelStudioWriter(
        alignment_mode=alignment.AlignmentMode.GEOMETRY,
        image_prefix="images",
    ).to_data(page)

    labels = output["predictions"][0]["result"][0]["value"]["rectanglelabels"]
    assert labels == ["heading"]
