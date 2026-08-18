"""Tests for the shared alignment model hierarchy."""

import dataclasses
import logging
from pathlib import Path

import pytest

import text_geometry_aligner as alignment
from text_geometry_aligner import models as models_module


def test_normal_input_geometry_does_not_warn(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=models_module.logger.name):
        alignment.AlignmentRegion(
            region_id=0,
            label="Title",
            input_geometry=alignment.BoundingBox(0, 0, 10, 10),
        )

    assert not any(
        record.levelno >= logging.WARNING for record in caplog.records
    )


def test_suspicious_geometry_warning_combines_source_details(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=models_module.logger.name):
        alignment.AlignmentRegion(
            region_id=7,
            label="Page Number",
            input_geometry=alignment.BoundingBox(-1, -2, 2, 3),
            category_id=4,
            json_geometry_path=("groups", 0, "number_bbox"),
            json_text_path=("groups", 0, "number"),
        )

    assert len(caplog.records) == 1
    warning = caplog.records[0].getMessage()
    assert "region_id=7" in warning
    assert "label='Page Number'" in warning
    assert "category_id=4" in warning
    assert "json_geometry_path='$.groups[0].number_bbox'" in warning
    assert "json_text_path='$.groups[0].number'" in warning
    assert "negative coordinates" in warning
    assert "width=2 is below 5" in warning
    assert "height=3 is below 5" in warning
    assert "area=6 is below 100" in warning


def test_polygon_area_and_non_finite_values_warn(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=models_module.logger.name):
        alignment.AlignmentRegion(
            region_id=0,
            label="Triangle",
            input_geometry=alignment.Polygon(
                ((0, 0), (10, 0), (0, 10), (0, 0))
            ),
        )
        alignment.AlignmentRegion(
            region_id=1,
            label="NonFinite",
            input_geometry=alignment.BoundingBox(
                0,
                0,
                float("inf"),
                10,
            ),
        )

    assert len(caplog.records) == 2
    assert "area=50 is below 100" in caplog.records[0].getMessage()
    assert "non-finite values: width=inf" in caplog.records[1].getMessage()


def test_input_region_is_populated_in_place() -> None:
    input_geometry = alignment.BoundingBox(10, 20, 30, 10)
    region = alignment.AlignmentRegion(
        region_id=0,
        label="PageNumber",
        input_geometry=input_geometry,
        category_id=4,
        input_geometry_confidence=0.92,
    )
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.YOLO,
        regions=[region],
        input_file_path=Path("input/page.labels"),
        alto_file_path=Path("alto/page.xml"),
        alto_page_id="page-id",
        alto_width=100,
        alto_height=200,
    )
    document = alignment.AlignmentDocument(
        alignment_mode=alignment.AlignmentMode.GEOMETRY,
        pages=[page],
        input_path=Path("input"),
        alto_path=Path("alto"),
    )

    assert region.input_text is None
    assert region.input_text_normalized is None
    assert region.text_alignment_candidate is None
    assert region.alto_text is None
    assert region.alto_text_normalized is None
    assert region.alto_geometry is None
    assert region.words is None
    assert document.unmatched_count == 1

    alignment.GeometryAligner().align_page(
        alignment.ALTOPage(
            source_path=Path("alto/page.xml"),
            words=(
                alignment.ALTOWord(
                    index=7,
                    text="12",
                    bbox=input_geometry,
                    line_index=2,
                    block_index=1,
                    element_id="word-id",
                ),
            ),
        ),
        page,
    )

    assert region.alto_text == "12"
    assert region.alto_geometry == input_geometry
    assert region.words[0].word_index == 7
    assert region.words[0].text_normalized is None
    assert region.words[0].word_coverage == 1.0
    assert region.words[0].input_geometry_coverage == 1.0
    assert region.words[0].overlap_score == 1.0
    assert region.input_geometry == input_geometry
    assert region.input_text_normalized is None
    assert region.text_alignment_candidate is None
    assert region.alto_text_normalized is None
    assert document.matched_count == 1


def test_deep_json_lists_and_dicts_reconstruct_from_region_paths(
    alignment_output,
) -> None:
    alto_page = alignment.ALTOPage(
        source_path=Path("page.xml"),
        words=tuple(
            alignment.ALTOWord(
                index=index,
                text=text,
                bbox=alignment.BoundingBox(index * 10, 0, 9, 10),
            )
            for index, text in enumerate(("A", "B", "C"))
        ),
    )
    aligner = alignment.TextAligner(
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=alignment.PassThroughCandidateSelector(),
    )
    document = aligner.align_data(
        alto_page,
        {
            "groups": [
                ["A", "B"],
                [{"name": "C"}],
            ]
        },
    )
    output = alignment_output(aligner, document)

    assert output["groups_bbox"] == [
        [
            {"x": 0, "y": 0, "width": 9, "height": 10},
            {"x": 10, "y": 0, "width": 9, "height": 10},
        ],
        [None],
    ]
    assert output["groups"][1][0]["name_bbox"] == {
        "x": 20,
        "y": 0,
        "width": 9,
        "height": 10,
    }


def test_mapper_preserves_source_label_and_sets_export_label() -> None:
    region = alignment.AlignmentRegion(
        region_id=0,
        label="Original",
        label_mapper=alignment.LabelMapper.from_data({"Original": "Exported"}),
    )

    assert region.label == "Original"
    assert region.label_export == "Exported"
    assert region.label_for_export == "Exported"
    assert "label_mapper" not in {
        field.name for field in dataclasses.fields(region)
    }


def test_unmapped_and_explicit_export_labels() -> None:
    unmapped = alignment.AlignmentRegion(
        region_id=0,
        label="Original",
        label_mapper=alignment.LabelMapper.from_data({"Other": "Out"}),
    )
    explicit = alignment.AlignmentRegion(
        region_id=1,
        label="Original",
        label_export="Exported",
    )

    assert unmapped.label_export is None
    assert unmapped.label_for_export == "Original"
    assert explicit.label_for_export == "Exported"


def test_mapper_and_explicit_export_label_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="must not both be provided"):
        alignment.AlignmentRegion(
            region_id=0,
            label="Original",
            label_export="Exported",
            label_mapper=alignment.LabelMapper.from_data(
                {"Original": "Mapped"}
            ),
        )

    with pytest.raises(ValueError, match="must not be empty"):
        alignment.AlignmentRegion(
            region_id=0,
            label="Original",
            label_export="   ",
        )


def test_suspicious_geometry_warning_has_both_labels(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=models_module.logger.name):
        alignment.AlignmentRegion(
            region_id=0,
            label="Original",
            label_mapper=alignment.LabelMapper.from_data(
                {"Original": "Exported"}
            ),
            input_geometry=alignment.BoundingBox(-1, 0, 2, 3),
        )

    message = caplog.records[0].getMessage()
    assert "label='Original'" in message
    assert "label_export='Exported'" in message
