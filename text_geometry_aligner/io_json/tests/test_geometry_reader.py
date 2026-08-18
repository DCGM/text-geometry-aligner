"""Tests for the JSON geometry reader."""

import logging
from pathlib import Path

import pytest

import text_geometry_aligner as alignment


def _bbox(
    x: float,
    y: float = 0,
    width: float = 10,
    height: float = 10,
) -> dict[str, float]:
    return {"x": x, "y": y, "width": width, "height": height}


def test_reader_logs_source_before_suspicious_region_warning(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="text_geometry_aligner"):
        alignment.JSONGeometryReader().from_data(
            {
                "groups": [
                    {
                        "title_bbox": _bbox(
                            -1,
                            width=2,
                            height=3,
                        )
                    }
                ]
            },
            page_key="page-7",
            input_file_path=Path("input/page-7.json"),
        )

    assert len(caplog.records) == 2
    assert caplog.records[0].levelno == logging.INFO
    assert (
        "Loading JSON geometry page 'page-7' from input/page-7.json"
        in caplog.records[0].getMessage()
    )
    assert caplog.records[1].levelno == logging.WARNING
    assert (
        "json_geometry_path='$.groups[0].title_bbox'"
        in caplog.records[1].getMessage()
    )


def test_extracts_nested_bbox_lists_and_retains_parallel_paths() -> None:
    regions = alignment.JSONGeometryReader().from_data(
        {
            "groups": [
                {
                    "publisher_bbox": [
                        _bbox(0),
                        None,
                        _bbox(20),
                    ]
                }
            ]
        }
    ).regions

    assert [region.json_geometry_path for region in regions] == [
        ("groups", 0, "publisher_bbox", 0),
        ("groups", 0, "publisher_bbox", 2),
    ]
    assert [region.json_text_path for region in regions] == [
        ("groups", 0, "publisher", 0),
        ("groups", 0, "publisher", 2),
    ]
    assert all(
        isinstance(region.input_geometry, alignment.BoundingBox)
        for region in regions
    )


def test_existing_destination_is_skipped_even_when_null_or_empty() -> None:
    data = {
        "nullText": None,
        "nullText_bbox": _bbox(0),
        "emptyText": "",
        "emptyText_bbox": _bbox(10),
    }

    regions = alignment.JSONGeometryReader().from_data(data).regions
    overwritten = (
        alignment.JSONGeometryReader(overwrite_existing_text=True)
        .from_data(data)
        .regions
    )

    assert regions == []
    assert len(overwritten) == 2


def test_protected_destination_skips_geometry_parsing() -> None:
    regions = alignment.JSONGeometryReader().from_data(
        {
            "title": "existing",
            "title_bbox": {"not": "a bbox"},
        }
    ).regions

    assert regions == []


def test_custom_suffix_can_extract_polygon() -> None:
    regions = alignment.JSONGeometryReader(geometry_suffix="_shape").from_data(
        {
            "title_shape": [
                [0, 0],
                [10, 0],
                [10, 10],
                [0, 10],
                [0, 0],
            ]
        }
    ).regions

    assert regions[0].json_text_path == ("title",)
    assert isinstance(regions[0].input_geometry, alignment.Polygon)


def test_invalid_geometry_reports_its_json_path() -> None:
    with pytest.raises(ValueError, match=r"\$\.title_bbox"):
        alignment.JSONGeometryReader().from_data(
            {"title_bbox": {"x": 0, "y": 0, "width": 10}}
        )


def test_json_geometry_reader_maps_regions(label_mapper) -> None:
    region = (
        alignment.JSONGeometryReader(label_mapper=label_mapper)
        .from_data({"Title_bbox": {"x": 0, "y": 0, "width": 10, "height": 10}})
        .regions[0]
    )

    assert (region.label, region.label_export) == ("Title", "heading")
