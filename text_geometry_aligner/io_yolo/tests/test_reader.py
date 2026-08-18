"""Tests for the YOLO reader."""

import logging

import pytest

import text_geometry_aligner as alignment


def _detection(
    label: str,
    confidence: float,
    *,
    center_x: float = 50,
    center_y: float = 50,
    width: float = 40,
    height: float = 20,
) -> alignment.YOLODetection:
    return alignment.YOLODetection(
        category_id=0,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
        confidence=confidence,
        class_name=label,
    )


def test_reader_logs_page_before_suspicious_region_warning(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="text_geometry_aligner"):
        alignment.YOLOReader().from_data(
            [
                alignment.YOLODetection(
                    category_id=3,
                    center_x=0,
                    center_y=0,
                    width=2,
                    height=3,
                    confidence=0.875,
                    class_name="Page Number",
                )
            ],
            page_key="memory-page",
        )

    assert len(caplog.records) == 2
    assert caplog.records[0].levelno == logging.INFO
    assert (
        "Loading YOLO geometry page 'memory-page' from in-memory data"
        in caplog.records[0].getMessage()
    )
    assert caplog.records[1].levelno == logging.WARNING
    assert "label='Page Number'" in caplog.records[1].getMessage()
    assert "category_id=3" in caplog.records[1].getMessage()


def test_reader_accepts_in_memory_detections() -> None:
    page = alignment.YOLOReader().from_data(
        [
            alignment.YOLODetection(
                category_id=3,
                center_x=25,
                center_y=30,
                width=30,
                height=10,
                confidence=0.875,
                class_name="Page Number",
            )
        ],
        page_key="memory-page",
    )

    assert page.page_key == "memory-page"
    assert page.input_file_path is None
    assert page.regions[0].label == "Page Number"
    assert page.regions[0].input_geometry == alignment.BoundingBox(
        10, 25, 30, 10
    )


def test_reader_preserves_detection_metadata(tmp_path, caplog) -> None:
    path = tmp_path / "page.detections"
    path.write_text(
        "3 25 30 30 10 0.875 Page Number\n",
        encoding="utf-8",
    )
    with caplog.at_level(
        logging.INFO, logger="text_geometry_aligner.io_yolo.reader"
    ):
        page = alignment.YOLOReader().read(
            path,
            page_key="page",
        )

    region = page.regions[0]
    assert (
        f"Loading YOLO geometry page 'page' from {path}"
        in caplog.records[0].getMessage()
    )
    assert page.input_format is alignment.InputFormat.YOLO
    assert page.json_source_data is None
    assert region.label == "Page Number"
    assert region.category_id == 3
    assert region.input_geometry_confidence == 0.875
    assert region.input_geometry == alignment.BoundingBox(10, 25, 30, 10)


def test_reader_rejects_inconsistent_id_name_mapping(tmp_path) -> None:
    path = tmp_path / "page"
    path.write_text(
        "0 5 5 10 10 0.9 First\n" "0 5 5 10 10 0.8 Second\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Inconsistent YOLO class mapping"):
        alignment.YOLOReader().read(path)


def test_reader_optionally_deduplicates_cross_class_regions() -> None:
    reader = alignment.YOLOReader(
        label_deduplication_groups=[
            alignment.LabelDeduplicationGroup(
                labels=frozenset({"Chapter", "Subchapter"}),
                minimum_coverage=0.8,
            )
        ]
    )

    page = reader.from_data(
        [
            _detection("Chapter", 0.7),
            _detection("Subchapter", 0.9),
            _detection("Other", 0.5),
        ]
    )

    assert [(region.region_id, region.label) for region in page.regions] == [
        (1, "Subchapter"),
        (2, "Other"),
    ]


def test_reader_deduplication_preserves_same_class_regions() -> None:
    reader = alignment.YOLOReader(
        label_deduplication_groups=[
            alignment.LabelDeduplicationGroup(
                labels=frozenset({"Chapter", "Subchapter"}),
                minimum_coverage=0.8,
            )
        ]
    )

    page = reader.from_data(
        [
            _detection("Chapter", 0.7),
            _detection("Chapter", 0.9),
        ]
    )

    assert [region.region_id for region in page.regions] == [0, 1]


def test_reader_deduplication_requires_coverage_of_larger_box() -> None:
    reader = alignment.YOLOReader(
        label_deduplication_groups=[
            alignment.LabelDeduplicationGroup(
                labels=frozenset({"Chapter", "Subchapter"}),
                minimum_coverage=0.8,
            )
        ]
    )

    page = reader.from_data(
        [
            _detection("Chapter", 0.9),
            _detection(
                "Subchapter",
                0.8,
                width=10,
                height=10,
            ),
        ]
    )

    assert [region.region_id for region in page.regions] == [0, 1]


def test_reader_deduplication_tie_retains_first_detection() -> None:
    reader = alignment.YOLOReader(
        label_deduplication_groups=[
            alignment.LabelDeduplicationGroup(
                labels=frozenset({"Chapter", "Subchapter"}),
                minimum_coverage=1.0,
            )
        ]
    )

    page = reader.from_data(
        [
            _detection("Chapter", 0.9),
            _detection("Subchapter", 0.9),
        ]
    )

    assert [(region.region_id, region.label) for region in page.regions] == [
        (0, "Chapter")
    ]


def test_reader_rejects_label_in_multiple_deduplication_groups() -> None:
    with pytest.raises(
        ValueError, match="cannot belong to multiple deduplication groups"
    ):
        alignment.YOLOReader(
            label_deduplication_groups=[
                alignment.LabelDeduplicationGroup(
                    labels=frozenset({"Chapter", "Subchapter"}),
                    minimum_coverage=0.8,
                ),
                alignment.LabelDeduplicationGroup(
                    labels=frozenset({"Subchapter", "Other"}),
                    minimum_coverage=0.7,
                ),
            ]
        )


def test_yolo_reader_maps_regions(label_mapper) -> None:
    region = alignment.YOLOReader(label_mapper=label_mapper).from_data(
        [
            alignment.YOLODetection(
                category_id=1,
                center_x=10,
                center_y=10,
                width=10,
                height=10,
                confidence=0.8,
                class_name="YOLO title",
            )
        ]
    ).regions[0]

    assert (region.label, region.label_export) == ("YOLO title", "heading")
