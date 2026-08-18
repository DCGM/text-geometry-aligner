"""Tests for the Label Studio reader."""

import json
import logging

import pytest

import text_geometry_aligner as alignment


def _rectangle(
    label: str,
    *,
    x: float = 10,
    y: float = 20,
    width: float = 30,
    height: float = 40,
    rotation: float = 0,
    image_rotation: float = 0,
) -> dict[str, object]:
    return {
        "type": "rectanglelabels",
        "value": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rotation": rotation,
            "rectanglelabels": [label],
        },
        "image_rotation": image_rotation,
        "original_width": 1000,
        "original_height": 500,
    }


def _annotation(
    results: list[dict[str, object]],
    *,
    updated_at: str | None = None,
    created_at: str | None = None,
    was_cancelled: bool = False,
) -> dict[str, object]:
    annotation: dict[str, object] = {
        "result": results,
        "was_cancelled": was_cancelled,
    }
    if updated_at is not None:
        annotation["updated_at"] = updated_at
    if created_at is not None:
        annotation["created_at"] = created_at
    return annotation


def _task(
    image: str,
    annotations: list[dict[str, object]],
    *,
    task_id: int = 1,
) -> dict[str, object]:
    return {
        "id": task_id,
        "data": {"image": image},
        "annotations": annotations,
    }


def test_from_data_converts_percentages_and_exports_grouped_json() -> None:
    document = alignment.LabelStudioReader().from_data(
        [
            _task(
                "/data/local-files/?d=images%2Fpage.one.jpg",
                [
                    _annotation(
                        [
                            _rectangle("Title"),
                            _rectangle(
                                "Title",
                                x=50,
                                y=10,
                                width=20,
                                height=10,
                            ),
                        ]
                    )
                ],
            )
        ]
    )

    assert document.alignment_mode == alignment.AlignmentMode.GEOMETRY
    assert document.input_path is None
    assert len(document.pages) == 1
    page = document.pages[0]
    assert page.page_key == "page.one"
    assert page.input_format is alignment.InputFormat.LABEL_STUDIO
    assert page.input_file_path is None
    assert page.json_source_data is None
    first, second = page.regions
    assert first.input_geometry == alignment.BoundingBox(
        x=100, y=100, width=300, height=200
    )
    assert second.input_geometry == alignment.BoundingBox(
        x=500, y=50, width=200, height=50
    )
    assert first.label == "Title"
    assert first.category_id is None
    assert first.input_geometry_confidence is None
    assert first.input_text is None
    assert first.json_geometry_path == (
        0,
        "annotations",
        0,
        "result",
        0,
        "value",
    )
    first.alto_text = "First title"

    output = alignment.AlignmentJSONWriter(
        alignment_mode=alignment.AlignmentMode.GEOMETRY,
        geometry_suffix="_bbox",
        output_geometry_format=alignment.OutputGeometryFormat.BBOX,
        output_geometry_source=alignment.OutputGeometrySource.INPUT,
    ).to_data(page)

    assert output["Title"] == ["First title", None]
    assert output["Title_bbox"] == [
        {"x": 100, "y": 100, "width": 300, "height": 200},
        {"x": 500, "y": 50, "width": 200, "height": 50},
    ]


def test_read_sets_project_path_on_document_and_pages(tmp_path) -> None:
    path = tmp_path / "project.json"
    path.write_text(
        json.dumps([_task("https://example.test/images/page.jpg", [])]),
        encoding="utf-8",
    )

    document = alignment.LabelStudioReader().read(path)

    assert document.input_path == path
    assert document.pages[0].input_file_path == path
    assert document.pages[0].page_key == "page"


def test_newest_non_cancelled_annotation_is_selected() -> None:
    annotations = [
        _annotation(
            [_rectangle("Old")],
            updated_at="2026-08-01T10:00:00Z",
        ),
        _annotation(
            [_rectangle("New")],
            updated_at="2026-08-02T10:00:00Z",
        ),
        _annotation(
            [_rectangle("Cancelled")],
            updated_at="2026-08-03T10:00:00Z",
            was_cancelled=True,
        ),
    ]

    page = alignment.LabelStudioReader().from_data(
        [_task("page.jpg", annotations)]
    ).pages[0]

    assert [region.label for region in page.regions] == ["New"]
    assert page.regions[0].json_geometry_path == (
        0,
        "annotations",
        1,
        "result",
        0,
        "value",
    )


def test_annotation_selection_falls_back_to_creation_and_array_order() -> None:
    annotations = [
        _annotation(
            [_rectangle("First")],
            created_at="2026-08-01T10:00:00Z",
        ),
        _annotation(
            [_rectangle("Second")],
            created_at="2026-08-02T10:00:00Z",
        ),
        _annotation(
            [_rectangle("Last")],
            created_at="2026-08-02T10:00:00Z",
        ),
    ]

    page = alignment.LabelStudioReader().from_data(
        [_task("page.jpg", annotations)]
    ).pages[0]

    assert [region.label for region in page.regions] == ["Last"]


def test_task_without_active_annotation_creates_empty_page() -> None:
    page = alignment.LabelStudioReader().from_data(
        [
            _task(
                "page.jpg",
                [
                    _annotation(
                        [_rectangle("Cancelled")],
                        was_cancelled=True,
                    )
                ],
            )
        ]
    ).pages[0]

    assert page.regions == []


def test_non_rectangle_results_are_ignored_and_summarized(caplog) -> None:
    with caplog.at_level(
        logging.INFO, logger="text_geometry_aligner.io_label_studio.reader"
    ):
        page = alignment.LabelStudioReader().from_data(
            [
                _task(
                    "page.jpg",
                    [
                        _annotation(
                            [
                                {"type": "textarea", "value": {}},
                                _rectangle("Title"),
                            ]
                        )
                    ],
                )
            ]
        ).pages[0]

    assert [region.label for region in page.regions] == ["Title"]
    assert (
        "Ignored 1 non-rectangle Label Studio annotation results"
        in caplog.records[-1].getMessage()
    )


def test_task_log_precedes_suspicious_geometry_warning(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="text_geometry_aligner"):
        alignment.LabelStudioReader().from_data(
            [
                _task(
                    "page.jpg",
                    [
                        _annotation(
                            [
                                _rectangle(
                                    "Tiny",
                                    x=-0.1,
                                    y=0,
                                    width=0.2,
                                    height=0.2,
                                )
                            ]
                        )
                    ],
                    task_id=87,
                )
            ]
        )

    task_log_index = next(
        index
        for index, record in enumerate(caplog.records)
        if "task index=0 task_id=87" in record.getMessage()
    )
    warning_index = next(
        index
        for index, record in enumerate(caplog.records)
        if record.levelno == logging.WARNING
    )
    assert task_log_index < warning_index
    assert (
        "json_geometry_path='$[0].annotations[0].result[0].value'"
        in caplog.records[warning_index].getMessage()
    )


@pytest.mark.parametrize(
    "rectangle",
    [
        _rectangle("Title", rotation=15),
        _rectangle("Title", image_rotation=90),
    ],
)
def test_rotated_rectangles_are_rejected(rectangle) -> None:
    with pytest.raises(
        ValueError,
        match="Rotated Label Studio rectangles are not supported",
    ):
        alignment.LabelStudioReader().from_data(
            [_task("page.jpg", [_annotation([rectangle])])]
        )


def test_duplicate_page_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate Label Studio page key"):
        alignment.LabelStudioReader().from_data(
            [
                _task("first/page.jpg", [], task_id=1),
                _task("second/page.png", [], task_id=2),
            ]
        )


def test_malformed_rectangle_reports_source_path() -> None:
    rectangle = _rectangle("Title")
    rectangle["value"]["width"] = 0  # type: ignore[index]

    with pytest.raises(
        ValueError,
        match=(
            r"Invalid Label Studio width at \$\[0\]\.annotations\[0\]"
            r"\.result\[0\]\.value"
        ),
    ):
        alignment.LabelStudioReader().from_data(
            [_task("page.jpg", [_annotation([rectangle])])]
        )


def test_label_studio_reader_maps_regions(label_mapper) -> None:
    document = alignment.LabelStudioReader(label_mapper=label_mapper).from_data(
        [
            {
                "data": {"image": "page.jpg"},
                "annotations": [
                    {
                        "result": [
                            {
                                "type": "rectanglelabels",
                                "original_width": 100,
                                "original_height": 100,
                                "value": {
                                    "x": 0,
                                    "y": 0,
                                    "width": 20,
                                    "height": 20,
                                    "rectanglelabels": ["Title"],
                                },
                            }
                        ]
                    }
                ],
            }
        ]
    )

    region = document.pages[0].regions[0]
    assert (region.label, region.label_export) == ("Title", "heading")
