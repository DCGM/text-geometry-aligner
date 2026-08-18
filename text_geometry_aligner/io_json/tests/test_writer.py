"""Tests for the alignment JSON writer."""

import json

import pytest

import text_geometry_aligner as alignment


def _bbox(
    x: float,
    y: float = 0,
    width: float = 10,
    height: float = 10,
) -> dict[str, float]:
    return {"x": x, "y": y, "width": width, "height": height}


def _page(
    data,
    *,
    overwrite_existing_text: bool = False,
) -> alignment.AlignmentPage:
    reader = alignment.JSONGeometryReader(
        overwrite_existing_text=overwrite_existing_text
    )
    return reader.from_data(
        data,
        page_key="page",
    )


def _export(page: alignment.AlignmentPage):
    return alignment.AlignmentJSONWriter(
        alignment_mode=alignment.AlignmentMode.GEOMETRY,
        geometry_suffix="_bbox",
        output_geometry_format=alignment.OutputGeometryFormat.BBOX,
    ).to_data(page)


def test_missing_list_destination_mirrors_trailing_null_shape() -> None:
    page = _page(
        {"publisher_bbox": [_bbox(0), None]},
    )
    page.regions[0].alto_text = "FIRST"
    output = _export(page)
    assert output["publisher"] == ["FIRST", None]


def test_incompatible_existing_container_is_not_replaced() -> None:
    page = _page(
        {"publisher": "existing", "publisher_bbox": [_bbox(0)]},
        overwrite_existing_text=True,
    )
    page.regions[0].alto_text = "FIRST"
    with pytest.raises(TypeError, match="Incompatible container"):
        _export(page)


def test_page_extraction_preserves_an_existing_destination() -> None:
    page = _page({"title": None, "title_bbox": _bbox(0)})
    output = _export(page)
    assert page.regions == []
    assert output["title"] is None


def test_reconstructs_nested_output_from_retained_paths() -> None:
    input_data = {"groups": [{"names": ["Original"]}]}
    page = alignment.JSONTextReader().from_data(
        input_data,
        page_key="page",
    )
    region = page.regions[0]
    region.alto_text = "ALTO"
    region.alto_geometry = alignment.BoundingBox(1, 2, 3, 4)
    output_data = alignment.AlignmentJSONWriter(
        alignment_mode=alignment.AlignmentMode.TEXT,
        geometry_suffix="_bbox",
        output_geometry_format=alignment.OutputGeometryFormat.BBOX,
        output_text_source=alignment.OutputTextSource.ALTO,
        output_geometry_source=alignment.OutputGeometrySource.ALTO,
    ).to_data(
        page,
    )

    assert input_data == {"groups": [{"names": ["Original"]}]}
    assert output_data == {
        "groups": [
            {
                "names": ["ALTO"],
                "names_bbox": [{"x": 1, "y": 2, "width": 3, "height": 4}],
            }
        ]
    }


def test_alignment_json_writer_writes_utf8_atomically(tmp_path) -> None:
    data = {"title": "Řím", "publishers": ["Šolc"]}
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.JSON,
        regions=[],
        json_source_data=data,
    )
    writer = alignment.AlignmentJSONWriter(
        alignment_mode=alignment.AlignmentMode.TEXT,
        geometry_suffix="_bbox",
        output_geometry_format=alignment.OutputGeometryFormat.BBOX,
    )

    output_path = tmp_path / "nested" / "document.json"
    writer.write(page, output_path)

    assert json.loads(output_path.read_text()) == data
    assert writer.to_data(page) == data
    output_text = output_path.read_text(encoding="utf-8")
    assert "Řím" in output_text
    assert output_text.endswith("\n")
    assert not output_path.with_name(f".{output_path.name}.tmp").exists()


def test_grouped_json_merges_source_labels_mapped_to_one_export_label() -> None:
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.YOLO,
        regions=[
            alignment.AlignmentRegion(
                region_id=index,
                label=label,
                label_export="heading",
                input_geometry=alignment.BoundingBox(index * 20, 0, 10, 10),
                alto_text=text,
                category_id=index,
            )
            for index, (label, text) in enumerate(
                (("Title", "First"), ("Subtitle", "Second"))
            )
        ],
    )

    output = alignment.AlignmentJSONWriter(
        alignment_mode=alignment.AlignmentMode.GEOMETRY,
        geometry_suffix="_bbox",
        output_geometry_format=alignment.OutputGeometryFormat.BBOX,
        output_geometry_source=alignment.OutputGeometrySource.INPUT,
    ).to_data(page)

    assert output["heading"] == ["First", "Second"]
    assert len(output["heading_bbox"]) == 2
    assert "Title" not in output
    assert "Subtitle" not in output


def test_source_json_keys_remain_original() -> None:
    page = alignment.JSONTextReader(
        label_mapper=alignment.LabelMapper.from_data({"Title": "heading"})
    ).from_data({"Title": "Original text"})
    page.regions[0].alto_geometry = alignment.BoundingBox(0, 0, 10, 10)

    output = alignment.AlignmentJSONWriter(
        alignment_mode=alignment.AlignmentMode.TEXT,
        geometry_suffix="_bbox",
        output_geometry_format=alignment.OutputGeometryFormat.BBOX,
        output_geometry_source=alignment.OutputGeometrySource.ALTO,
    ).to_data(page)

    assert "Title" in output
    assert "Title_bbox" in output
    assert "heading" not in output


def test_writer_groups_repeated_labels_and_retains_input_boxes() -> None:
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.YOLO,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="PageNumber",
                input_geometry=alignment.BoundingBox(1, 2, 3, 4),
                alto_text="12",
                category_id=0,
                input_geometry_confidence=0.9,
            ),
            alignment.AlignmentRegion(
                region_id=1,
                label="PageNumber",
                input_geometry=alignment.BoundingBox(5, 6, 7, 8),
                alto_text=None,
                category_id=0,
                input_geometry_confidence=0.8,
            ),
        ],
    )
    output = alignment.AlignmentJSONWriter(
        alignment_mode=alignment.AlignmentMode.GEOMETRY,
        geometry_suffix="_bbox",
        output_geometry_format=alignment.OutputGeometryFormat.BBOX,
        output_geometry_source=alignment.OutputGeometrySource.INPUT,
    ).to_data(page)

    assert output["PageNumber"] == ["12", None]
    assert output["PageNumber_bbox"] == [
        {"x": 1, "y": 2, "width": 3, "height": 4},
        {"x": 5, "y": 6, "width": 7, "height": 8},
    ]
    assert page.regions[0].category_id == 0


def test_writer_rejects_labels_that_collide_with_suffix_keys() -> None:
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.YOLO,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="Title",
                input_geometry=alignment.BoundingBox(0, 0, 1, 1),
            ),
            alignment.AlignmentRegion(
                region_id=1,
                label="Title_bbox",
                input_geometry=alignment.BoundingBox(1, 0, 1, 1),
            ),
        ],
    )
    writer = alignment.AlignmentJSONWriter(
        alignment_mode=alignment.AlignmentMode.GEOMETRY,
        geometry_suffix="_bbox",
        output_geometry_format=alignment.OutputGeometryFormat.BBOX,
    )

    with pytest.raises(ValueError, match="collide"):
        writer.to_data(page)
