"""Tests for the Pillow alignment renderer."""

from unittest import mock

import text_geometry_aligner as alignment


def test_polygon_points_are_scaled_and_clamped() -> None:
    polygon = alignment.Polygon(((-1, 1), (10, 1), (10, 20), (-1, 1)))

    scaled = alignment.PillowAlignmentRenderer._scaled_polygon(
        polygon,
        scale_x=2,
        scale_y=3,
        image_width=15,
        image_height=30,
    )

    assert scaled == [(0, 3), (14, 3), (14, 29), (0, 3)]


def test_renderer_draws_polygon_outline_instead_of_rectangle() -> None:
    renderer = alignment.PillowAlignmentRenderer(line_width=4)
    draw = mock.Mock()
    polygon = alignment.Polygon(((0, 0), (10, 0), (10, 10), (0, 0)))

    renderer._draw_geometry_outline(
        draw=draw,
        geometry=polygon,
        color=(1, 2, 3),
        scale_x=1,
        scale_y=1,
        image_width=20,
        image_height=20,
    )

    draw.line.assert_called_once_with(
        [(0, 0), (10, 0), (10, 10), (0, 0)],
        fill=(1, 2, 3),
        width=4,
    )
    draw.rectangle.assert_not_called()


def test_renderer_keeps_rectangle_path_for_bbox() -> None:
    renderer = alignment.PillowAlignmentRenderer(line_width=2)
    draw = mock.Mock()
    bbox = alignment.BoundingBox(1, 2, 3, 4)

    renderer._draw_geometry_outline(
        draw=draw,
        geometry=bbox,
        color=(3, 2, 1),
        scale_x=1,
        scale_y=1,
        image_width=20,
        image_height=20,
    )

    draw.rectangle.assert_called_once_with(
        (1, 2, 4, 6),
        outline=(3, 2, 1),
        width=2,
    )
    draw.line.assert_not_called()


def test_rendering_converts_selected_geometry_to_requested_format() -> None:
    page = alignment.AlignmentPage(
        page_key="page",
        input_format=alignment.InputFormat.YOLO,
        regions=[
            alignment.AlignmentRegion(
                region_id=0,
                label="Title",
                input_geometry=alignment.BoundingBox(1, 2, 3, 4),
            )
        ],
    )

    rendered = alignment.PillowAlignmentRenderer._render_alignments(
        page,
        alignment.OutputTextSource.ALTO,
        alignment.OutputGeometrySource.INPUT,
        alignment.OutputGeometryFormat.POLYGON,
    )

    assert isinstance(rendered[0].geometry, alignment.Polygon)
    assert rendered[0].geometry.points == (
        (1, 2),
        (4, 2),
        (4, 6),
        (1, 6),
        (1, 2),
    )
