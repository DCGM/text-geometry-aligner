"""Tests for Label Studio rectangle input."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path

import text_geometry_aligner as alignment
from text_geometry_aligner import base_aligner
from text_geometry_aligner import geometry_aligner
from text_geometry_aligner import text_aligner


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


class LabelStudioReaderTests(unittest.TestCase):
    def test_from_data_converts_percentages_and_exports_grouped_json(
        self,
    ) -> None:
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

        self.assertEqual(document.alignment_mode, alignment.AlignmentMode.GEOMETRY)
        self.assertIsNone(document.input_path)
        self.assertEqual(len(document.pages), 1)
        page = document.pages[0]
        self.assertEqual(page.page_key, "page.one")
        self.assertIs(page.input_format, alignment.InputFormat.LABEL_STUDIO)
        self.assertIsNone(page.input_file_path)
        self.assertIsNone(page.json_source_data)
        first, second = page.regions
        self.assertEqual(
            first.input_geometry,
            alignment.BoundingBox(x=100, y=100, width=300, height=200),
        )
        self.assertEqual(
            second.input_geometry,
            alignment.BoundingBox(x=500, y=50, width=200, height=50),
        )
        self.assertEqual(first.label, "Title")
        self.assertIsNone(first.category_id)
        self.assertIsNone(first.input_geometry_confidence)
        self.assertIsNone(first.input_text)
        self.assertEqual(
            first.json_geometry_path,
            (0, "annotations", 0, "result", 0, "value"),
        )
        first.alto_text = "First title"

        output = alignment.AlignmentJSONWriter(
            alignment_mode=alignment.AlignmentMode.GEOMETRY,
            geometry_suffix="_bbox",
            output_geometry_format=alignment.OutputGeometryFormat.BBOX,
            output_geometry_source=alignment.OutputGeometrySource.INPUT,
        ).to_data(page)

        self.assertEqual(output["Title"], ["First title", None])
        self.assertEqual(
            output["Title_bbox"],
            [
                {"x": 100, "y": 100, "width": 300, "height": 200},
                {"x": 500, "y": 50, "width": 200, "height": 50},
            ],
        )

    def test_read_sets_project_path_on_document_and_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "project.json"
            path.write_text(
                json.dumps([_task("https://example.test/images/page.jpg", [])]),
                encoding="utf-8",
            )

            document = alignment.LabelStudioReader().read(path)

        self.assertEqual(document.input_path, path)
        self.assertEqual(document.pages[0].input_file_path, path)
        self.assertEqual(document.pages[0].page_key, "page")

    def test_newest_non_cancelled_annotation_is_selected(self) -> None:
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

        self.assertEqual([region.label for region in page.regions], ["New"])
        self.assertEqual(
            page.regions[0].json_geometry_path,
            (0, "annotations", 1, "result", 0, "value"),
        )

    def test_annotation_selection_falls_back_to_creation_and_array_order(
        self,
    ) -> None:
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

        self.assertEqual([region.label for region in page.regions], ["Last"])

    def test_task_without_active_annotation_creates_empty_page(self) -> None:
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

        self.assertEqual(page.regions, [])

    def test_non_rectangle_results_are_ignored_and_summarized(self) -> None:
        with self.assertLogs(
            "text_geometry_aligner.io_label_studio.reader",
            logging.INFO,
        ) as captured:
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

        self.assertEqual([region.label for region in page.regions], ["Title"])
        self.assertIn(
            "Ignored 1 non-rectangle Label Studio annotation results",
            captured.records[-1].getMessage(),
        )

    def test_task_log_precedes_suspicious_geometry_warning(self) -> None:
        with self.assertLogs("text_geometry_aligner", logging.INFO) as captured:
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
            for index, record in enumerate(captured.records)
            if "task index=0 task_id=87" in record.getMessage()
        )
        warning_index = next(
            index
            for index, record in enumerate(captured.records)
            if record.levelno == logging.WARNING
        )
        self.assertLess(task_log_index, warning_index)
        self.assertIn(
            "json_geometry_path='$[0].annotations[0].result[0].value'",
            captured.records[warning_index].getMessage(),
        )

    def test_rotated_rectangles_are_rejected(self) -> None:
        for rectangle in (
            _rectangle("Title", rotation=15),
            _rectangle("Title", image_rotation=90),
        ):
            with self.subTest(rectangle=rectangle):
                with self.assertRaisesRegex(
                    ValueError,
                    "Rotated Label Studio rectangles are not supported",
                ):
                    alignment.LabelStudioReader().from_data(
                        [_task("page.jpg", [_annotation([rectangle])])]
                    )

    def test_duplicate_page_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate Label Studio page key"):
            alignment.LabelStudioReader().from_data(
                [
                    _task("first/page.jpg", [], task_id=1),
                    _task("second/page.png", [], task_id=2),
                ]
            )

    def test_malformed_rectangle_reports_source_path(self) -> None:
        rectangle = _rectangle("Title")
        rectangle["value"]["width"] = 0  # type: ignore[index]

        with self.assertRaisesRegex(
            ValueError,
            r"Invalid Label Studio width at \$\[0\]\.annotations\[0\]"
            r"\.result\[0\]\.value",
        ):
            alignment.LabelStudioReader().from_data(
                [_task("page.jpg", [_annotation([rectangle])])]
            )


class LabelStudioWriterTests(unittest.TestCase):
    def test_text_alignment_export_uses_json_text_polygon_bounds_and_scores(
        self,
    ) -> None:
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

        self.assertEqual(
            output["data"]["image"],
            "/data/local-files/?d=books/images/page-1.jpg",
        )
        prediction = output["predictions"][0]
        self.assertEqual(prediction["score"], 0.6)
        first, second = prediction["result"]
        self.assertEqual(
            first,
            {
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
            },
        )
        self.assertEqual(second["meta"]["text"], ["Žluťoučký kůň"])
        self.assertNotIn("input_geometry_confidence", second["meta"])
        self.assertEqual(second["score"], 0.7)

    def test_text_alignment_can_export_alto_text(self) -> None:
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
        self.assertEqual(result["meta"], {"text": ["ALTO"]})
        self.assertNotIn("score", result)
        self.assertNotIn("score", output["predictions"][0])

    def test_geometry_alignment_uses_alto_text_and_selected_geometry(
        self,
    ) -> None:
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
        self.assertEqual(input_result["meta"]["text"], ["Matched text"])
        self.assertEqual(
            input_result["value"],
            {
                "x": -5,
                "y": 10,
                "width": 10,
                "height": 30,
                "rectanglelabels": ["Title"],
            },
        )
        self.assertEqual(
            alto_result["value"],
            {
                "x": 15,
                "y": 20,
                "width": 20,
                "height": 10,
                "rectanglelabels": ["Title"],
            },
        )

    def test_incomplete_regions_are_skipped_and_logged(self) -> None:
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

        with self.assertLogs(
            "text_geometry_aligner.io_label_studio.writer",
            logging.INFO,
        ) as captured:
            output = alignment.LabelStudioWriter(
                alignment_mode=alignment.AlignmentMode.TEXT,
                image_prefix="images",
            ).to_data(page)

        self.assertEqual(output["predictions"], [{"result": []}])
        self.assertIn("Skipped 2 incomplete", captured.records[0].getMessage())

    def test_invalid_dimensions_and_geometry_are_rejected(self) -> None:
        for width, height in ((None, 100), (0, 100), (100, float("inf"))):
            with self.subTest(width=width, height=height):
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
                with self.assertRaisesRegex(ValueError, "ALTO dimensions"):
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
        with self.assertRaisesRegex(ValueError, "positive finite geometry"):
            alignment.LabelStudioWriter(
                alignment_mode=alignment.AlignmentMode.TEXT,
                image_prefix="images",
            ).to_data(invalid_page)

    def test_write_is_utf8_and_atomic(self) -> None:
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

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "page.json"
            writer.write(page, path)
            output = json.loads(path.read_text(encoding="utf-8"))
            temporary_path = path.with_name(f".{path.name}.tmp")

        self.assertFalse(temporary_path.exists())
        self.assertEqual(
            output["predictions"][0]["result"][0]["meta"]["text"],
            ["Příliš žluťoučký kůň"],
        )

    def test_cli_exposes_format_and_requires_label_studio_prefix(self) -> None:
        required = [
            "--alto-dir",
            "alto",
            "--input-dir",
            "input",
            "--json-output-dir",
            "output",
        ]
        for parser in (
            text_aligner.build_argument_parser(),
            geometry_aligner.build_argument_parser(),
        ):
            with self.subTest(prog=parser.prog):
                defaults = parser.parse_args(required)
                self.assertEqual(defaults.output_json_format, "package")
                self.assertIsNone(defaults.label_studio_image_prefix)
                base_aligner.validate_common_cli_arguments(parser, defaults)

                missing_prefix = parser.parse_args(
                    [*required, "--output-json-format", "label-studio"]
                )
                with (
                    self.assertRaises(SystemExit),
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


if __name__ == "__main__":
    unittest.main()
