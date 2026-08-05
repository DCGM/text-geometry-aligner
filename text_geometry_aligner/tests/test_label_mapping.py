"""Tests for source-label preservation and export-label mapping."""

from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
import unittest
from pathlib import Path

import text_geometry_aligner as alignment
from text_geometry_aligner import geometry_aligner, text_aligner
from text_geometry_aligner import models as models_module


class LabelMapperTests(unittest.TestCase):
    def test_mapping_is_exact_single_step_and_allows_many_to_one(self) -> None:
        mapper = alignment.LabelMapper.from_data(
            {
                "Title": "heading",
                "Subtitle": "heading",
                "heading": "renamed-again",
            }
        )

        self.assertEqual(mapper.map("Title"), "heading")
        self.assertEqual(mapper.map("Subtitle"), "heading")
        self.assertEqual(mapper.map("heading"), "renamed-again")
        self.assertIsNone(mapper.map("title"))
        self.assertIsNone(mapper.map("Unknown"))

    def test_mapping_file_is_utf8_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mapping.json"
            path.write_text(
                json.dumps({"Místo vydání": "publication_place"}),
                encoding="utf-8",
            )
            mapper = alignment.LabelMapper.from_file(path)

            self.assertEqual(
                mapper.map("Místo vydání"),
                "publication_place",
            )

            invalid_cases = (
                ('["Title"]', TypeError, "root must be an object"),
                ('{"Title": 2}', TypeError, "names must be strings"),
                ('{"Title": "   "}', ValueError, "must not be empty"),
                (
                    '{"Title": "one", "Title": "two"}',
                    ValueError,
                    "Duplicate class mapping key",
                ),
            )
            for raw_data, exception, message in invalid_cases:
                with self.subTest(raw_data=raw_data):
                    path.write_text(raw_data, encoding="utf-8")
                    with self.assertRaisesRegex(exception, message):
                        alignment.LabelMapper.from_file(path)


class AlignmentRegionLabelTests(unittest.TestCase):
    def test_mapper_preserves_source_label_and_sets_export_label(self) -> None:
        region = alignment.AlignmentRegion(
            region_id=0,
            label="Original",
            label_mapper=alignment.LabelMapper.from_data(
                {"Original": "Exported"}
            ),
        )

        self.assertEqual(region.label, "Original")
        self.assertEqual(region.label_export, "Exported")
        self.assertEqual(region.label_for_export, "Exported")
        self.assertNotIn(
            "label_mapper",
            {field.name for field in dataclasses.fields(region)},
        )

    def test_unmapped_and_explicit_export_labels(self) -> None:
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

        self.assertIsNone(unmapped.label_export)
        self.assertEqual(unmapped.label_for_export, "Original")
        self.assertEqual(explicit.label_for_export, "Exported")

    def test_mapper_and_explicit_export_label_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not both be provided"):
            alignment.AlignmentRegion(
                region_id=0,
                label="Original",
                label_export="Exported",
                label_mapper=alignment.LabelMapper.from_data(
                    {"Original": "Mapped"}
                ),
            )

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            alignment.AlignmentRegion(
                region_id=0,
                label="Original",
                label_export="   ",
            )

    def test_suspicious_geometry_warning_has_both_labels(self) -> None:
        with self.assertLogs(models_module.logger, logging.WARNING) as captured:
            alignment.AlignmentRegion(
                region_id=0,
                label="Original",
                label_mapper=alignment.LabelMapper.from_data(
                    {"Original": "Exported"}
                ),
                input_geometry=alignment.BoundingBox(-1, 0, 2, 3),
            )

        message = captured.records[0].getMessage()
        self.assertIn("label='Original'", message)
        self.assertIn("label_export='Exported'", message)


class ReaderLabelMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = alignment.LabelMapper.from_data(
            {
                "Title": "heading",
                "YOLO title": "heading",
            }
        )

    def test_json_text_reader_maps_regions(self) -> None:
        region = alignment.JSONTextReader(
            label_mapper=self.mapper
        ).from_data({"Title": "Text"}).regions[0]

        self.assertEqual((region.label, region.label_export), ("Title", "heading"))

    def test_json_geometry_reader_maps_regions(self) -> None:
        region = alignment.JSONGeometryReader(
            label_mapper=self.mapper
        ).from_data(
            {"Title_bbox": {"x": 0, "y": 0, "width": 10, "height": 10}}
        ).regions[0]

        self.assertEqual((region.label, region.label_export), ("Title", "heading"))

    def test_yolo_reader_maps_regions(self) -> None:
        region = alignment.YOLOReader(label_mapper=self.mapper).from_data(
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

        self.assertEqual(
            (region.label, region.label_export),
            ("YOLO title", "heading"),
        )

    def test_label_studio_reader_maps_regions(self) -> None:
        document = alignment.LabelStudioReader(
            label_mapper=self.mapper
        ).from_data(
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
        self.assertEqual((region.label, region.label_export), ("Title", "heading"))


class ExportLabelTests(unittest.TestCase):
    def test_grouped_json_merges_source_labels_mapped_to_one_export_label(
        self,
    ) -> None:
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

        self.assertEqual(output["heading"], ["First", "Second"])
        self.assertEqual(len(output["heading_bbox"]), 2)
        self.assertNotIn("Title", output)
        self.assertNotIn("Subtitle", output)

    def test_source_json_keys_remain_original(self) -> None:
        page = alignment.JSONTextReader(
            label_mapper=alignment.LabelMapper.from_data(
                {"Title": "heading"}
            )
        ).from_data({"Title": "Original text"})
        page.regions[0].alto_geometry = alignment.BoundingBox(0, 0, 10, 10)

        output = alignment.AlignmentJSONWriter(
            alignment_mode=alignment.AlignmentMode.TEXT,
            geometry_suffix="_bbox",
            output_geometry_format=alignment.OutputGeometryFormat.BBOX,
            output_geometry_source=alignment.OutputGeometrySource.ALTO,
        ).to_data(page)

        self.assertIn("Title", output)
        self.assertIn("Title_bbox", output)
        self.assertNotIn("heading", output)

    def test_label_studio_writer_uses_export_label(self) -> None:
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

        labels = output["predictions"][0]["result"][0]["value"][
            "rectanglelabels"
        ]
        self.assertEqual(labels, ["heading"])


class AlignerLabelMappingTests(unittest.TestCase):
    def test_cli_parsers_accept_mapping_file(self) -> None:
        for parser in (
            text_aligner.build_argument_parser(),
            geometry_aligner.build_argument_parser(),
        ):
            with self.subTest(parser=parser.prog):
                self.assertIsNone(parser.get_default("class_mapping_file"))
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
                self.assertEqual(namespace.class_mapping_file, "mapping.json")

    def test_text_aligner_rejects_mapper_with_custom_reader(self) -> None:
        with self.assertRaisesRegex(ValueError, "custom json_reader"):
            alignment.TextAligner(
                candidate_generator=alignment.ExactTextCandidateGenerator(),
                candidate_selector=alignment.PassThroughCandidateSelector(),
                json_reader=alignment.JSONTextReader(),
                label_mapper=alignment.LabelMapper.from_data(
                    {"Title": "heading"}
                ),
            )

    def test_text_aligner_passes_mapper_to_default_reader(self) -> None:
        mapper = alignment.LabelMapper.from_data({"Title": "heading"})
        aligner = alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            label_mapper=mapper,
        )

        self.assertIs(aligner.json_reader.label_mapper, mapper)

    def test_geometry_aligner_rejects_mapper_with_custom_reader(self) -> None:
        with self.assertRaisesRegex(ValueError, "custom input readers"):
            alignment.GeometryAligner(
                yolo_reader=alignment.YOLOReader(),
                label_mapper=alignment.LabelMapper.from_data(
                    {"Title": "heading"}
                ),
            )

    def test_geometry_aligner_passes_mapper_to_both_default_readers(
        self,
    ) -> None:
        mapper = alignment.LabelMapper.from_data({"Title": "heading"})
        aligner = alignment.GeometryAligner(label_mapper=mapper)

        self.assertIs(aligner.json_reader.label_mapper, mapper)
        self.assertIs(aligner.yolo_reader.label_mapper, mapper)


if __name__ == "__main__":
    unittest.main()
