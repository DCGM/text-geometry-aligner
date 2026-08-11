"""Tests for the shared alignment hierarchy and YOLO input adapter."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import text_geometry_aligner as alignment
from text_geometry_aligner import models as models_module


def _alto_xml(word: str) -> str:
    return f"""\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout>
    <Page ID="page-id" WIDTH="100" HEIGHT="200">
      <TextBlock>
        <TextLine>
          <String ID="word-id" CONTENT="{word}"
                  HPOS="10" VPOS="20" WIDTH="30" HEIGHT="10"/>
        </TextLine>
      </TextBlock>
    </Page>
  </Layout>
</alto>
"""


class SharedAlignmentModelTests(unittest.TestCase):
    def test_normal_input_geometry_does_not_warn(self) -> None:
        with self.assertNoLogs(models_module.logger, logging.WARNING):
            alignment.AlignmentRegion(
                region_id=0,
                label="Title",
                input_geometry=alignment.BoundingBox(0, 0, 10, 10),
            )

    def test_suspicious_geometry_warning_combines_source_details(self) -> None:
        with self.assertLogs(
            models_module.logger,
            logging.WARNING,
        ) as captured:
            alignment.AlignmentRegion(
                region_id=7,
                label="Page Number",
                input_geometry=alignment.BoundingBox(-1, -2, 2, 3),
                category_id=4,
                json_geometry_path=("groups", 0, "number_bbox"),
                json_text_path=("groups", 0, "number"),
            )

        self.assertEqual(len(captured.records), 1)
        warning = captured.records[0].getMessage()
        self.assertIn("region_id=7", warning)
        self.assertIn("label='Page Number'", warning)
        self.assertIn("category_id=4", warning)
        self.assertIn(
            "json_geometry_path='$.groups[0].number_bbox'",
            warning,
        )
        self.assertIn("json_text_path='$.groups[0].number'", warning)
        self.assertIn("negative coordinates", warning)
        self.assertIn("width=2 is below 5", warning)
        self.assertIn("height=3 is below 5", warning)
        self.assertIn("area=6 is below 100", warning)

    def test_polygon_area_and_non_finite_values_warn(self) -> None:
        with self.assertLogs(
            models_module.logger,
            logging.WARNING,
        ) as captured:
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

        self.assertEqual(len(captured.records), 2)
        self.assertIn("area=50 is below 100", captured.output[0])
        self.assertIn("non-finite values: width=inf", captured.output[1])

    def test_input_region_is_populated_in_place(self) -> None:
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

        self.assertIsNone(region.input_text)
        self.assertIsNone(region.input_text_normalized)
        self.assertIsNone(region.text_alignment_candidate)
        self.assertIsNone(region.alto_text)
        self.assertIsNone(region.alto_text_normalized)
        self.assertIsNone(region.alto_geometry)
        self.assertIsNone(region.words)
        self.assertEqual(document.unmatched_count, 1)

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

        self.assertEqual(region.alto_text, "12")
        self.assertEqual(region.alto_geometry, input_geometry)
        self.assertEqual(region.words[0].word_index, 7)
        self.assertIsNone(region.words[0].text_normalized)
        self.assertEqual(region.words[0].word_coverage, 1.0)
        self.assertEqual(
            region.words[0].input_geometry_coverage,
            1.0,
        )
        self.assertEqual(region.words[0].overlap_score, 1.0)
        self.assertEqual(region.input_geometry, input_geometry)
        self.assertIsNone(region.input_text_normalized)
        self.assertIsNone(region.text_alignment_candidate)
        self.assertIsNone(region.alto_text_normalized)
        self.assertEqual(document.matched_count, 1)

    def test_deep_json_lists_and_dicts_reconstruct_from_region_paths(
        self,
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
        output = aligner.json_writer.to_data(document.pages[0])

        self.assertEqual(
            output["groups_bbox"],
            [
                [
                    {"x": 0, "y": 0, "width": 9, "height": 10},
                    {"x": 10, "y": 0, "width": 9, "height": 10},
                ],
                [None],
            ],
        )
        self.assertEqual(
            output["groups"][1][0]["name_bbox"],
            {"x": 20, "y": 0, "width": 9, "height": 10},
        )


class YOLOAdapterTests(unittest.TestCase):
    @staticmethod
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

    def test_reader_logs_page_before_suspicious_region_warning(self) -> None:
        with self.assertLogs(
            "text_geometry_aligner",
            logging.INFO,
        ) as captured:
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

        self.assertEqual(len(captured.records), 2)
        self.assertEqual(captured.records[0].levelno, logging.INFO)
        self.assertIn(
            "Loading YOLO geometry page 'memory-page' from in-memory data",
            captured.records[0].getMessage(),
        )
        self.assertEqual(captured.records[1].levelno, logging.WARNING)
        self.assertIn("label='Page Number'", captured.records[1].getMessage())
        self.assertIn("category_id=3", captured.records[1].getMessage())

    def test_reader_accepts_in_memory_detections(self) -> None:
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

        self.assertEqual(page.page_key, "memory-page")
        self.assertIsNone(page.input_file_path)
        self.assertEqual(page.regions[0].label, "Page Number")
        self.assertEqual(
            page.regions[0].input_geometry,
            alignment.BoundingBox(10, 25, 30, 10),
        )

    def test_reader_preserves_detection_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "page.detections"
            path.write_text(
                "3 25 30 30 10 0.875 Page Number\n",
                encoding="utf-8",
            )
            with self.assertLogs(
                "text_geometry_aligner.io_yolo.reader",
                logging.INFO,
            ) as captured:
                page = alignment.YOLOReader().read(
                    path,
                    page_key="page",
                )

        region = page.regions[0]
        self.assertIn(
            f"Loading YOLO geometry page 'page' from {path}",
            captured.records[0].getMessage(),
        )
        self.assertIs(page.input_format, alignment.InputFormat.YOLO)
        self.assertIsNone(page.json_source_data)
        self.assertEqual(region.label, "Page Number")
        self.assertEqual(region.category_id, 3)
        self.assertEqual(region.input_geometry_confidence, 0.875)
        self.assertEqual(
            region.input_geometry,
            alignment.BoundingBox(10, 25, 30, 10),
        )

    def test_reader_rejects_inconsistent_id_name_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "page"
            path.write_text(
                "0 5 5 10 10 0.9 First\n"
                "0 5 5 10 10 0.8 Second\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "Inconsistent YOLO class mapping",
            ):
                alignment.YOLOReader().read(path)

    def test_reader_optionally_deduplicates_cross_class_regions(self) -> None:
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
                self._detection("Chapter", 0.7),
                self._detection("Subchapter", 0.9),
                self._detection("Other", 0.5),
            ]
        )

        self.assertEqual(
            [(region.region_id, region.label) for region in page.regions],
            [(1, "Subchapter"), (2, "Other")],
        )

    def test_reader_deduplication_preserves_same_class_regions(self) -> None:
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
                self._detection("Chapter", 0.7),
                self._detection("Chapter", 0.9),
            ]
        )

        self.assertEqual([region.region_id for region in page.regions], [0, 1])

    def test_reader_deduplication_requires_coverage_of_larger_box(
        self,
    ) -> None:
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
                self._detection("Chapter", 0.9),
                self._detection(
                    "Subchapter",
                    0.8,
                    width=10,
                    height=10,
                ),
            ]
        )

        self.assertEqual([region.region_id for region in page.regions], [0, 1])

    def test_reader_deduplication_tie_retains_first_detection(self) -> None:
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
                self._detection("Chapter", 0.9),
                self._detection("Subchapter", 0.9),
            ]
        )

        self.assertEqual(
            [(region.region_id, region.label) for region in page.regions],
            [(0, "Chapter")],
        )

    def test_reader_rejects_label_in_multiple_deduplication_groups(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "cannot belong to multiple deduplication groups",
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

    def test_yolo_export_groups_repeated_labels_and_retains_input_boxes(
        self,
    ) -> None:
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

        self.assertEqual(output["PageNumber"], ["12", None])
        self.assertEqual(
            output["PageNumber_bbox"],
            [
                {"x": 1, "y": 2, "width": 3, "height": 4},
                {"x": 5, "y": 6, "width": 7, "height": 8},
            ],
        )
        self.assertEqual(page.regions[0].category_id, 0)

    def test_yolo_export_rejects_class_names_that_collide_with_suffix_keys(
        self,
    ) -> None:
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

        with self.assertRaisesRegex(ValueError, "collide"):
            writer.to_data(page)

    def test_process_files_pairs_by_key_and_preserves_input_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            alto_dir = root / "alto"
            input_dir.mkdir()
            alto_dir.mkdir()
            first_input = input_dir / "first"
            second_input = input_dir / "second.labels"
            first_alto = alto_dir / "first.xml"
            second_alto = alto_dir / "second.xml"
            first_input.write_text(
                "0 25 25 30 10 0.8 PageNumber\n",
                encoding="utf-8",
            )
            second_input.write_text(
                "0 25 25 30 10 0.9 PageNumber\n",
                encoding="utf-8",
            )
            first_alto.write_text(_alto_xml("1"), encoding="utf-8")
            second_alto.write_text(_alto_xml("2"), encoding="utf-8")
            json_writer = mock.create_autospec(
                alignment.AlignmentJSONWriter,
                instance=True,
            )

            document = alignment.GeometryAligner(
                json_writer=json_writer,
            ).process_files(
                alto_files=[first_alto, second_alto],
                input_files=[second_input, first_input],
                input_format=alignment.InputFormat.YOLO,
            )

            json_writer.write.assert_not_called()
            self.assertEqual(
                [page.page_key for page in document.pages],
                ["second", "first"],
            )
            self.assertEqual(
                [
                    page.regions[0].alto_text
                    for page in document.pages
                ],
                ["2", "1"],
            )
            self.assertEqual(document.input_path, input_dir)
            self.assertEqual(document.alto_path, alto_dir)
            self.assertEqual(
                document.pages[0].input_file_path,
                second_input,
            )
            self.assertEqual(
                document.pages[0].alto_file_path,
                second_alto,
            )

    def test_process_files_supports_json_input_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_file = root / "page.json"
            alto_file = root / "page.xml"
            output_dir = root / "output"
            input_file.write_text(
                json.dumps(
                    {
                        "PageNumber_bbox": {
                            "x": 10,
                            "y": 20,
                            "width": 30,
                            "height": 10,
                        }
                    }
                ),
                encoding="utf-8",
            )
            alto_file.write_text(_alto_xml("12"), encoding="utf-8")

            document = alignment.GeometryAligner().process_files(
                alto_files=[alto_file],
                input_files=[input_file],
                json_output_dir=output_dir,
            )

            self.assertEqual(document.pages[0].regions[0].alto_text, "12")
            self.assertEqual(
                json.loads(
                    (output_dir / "page.json").read_text(encoding="utf-8")
                )["PageNumber"],
                "12",
            )

    def test_process_files_validates_page_keys_and_missing_alto(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_input = root / "page.labels"
            duplicate_input = root / "page.txt"
            first_input.write_text("", encoding="utf-8")
            duplicate_input.write_text("", encoding="utf-8")

            aligner = alignment.GeometryAligner()
            with self.assertRaisesRegex(ValueError, "Multiple yolo input"):
                aligner.process_files(
                    alto_files=[],
                    input_files=[first_input, duplicate_input],
                    input_format=alignment.InputFormat.YOLO,
                )

            document = aligner.process_files(
                alto_files=[],
                input_files=[first_input],
                input_format=alignment.InputFormat.YOLO,
            )
            self.assertEqual(document.pages, [])

            with self.assertRaisesRegex(FileNotFoundError, "No ALTO XML"):
                aligner.process_files(
                    alto_files=[],
                    input_files=[first_input],
                    input_format=alignment.InputFormat.YOLO,
                    fail_on_missing_alto=True,
                )

    def test_directory_pairing_removes_one_suffix_or_uses_full_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            alto_dir = root / "alto"
            output_dir = root / "output"
            input_dir.mkdir()
            alto_dir.mkdir()

            (input_dir / "page.full.labels").write_text(
                "0 25 25 30 10 0.9 PageNumber\n",
                encoding="utf-8",
            )
            (alto_dir / "page.full.xml").write_text(
                _alto_xml("12"),
                encoding="utf-8",
            )
            (input_dir / "plain").write_text(
                "0 25 25 30 10 0.8 PageNumber\n",
                encoding="utf-8",
            )
            (alto_dir / "plain.xml").write_text(
                _alto_xml("13"),
                encoding="utf-8",
            )

            document = alignment.GeometryAligner().process_directories(
                alto_dir,
                input_dir,
                output_dir,
                input_format=alignment.InputFormat.YOLO,
            )

            self.assertEqual(
                [page.page_key for page in document.pages],
                ["page.full", "plain"],
            )
            self.assertEqual(document.input_path, input_dir)
            self.assertEqual(document.alto_path, alto_dir)
            self.assertEqual(document.matched_count, 2)
            self.assertIsNone(document.pages[0].json_source_data)
            self.assertEqual(document.pages[0].alto_width, 100)
            self.assertEqual(document.pages[0].alto_height, 200)
            self.assertEqual(
                json.loads(
                    (output_dir / "page.full.json").read_text(
                        encoding="utf-8"
                    )
                )["PageNumber"],
                ["12"],
            )

    def test_directory_processing_can_return_document_without_export(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            alto_dir = root / "alto"
            input_dir.mkdir()
            alto_dir.mkdir()
            (input_dir / "page.labels").write_text(
                "0 25 25 30 10 0.9 PageNumber\n",
                encoding="utf-8",
            )
            (alto_dir / "page.xml").write_text(
                _alto_xml("12"),
                encoding="utf-8",
            )
            json_writer = mock.create_autospec(
                alignment.AlignmentJSONWriter,
                instance=True,
            )

            document = alignment.GeometryAligner(
                json_writer=json_writer,
            ).process_directories(
                alto_dir,
                input_dir,
                input_format=alignment.InputFormat.YOLO,
            )

            json_writer.write.assert_not_called()
            self.assertEqual(len(document.pages), 1)
            self.assertEqual(document.pages[0].page_key, "page")
            self.assertEqual(document.pages[0].regions[0].alto_text, "12")
            self.assertEqual(document.matched_count, 1)

    def test_rendering_converts_selected_geometry_to_requested_format(
        self,
    ) -> None:
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

        self.assertIsInstance(rendered[0].geometry, alignment.Polygon)
        self.assertEqual(
            rendered[0].geometry.points,
            ((1, 2), (4, 2), (4, 6), (1, 6), (1, 2)),
        )


if __name__ == "__main__":
    unittest.main()
