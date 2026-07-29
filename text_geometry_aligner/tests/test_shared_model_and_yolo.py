"""Tests for the shared alignment hierarchy and YOLO input adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import text_geometry_aligner as alignment


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
        output = aligner.export_page(document.pages[0])

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
    def test_reader_and_extractor_preserve_detection_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "page.detections"
            path.write_text(
                "3 25 30 30 10 0.875 Page Number\n",
                encoding="utf-8",
            )
            page = alignment.YOLOGeometryExtractor().extract_alignment_page(
                path,
                page_key="page",
            )

        region = page.regions[0]
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
        output = alignment.AlignmentJSONExporter(
            alignment_mode=alignment.AlignmentMode.GEOMETRY,
            geometry_suffix="_bbox",
            output_geometry_format=alignment.OutputGeometryFormat.BBOX,
            output_geometry_source=alignment.OutputGeometrySource.INPUT,
        ).export(page)

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
        exporter = alignment.AlignmentJSONExporter(
            alignment_mode=alignment.AlignmentMode.GEOMETRY,
            geometry_suffix="_bbox",
            output_geometry_format=alignment.OutputGeometryFormat.BBOX,
        )

        with self.assertRaisesRegex(ValueError, "collide"):
            exporter.export(page)

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
                alignment.JSONWriter,
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
