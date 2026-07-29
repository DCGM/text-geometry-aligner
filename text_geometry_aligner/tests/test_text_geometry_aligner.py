"""Tests for the modular text-geometry aligner package."""

import contextlib
import importlib.util
import io
import logging
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import text_geometry_aligner as alignment
from text_geometry_aligner import (
    base_aligner as base_aligner_module,
    text_aligner as aligner_module,
)
from text_geometry_aligner.text_matching.candidate_generators import (
    anchored_fuzzy as candidate_generation,
)
from text_geometry_aligner.text_matching.candidate_generators import (
    ordered_alignment as ordered_candidate_generation,
)


def _distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _editops(source: str, target: str) -> list[tuple[str, int, int]]:
    """Small test-only equivalent of Levenshtein.editops."""

    distances = [
        [0] * (len(target) + 1)
        for _ in range(len(source) + 1)
    ]
    for source_index in range(len(source) + 1):
        distances[source_index][0] = source_index
    for target_index in range(len(target) + 1):
        distances[0][target_index] = target_index

    for source_index, source_character in enumerate(source, start=1):
        for target_index, target_character in enumerate(target, start=1):
            distances[source_index][target_index] = min(
                distances[source_index - 1][target_index] + 1,
                distances[source_index][target_index - 1] + 1,
                distances[source_index - 1][target_index - 1]
                + int(source_character != target_character),
            )

    operations: list[tuple[str, int, int]] = []
    source_index = len(source)
    target_index = len(target)
    while source_index or target_index:
        if (
            source_index
            and target_index
            and source[source_index - 1] == target[target_index - 1]
            and distances[source_index][target_index]
            == distances[source_index - 1][target_index - 1]
        ):
            source_index -= 1
            target_index -= 1
        elif (
            source_index
            and target_index
            and distances[source_index][target_index]
            == distances[source_index - 1][target_index - 1] + 1
        ):
            operations.append(
                ("replace", source_index - 1, target_index - 1)
            )
            source_index -= 1
            target_index -= 1
        elif (
            source_index
            and distances[source_index][target_index]
            == distances[source_index - 1][target_index] + 1
        ):
            operations.append(("delete", source_index - 1, target_index))
            source_index -= 1
        else:
            operations.append(("insert", source_index, target_index - 1))
            target_index -= 1

    operations.reverse()
    return operations


def _page(*texts: str, block_indexes: tuple[int, ...] | None = None) -> alignment.ALTOPage:
    if block_indexes is None:
        block_indexes = (0,) * len(texts)
    words = tuple(
        alignment.ALTOWord(
            index=index,
            text=text,
            bbox=alignment.BoundingBox(index * 10, 0, 9, 10),
            line_index=0,
            block_index=block_indexes[index],
        )
        for index, text in enumerate(texts)
    )
    return alignment.ALTOPage(source_path=Path("test.xml"), words=words)


def _value(value_id: int, text: str) -> alignment.AlignmentRegion:
    return alignment.AlignmentRegion(
        region_id=value_id,
        label=f"value_{value_id}",
        input_text=text,
        input_text_normalized=text.casefold(),
        json_text_path=(f"value_{value_id}",),
    )


def _lowercase_normalizer() -> alignment.TextNormalizationPipeline:
    return alignment.TextNormalizationPipeline.from_optional_names(
        ("lowercase",)
    )


def _normalize_regions(
    regions: tuple[alignment.AlignmentRegion, ...],
    normalizer: alignment.TextNormalizer,
) -> tuple[alignment.AlignmentRegion, ...]:
    for region in regions:
        region.input_text_normalized = (
            None
            if region.input_text is None
            else normalizer.normalize(str(region.input_text))
        )
    return regions


def _combined_generator(
    fuzzy_config: alignment.FuzzyCandidateConfig | None = None,
) -> alignment.CompositeCandidateGenerator:
    return alignment.CompositeCandidateGenerator(
        (
            alignment.ExactTextCandidateGenerator(),
            alignment.AnchoredFuzzyTextCandidateGenerator(fuzzy_config),
        )
    )


def _output(aligner, document):
    return aligner.export_page(document.pages[0])


class _FirstCandidateSelector:
    def select(self, candidates, regions):
        return tuple(candidates[:1])


class _AllCandidateSelector:
    def select(self, candidates, regions):
        return tuple(candidates)


class _StaticCandidateGenerator(alignment.CandidateGenerator):
    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        self.calls = []

    def generate(self, regions, alto_index):
        self.calls.append((regions, alto_index))
        return self.candidates


class FormatIOTests(unittest.TestCase):
    def test_json_reader_and_writer_round_trip_utf8_atomically(self) -> None:
        data = {"title": "Řím", "publishers": ["Šolc"]}

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory) / "nested" / "document.json"
            )
            alignment.JSONWriter().write(data, output_path)

            self.assertEqual(alignment.JSONReader().read(output_path), data)
            output_text = output_path.read_text(encoding="utf-8")
            self.assertIn("Řím", output_text)
            self.assertTrue(output_text.endswith("\n"))
            self.assertFalse(
                output_path.with_name(f".{output_path.name}.tmp").exists()
            )

    def test_alto_reader_only_parses_xml_into_page_model(self) -> None:
        alto_xml = """\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout>
    <Page ID="page-1" WIDTH="100" HEIGHT="200">
      <TextBlock>
        <TextLine>
          <String ID="word-1" CONTENT="ROME"
                  HPOS="1" VPOS="2" WIDTH="30" HEIGHT="10"/>
        </TextLine>
      </TextBlock>
    </Page>
  </Layout>
</alto>
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "page.xml"
            input_path.write_text(alto_xml, encoding="utf-8")

            page = alignment.ALTOReader().read(input_path)

        self.assertEqual(page.page_id, "page-1")
        self.assertEqual((page.width, page.height), (100.0, 200.0))
        self.assertEqual(len(page.words), 1)
        self.assertEqual(page.words[0].text, "ROME")
        self.assertEqual(page.words[0].element_id, "word-1")
        self.assertEqual(
            page.words[0].bbox,
            alignment.BoundingBox(1.0, 2.0, 30.0, 10.0),
        )

    def test_align_files_delegates_format_io(self) -> None:
        alto_reader = mock.create_autospec(
            alignment.ALTOReader,
            instance=True,
        )
        json_reader = mock.create_autospec(
            alignment.JSONReader,
            instance=True,
        )
        json_writer = mock.create_autospec(
            alignment.JSONWriter,
            instance=True,
        )
        alto_reader.read.return_value = _page("ROME")
        json_reader.read.return_value = {"title": "Rome"}
        aligner = alignment.TextAligner(
            alto_reader=alto_reader,
            json_reader=json_reader,
            json_writer=json_writer,
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=_FirstCandidateSelector(),
        )

        result = aligner.align_files(
            "page.xml",
            "input.json",
            "output.json",
        )

        alto_reader.read.assert_called_once_with(Path("page.xml"))
        json_reader.read.assert_called_once_with(Path("input.json"))
        json_writer.write.assert_called_once_with(
            _output(aligner, result),
            Path("output.json"),
        )


class NormalizationTests(unittest.TestCase):
    @staticmethod
    def _candidate(**changes) -> alignment.AlignmentCandidate:
        candidate = alignment.AlignmentCandidate(
            candidate_id=0,
            region_id=0,
            json_text_path=("title",),
            start_word=0,
            end_word=0,
            start_char=0,
            end_char=4,
            query_text="Raw input",
            matched_text="ALTO",
            normalized_query_text="candidate query",
            normalized_matched_text="candidate match",
            exact=False,
            edit_distance=1,
            cer_int=1000,
            similarity_int=9000,
            query_length=15,
            quality_chars=14,
            source="test",
        )
        return replace(candidate, **changes)

    def test_default_pipeline_preserves_case(self) -> None:
        pipeline = alignment.TextNormalizationPipeline.from_optional_names()
        self.assertEqual(pipeline.normalize(" Straße  TEST "), "Straße TEST")

    def test_optional_normalizers_are_composable_and_ordered(self) -> None:
        pipeline = alignment.TextNormalizationPipeline.from_optional_names(
            ("lowercase", "strip-diacritics", "strip-punctuation")
        )
        self.assertEqual(pipeline.normalize(" ČESKÝ—Krumlov! "), "cesky krumlov")

    def test_alignment_retains_raw_and_normalized_text(self) -> None:
        pipeline = alignment.TextNormalizationPipeline.from_optional_names(
            ("lowercase", "strip-diacritics", "strip-punctuation")
        )
        page = _page("ŘÍM!")
        aligner = alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            normalizer=pipeline,
        )
        result = aligner.align_data(page, {"city": "ŘÍM"})
        region = result.pages[0].regions[0]
        word = region.words[0]

        self.assertEqual(region.input_text, "ŘÍM")
        self.assertEqual(region.input_text_normalized, "rim")
        self.assertEqual(region.alto_text, "ŘÍM!")
        self.assertEqual(region.alto_text_normalized, "rim")
        self.assertEqual(word.text, "ŘÍM!")
        self.assertEqual(word.text_normalized, "rim")
        self.assertEqual(page.words[0].text, "ŘÍM!")

    def test_unmatched_region_retains_only_normalized_input_text(self) -> None:
        pipeline = alignment.TextNormalizationPipeline.from_optional_names(
            ("lowercase",)
        )
        aligner = alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            normalizer=pipeline,
        )

        result = aligner.align_data(_page("different"), {"title": "TITLE"})
        region = result.pages[0].regions[0]

        self.assertEqual(region.input_text, "TITLE")
        self.assertEqual(region.input_text_normalized, "title")
        self.assertIsNone(region.text_alignment_candidate)
        self.assertIsNone(region.alto_text)
        self.assertIsNone(region.alto_text_normalized)
        self.assertIsNone(region.words)

    def test_normalization_to_empty_produces_no_candidate(self) -> None:
        pipeline = alignment.TextNormalizationPipeline.from_optional_names(
            ("strip-punctuation",)
        )
        aligner = alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
            normalizer=pipeline,
        )

        result = aligner.align_data(_page("text"), {"mark": "!!!"})
        region = result.pages[0].regions[0]

        self.assertEqual(region.input_text_normalized, "")
        self.assertIsNone(region.alto_text_normalized)
        self.assertIsNone(region.words)

    def test_source_normalization_is_independent_of_candidate_snapshots(
        self,
    ) -> None:
        candidate = self._candidate()
        aligner = alignment.TextAligner(
            candidate_generator=_StaticCandidateGenerator((candidate,)),
            candidate_selector=_AllCandidateSelector(),
        )

        result = aligner.align_data(_page("ALTO"), {"title": "Raw input"})
        region = result.pages[0].regions[0]

        self.assertEqual(region.input_text_normalized, "Raw input")
        self.assertEqual(region.alto_text_normalized, "ALTO")
        self.assertIs(region.text_alignment_candidate, candidate)
        self.assertEqual(
            region.text_alignment_candidate.normalized_query_text,
            "candidate query",
        )
        self.assertEqual(
            region.text_alignment_candidate.normalized_matched_text,
            "candidate match",
        )
        self.assertEqual(region.words[0].text_normalized, "ALTO")

    def test_selected_candidate_survives_failed_text_building(self) -> None:
        candidate = self._candidate()
        text_builder = mock.create_autospec(
            alignment.TextBuilder,
            instance=True,
        )
        text_builder.build.return_value = None
        aligner = alignment.TextAligner(
            candidate_generator=_StaticCandidateGenerator((candidate,)),
            candidate_selector=_AllCandidateSelector(),
            text_builder=text_builder,
        )

        result = aligner.align_data(_page("ALTO"), {"title": "Raw input"})
        region = result.pages[0].regions[0]

        self.assertIs(region.text_alignment_candidate, candidate)
        self.assertEqual(region.input_text_normalized, "Raw input")
        self.assertEqual(region.alto_text_normalized, "ALTO")
        self.assertIsNone(region.alto_text)
        self.assertIsNone(region.words)
        self.assertFalse(region.matched)


class GeometryBuilderTests(unittest.TestCase):
    def test_base_builder_requires_a_build_implementation(self) -> None:
        with self.assertRaises(TypeError):
            alignment.GeometryBuilder()

    def test_single_word_polygon_is_a_closed_rectangle(self) -> None:
        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(
            _page("WORD").words
        )

        self.assertEqual(
            polygon.points,
            ((0, 0), (9, 0), (9, 10), (0, 10), (0, 0)),
        )
        self.assertEqual(
            polygon.to_json(),
            [[0, 0], [9, 0], [9, 10], [0, 10], [0, 0]],
        )
        self.assertEqual(
            polygon.bounds,
            alignment.BoundingBox(0, 0, 9, 10),
        )

    def test_multiline_polygon_breaks_at_bottom_of_upper_line(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="FULL",
                bbox=alignment.BoundingBox(0, 0, 100, 10),
                line_index=0,
            ),
            alignment.ALTOWord(
                index=1,
                text="HALF",
                bbox=alignment.BoundingBox(0, 12, 50, 10),
                line_index=1,
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            (
                (0, 0),
                (100, 0),
                (100, 10),
                (50, 10),
                (50, 22),
                (0, 22),
                (0, 0),
            ),
        )

    def test_multiline_polygon_steps_on_both_sides(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="UPPER",
                bbox=alignment.BoundingBox(20, 0, 80, 10),
                line_index=0,
            ),
            alignment.ALTOWord(
                index=1,
                text="LOWER",
                bbox=alignment.BoundingBox(0, 12, 50, 10),
                line_index=1,
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            (
                (20, 0),
                (100, 0),
                (100, 10),
                (50, 10),
                (50, 22),
                (0, 22),
                (0, 12),
                (20, 12),
                (20, 0),
            ),
        )

    def test_each_edge_uses_its_own_inward_or_outward_break(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="UPPER",
                bbox=alignment.BoundingBox(0, 0, 100, 10),
                line_index=0,
            ),
            alignment.ALTOWord(
                index=1,
                text="LOWER",
                bbox=alignment.BoundingBox(20, 12, 100, 10),
                line_index=1,
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            (
                (0, 0),
                (100, 0),
                (100, 12),
                (120, 12),
                (120, 22),
                (20, 22),
                (20, 10),
                (0, 10),
                (0, 0),
            ),
        )

    def test_both_outward_edges_break_at_lower_line_top(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="UPPER",
                bbox=alignment.BoundingBox(20, 0, 80, 10),
                line_index=0,
            ),
            alignment.ALTOWord(
                index=1,
                text="LOWER",
                bbox=alignment.BoundingBox(0, 12, 120, 10),
                line_index=1,
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            (
                (20, 0),
                (100, 0),
                (100, 12),
                (120, 12),
                (120, 22),
                (0, 22),
                (0, 12),
                (20, 12),
                (20, 0),
            ),
        )

    def test_words_in_one_alto_line_become_one_clean_rectangle(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="FIRST",
                bbox=alignment.BoundingBox(0, 1, 20, 9),
                line_index=0,
            ),
            alignment.ALTOWord(
                index=1,
                text="SECOND",
                bbox=alignment.BoundingBox(25, 0, 30, 10),
                line_index=0,
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            ((0, 0), (55, 0), (55, 10), (0, 10), (0, 0)),
        )

    def test_words_without_line_ids_retain_word_box_bands(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="FIRST",
                bbox=alignment.BoundingBox(0, 1, 20, 9),
                line_index=None,
                block_index=0,
            ),
            alignment.ALTOWord(
                index=1,
                text="SECOND",
                bbox=alignment.BoundingBox(25, 0, 30, 10),
                line_index=None,
                block_index=0,
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            (
                (25, 0),
                (55, 0),
                (55, 10),
                (0, 10),
                (0, 1),
                (25, 1),
                (25, 0),
            ),
        )

    def test_equal_line_indexes_in_different_blocks_are_not_merged(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="UPPER",
                bbox=alignment.BoundingBox(0, 0, 100, 10),
                line_index=0,
                block_index=0,
            ),
            alignment.ALTOWord(
                index=1,
                text="LOWER",
                bbox=alignment.BoundingBox(0, 12, 50, 10),
                line_index=0,
                block_index=1,
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            (
                (0, 0),
                (100, 0),
                (100, 10),
                (50, 10),
                (50, 22),
                (0, 22),
                (0, 0),
            ),
        )

    def test_ocr_line_splitting_does_not_split_a_visual_band(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="FIRST",
                bbox=alignment.BoundingBox(0, 0, 20, 10),
                line_index=4,
                block_index=0,
            ),
            alignment.ALTOWord(
                index=1,
                text="SECOND",
                bbox=alignment.BoundingBox(25, 0, 30, 10),
                line_index=9,
                block_index=3,
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            ((0, 0), (55, 0), (55, 10), (0, 10), (0, 0)),
        )

    def test_word_input_order_does_not_affect_polygon(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="UPPER",
                bbox=alignment.BoundingBox(0, 0, 100, 10),
                line_index=8,
            ),
            alignment.ALTOWord(
                index=1,
                text="LOWER",
                bbox=alignment.BoundingBox(20, 12, 60, 10),
                line_index=2,
            ),
        )
        builder = alignment.OrthogonalPolygonGeometryBuilder()

        forward = builder.build(words)
        reversed_order = builder.build(tuple(reversed(words)))

        self.assertEqual(forward, reversed_order)

    def test_wide_narrow_wide_layout_retains_middle_concavity(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="WIDE",
                bbox=alignment.BoundingBox(0, 0, 100, 10),
            ),
            alignment.ALTOWord(
                index=1,
                text="NARROW",
                bbox=alignment.BoundingBox(0, 12, 50, 10),
            ),
            alignment.ALTOWord(
                index=2,
                text="WIDE",
                bbox=alignment.BoundingBox(0, 24, 100, 10),
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            (
                (0, 0),
                (100, 0),
                (100, 10),
                (50, 10),
                (50, 24),
                (100, 24),
                (100, 34),
                (0, 34),
                (0, 0),
            ),
        )

    def test_nonoverlapping_bands_are_connected_without_dropping_words(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="UPPER",
                bbox=alignment.BoundingBox(0, 0, 20, 10),
            ),
            alignment.ALTOWord(
                index=1,
                text="LOWER",
                bbox=alignment.BoundingBox(40, 12, 20, 10),
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            (
                (0, 0),
                (20, 0),
                (20, 10),
                (60, 10),
                (60, 22),
                (40, 22),
                (40, 12),
                (0, 12),
                (0, 0),
            ),
        )
        self.assertEqual(
            polygon.bounds,
            alignment.BoundingBox(0, 0, 60, 22),
        )

    def test_touching_disjoint_bands_fall_back_without_dropping_words(self) -> None:
        words = (
            alignment.ALTOWord(
                index=0,
                text="UPPER",
                bbox=alignment.BoundingBox(0, 0, 20, 10),
            ),
            alignment.ALTOWord(
                index=1,
                text="LOWER",
                bbox=alignment.BoundingBox(40, 10, 20, 10),
            ),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build(words)

        self.assertEqual(
            polygon.points,
            ((0, 0), (60, 0), (60, 20), (0, 20), (0, 0)),
        )

    def test_degenerate_word_box_still_produces_closed_output(self) -> None:
        word = alignment.ALTOWord(
            index=0,
            text="POINT",
            bbox=alignment.BoundingBox(5, 7, 0, 0),
        )

        polygon = alignment.OrthogonalPolygonGeometryBuilder().build((word,))

        self.assertEqual(
            polygon.points,
            ((5, 7), (5, 7), (5, 7), (5, 7), (5, 7)),
        )
        self.assertEqual(
            polygon.bounds,
            alignment.BoundingBox(5, 7, 0, 0),
        )


class ListValueTests(unittest.TestCase):
    def test_extractor_exposes_each_scalar_list_element(self) -> None:
        values = alignment.JSONTextExtractor().extract_alignment_region(
            {"publisher": ["First", "Second"]}
        )

        self.assertEqual(
            [value.json_text_path for value in values],
            [("publisher", 0), ("publisher", 1)],
        )
        self.assertEqual(
            [value.json_geometry_path for value in values],
            [("publisher_bbox", 0), ("publisher_bbox", 1)],
        )

    def test_each_list_value_gets_parallel_geometry_and_alto_text(self) -> None:
        aligner = alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=_AllCandidateSelector(),
            normalizer=_lowercase_normalizer(),
            output_text_source=alignment.OutputTextSource.ALTO,
        )
        result = aligner.align_data(
            _page("FIRST", "SECOND"),
            {"publisher": ["First", "Second"]},
        )
        output = _output(aligner, result)

        self.assertEqual(output["publisher"], ["FIRST", "SECOND"])
        self.assertEqual(
            output["publisher_bbox"],
            [
                {"x": 0, "y": 0, "width": 9, "height": 10},
                {"x": 10, "y": 0, "width": 9, "height": 10},
            ],
        )
        self.assertEqual(
            [
                region.json_text_path
                for region in result.pages[0].regions
                if region.matched
            ],
            [("publisher", 0), ("publisher", 1)],
        )

    def test_publisher_list_phrase_participates_in_candidate_generation(self) -> None:
        page = _page(
            "ŠOLC",
            "a",
            "ŠIMÁČEK,",
            "společnost",
            "s",
            "r.",
            "o.",
            "v",
            "Praze.",
        )
        normalizer = alignment.TextNormalizationPipeline.from_optional_names()
        regions = _normalize_regions(
            alignment.JSONTextExtractor().extract_alignment_region(
                {
                    "partNumber": "I.",
                    "placeTerm": "Praze.",
                    "publisher": [
                        "ŠOLC a ŠIMÁČEK, společnost s r. o. v Praze."
                    ],
                }
            ),
            normalizer,
        )
        candidates = alignment.ExactTextCandidateGenerator().generate(
            regions,
            alignment.ALTOTextIndex(page, normalizer),
        )

        publisher_candidate = next(
            candidate
            for candidate in candidates
            if candidate.json_text_path == ("publisher", 0)
        )
        self.assertEqual(
            (publisher_candidate.start_word, publisher_candidate.end_word),
            (0, 8),
        )
        self.assertGreater(
            publisher_candidate.quality_chars,
            next(
                candidate.quality_chars
                for candidate in candidates
                if candidate.json_text_path == ("placeTerm",)
            ),
        )

    def test_unmatched_list_elements_keep_a_null_geometry_slot(self) -> None:
        aligner = alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=_AllCandidateSelector(),
            normalizer=_lowercase_normalizer(),
        )
        result = aligner.align_data(
            _page("FIRST"),
            {"publisher": ["First", "Missing"]},
        )

        self.assertEqual(
            _output(aligner, result)["publisher_bbox"],
            [{"x": 0, "y": 0, "width": 9, "height": 10}, None],
        )

    def test_polygon_output_preserves_list_shape(self) -> None:
        aligner = alignment.TextAligner(
            output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=_AllCandidateSelector(),
            normalizer=_lowercase_normalizer(),
        )
        result = aligner.align_data(
            _page("FIRST", "SECOND"),
            {"publisher": ["First", "Second"]},
        )
        output = _output(aligner, result)

        self.assertEqual(
            output["publisher_polygon"],
            [
                [[0, 0], [9, 0], [9, 10], [0, 10], [0, 0]],
                [[10, 0], [19, 0], [19, 10], [10, 10], [10, 0]],
            ],
        )
        self.assertNotIn("publisher_bbox", output)


class AlignmentJSONExporterTests(unittest.TestCase):
    def test_reconstructs_nested_output_from_retained_paths(self) -> None:
        input_data = {"groups": [{"names": ["Original"]}]}
        page = alignment.JSONTextExtractor().extract_alignment_page(
            input_data,
            page_key="page",
        )
        region = page.regions[0]
        region.alto_text = "ALTO"
        region.alto_geometry = alignment.BoundingBox(1, 2, 3, 4)
        output_data = alignment.AlignmentJSONExporter(
            alignment_mode=alignment.AlignmentMode.TEXT,
            geometry_suffix="_bbox",
            output_geometry_format=alignment.OutputGeometryFormat.BBOX,
            output_text_source=alignment.OutputTextSource.ALTO,
            output_geometry_source=alignment.OutputGeometrySource.ALTO,
        ).export(
            page,
        )

        self.assertEqual(input_data, {"groups": [{"names": ["Original"]}]})
        self.assertEqual(
            output_data,
            {
                "groups": [
                    {
                        "names": ["ALTO"],
                        "names_bbox": [
                            {"x": 1, "y": 2, "width": 3, "height": 4}
                        ],
                    }
                ]
            },
        )


class PolygonOutputTests(unittest.TestCase):
    def _align(
        self,
        geometry_suffix: str | None = None,
    ):
        aligner = alignment.TextAligner(
            geometry_suffix=geometry_suffix,
            output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=_AllCandidateSelector(),
            normalizer=_lowercase_normalizer(),
        )
        return (
            aligner,
            aligner.align_data(
                _page("FIRST", "SECOND"),
                {"title": "First Second"},
            ),
        )

    def test_polygon_format_selects_builder_suffix_and_result_metadata(self) -> None:
        aligner, result = self._align()
        output = _output(aligner, result)
        region = result.pages[0].regions[0]

        self.assertEqual(
            output["title_polygon"],
            [[0, 0], [19, 0], [19, 10], [0, 10], [0, 0]],
        )
        self.assertNotIn("title_bbox", output)
        self.assertIs(
            aligner.output_geometry_format,
            alignment.OutputGeometryFormat.POLYGON,
        )
        self.assertIsInstance(
            region.alto_geometry,
            alignment.Polygon,
        )

    def test_explicit_geometry_suffix_overrides_format_default(self) -> None:
        aligner, result = self._align("_shape")
        output = _output(aligner, result)

        self.assertIn("title_shape", output)
        self.assertNotIn("title_polygon", output)

    def test_extractor_ignores_geometry_from_both_formats(self) -> None:
        values = alignment.JSONTextExtractor(
            geometry_suffix="_polygon",
            overwrite_existing_geometry=True,
        ).extract_alignment_region(
            {
                "title": "Rome",
                "title_bbox": {
                    "x": 1,
                    "y": 2,
                    "width": 3,
                    "height": 4,
                },
                "title_polygon": [[1, 2], [4, 2], [1, 2]],
            }
        )

        self.assertEqual(
            [value.json_text_path for value in values],
            [("title",)],
        )

    def test_existing_bbox_is_retained_when_polygon_is_added(self) -> None:
        existing_bbox = {"x": 1, "y": 2, "width": 3, "height": 4}
        aligner = alignment.TextAligner(
            output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=_AllCandidateSelector(),
            normalizer=_lowercase_normalizer(),
        )

        result = aligner.align_data(
            _page("ROME"),
            {"title": "Rome", "title_bbox": existing_bbox},
        )
        output = _output(aligner, result)

        self.assertEqual(output["title_bbox"], existing_bbox)
        self.assertIn("title_polygon", output)

    def test_existing_geometry_is_preserved_by_default(self) -> None:
        existing_polygon = [[1, 2], [4, 2], [1, 2]]
        aligner = alignment.TextAligner(
            output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=_AllCandidateSelector(),
        )

        result = aligner.align_data(
            _page("ROME"),
            {"title": "Rome", "title_polygon": existing_polygon},
        )

        self.assertEqual(
            _output(aligner, result)["title_polygon"],
            existing_polygon,
        )
        self.assertEqual(result.matched_count, 0)

    def test_overwrite_existing_geometry_realigns_selected_format(self) -> None:
        aligner = alignment.TextAligner(
            output_geometry_format=alignment.OutputGeometryFormat.POLYGON,
            overwrite_existing_geometry=True,
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=_AllCandidateSelector(),
        )

        result = aligner.align_data(
            _page("Rome"),
            {
                "title": "Rome",
                "title_polygon": [[1, 2], [4, 2], [1, 2]],
            },
        )

        self.assertEqual(
            _output(aligner, result)["title_polygon"],
            [[0, 0], [9, 0], [9, 10], [0, 10], [0, 0]],
        )
        self.assertEqual(result.matched_count, 1)


class PolygonRenderingTests(unittest.TestCase):
    def test_polygon_points_are_scaled_and_clamped(self) -> None:
        polygon = alignment.Polygon(
            ((-1, 1), (10, 1), (10, 20), (-1, 1))
        )

        scaled = alignment.PillowAlignmentRenderer._scaled_polygon(
            polygon,
            scale_x=2,
            scale_y=3,
            image_width=15,
            image_height=30,
        )

        self.assertEqual(
            scaled,
            [(0, 3), (14, 3), (14, 29), (0, 3)],
        )

    def test_renderer_draws_polygon_outline_instead_of_rectangle(self) -> None:
        renderer = alignment.PillowAlignmentRenderer(line_width=4)
        draw = mock.Mock()
        polygon = alignment.Polygon(
            ((0, 0), (10, 0), (10, 10), (0, 0))
        )

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

    def test_renderer_keeps_rectangle_path_for_bbox(self) -> None:
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


class FuzzyConfigurationTests(unittest.TestCase):
    def test_boundary_zero_is_valid_and_does_not_warn_about_a_cliff(self) -> None:
        with self.assertNoLogs(candidate_generation.logger, logging.WARNING):
            config = alignment.FuzzyCandidateConfig(query_length_boundary=0)
        self.assertEqual(config.query_length_boundary, 0)

    def test_inconsistent_boundary_tolerances_warn(self) -> None:
        with self.assertLogs(candidate_generation.logger, logging.WARNING) as captured:
            alignment.FuzzyCandidateConfig(
                query_length_boundary=6,
                max_cer_at_or_above_boundary=0.15,
                max_edit_distance_below_boundary=1,
            )
        self.assertIn("boundary cliff", "\n".join(captured.output))


class FuzzyCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = alignment.TextNormalizationPipeline.from_optional_names(
            ("lowercase",)
        )
        self.distance_patch = mock.patch.object(
            candidate_generation,
            "_load_levenshtein_distance",
            return_value=_distance,
        )
        self.distance_patch.start()
        self.addCleanup(self.distance_patch.stop)

    def _generate(
        self,
        query: str,
        page: alignment.ALTOPage,
        config: alignment.FuzzyCandidateConfig,
    ) -> tuple[alignment.AlignmentCandidate, ...]:
        index = alignment.ALTOTextIndex(page, self.normalizer)
        generator = alignment.AnchoredFuzzyTextCandidateGenerator(config)
        return generator.generate((_value(0, query),), index)

    def test_short_query_uses_absolute_edit_distance_without_trigrams(self) -> None:
        candidates = self._generate(
            "rome",
            _page("reme"),
            alignment.FuzzyCandidateConfig(max_word_delta=0, max_start_shift=0),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].edit_distance, 1)
        self.assertEqual(candidates[0].source, "fuzzy-length-fallback")

    def test_boundary_zero_routes_one_character_query_through_cer(self) -> None:
        candidates = self._generate(
            "a",
            _page("b"),
            alignment.FuzzyCandidateConfig(
                query_length_boundary=0,
                max_cer_at_or_above_boundary=1.0,
                max_word_delta=0,
                max_start_shift=0,
            ),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].cer_int, alignment.CER_SCALE)

    def test_query_at_boundary_uses_cer(self) -> None:
        accepted = self._generate(
            "abcdef",
            _page("abcxef"),
            alignment.FuzzyCandidateConfig(
                query_length_boundary=6,
                max_cer_at_or_above_boundary=0.20,
                max_word_delta=0,
                max_start_shift=0,
            ),
        )
        with self.assertLogs(candidate_generation.logger, logging.WARNING):
            rejected_config = alignment.FuzzyCandidateConfig(
                query_length_boundary=6,
                max_cer_at_or_above_boundary=0.15,
                max_word_delta=0,
                max_start_shift=0,
            )
        rejected = self._generate(
            "abcdef",
            _page("abcxef"),
            rejected_config,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, ())

    def test_candidate_cap_prefers_distinct_occurrences(self) -> None:
        candidates = self._generate(
            "rome",
            _page("reme", "noise", "reme"),
            alignment.FuzzyCandidateConfig(
                max_candidates_per_value=2,
                max_word_delta=0,
                max_start_shift=0,
            ),
        )
        self.assertEqual(
            [(candidate.start_word, candidate.end_word) for candidate in candidates],
            [(0, 0), (2, 2)],
        )

    def test_anchored_multiword_candidate_records_quality(self) -> None:
        candidates = self._generate(
            "library of reme",
            _page("Library", "of", "Rome"),
            alignment.FuzzyCandidateConfig(),
        )
        full_span = next(
            candidate
            for candidate in candidates
            if (candidate.start_word, candidate.end_word) == (0, 2)
        )
        self.assertEqual(full_span.edit_distance, 1)
        self.assertEqual(full_span.quality_chars, len("library of reme") - 1)
        self.assertEqual(full_span.source, "fuzzy-anchor")

    def test_cross_block_search_is_used_only_as_a_fallback(self) -> None:
        candidates = self._generate(
            "library of reme",
            _page("Library", "of", "Rome", block_indexes=(0, 0, 1)),
            alignment.FuzzyCandidateConfig(),
        )
        full_span = next(
            candidate
            for candidate in candidates
            if (candidate.start_word, candidate.end_word) == (0, 2)
        )
        self.assertEqual(full_span.source, "fuzzy-anchor-cross-block")

    def test_combined_generator_keeps_normalized_exact_candidate(self) -> None:
        index = alignment.ALTOTextIndex(_page("Rome"), self.normalizer)
        candidates = _combined_generator().generate(
            (_value(0, "rome"),),
            index,
        )
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].exact)
        self.assertEqual(candidates[0].edit_distance, 0)

    def test_long_fuzzy_phrase_has_more_quality_than_short_exact_substring(self) -> None:
        index = alignment.ALTOTextIndex(
            _page("Library", "of", "Rome"),
            self.normalizer,
        )
        candidates = _combined_generator().generate(
            (
                _value(0, "rome"),
                _value(1, "library of reme"),
            ),
            index,
        )
        short_exact = next(
            candidate
            for candidate in candidates
            if candidate.region_id == 0 and candidate.exact
        )
        long_fuzzy = next(
            candidate
            for candidate in candidates
            if candidate.region_id == 1
            and (candidate.start_word, candidate.end_word) == (0, 2)
        )
        self.assertGreater(long_fuzzy.quality_chars, short_exact.quality_chars)
        self.assertIn(short_exact.start_word, long_fuzzy.word_indexes)

    def test_unanchored_large_page_search_remains_word_window_bounded(self) -> None:
        distance_calls = 0

        def counted_distance(left: str, right: str) -> int:
            nonlocal distance_calls
            distance_calls += 1
            return _distance(left, right)

        page_word_count = 2_000
        with mock.patch.object(
            candidate_generation,
            "_load_levenshtein_distance",
            return_value=counted_distance,
        ):
            candidates = self._generate(
                "abcdef",
                _page(*(("uvwxyz",) * page_word_count)),
                alignment.FuzzyCandidateConfig(),
            )

        maximum_windows = page_word_count * (
            2 * alignment.FuzzyCandidateConfig().max_word_delta + 1
        )
        self.assertLessEqual(distance_calls, maximum_windows)
        self.assertLessEqual(
            len(candidates),
            alignment.FuzzyCandidateConfig().max_candidates_per_value,
        )


class OrderedAlignmentCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = alignment.TextNormalizationPipeline.from_optional_names(
            ("lowercase",)
        )
        dependency_patch = mock.patch.object(
            ordered_candidate_generation,
            "_load_levenshtein_functions",
            return_value=(_distance, _editops),
        )
        dependency_patch.start()
        self.addCleanup(dependency_patch.stop)

    def _generate(
        self,
        regions: tuple[alignment.AlignmentRegion, ...],
        page: alignment.ALTOPage,
        config: alignment.OrderedAlignmentCandidateConfig | None = None,
    ) -> tuple[alignment.AlignmentCandidate, ...]:
        generator = alignment.OrderedAlignmentCandidateGenerator(config)
        index = alignment.ALTOTextIndex(page, self.normalizer)
        return generator.generate(regions, index)

    def test_one_global_alignment_maps_values_in_reading_order(self) -> None:
        candidates = self._generate(
            (
                _value(0, "First"),
                _value(1, "Second"),
                _value(2, "Third"),
            ),
            _page("FIRST", "inserted", "SECOND", "THIRD"),
        )

        self.assertEqual(
            [
                (
                    candidate.region_id,
                    candidate.start_word,
                    candidate.end_word,
                )
                for candidate in candidates
            ],
            [(0, 0, 0), (1, 2, 2), (2, 3, 3)],
        )
        self.assertTrue(all(candidate.exact for candidate in candidates))
        self.assertTrue(
            all(
                candidate.source == "ordered-global-alignment"
                for candidate in candidates
            )
        )

    def test_repeated_values_are_resolved_by_reading_order(self) -> None:
        candidates = self._generate(
            (_value(0, "Rome"), _value(1, "Rome")),
            _page("ROME", "ROME"),
        )

        self.assertEqual(
            [
                (candidate.region_id, candidate.start_word)
                for candidate in candidates
            ],
            [(0, 0), (1, 1)],
        )

    def test_per_value_threshold_rejects_bad_part_without_losing_later_match(
        self,
    ) -> None:
        candidates = self._generate(
            (_value(0, "abc"), _value(1, "Rome")),
            _page("123", "ROME"),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].region_id, 1)
        self.assertEqual(candidates[0].start_word, 1)

    def test_fuzzy_value_uses_the_shared_boundary_tolerances(self) -> None:
        candidates = self._generate(
            (_value(0, "Reme"),),
            _page("ROME"),
        )
        rejected = self._generate(
            (_value(0, "Reme"),),
            _page("ROME"),
            alignment.OrderedAlignmentCandidateConfig(
                max_edit_distance_below_boundary=0
            ),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].edit_distance, 1)
        self.assertFalse(candidates[0].exact)
        self.assertEqual(rejected, ())

    def test_boundary_zero_routes_short_value_through_cer(self) -> None:
        candidates = self._generate(
            (_value(0, "a"),),
            _page("B"),
            alignment.OrderedAlignmentCandidateConfig(
                query_length_boundary=0,
                max_cer_at_or_above_boundary=1.0,
                max_edit_distance_below_boundary=0,
            ),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].cer_int, alignment.CER_SCALE)

    def test_aligner_preserves_recursive_json_order_and_normalizes_both_sides(
        self,
    ) -> None:
        normalizer = alignment.TextNormalizationPipeline.from_optional_names(
            ("lowercase", "strip-diacritics", "strip-punctuation")
        )
        aligner = alignment.TextAligner(
            normalizer=normalizer,
            candidate_generator=alignment.OrderedAlignmentCandidateGenerator(),
            candidate_selector=alignment.PassThroughCandidateSelector(),
        )

        result = aligner.align_data(
            _page("ROME", "PRAGUE"),
            {
                "nested": {
                    "cities": ["Róme!", {"name": "Prague"}],
                }
            },
        )

        self.assertEqual(result.matched_count, 2)
        output = _output(aligner, result)
        self.assertEqual(
            output["nested"]["cities_bbox"],
            [
                {"x": 0, "y": 0, "width": 9, "height": 10},
                None,
            ],
        )
        self.assertEqual(
            output["nested"]["cities"][1]["name_bbox"],
            {"x": 10, "y": 0, "width": 9, "height": 10},
        )

    def test_empty_inputs_produce_no_candidates(self) -> None:
        self.assertEqual(self._generate((), _page("ROME")), ())
        self.assertEqual(
            self._generate((_value(0, "Rome"),), _page()),
            (),
        )


class PassThroughCandidateSelectorTests(unittest.TestCase):
    def test_returns_candidates_unchanged_and_in_the_same_order(self) -> None:
        normalizer = _lowercase_normalizer()
        values = (_value(0, "rome"),)
        candidates = alignment.ExactTextCandidateGenerator().generate(
            values,
            alignment.ALTOTextIndex(_page("Rome", "Rome"), normalizer),
        )

        selected = alignment.PassThroughCandidateSelector().select(
            candidates,
            values,
        )

        self.assertIs(selected, candidates)


class CompositeCandidateGeneratorTests(unittest.TestCase):
    def test_requires_at_least_one_generator(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one generator"):
            alignment.CompositeCandidateGenerator(())

    def test_earlier_generator_wins_duplicate_span_and_ids_are_reassigned(
        self,
    ) -> None:
        normalizer = _lowercase_normalizer()
        index = alignment.ALTOTextIndex(_page("Rome"), normalizer)
        values = (_value(0, "rome"),)
        exact_candidate = alignment.ExactTextCandidateGenerator().generate(
            values,
            index,
        )[0]
        later_duplicate = replace(
            exact_candidate,
            candidate_id=17,
            exact=False,
            edit_distance=1,
            source="later",
        )
        first = _StaticCandidateGenerator((exact_candidate,))
        second = _StaticCandidateGenerator((later_duplicate,))

        candidates = alignment.CompositeCandidateGenerator(
            (first, second)
        ).generate(values, index)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].exact)
        self.assertEqual(candidates[0].candidate_id, 0)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)


class MatchingDependencyTests(unittest.TestCase):
    def test_aligner_requires_generator_and_selector(self) -> None:
        with self.assertRaises(TypeError):
            alignment.TextAligner()


class OutputTextSourceTests(unittest.TestCase):
    def _align(
        self,
        output_text_source: alignment.OutputTextSource | str | None = None,
    ):
        kwargs = {}
        if output_text_source is not None:
            kwargs["output_text_source"] = output_text_source
        aligner = alignment.TextAligner(
            candidate_generator=alignment.ExactTextCandidateGenerator(),
            candidate_selector=_FirstCandidateSelector(),
            normalizer=_lowercase_normalizer(),
            **kwargs,
        )
        return (
            aligner,
            aligner.align_data(_page("ROME"), {"title": "Rome"}),
        )

    def test_json_is_the_default_output_and_rendering_text_source(self) -> None:
        aligner, result = self._align()
        page = result.pages[0]
        region = page.regions[0]
        rendered = alignment.PillowAlignmentRenderer._render_alignments(
            page,
            aligner.render_text_source,
            aligner.render_geometry_source,
            aligner.render_geometry_format,
        )

        self.assertEqual(_output(aligner, result)["title"], "Rome")
        self.assertIs(
            aligner.output_text_source,
            alignment.OutputTextSource.JSON,
        )
        self.assertEqual(region.input_text, "Rome")
        self.assertEqual(region.alto_text, "ROME")
        self.assertEqual(
            alignment.PillowAlignmentRenderer()._build_label(
                rendered[0]
            ),
            "Rome [1.00]",
        )

    def test_alto_source_updates_output_and_rendering_text(self) -> None:
        aligner, result = self._align(alignment.OutputTextSource.ALTO)
        page = result.pages[0]
        rendered = alignment.PillowAlignmentRenderer._render_alignments(
            page,
            aligner.render_text_source,
            aligner.render_geometry_source,
            aligner.render_geometry_format,
        )

        self.assertEqual(_output(aligner, result)["title"], "ROME")
        self.assertIs(
            aligner.output_text_source,
            alignment.OutputTextSource.ALTO,
        )
        self.assertEqual(page.regions[0].alto_text, "ROME")
        self.assertEqual(
            alignment.PillowAlignmentRenderer()._build_label(
                rendered[0]
            ),
            "ROME [1.00]",
        )


@unittest.skipUnless(
    importlib.util.find_spec("ortools") is not None,
    "OR-Tools is not installed",
)
class CPSATSelectionTests(unittest.TestCase):
    def test_equal_matches_prefer_earlier_alto_start_independently_of_id(
        self,
    ) -> None:
        values = (_value(0, "Rome"),)
        generated = alignment.ExactTextCandidateGenerator().generate(
            values,
            alignment.ALTOTextIndex(
                _page("Rome", "other", "Rome"),
                _lowercase_normalizer(),
            ),
        )
        earlier, later = generated
        candidates = (
            replace(earlier, candidate_id=1),
            replace(later, candidate_id=0),
        )

        selected = alignment.CPSATCandidateSelector().select(
            candidates,
            values,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].start_word, 0)
        self.assertEqual(selected[0].candidate_id, 1)

    def test_long_near_exact_phrase_beats_short_exact_substring(self) -> None:
        normalizer = _lowercase_normalizer()
        regions = (
            _value(0, "Rome"),
            _value(1, "Library of Reme"),
        )
        index = alignment.ALTOTextIndex(
            _page("Library", "of", "Rome"),
            normalizer,
        )

        with mock.patch.object(
            candidate_generation,
            "_load_levenshtein_distance",
            return_value=_distance,
        ):
            candidates = _combined_generator().generate(
                regions,
                index,
            )
        selected = alignment.CPSATCandidateSelector().select(
            candidates,
            regions,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].region_id, 1)
        self.assertEqual((selected[0].start_word, selected[0].end_word), (0, 2))
        self.assertFalse(selected[0].exact)

class ArgumentParserTests(unittest.TestCase):
    def test_public_orchestrator_uses_new_name_and_module(self) -> None:
        self.assertEqual(alignment.TextAligner.__name__, "TextAligner")
        self.assertEqual(
            alignment.TextAligner.__module__,
            "text_geometry_aligner.text_aligner",
        )

    def test_specific_models_live_with_their_processing_domain(self) -> None:
        self.assertEqual(
            alignment.ALTOWord.__module__,
            "text_geometry_aligner.alto_io.reader",
        )
        self.assertEqual(
            alignment.ALTOPage.__module__,
            "text_geometry_aligner.alto_io.reader",
        )
        self.assertFalse(hasattr(alignment, "OCRWord"))
        self.assertEqual(
            alignment.AlignmentCandidate.__module__,
            "text_geometry_aligner.text_matching.candidate",
        )
        self.assertEqual(
            alignment.ALTOWordSpan.__module__,
            "text_geometry_aligner.alto_processing.text_index",
        )
        self.assertFalse(hasattr(alignment, "OCRWordSpan"))

    def test_io_and_processing_apis_have_no_compatibility_aliases(self) -> None:
        self.assertEqual(
            alignment.ALTOReader.__module__,
            "text_geometry_aligner.alto_io.reader",
        )
        self.assertEqual(
            alignment.ALTOTextIndex.__module__,
            "text_geometry_aligner.alto_processing.text_index",
        )
        self.assertEqual(
            alignment.JSONReader.__module__,
            "text_geometry_aligner.json_io.reader",
        )
        self.assertEqual(
            alignment.JSONWriter.__module__,
            "text_geometry_aligner.json_io.writer",
        )
        self.assertEqual(
            alignment.JSONTextExtractor.__module__,
            "text_geometry_aligner.json_processing.text_extractor",
        )
        self.assertEqual(
            alignment.AlignmentJSONExporter.__module__,
            "text_geometry_aligner.json_processing.alignment_exporter",
        )
        self.assertFalse(hasattr(alignment, "JSONGeometryMerger"))
        self.assertFalse(hasattr(alignment, "JSONTextMerger"))
        self.assertFalse(hasattr(alignment, "ALTOParser"))
        self.assertFalse(hasattr(alignment, "JSONValueWriter"))
        self.assertFalse(hasattr(alignment, "AlignmentInputNormalizer"))
        self.assertFalse(hasattr(alignment, "PreparedTextRegion"))
        self.assertFalse(hasattr(alignment, "StrictTextNormalizer"))
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.preprocessing"
            )
        )
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.alto"
            )
        )
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.json_io.extractor"
            )
        )
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.json_processing.extractor"
            )
        )
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.json_processing.merger"
            )
        )
        self.assertFalse(hasattr(alignment, "JSONValueExtractor"))
        self.assertFalse(hasattr(alignment, "JSONAlignmentMerger"))
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.text_matching.candidates"
            )
        )
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.text_matching.selection"
            )
        )
        self.assertEqual(
            alignment.GeometryBuilder.__module__,
            "text_geometry_aligner.geometry_building.base",
        )
        self.assertEqual(
            alignment.TextBuilder.__module__,
            "text_geometry_aligner.text_building.base",
        )
        self.assertEqual(
            alignment.UnionBoundingBoxGeometryBuilder.__module__,
            (
                "text_geometry_aligner.geometry_building."
                "union_bounding_box"
            ),
        )
        self.assertEqual(
            alignment.OrthogonalPolygonGeometryBuilder.__module__,
            (
                "text_geometry_aligner.geometry_building."
                "orthogonal_polygon"
            ),
        )
        self.assertEqual(
            alignment.SpaceSeparatedTextBuilder.__module__,
            (
                "text_geometry_aligner.text_building."
                "space_separated"
            ),
        )
        self.assertIn(
            ".text_matching.",
            alignment.ExactTextCandidateGenerator.__module__,
        )
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.geometry"
            )
        )
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.geometry_builder"
            )
        )
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.text_builder"
            )
        )
        self.assertFalse(hasattr(alignment, "create_geometry_builder"))
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.matching"
            )
        )
        self.assertFalse(hasattr(alignment, "CandidateGenerationStrategy"))
        self.assertFalse(hasattr(alignment, "HybridTextCandidateGenerator"))

    def test_no_non_cp_sat_fallback_selectors_remain(self) -> None:
        self.assertFalse(hasattr(alignment, "AutoCandidateSelector"))
        self.assertFalse(hasattr(alignment, "BranchAndBoundCandidateSelector"))

    def test_matching_strategy_arguments_are_exposed(self) -> None:
        parser = aligner_module.build_argument_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--text-normalizer", option_strings)
        self.assertIn("--output-text-source", option_strings)
        self.assertIn("--output-alto-text-format", option_strings)
        self.assertIn("--output-alto-geometry-format", option_strings)
        self.assertNotIn("--output-geometry-format", option_strings)
        self.assertIn("--candidate-generator", option_strings)
        self.assertIn("--candidate-selector", option_strings)
        self.assertIn("--overwrite-existing-geometry", option_strings)
        self.assertIn("--fuzzy-query-length-boundary", option_strings)
        self.assertIn("--fuzzy-max-cer-at-or-above-boundary", option_strings)
        self.assertIn(
            "--fuzzy-max-edit-distance-below-boundary",
            option_strings,
        )

        required_arguments = [
            "--alto-dir",
            "alto",
            "--input-dir",
            "json",
            "--json-output-dir",
            "output",
        ]
        default_args = parser.parse_args(required_arguments)
        alto_args = parser.parse_args(
            [*required_arguments, "--output-text-source", "alto"]
        )
        polygon_args = parser.parse_args(
            [
                *required_arguments,
                "--output-alto-geometry-format",
                "polygon",
            ]
        )
        exact_args = parser.parse_args(
            [*required_arguments, "--candidate-generator", "exact"]
        )
        ordered_args = parser.parse_args(
            [
                *required_arguments,
                "--candidate-generator",
                "ordered-alignment",
                "--candidate-selector",
                "pass-through",
            ]
        )
        stacked_normalizer_args = parser.parse_args(
            [
                *required_arguments,
                "--text-normalizer",
                "strip-diacritics",
                "--text-normalizer",
                "lowercase",
            ]
        )
        overwrite_args = parser.parse_args(
            [*required_arguments, "--overwrite-existing-geometry"]
        )
        self.assertEqual(default_args.output_text_source, "json")
        self.assertEqual(
            default_args.output_alto_text_format,
            "space-separated",
        )
        self.assertEqual(
            default_args.output_alto_geometry_format,
            "bbox",
        )
        self.assertEqual(default_args.candidate_generator, "combined")
        self.assertEqual(default_args.candidate_selector, "cp-sat")
        self.assertIsNone(default_args.text_normalizer)
        self.assertIsNone(default_args.geometry_suffix)
        self.assertFalse(default_args.overwrite_existing_geometry)
        self.assertEqual(alto_args.output_text_source, "alto")
        self.assertEqual(
            polygon_args.output_alto_geometry_format,
            "polygon",
        )
        self.assertEqual(exact_args.candidate_generator, "exact")
        self.assertEqual(
            ordered_args.candidate_generator,
            "ordered-alignment",
        )
        self.assertEqual(ordered_args.candidate_selector, "pass-through")
        self.assertEqual(
            stacked_normalizer_args.text_normalizer,
            ["strip-diacritics", "lowercase"],
        )
        self.assertTrue(overwrite_args.overwrite_existing_geometry)

        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            parser.parse_args(
                [*required_arguments, "--candidate-generator", "fuzzy"]
            )

    def test_cli_factories_build_selected_matching_strategies(self) -> None:
        parser = aligner_module.build_argument_parser()
        required_arguments = [
            "--alto-dir",
            "alto",
            "--input-dir",
            "json",
            "--json-output-dir",
            "output",
        ]
        exact_args = parser.parse_args(
            [
                *required_arguments,
                "--candidate-generator",
                "exact",
                "--fuzzy-query-length-boundary",
                "-1",
            ]
        )
        combined_args = parser.parse_args(required_arguments)
        ordered_args = parser.parse_args(
            [
                *required_arguments,
                "--candidate-generator",
                "ordered-alignment",
                "--candidate-selector",
                "pass-through",
            ]
        )

        exact = aligner_module._build_candidate_generator(exact_args)
        combined = aligner_module._build_candidate_generator(combined_args)
        ordered = aligner_module._build_candidate_generator(ordered_args)
        cp_sat = aligner_module._build_candidate_selector(combined_args)
        pass_through = aligner_module._build_candidate_selector(ordered_args)
        text_builder = base_aligner_module._build_text_builder(
            combined_args.output_alto_text_format
        )
        geometry_builder = base_aligner_module._build_geometry_builder(
            alignment.OutputGeometryFormat(
                combined_args.output_alto_geometry_format
            )
        )

        self.assertIsInstance(exact, alignment.ExactTextCandidateGenerator)
        self.assertIsInstance(combined, alignment.CompositeCandidateGenerator)
        self.assertIsInstance(
            ordered,
            alignment.OrderedAlignmentCandidateGenerator,
        )
        self.assertIsInstance(cp_sat, alignment.CPSATCandidateSelector)
        self.assertIsInstance(
            pass_through,
            alignment.PassThroughCandidateSelector,
        )
        self.assertIsInstance(
            text_builder,
            alignment.SpaceSeparatedTextBuilder,
        )
        self.assertIsInstance(
            geometry_builder,
            alignment.UnionBoundingBoxGeometryBuilder,
        )
        self.assertEqual(
            [type(generator) for generator in combined.generators],
            [
                alignment.ExactTextCandidateGenerator,
                alignment.AnchoredFuzzyTextCandidateGenerator,
            ],
        )


if __name__ == "__main__":
    unittest.main()
