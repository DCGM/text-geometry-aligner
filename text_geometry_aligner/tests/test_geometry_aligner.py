"""Tests for geometry-to-text alignment."""

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock

import text_geometry_aligner as alignment
from text_geometry_aligner import (
    base_aligner as base_aligner_module,
    geometry_aligner as geometry_aligner_module,
)
from text_geometry_aligner.geometry_matching import (
    overlap as overlap_module,
)


def _word(
    index: int,
    text: str,
    x: float,
    y: float = 0,
    width: float = 10,
    height: float = 10,
) -> alignment.ALTOWord:
    return alignment.ALTOWord(
        index=index,
        text=text,
        bbox=alignment.BoundingBox(x, y, width, height),
        line_index=0,
        block_index=0,
    )


def _page(*words: alignment.ALTOWord) -> alignment.ALTOPage:
    return alignment.ALTOPage(
        source_path=Path("page.xml"),
        words=tuple(words),
        width=100,
        height=100,
    )


def _bbox(
    x: float,
    y: float = 0,
    width: float = 10,
    height: float = 10,
) -> dict[str, float]:
    return {"x": x, "y": y, "width": width, "height": height}


def _output(aligner, document):
    return aligner.json_writer.to_data(document.pages[0])


class JSONGeometryReaderTests(unittest.TestCase):
    def test_extracts_nested_bbox_lists_and_retains_parallel_paths(self) -> None:
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

        self.assertEqual(
            [region.json_geometry_path for region in regions],
            [
                ("groups", 0, "publisher_bbox", 0),
                ("groups", 0, "publisher_bbox", 2),
            ],
        )
        self.assertEqual(
            [region.json_text_path for region in regions],
            [
                ("groups", 0, "publisher", 0),
                ("groups", 0, "publisher", 2),
            ],
        )
        self.assertTrue(
            all(
                isinstance(region.input_geometry, alignment.BoundingBox)
                for region in regions
            )
        )

    def test_existing_destination_is_skipped_even_when_null_or_empty(self) -> None:
        data = {
            "nullText": None,
            "nullText_bbox": _bbox(0),
            "emptyText": "",
            "emptyText_bbox": _bbox(10),
        }

        regions = (
            alignment.JSONGeometryReader().from_data(data).regions
        )
        overwritten = alignment.JSONGeometryReader(
            overwrite_existing_text=True
        ).from_data(data).regions

        self.assertEqual(regions, [])
        self.assertEqual(len(overwritten), 2)

    def test_protected_destination_skips_geometry_parsing(self) -> None:
        regions = alignment.JSONGeometryReader().from_data(
            {
                "title": "existing",
                "title_bbox": {"not": "a bbox"},
            }
        ).regions

        self.assertEqual(regions, [])

    def test_custom_suffix_can_extract_polygon(self) -> None:
        regions = alignment.JSONGeometryReader(
            geometry_suffix="_shape"
        ).from_data(
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

        self.assertEqual(regions[0].json_text_path, ("title",))
        self.assertIsInstance(regions[0].input_geometry, alignment.Polygon)

    def test_invalid_geometry_reports_its_json_path(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\$\.title_bbox"):
            alignment.JSONGeometryReader().from_data(
                {"title_bbox": {"x": 0, "y": 0, "width": 10}}
            )


class AlignmentJSONWriterTests(unittest.TestCase):
    @staticmethod
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

    @staticmethod
    def _export(page: alignment.AlignmentPage):
        return alignment.AlignmentJSONWriter(
            alignment_mode=alignment.AlignmentMode.GEOMETRY,
            geometry_suffix="_bbox",
            output_geometry_format=alignment.OutputGeometryFormat.BBOX,
        ).to_data(page)

    def test_missing_list_destination_mirrors_trailing_null_shape(self) -> None:
        page = self._page(
            {"publisher_bbox": [_bbox(0), None]},
        )
        page.regions[0].alto_text = "FIRST"
        output = self._export(page)
        self.assertEqual(output["publisher"], ["FIRST", None])

    def test_incompatible_existing_container_is_not_replaced(self) -> None:
        page = self._page(
            {"publisher": "existing", "publisher_bbox": [_bbox(0)]},
            overwrite_existing_text=True,
        )
        page.regions[0].alto_text = "FIRST"
        with self.assertRaisesRegex(TypeError, "Incompatible container"):
            self._export(page)

    def test_page_extraction_preserves_an_existing_destination(self) -> None:
        page = self._page(
            {"title": None, "title_bbox": _bbox(0)}
        )
        output = self._export(page)
        self.assertEqual(page.regions, [])
        self.assertIsNone(output["title"])


class BoundingBoxOverlapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = alignment.BoundingBoxOverlapCalculator()
        self.word = _word(0, "WORD", 0)

    def _overlap(
        self,
        region_bbox: alignment.BoundingBox,
        threshold: float,
        strategy: alignment.GeometryOverlapStrategy = (
            alignment.GeometryOverlapStrategy.BIDIRECTIONAL_CONTAINMENT
        ),
    ) -> tuple[alignment.GeometryWordOverlap, ...]:
        region = alignment.AlignmentRegion(
            region_id=0,
            label="title",
            json_geometry_path=("title_bbox",),
            json_text_path=("title",),
            input_geometry=region_bbox,
        )
        return self.calculator.calculate(
            (region,),
            (self.word,),
            threshold,
            strategy,
        )

    def test_threshold_is_inclusive_and_uses_word_area(self) -> None:
        at_threshold = self._overlap(
            alignment.BoundingBox(0, 0, 6.5, 10),
            0.65,
            alignment.GeometryOverlapStrategy.WORD_COVERAGE,
        )
        below_threshold = self._overlap(
            alignment.BoundingBox(0, 0, 6.49, 10),
            0.65,
            alignment.GeometryOverlapStrategy.WORD_COVERAGE,
        )

        self.assertAlmostEqual(at_threshold[0].word_coverage, 0.65)
        self.assertAlmostEqual(
            at_threshold[0].input_geometry_coverage,
            1.0,
        )
        self.assertAlmostEqual(at_threshold[0].overlap_score, 0.65)
        self.assertEqual(below_threshold, ())

    def test_bidirectional_containment_accepts_tight_detection(self) -> None:
        overlaps = self._overlap(
            alignment.BoundingBox(2, 2, 4, 4),
            0.65,
        )

        self.assertEqual(len(overlaps), 1)
        self.assertAlmostEqual(overlaps[0].word_coverage, 0.16)
        self.assertAlmostEqual(
            overlaps[0].input_geometry_coverage,
            1.0,
        )
        self.assertAlmostEqual(overlaps[0].overlap_score, 1.0)

    def test_word_coverage_can_reject_tight_detection(self) -> None:
        self.assertEqual(
            self._overlap(
                alignment.BoundingBox(2, 2, 4, 4),
                0.65,
                alignment.GeometryOverlapStrategy.WORD_COVERAGE,
            ),
            (),
        )

    def test_bidirectional_containment_accepts_word_inside_region(self) -> None:
        overlaps = self._overlap(
            alignment.BoundingBox(-5, -5, 20, 20),
            0.65,
        )

        self.assertEqual(len(overlaps), 1)
        self.assertAlmostEqual(overlaps[0].word_coverage, 1.0)
        self.assertAlmostEqual(
            overlaps[0].input_geometry_coverage,
            0.25,
        )
        self.assertAlmostEqual(overlaps[0].overlap_score, 1.0)

    def test_insufficient_overlap_in_both_directions_is_rejected(self) -> None:
        self.assertEqual(
            self._overlap(
                alignment.BoundingBox(8, 8, 10, 10),
                0.65,
            ),
            (),
        )

    def test_zero_area_input_geometry_has_no_overlap(self) -> None:
        self.assertEqual(
            self._overlap(
                alignment.BoundingBox(0, 0, 0, 10),
                0.0,
            ),
            (),
        )

    def test_boundary_contact_has_no_coverage_even_at_zero_threshold(self) -> None:
        self.assertEqual(
            self._overlap(
                alignment.BoundingBox(10, 0, 5, 10),
                0.0,
            ),
            (),
        )

    def test_polygon_requires_optional_shapely_dependency(self) -> None:
        region = alignment.AlignmentRegion(
            region_id=0,
            label="title",
            json_geometry_path=("title_polygon",),
            json_text_path=("title",),
            input_geometry=alignment.Polygon(
                ((0, 0), (10, 0), (10, 10), (0, 10), (0, 0))
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "requires Shapely"):
            self.calculator.calculate((region,), (self.word,), 0.65)

    def test_factory_falls_back_to_bbox_when_shapely_is_unavailable(self) -> None:
        region = alignment.AlignmentRegion(
            region_id=0,
            label="title",
            json_geometry_path=("title_bbox",),
            json_text_path=("title",),
            input_geometry=alignment.BoundingBox(0, 0, 10, 10),
        )
        with mock.patch.object(
            overlap_module,
            "_load_shapely_box_factory",
            side_effect=ImportError,
        ):
            calculator = alignment.create_overlap_calculator((region,))

        self.assertIsInstance(
            calculator,
            alignment.BoundingBoxOverlapCalculator,
        )

    def test_factory_reports_missing_shapely_for_polygon(self) -> None:
        region = alignment.AlignmentRegion(
            region_id=0,
            label="title",
            json_geometry_path=("title_polygon",),
            json_text_path=("title",),
            input_geometry=alignment.Polygon(
                ((0, 0), (10, 0), (10, 10), (0, 10), (0, 0))
            ),
        )
        with (
            mock.patch.object(
                overlap_module,
                "_load_shapely_box_factory",
                side_effect=ImportError,
            ),
            self.assertRaisesRegex(RuntimeError, "requires Shapely"),
        ):
            alignment.create_overlap_calculator((region,))

    def test_polygon_uses_bidirectional_containment(self) -> None:
        class FakeShape:
            def __init__(self, points):
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                self.bounds = (min(xs), min(ys), max(xs), max(ys))
                self.area = (
                    (self.bounds[2] - self.bounds[0])
                    * (self.bounds[3] - self.bounds[1])
                )
                self.is_empty = False
                self.is_valid = True

            def intersection(self, other):
                x_min = max(self.bounds[0], other.bounds[0])
                y_min = max(self.bounds[1], other.bounds[1])
                x_max = min(self.bounds[2], other.bounds[2])
                y_max = min(self.bounds[3], other.bounds[3])
                area = max(0, x_max - x_min) * max(0, y_max - y_min)
                return type("Intersection", (), {"area": area})()

        def fake_box(x_min, y_min, x_max, y_max):
            return FakeShape(
                (
                    (x_min, y_min),
                    (x_max, y_min),
                    (x_max, y_max),
                    (x_min, y_max),
                )
            )

        region = alignment.AlignmentRegion(
            region_id=0,
            label="title",
            input_geometry=alignment.Polygon(
                ((2, 2), (6, 2), (6, 6), (2, 6), (2, 2))
            ),
        )

        overlaps = alignment.ShapelyOverlapCalculator(
            geometry_factory=fake_box,
            polygon_class=FakeShape,
        ).calculate(
            (region,),
            (self.word,),
            0.65,
        )

        self.assertEqual(len(overlaps), 1)
        self.assertAlmostEqual(overlaps[0].word_coverage, 0.16)
        self.assertAlmostEqual(
            overlaps[0].input_geometry_coverage,
            1.0,
        )
        self.assertAlmostEqual(overlaps[0].overlap_score, 1.0)

    def test_shapely_and_fallback_backends_agree_for_bboxes(self) -> None:
        class FakeShape:
            def __init__(self, x_min, y_min, x_max, y_max):
                self.bounds = (x_min, y_min, x_max, y_max)

            def intersection(self, other):
                x_min = max(self.bounds[0], other.bounds[0])
                y_min = max(self.bounds[1], other.bounds[1])
                x_max = min(self.bounds[2], other.bounds[2])
                y_max = min(self.bounds[3], other.bounds[3])
                area = max(0, x_max - x_min) * max(0, y_max - y_min)
                return type("Intersection", (), {"area": area})()

        def fake_box(x_min, y_min, x_max, y_max):
            return FakeShape(x_min, y_min, x_max, y_max)

        region = alignment.AlignmentRegion(
            region_id=0,
            label="title",
            json_geometry_path=("title_bbox",),
            json_text_path=("title",),
            input_geometry=alignment.BoundingBox(0, 0, 6.5, 10),
        )
        fallback = alignment.BoundingBoxOverlapCalculator().calculate(
            (region,),
            (self.word,),
            0.65,
        )
        shapely = alignment.ShapelyOverlapCalculator(
            geometry_factory=fake_box,
            polygon_class=object(),
        ).calculate(
            (region,),
            (self.word,),
            0.65,
        )

        self.assertEqual(fallback, shapely)


class WordAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.words = (
            _word(0, "FIRST", 0),
            _word(1, "SECOND", 10),
        )
        self.regions = (
            alignment.AlignmentRegion(
                region_id=0,
                label="first",
                json_geometry_path=("first_bbox",),
                json_text_path=("first",),
                input_geometry=alignment.BoundingBox(0, 0, 20, 10),
            ),
            alignment.AlignmentRegion(
                region_id=1,
                label="second",
                json_geometry_path=("second_bbox",),
                json_text_path=("second",),
                input_geometry=alignment.BoundingBox(0, 0, 20, 10),
            ),
        )

    def test_greatest_coverage_wins_and_ties_use_region_order(self) -> None:
        overlaps = (
            alignment.GeometryWordOverlap(0, 0, 0.7, 0.5, 0.7),
            alignment.GeometryWordOverlap(1, 0, 0.8, 0.4, 0.8),
            alignment.GeometryWordOverlap(0, 1, 0.9, 0.3, 0.9),
            alignment.GeometryWordOverlap(1, 1, 0.9, 0.3, 0.9),
        )

        assigned = alignment.GreatestCoverageWordAssigner().assign(
            self.regions,
            self.words,
            overlaps,
        )

        self.assertEqual(
            tuple(item.word_index for item in assigned[0].overlaps),
            (1,),
        )
        self.assertEqual(
            tuple(item.overlap_score for item in assigned[0].overlaps),
            (0.9,),
        )
        self.assertEqual(
            tuple(item.word_index for item in assigned[1].overlaps),
            (0,),
        )
        self.assertEqual(
            tuple(item.overlap_score for item in assigned[1].overlaps),
            (0.8,),
        )

    def test_equal_scores_prefer_directional_coverages_then_region(self) -> None:
        overlaps = (
            alignment.GeometryWordOverlap(0, 0, 0.4, 1.0, 1.0),
            alignment.GeometryWordOverlap(1, 0, 1.0, 0.4, 1.0),
            alignment.GeometryWordOverlap(0, 1, 1.0, 0.6, 1.0),
            alignment.GeometryWordOverlap(1, 1, 1.0, 0.6, 1.0),
        )

        assigned = alignment.GreatestCoverageWordAssigner().assign(
            self.regions,
            self.words,
            overlaps,
        )

        self.assertEqual(
            tuple(item.word_index for item in assigned[0].overlaps),
            (1,),
        )
        self.assertEqual(
            tuple(item.word_index for item in assigned[1].overlaps),
            (0,),
        )

    def test_all_over_threshold_retains_shared_words(self) -> None:
        overlaps = (
            alignment.GeometryWordOverlap(0, 0, 0.7, 0.5, 0.7),
            alignment.GeometryWordOverlap(1, 0, 0.8, 0.4, 0.8),
        )

        assigned = alignment.AllOverThresholdWordAssigner().assign(
            self.regions,
            self.words,
            overlaps,
        )

        self.assertEqual(
            tuple(item.word_index for item in assigned[0].overlaps),
            (0,),
        )
        self.assertEqual(
            tuple(item.word_index for item in assigned[1].overlaps),
            (0,),
        )

    def test_aligner_uses_builders_after_assignment(self) -> None:
        text_builder = mock.create_autospec(
            alignment.TextBuilder,
            instance=True,
        )
        text_builder.build.return_value = "CUSTOM TEXT"
        geometry_builder = mock.create_autospec(
            alignment.GeometryBuilder,
            instance=True,
        )
        geometry_builder.build.return_value = alignment.BoundingBox(
            0, 0, 20, 10
        )
        aligner = alignment.GeometryAligner(
            text_builder=text_builder,
            geometry_builder=geometry_builder,
        )
        result = aligner.align_data(
            _page(*self.words),
            {"title_bbox": _bbox(2, width=17)},
        )
        region = result.pages[0].regions[0]

        text_builder.build.assert_called_once_with(self.words)
        geometry_builder.build.assert_called_once_with(self.words)
        self.assertEqual(region.alto_text, "CUSTOM TEXT")
        self.assertEqual(
            region.alto_geometry,
            alignment.BoundingBox(0, 0, 20, 10),
        )
        self.assertEqual(
            tuple(word.word_coverage for word in region.words or ()),
            (0.8, 0.9),
        )
        self.assertEqual(
            tuple(word.overlap_score for word in region.words or ()),
            (0.8, 0.9),
        )
        self.assertAlmostEqual(region.alignment_score or 0.0, 0.85)


class TextBuilderTests(unittest.TestCase):
    def test_base_builder_requires_a_build_implementation(self) -> None:
        with self.assertRaises(TypeError):
            alignment.TextBuilder()

    def test_space_separated_builder_joins_words_in_input_order(self) -> None:
        builder = alignment.SpaceSeparatedTextBuilder()

        self.assertEqual(builder.build((_word(1, "ONE", 0),)), "ONE")
        self.assertEqual(
            builder.build((_word(1, "ONE", 0), _word(2, "TWO", 10))),
            "ONE TWO",
        )
        self.assertIsNone(builder.build(()))


class GeometryAlignerTests(unittest.TestCase):
    def test_extracts_noncontiguous_words_in_alto_order(self) -> None:
        page = _page(
            _word(0, "FIRST", 0),
            _word(1, "MIDDLE", 10),
            _word(2, "LAST", 20),
        )
        aligner = alignment.GeometryAligner()
        result = aligner.align_data(
            page,
            {
                "outer_bbox": _bbox(0, width=30, height=8),
                "middle_bbox": _bbox(10),
            },
        )

        output = _output(aligner, result)
        outer, middle = result.pages[0].regions
        self.assertEqual(output["outer"], "FIRST LAST")
        self.assertEqual(output["middle"], "MIDDLE")
        self.assertEqual(
            tuple(word.word_index for word in outer.words or ()),
            (0, 2),
        )
        self.assertAlmostEqual(
            outer.alignment_score or 0.0,
            0.8,
        )
        self.assertEqual(middle.alto_text, "MIDDLE")

    def test_all_over_threshold_can_retain_shared_word(self) -> None:
        aligner = alignment.GeometryAligner(
            word_assignment_strategy="all-over-threshold"
        )
        result = aligner.align_data(
            _page(_word(0, "WORD", 0)),
            {
                "first_bbox": _bbox(0),
                "second_bbox": _bbox(0),
            },
        )

        output = _output(aligner, result)
        self.assertEqual(output["first"], "WORD")
        self.assertEqual(output["second"], "WORD")

    def test_custom_text_builder_is_used_by_default_assigner(self) -> None:
        text_builder = mock.create_autospec(
            alignment.TextBuilder,
            instance=True,
        )
        text_builder.build.return_value = "BUILT"

        aligner = alignment.GeometryAligner(
            text_builder=text_builder,
        )
        result = aligner.align_data(
            _page(_word(0, "WORD", 0)),
            {"title_bbox": _bbox(0)},
        )

        self.assertEqual(_output(aligner, result)["title"], "BUILT")

    def test_custom_assigner_and_text_builder_are_independent_components(
        self,
    ) -> None:
        assigner = mock.create_autospec(
            alignment.GeometryWordAssigner,
            instance=True,
        )
        text_builder = mock.create_autospec(
            alignment.TextBuilder,
            instance=True,
        )
        aligner = alignment.GeometryAligner(
            word_assigner=assigner,
            text_builder=text_builder,
        )

        self.assertIs(aligner.word_assigner, assigner)
        self.assertIs(aligner.text_builder, text_builder)

    def test_unmatched_region_creates_null_and_render_score_zero(self) -> None:
        aligner = alignment.GeometryAligner()
        result = aligner.align_data(
            _page(_word(0, "WORD", 0)),
            {"missing_bbox": _bbox(50)},
        )

        page = result.pages[0]
        self.assertIsNone(_output(aligner, result)["missing"])
        self.assertEqual(page.unmatched_count, 1)
        rendered = alignment.PillowAlignmentRenderer._render_alignments(
            page,
            alignment.OutputTextSource.ALTO,
            alignment.OutputGeometrySource.INPUT,
            alignment.OutputGeometryFormat.BBOX,
        )
        self.assertEqual(rendered[0].text, "null")
        self.assertEqual(rendered[0].score, 0.0)

    def test_existing_destination_is_skipped_before_overlap_calculation(
        self,
    ) -> None:
        calculator = mock.create_autospec(
            alignment.GeometryOverlapCalculator,
            instance=True,
        )
        aligner = alignment.GeometryAligner(
            overlap_calculator=calculator,
        )
        result = aligner.align_data(
            _page(_word(0, "WORD", 0)),
            {"title": "", "title_bbox": _bbox(0)},
        )

        calculator.calculate.assert_not_called()
        self.assertEqual(result.pages[0].regions, [])
        self.assertEqual(_output(aligner, result)["title"], "")

    def test_overwrite_processes_existing_destination(self) -> None:
        aligner = alignment.GeometryAligner(
            overwrite_existing_text=True
        )
        result = aligner.align_data(
            _page(_word(0, "ALTO", 0)),
            {"title": "JSON", "title_bbox": _bbox(0)},
        )

        self.assertEqual(_output(aligner, result)["title"], "ALTO")

    def test_overwrite_replaces_existing_text_with_null_when_unmatched(
        self,
    ) -> None:
        aligner = alignment.GeometryAligner(
            overwrite_existing_text=True
        )
        result = aligner.align_data(
            _page(_word(0, "ALTO", 0)),
            {"title": "JSON", "title_bbox": _bbox(50)},
        )

        self.assertIsNone(_output(aligner, result)["title"])

    def test_geometry_render_label_uses_average_overlap_score(self) -> None:
        aligner = alignment.GeometryAligner()
        result = aligner.align_data(
            _page(_word(0, "WORD", 0)),
            {"title_bbox": _bbox(0, width=8, height=20)},
        )

        rendered = alignment.PillowAlignmentRenderer._render_alignments(
            result.pages[0],
            alignment.OutputTextSource.ALTO,
            alignment.OutputGeometrySource.INPUT,
            alignment.OutputGeometryFormat.BBOX,
        )
        self.assertEqual(
            alignment.PillowAlignmentRenderer()._build_label(rendered[0]),
            "WORD [0.80]",
        )


class GeometryArgumentParserTests(unittest.TestCase):
    def test_defaults_and_strategies_are_exposed(self) -> None:
        parser = geometry_aligner_module.build_argument_parser()
        required = [
            "--alto-dir",
            "alto",
            "--input-dir",
            "json",
            "--json-output-dir",
            "output",
        ]

        defaults = parser.parse_args(required)
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        all_matches = parser.parse_args(
            [
                *required,
                "--geometry-suffix",
                "_polygon",
                "--minimum-overlap-coverage",
                "0.4",
                "--overlap-strategy",
                "word-coverage",
                "--word-assignment-strategy",
                "all-over-threshold",
                "--output-alto-text-format",
                "space-separated",
                "--output-alto-geometry-format",
                "polygon",
                "--overwrite-existing-text",
            ]
        )

        self.assertEqual(defaults.geometry_suffix, "_bbox")
        self.assertEqual(defaults.minimum_overlap_coverage, 0.65)
        self.assertEqual(
            defaults.overlap_strategy,
            "bidirectional-containment",
        )
        self.assertEqual(
            defaults.word_assignment_strategy,
            "greatest-coverage",
        )
        self.assertEqual(
            defaults.output_alto_text_format,
            "space-separated",
        )
        self.assertEqual(
            defaults.output_alto_geometry_format,
            "bbox",
        )
        self.assertNotIn("--text-builder", option_strings)
        self.assertNotIn("--output-geometry-format", option_strings)
        self.assertFalse(defaults.overwrite_existing_text)
        self.assertEqual(all_matches.geometry_suffix, "_polygon")
        self.assertEqual(all_matches.minimum_overlap_coverage, 0.4)
        self.assertEqual(all_matches.overlap_strategy, "word-coverage")
        self.assertEqual(
            all_matches.word_assignment_strategy,
            "all-over-threshold",
        )
        self.assertEqual(
            all_matches.output_alto_text_format,
            "space-separated",
        )
        self.assertEqual(
            all_matches.output_alto_geometry_format,
            "polygon",
        )
        self.assertTrue(all_matches.overwrite_existing_text)

    def test_invalid_assignment_strategy_is_rejected(self) -> None:
        parser = geometry_aligner_module.build_argument_parser()
        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            parser.parse_args(
                [
                    "--alto-dir",
                    "alto",
                    "--input-dir",
                    "json",
                    "--json-output-dir",
                    "output",
                    "--word-assignment-strategy",
                    "unknown",
                ]
            )

    def test_invalid_overlap_strategy_is_rejected(self) -> None:
        parser = geometry_aligner_module.build_argument_parser()
        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            parser.parse_args(
                [
                    "--alto-dir",
                    "alto",
                    "--input-dir",
                    "json",
                    "--json-output-dir",
                    "output",
                    "--overlap-strategy",
                    "unknown",
                ]
            )

    def test_invalid_alto_text_format_is_rejected(self) -> None:
        parser = geometry_aligner_module.build_argument_parser()
        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            parser.parse_args(
                [
                    "--alto-dir",
                    "alto",
                    "--input-dir",
                    "json",
                    "--json-output-dir",
                    "output",
                    "--output-alto-text-format",
                    "unknown",
                ]
            )

    def test_cli_alto_builder_resolution(self) -> None:
        builder = base_aligner_module._build_text_builder(
            "space-separated"
        )

        self.assertIsInstance(
            builder,
            alignment.SpaceSeparatedTextBuilder,
        )
        self.assertIsInstance(
            base_aligner_module._build_geometry_builder(
                alignment.OutputGeometryFormat.POLYGON
            ),
            alignment.OrthogonalPolygonGeometryBuilder,
        )
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported output ALTO text format",
        ):
            base_aligner_module._build_text_builder("unknown")


class RefactorAPITests(unittest.TestCase):
    def test_direction_specific_aligners_replace_old_orchestrator(self) -> None:
        self.assertEqual(
            alignment.TextAligner.__module__,
            "text_geometry_aligner.text_aligner",
        )
        self.assertEqual(
            alignment.GeometryAligner.__module__,
            "text_geometry_aligner.geometry_aligner",
        )
        self.assertFalse(hasattr(alignment, "TextGeometryAligner"))
        self.assertFalse(hasattr(alignment, "AlignmentDirection"))
        self.assertIsNone(
            importlib.util.find_spec(
                "text_geometry_aligner.text_geometry_aligner"
            )
        )
