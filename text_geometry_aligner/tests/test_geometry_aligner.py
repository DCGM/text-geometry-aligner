"""Tests for geometry-to-text alignment."""

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock

import text_geometry_aligner as alignment
from text_geometry_aligner import (
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
) -> alignment.OCRWord:
    return alignment.OCRWord(
        index=index,
        text=text,
        bbox=alignment.BoundingBox(x, y, width, height),
        line_index=0,
        block_index=0,
    )


def _page(*words: alignment.OCRWord) -> alignment.ALTOPage:
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


class JSONGeometryExtractorTests(unittest.TestCase):
    def test_extracts_nested_bbox_lists_and_retains_parallel_paths(self) -> None:
        regions = alignment.JSONGeometryExtractor().extract(
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
        )

        self.assertEqual(
            [region.geometry_path for region in regions],
            [
                ("groups", 0, "publisher_bbox", 0),
                ("groups", 0, "publisher_bbox", 2),
            ],
        )
        self.assertEqual(
            [region.text_path for region in regions],
            [
                ("groups", 0, "publisher", 0),
                ("groups", 0, "publisher", 2),
            ],
        )
        self.assertTrue(
            all(
                isinstance(region.geometry, alignment.BoundingBox)
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

        regions = alignment.JSONGeometryExtractor().extract(data)
        overwritten = alignment.JSONGeometryExtractor(
            overwrite_existing_text=True
        ).extract(data)

        self.assertEqual(regions, ())
        self.assertEqual(len(overwritten), 2)

    def test_protected_destination_skips_geometry_parsing(self) -> None:
        regions = alignment.JSONGeometryExtractor().extract(
            {
                "title": "existing",
                "title_bbox": {"not": "a bbox"},
            }
        )

        self.assertEqual(regions, ())

    def test_custom_suffix_can_extract_polygon(self) -> None:
        regions = alignment.JSONGeometryExtractor(
            geometry_suffix="_shape"
        ).extract(
            {
                "title_shape": [
                    [0, 0],
                    [10, 0],
                    [10, 10],
                    [0, 10],
                    [0, 0],
                ]
            }
        )

        self.assertEqual(regions[0].text_path, ("title",))
        self.assertIsInstance(regions[0].geometry, alignment.Polygon)

    def test_invalid_geometry_reports_its_json_path(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\$\.title_bbox"):
            alignment.JSONGeometryExtractor().extract(
                {"title_bbox": {"x": 0, "y": 0, "width": 10}}
            )


class JSONTextMergerTests(unittest.TestCase):
    def test_missing_list_destination_mirrors_trailing_null_shape(self) -> None:
        merger = alignment.JSONTextMerger(geometry_suffix="_bbox")
        output = merger.create_output(
            {"publisher_bbox": [_bbox(0), None]}
        )

        self.assertEqual(output["publisher"], [None, None])

        merger.set_text(output, ("publisher", 0), "FIRST")
        self.assertEqual(output["publisher"], ["FIRST", None])

    def test_incompatible_existing_container_is_not_replaced(self) -> None:
        merger = alignment.JSONTextMerger(overwrite_existing_text=True)
        output = merger.create_output(
            {"publisher": "existing", "publisher_bbox": [_bbox(0)]}
        )

        with self.assertRaisesRegex(TypeError, "Incompatible container"):
            merger.set_text(output, ("publisher", 0), "FIRST")

    def test_merger_itself_preserves_an_existing_destination(self) -> None:
        merger = alignment.JSONTextMerger()
        output = merger.create_output(
            {"title": None, "title_bbox": _bbox(0)}
        )

        merger.set_text(output, ("title",), "ALTO")

        self.assertIsNone(output["title"])


class BoundingBoxOverlapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = alignment.BoundingBoxOverlapCalculator()
        self.word = _word(0, "WORD", 0)

    def _coverage(
        self,
        region_bbox: alignment.BoundingBox,
        threshold: float,
    ) -> tuple[alignment.WordCoverage, ...]:
        region = alignment.JSONGeometryRegion(
            region_id=0,
            geometry_path=("title_bbox",),
            text_path=("title",),
            geometry=region_bbox,
        )
        return self.calculator.calculate(
            (region,),
            (self.word,),
            threshold,
        )

    def test_threshold_is_inclusive_and_uses_word_area(self) -> None:
        at_threshold = self._coverage(
            alignment.BoundingBox(0, 0, 6.5, 10),
            0.65,
        )
        below_threshold = self._coverage(
            alignment.BoundingBox(0, 0, 6.49, 10),
            0.65,
        )

        self.assertAlmostEqual(at_threshold[0].coverage, 0.65)
        self.assertEqual(below_threshold, ())

    def test_boundary_contact_has_no_coverage_even_at_zero_threshold(self) -> None:
        self.assertEqual(
            self._coverage(
                alignment.BoundingBox(10, 0, 5, 10),
                0.0,
            ),
            (),
        )

    def test_polygon_requires_optional_shapely_dependency(self) -> None:
        region = alignment.JSONGeometryRegion(
            region_id=0,
            geometry_path=("title_polygon",),
            text_path=("title",),
            geometry=alignment.Polygon(
                ((0, 0), (10, 0), (10, 10), (0, 10), (0, 0))
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "requires Shapely"):
            self.calculator.calculate((region,), (self.word,), 0.65)

    def test_factory_falls_back_to_bbox_when_shapely_is_unavailable(self) -> None:
        region = alignment.JSONGeometryRegion(
            region_id=0,
            geometry_path=("title_bbox",),
            text_path=("title",),
            geometry=alignment.BoundingBox(0, 0, 10, 10),
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
        region = alignment.JSONGeometryRegion(
            region_id=0,
            geometry_path=("title_polygon",),
            text_path=("title",),
            geometry=alignment.Polygon(
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

        region = alignment.JSONGeometryRegion(
            region_id=0,
            geometry_path=("title_bbox",),
            text_path=("title",),
            geometry=alignment.BoundingBox(0, 0, 6.5, 10),
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
            alignment.JSONGeometryRegion(
                0,
                ("first_bbox",),
                ("first",),
                alignment.BoundingBox(0, 0, 20, 10),
            ),
            alignment.JSONGeometryRegion(
                1,
                ("second_bbox",),
                ("second",),
                alignment.BoundingBox(0, 0, 20, 10),
            ),
        )

    def test_greatest_coverage_wins_and_ties_use_region_order(self) -> None:
        coverages = (
            alignment.WordCoverage(0, 0, 0.7),
            alignment.WordCoverage(1, 0, 0.8),
            alignment.WordCoverage(0, 1, 0.9),
            alignment.WordCoverage(1, 1, 0.9),
        )

        assigned = alignment.GreatestCoverageWordAssigner().assign(
            self.regions,
            self.words,
            coverages,
        )

        self.assertEqual(assigned[0].word_indexes, (1,))
        self.assertEqual(assigned[0].extracted_text, "SECOND")
        self.assertEqual(assigned[1].word_indexes, (0,))
        self.assertEqual(assigned[1].extracted_text, "FIRST")

    def test_all_over_threshold_retains_shared_words(self) -> None:
        coverages = (
            alignment.WordCoverage(0, 0, 0.7),
            alignment.WordCoverage(1, 0, 0.8),
        )

        assigned = alignment.AllOverThresholdWordAssigner().assign(
            self.regions,
            self.words,
            coverages,
        )

        self.assertEqual(assigned[0].word_indexes, (0,))
        self.assertEqual(assigned[1].word_indexes, (0,))

    def test_text_builder_constructs_the_assigned_output(self) -> None:
        text_builder = mock.create_autospec(
            alignment.TextBuilder,
            instance=True,
        )
        text_builder.build.return_value = "CUSTOM TEXT"

        assigned = alignment.GreatestCoverageWordAssigner(
            text_builder=text_builder,
        ).assign(
            self.regions[:1],
            self.words,
            (
                alignment.WordCoverage(0, 1, 0.9),
                alignment.WordCoverage(0, 0, 0.8),
            ),
        )

        text_builder.build.assert_called_once_with(self.words)
        self.assertEqual(assigned[0].extracted_text, "CUSTOM TEXT")


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
        result = alignment.GeometryAligner().align_data(
            page,
            {
                "outer_bbox": _bbox(0, width=30, height=8),
                "middle_bbox": _bbox(10),
            },
        )

        self.assertEqual(result.output_data["outer"], "FIRST LAST")
        self.assertEqual(result.output_data["middle"], "MIDDLE")
        self.assertEqual(result.alignments[0].word_indexes, (0, 2))
        self.assertAlmostEqual(
            result.alignments[0].average_coverage,
            0.8,
        )

    def test_all_over_threshold_can_retain_shared_word(self) -> None:
        result = alignment.GeometryAligner(
            word_assignment_strategy="all-over-threshold"
        ).align_data(
            _page(_word(0, "WORD", 0)),
            {
                "first_bbox": _bbox(0),
                "second_bbox": _bbox(0),
            },
        )

        self.assertEqual(result.output_data["first"], "WORD")
        self.assertEqual(result.output_data["second"], "WORD")

    def test_custom_text_builder_is_used_by_default_assigner(self) -> None:
        text_builder = mock.create_autospec(
            alignment.TextBuilder,
            instance=True,
        )
        text_builder.build.return_value = "BUILT"

        result = alignment.GeometryAligner(
            text_builder=text_builder,
        ).align_data(
            _page(_word(0, "WORD", 0)),
            {"title_bbox": _bbox(0)},
        )

        self.assertEqual(result.output_data["title"], "BUILT")

    def test_custom_assigner_and_text_builder_are_mutually_exclusive(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            alignment.GeometryAligner(
                word_assigner=mock.create_autospec(
                    alignment.GeometryWordAssigner,
                    instance=True,
                ),
                text_builder=mock.create_autospec(
                    alignment.TextBuilder,
                    instance=True,
                ),
            )

    def test_unmatched_region_creates_null_and_render_score_zero(self) -> None:
        result = alignment.GeometryAligner().align_data(
            _page(_word(0, "WORD", 0)),
            {"missing_bbox": _bbox(50)},
        )

        self.assertIsNone(result.output_data["missing"])
        self.assertEqual(result.unmatched_region_ids, (0,))
        self.assertEqual(result.render_alignments[0].text, "null")
        self.assertEqual(result.render_alignments[0].score, 0.0)

    def test_existing_destination_is_skipped_before_overlap_calculation(
        self,
    ) -> None:
        calculator = mock.create_autospec(
            alignment.GeometryOverlapCalculator,
            instance=True,
        )
        result = alignment.GeometryAligner(
            overlap_calculator=calculator,
        ).align_data(
            _page(_word(0, "WORD", 0)),
            {"title": "", "title_bbox": _bbox(0)},
        )

        calculator.calculate.assert_not_called()
        self.assertEqual(result.regions, ())
        self.assertEqual(result.output_data["title"], "")

    def test_overwrite_processes_existing_destination(self) -> None:
        result = alignment.GeometryAligner(
            overwrite_existing_text=True
        ).align_data(
            _page(_word(0, "ALTO", 0)),
            {"title": "JSON", "title_bbox": _bbox(0)},
        )

        self.assertEqual(result.output_data["title"], "ALTO")

    def test_overwrite_replaces_existing_text_with_null_when_unmatched(
        self,
    ) -> None:
        result = alignment.GeometryAligner(
            overwrite_existing_text=True
        ).align_data(
            _page(_word(0, "ALTO", 0)),
            {"title": "JSON", "title_bbox": _bbox(50)},
        )

        self.assertIsNone(result.output_data["title"])

    def test_geometry_render_label_uses_average_coverage(self) -> None:
        result = alignment.GeometryAligner().align_data(
            _page(_word(0, "WORD", 0)),
            {"title_bbox": _bbox(0, width=8)},
        )

        self.assertEqual(
            alignment.PillowAlignmentRenderer()._build_label(
                result.render_alignments[0]
            ),
            "WORD [0.80]",
        )


class GeometryArgumentParserTests(unittest.TestCase):
    def test_defaults_and_strategies_are_exposed(self) -> None:
        parser = geometry_aligner_module.build_argument_parser()
        required = [
            "--alto-dir",
            "alto",
            "--json-input-dir",
            "json",
            "--json-output-dir",
            "output",
        ]

        defaults = parser.parse_args(required)
        all_matches = parser.parse_args(
            [
                *required,
                "--geometry-suffix",
                "_polygon",
                "--minimum-word-coverage",
                "0.4",
                "--word-assignment-strategy",
                "all-over-threshold",
                "--text-builder",
                "space-separated",
                "--overwrite-existing-text",
            ]
        )

        self.assertEqual(defaults.geometry_suffix, "_bbox")
        self.assertEqual(defaults.minimum_word_coverage, 0.65)
        self.assertEqual(
            defaults.word_assignment_strategy,
            "greatest-coverage",
        )
        self.assertEqual(defaults.text_builder, "space-separated")
        self.assertFalse(defaults.overwrite_existing_text)
        self.assertEqual(all_matches.geometry_suffix, "_polygon")
        self.assertEqual(all_matches.minimum_word_coverage, 0.4)
        self.assertEqual(
            all_matches.word_assignment_strategy,
            "all-over-threshold",
        )
        self.assertEqual(all_matches.text_builder, "space-separated")
        self.assertTrue(all_matches.overwrite_existing_text)

    def test_invalid_strategy_is_rejected(self) -> None:
        parser = geometry_aligner_module.build_argument_parser()
        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            parser.parse_args(
                [
                    "--alto-dir",
                    "alto",
                    "--json-input-dir",
                    "json",
                    "--json-output-dir",
                    "output",
                    "--word-assignment-strategy",
                    "unknown",
                ]
            )

    def test_invalid_text_builder_is_rejected(self) -> None:
        parser = geometry_aligner_module.build_argument_parser()
        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            parser.parse_args(
                [
                    "--alto-dir",
                    "alto",
                    "--json-input-dir",
                    "json",
                    "--json-output-dir",
                    "output",
                    "--text-builder",
                    "unknown",
                ]
            )

    def test_cli_text_builder_resolution(self) -> None:
        builder = geometry_aligner_module._build_text_builder(
            "space-separated"
        )

        self.assertIsInstance(
            builder,
            alignment.SpaceSeparatedTextBuilder,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported text builder"):
            geometry_aligner_module._build_text_builder("unknown")


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
