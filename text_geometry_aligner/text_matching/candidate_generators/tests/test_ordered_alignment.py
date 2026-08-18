"""Tests for the ordered-alignment candidate generator."""

from pathlib import Path
from unittest import mock

import pytest

import text_geometry_aligner as alignment
from text_geometry_aligner.text_matching.candidate_generators import (
    ordered_alignment as ordered_candidate_generation,
)


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


@pytest.fixture(autouse=True)
def _patch_levenshtein_functions():
    with mock.patch.object(
        ordered_candidate_generation,
        "_load_levenshtein_functions",
        return_value=(_distance, _editops),
    ):
        yield


def _generate(
    regions: tuple[alignment.AlignmentRegion, ...],
    page: alignment.ALTOPage,
    normalizer: alignment.TextNormalizer,
    config: alignment.OrderedAlignmentCandidateConfig | None = None,
) -> tuple[alignment.AlignmentCandidate, ...]:
    generator = alignment.OrderedAlignmentCandidateGenerator(config)
    index = alignment.ALTOTextIndex(page, normalizer)
    return generator.generate(regions, index)


def test_one_global_alignment_maps_values_in_reading_order(
    lowercase_normalizer,
) -> None:
    candidates = _generate(
        (
            _value(0, "First"),
            _value(1, "Second"),
            _value(2, "Third"),
        ),
        _page("FIRST", "inserted", "SECOND", "THIRD"),
        lowercase_normalizer,
    )

    assert [
        (candidate.region_id, candidate.start_word, candidate.end_word)
        for candidate in candidates
    ] == [(0, 0, 0), (1, 2, 2), (2, 3, 3)]
    assert all(candidate.exact for candidate in candidates)
    assert all(
        candidate.source == "ordered-global-alignment" for candidate in candidates
    )


def test_repeated_values_are_resolved_by_reading_order(
    lowercase_normalizer,
) -> None:
    candidates = _generate(
        (_value(0, "Rome"), _value(1, "Rome")),
        _page("ROME", "ROME"),
        lowercase_normalizer,
    )

    assert [
        (candidate.region_id, candidate.start_word) for candidate in candidates
    ] == [(0, 0), (1, 1)]


def test_per_value_threshold_rejects_bad_part_without_losing_later_match(
    lowercase_normalizer,
) -> None:
    candidates = _generate(
        (_value(0, "abc"), _value(1, "Rome")),
        _page("123", "ROME"),
        lowercase_normalizer,
    )

    assert len(candidates) == 1
    assert candidates[0].region_id == 1
    assert candidates[0].start_word == 1


def test_fuzzy_value_uses_the_shared_boundary_tolerances(
    lowercase_normalizer,
) -> None:
    candidates = _generate(
        (_value(0, "Reme"),),
        _page("ROME"),
        lowercase_normalizer,
    )
    rejected = _generate(
        (_value(0, "Reme"),),
        _page("ROME"),
        lowercase_normalizer,
        alignment.OrderedAlignmentCandidateConfig(
            max_edit_distance_below_boundary=0
        ),
    )

    assert len(candidates) == 1
    assert candidates[0].edit_distance == 1
    assert not candidates[0].exact
    assert rejected == ()


def test_boundary_zero_routes_short_value_through_cer(
    lowercase_normalizer,
) -> None:
    candidates = _generate(
        (_value(0, "a"),),
        _page("B"),
        lowercase_normalizer,
        alignment.OrderedAlignmentCandidateConfig(
            query_length_boundary=0,
            max_cer_at_or_above_boundary=1.0,
            max_edit_distance_below_boundary=0,
        ),
    )

    assert len(candidates) == 1
    assert candidates[0].cer_int == alignment.CER_SCALE


def test_aligner_preserves_recursive_json_order_and_normalizes_both_sides(
    alignment_output,
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

    assert result.matched_count == 2
    output = alignment_output(aligner, result)
    assert output["nested"]["cities_bbox"] == [
        {"x": 0, "y": 0, "width": 9, "height": 10},
        None,
    ]
    assert output["nested"]["cities"][1]["name_bbox"] == {
        "x": 10,
        "y": 0,
        "width": 9,
        "height": 10,
    }


def test_empty_inputs_produce_no_candidates(lowercase_normalizer) -> None:
    assert _generate((), _page("ROME"), lowercase_normalizer) == ()
    assert _generate((_value(0, "Rome"),), _page(), lowercase_normalizer) == ()
