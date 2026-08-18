"""Tests for the anchored fuzzy candidate generator."""

import logging
from pathlib import Path
from unittest import mock

import pytest

import text_geometry_aligner as alignment
from text_geometry_aligner.text_matching.candidate_generators import (
    anchored_fuzzy as candidate_generation,
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


def _combined_generator(
    fuzzy_config: alignment.FuzzyCandidateConfig | None = None,
) -> alignment.CompositeCandidateGenerator:
    return alignment.CompositeCandidateGenerator(
        (
            alignment.ExactTextCandidateGenerator(),
            alignment.AnchoredFuzzyTextCandidateGenerator(fuzzy_config),
        )
    )


def test_boundary_zero_is_valid_and_does_not_warn_about_a_cliff(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=candidate_generation.logger.name):
        config = alignment.FuzzyCandidateConfig(query_length_boundary=0)
    assert not any(
        record.levelno >= logging.WARNING for record in caplog.records
    )
    assert config.query_length_boundary == 0


def test_inconsistent_boundary_tolerances_warn(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=candidate_generation.logger.name):
        alignment.FuzzyCandidateConfig(
            query_length_boundary=6,
            max_cer_at_or_above_boundary=0.15,
            max_edit_distance_below_boundary=1,
        )
    assert "boundary cliff" in "\n".join(
        record.getMessage() for record in caplog.records
    )


@pytest.fixture(autouse=True)
def _patch_levenshtein_distance():
    with mock.patch.object(
        candidate_generation,
        "_load_levenshtein_distance",
        return_value=_distance,
    ):
        yield


def _generate(
    query: str,
    page: alignment.ALTOPage,
    config: alignment.FuzzyCandidateConfig,
    normalizer: alignment.TextNormalizer,
) -> tuple[alignment.AlignmentCandidate, ...]:
    index = alignment.ALTOTextIndex(page, normalizer)
    generator = alignment.AnchoredFuzzyTextCandidateGenerator(config)
    return generator.generate((_value(0, query),), index)


def test_short_query_uses_absolute_edit_distance_without_trigrams(
    lowercase_normalizer,
) -> None:
    candidates = _generate(
        "rome",
        _page("reme"),
        alignment.FuzzyCandidateConfig(max_word_delta=0, max_start_shift=0),
        lowercase_normalizer,
    )
    assert len(candidates) == 1
    assert candidates[0].edit_distance == 1
    assert candidates[0].source == "fuzzy-length-fallback"


def test_boundary_zero_routes_one_character_query_through_cer(
    lowercase_normalizer,
) -> None:
    candidates = _generate(
        "a",
        _page("b"),
        alignment.FuzzyCandidateConfig(
            query_length_boundary=0,
            max_cer_at_or_above_boundary=1.0,
            max_word_delta=0,
            max_start_shift=0,
        ),
        lowercase_normalizer,
    )
    assert len(candidates) == 1
    assert candidates[0].cer_int == alignment.CER_SCALE


def test_query_at_boundary_uses_cer(lowercase_normalizer, caplog) -> None:
    accepted = _generate(
        "abcdef",
        _page("abcxef"),
        alignment.FuzzyCandidateConfig(
            query_length_boundary=6,
            max_cer_at_or_above_boundary=0.20,
            max_word_delta=0,
            max_start_shift=0,
        ),
        lowercase_normalizer,
    )
    with caplog.at_level(logging.WARNING, logger=candidate_generation.logger.name):
        rejected_config = alignment.FuzzyCandidateConfig(
            query_length_boundary=6,
            max_cer_at_or_above_boundary=0.15,
            max_word_delta=0,
            max_start_shift=0,
        )
    assert caplog.records
    rejected = _generate(
        "abcdef",
        _page("abcxef"),
        rejected_config,
        lowercase_normalizer,
    )
    assert len(accepted) == 1
    assert rejected == ()


def test_candidate_cap_prefers_distinct_occurrences(lowercase_normalizer) -> None:
    candidates = _generate(
        "rome",
        _page("reme", "noise", "reme"),
        alignment.FuzzyCandidateConfig(
            max_candidates_per_value=2,
            max_word_delta=0,
            max_start_shift=0,
        ),
        lowercase_normalizer,
    )
    assert [
        (candidate.start_word, candidate.end_word) for candidate in candidates
    ] == [(0, 0), (2, 2)]


def test_anchored_multiword_candidate_records_quality(lowercase_normalizer) -> None:
    candidates = _generate(
        "library of reme",
        _page("Library", "of", "Rome"),
        alignment.FuzzyCandidateConfig(),
        lowercase_normalizer,
    )
    full_span = next(
        candidate
        for candidate in candidates
        if (candidate.start_word, candidate.end_word) == (0, 2)
    )
    assert full_span.edit_distance == 1
    assert full_span.quality_chars == len("library of reme") - 1
    assert full_span.source == "fuzzy-anchor"


def test_cross_block_search_is_used_only_as_a_fallback(lowercase_normalizer) -> None:
    candidates = _generate(
        "library of reme",
        _page("Library", "of", "Rome", block_indexes=(0, 0, 1)),
        alignment.FuzzyCandidateConfig(),
        lowercase_normalizer,
    )
    full_span = next(
        candidate
        for candidate in candidates
        if (candidate.start_word, candidate.end_word) == (0, 2)
    )
    assert full_span.source == "fuzzy-anchor-cross-block"


def test_combined_generator_keeps_normalized_exact_candidate(
    lowercase_normalizer,
) -> None:
    index = alignment.ALTOTextIndex(_page("Rome"), lowercase_normalizer)
    candidates = _combined_generator().generate(
        (_value(0, "rome"),),
        index,
    )
    assert len(candidates) == 1
    assert candidates[0].exact
    assert candidates[0].edit_distance == 0


def test_long_fuzzy_phrase_has_more_quality_than_short_exact_substring(
    lowercase_normalizer,
) -> None:
    index = alignment.ALTOTextIndex(
        _page("Library", "of", "Rome"),
        lowercase_normalizer,
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
    assert long_fuzzy.quality_chars > short_exact.quality_chars
    assert short_exact.start_word in long_fuzzy.word_indexes


def test_unanchored_large_page_search_remains_word_window_bounded(
    lowercase_normalizer,
) -> None:
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
        candidates = _generate(
            "abcdef",
            _page(*(("uvwxyz",) * page_word_count)),
            alignment.FuzzyCandidateConfig(),
            lowercase_normalizer,
        )

    maximum_windows = page_word_count * (
        2 * alignment.FuzzyCandidateConfig().max_word_delta + 1
    )
    assert distance_calls <= maximum_windows
    assert len(candidates) <= alignment.FuzzyCandidateConfig().max_candidates_per_value
