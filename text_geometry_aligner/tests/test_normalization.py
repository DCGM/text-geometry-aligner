"""Tests for text normalization."""

from dataclasses import replace
from pathlib import Path
from unittest import mock

import text_geometry_aligner as alignment


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


def test_default_pipeline_preserves_case() -> None:
    pipeline = alignment.TextNormalizationPipeline.from_optional_names()
    assert pipeline.normalize(" Straße  TEST ") == "Straße TEST"


def test_optional_normalizers_are_composable_and_ordered() -> None:
    pipeline = alignment.TextNormalizationPipeline.from_optional_names(
        ("lowercase", "strip-diacritics", "strip-punctuation")
    )
    assert pipeline.normalize(" ČESKÝ—Krumlov! ") == "cesky krumlov"


def test_alignment_retains_raw_and_normalized_text() -> None:
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

    assert region.input_text == "ŘÍM"
    assert region.input_text_normalized == "rim"
    assert region.alto_text == "ŘÍM!"
    assert region.alto_text_normalized == "rim"
    assert word.text == "ŘÍM!"
    assert word.text_normalized == "rim"
    assert page.words[0].text == "ŘÍM!"


def test_unmatched_region_retains_only_normalized_input_text() -> None:
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

    assert region.input_text == "TITLE"
    assert region.input_text_normalized == "title"
    assert region.text_alignment_candidate is None
    assert region.alto_text is None
    assert region.alto_text_normalized is None
    assert region.words is None


def test_normalization_to_empty_produces_no_candidate() -> None:
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

    assert region.input_text_normalized == ""
    assert region.alto_text_normalized is None
    assert region.words is None


def test_source_normalization_is_independent_of_candidate_snapshots() -> None:
    candidate = _candidate()
    aligner = alignment.TextAligner(
        candidate_generator=_StaticCandidateGenerator((candidate,)),
        candidate_selector=_AllCandidateSelector(),
    )

    result = aligner.align_data(_page("ALTO"), {"title": "Raw input"})
    region = result.pages[0].regions[0]

    assert region.input_text_normalized == "Raw input"
    assert region.alto_text_normalized == "ALTO"
    assert region.text_alignment_candidate is candidate
    assert (
        region.text_alignment_candidate.normalized_query_text
        == "candidate query"
    )
    assert (
        region.text_alignment_candidate.normalized_matched_text
        == "candidate match"
    )
    assert region.words[0].text_normalized == "ALTO"


def test_selected_candidate_survives_failed_text_building() -> None:
    candidate = _candidate()
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

    assert region.text_alignment_candidate is candidate
    assert region.input_text_normalized == "Raw input"
    assert region.alto_text_normalized == "ALTO"
    assert region.alto_text is None
    assert region.words is None
    assert not region.matched
