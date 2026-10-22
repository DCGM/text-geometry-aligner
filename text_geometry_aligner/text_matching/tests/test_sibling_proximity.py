"""Tests for re-picking equally good candidates by sibling proximity."""

from dataclasses import replace
from pathlib import Path

import text_geometry_aligner as alignment
from text_geometry_aligner.text_matching.sibling_proximity import (
    refine_by_sibling_proximity,
)


def _words(*texts: str) -> tuple[alignment.ALTOWord, ...]:
    return alignment.ALTOPage(
        source_path=Path("test.xml"),
        words=tuple(
            alignment.ALTOWord(
                index=index,
                text=text,
                bbox=alignment.BoundingBox(index * 10, 0, 9, 10),
                line_index=0,
                block_index=0,
            )
            for index, text in enumerate(texts)
        ),
    ).words


def _candidate(
    candidate_id: int,
    region_id: int,
    path: tuple,
    start_word: int,
    end_word: int,
    *,
    similarity_int: int = 1_000_000,
    exact: bool = True,
) -> alignment.AlignmentCandidate:
    return alignment.AlignmentCandidate(
        candidate_id=candidate_id,
        region_id=region_id,
        json_text_path=path,
        start_word=start_word,
        end_word=end_word,
        start_char=0,
        end_char=0,
        query_text="",
        matched_text="",
        normalized_query_text="",
        normalized_matched_text="",
        exact=exact,
        edit_distance=0,
        cer_int=0,
        similarity_int=similarity_int,
        query_length=1,
        quality_chars=1,
        source="exact",
    )


PLACE = ("publication", 0, "placeTerm", 0)
PUBLISHER = ("publication", 0, "publisher", 0)
DATE = ("publication", 0, "dateIssued")


def test_stray_word_span_gives_its_word_to_the_sibling_that_needs_it() -> None:
    words = _words("1934", "x", "x", "x", "Praha", "1934", "Otto")
    place = _candidate(0, 0, PLACE, 4, 4)
    padded_publisher = _candidate(1, 1, PUBLISHER, 5, 6, exact=False)
    publisher = _candidate(2, 1, PUBLISHER, 6, 6, exact=False)
    early_date = _candidate(3, 2, DATE, 0, 0)
    imprint_date = _candidate(4, 2, DATE, 5, 5)

    refined = refine_by_sibling_proximity(
        (place, padded_publisher, publisher, early_date, imprint_date),
        (place, padded_publisher, early_date),
        words,
    )

    assert {(c.region_id, c.start_word, c.end_word) for c in refined} == {
        (0, 4, 4),
        (1, 6, 6),
        (2, 5, 5),
    }


def test_values_without_siblings_and_owned_words_are_left_alone() -> None:
    words = _words("Rome", "x", "Rome", "Livy")
    title = _candidate(0, 0, ("title",), 0, 0)
    later_title = _candidate(1, 0, ("title",), 2, 2)
    name = _candidate(2, 1, ("agents", 0, "name"), 3, 3)
    worse_name = replace(name, candidate_id=3, start_word=0, end_word=0, similarity_int=1)

    refined = refine_by_sibling_proximity(
        (title, later_title, name, worse_name),
        (title, name),
        words,
    )

    assert refined == (title, name)
