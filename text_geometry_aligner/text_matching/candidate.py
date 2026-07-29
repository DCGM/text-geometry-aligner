"""Data carried between text candidate generators and selectors."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import JSONPath

SIMILARITY_SCALE = 1_000_000
CER_SCALE = 1_000_000


@dataclass(frozen=True)
class AlignmentCandidate:
    """Possible match between one alignment region and ALTO words."""

    candidate_id: int
    region_id: int
    json_text_path: JSONPath | None
    start_word: int
    end_word: int  # Inclusive.
    start_char: int
    end_char: int  # Exclusive.
    query_text: str
    matched_text: str
    normalized_query_text: str
    normalized_matched_text: str
    exact: bool
    edit_distance: int
    cer_int: int
    similarity_int: int
    query_length: int
    quality_chars: int
    source: str

    @property
    def word_indexes(self) -> range:
        return range(self.start_word, self.end_word + 1)
