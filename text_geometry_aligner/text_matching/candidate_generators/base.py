"""Base interface for text candidate generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..alto_text_index import ALTOTextIndex
from ...models import AlignmentRegion
from ..candidate import AlignmentCandidate
from ...io_alto import ALTOPage


class CandidateGenerator(ABC):
    """Generate deterministic alignment candidates with unique IDs."""

    def prepare_page(self, page: ALTOPage) -> ALTOPage:
        """Return the OCR view used by matching; default preserves file order."""
        return page

    @abstractmethod
    def generate(
        self,
        regions: Sequence[AlignmentRegion],
        alto_index: ALTOTextIndex,
    ) -> tuple[AlignmentCandidate, ...]:
        raise NotImplementedError
