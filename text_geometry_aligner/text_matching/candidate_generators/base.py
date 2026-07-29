"""Base interface for text candidate generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ...alto_processing import ALTOTextIndex
from ...models import AlignmentRegion
from ..candidate import AlignmentCandidate


class CandidateGenerator(ABC):
    """Generate deterministic alignment candidates with unique IDs."""

    @abstractmethod
    def generate(
        self,
        regions: Sequence[AlignmentRegion],
        alto_index: ALTOTextIndex,
    ) -> tuple[AlignmentCandidate, ...]:
        raise NotImplementedError
