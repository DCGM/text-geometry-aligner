"""Base interface for text candidate selectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ...models import AlignmentRegion
from ..candidate import AlignmentCandidate


class CandidateSelector(ABC):
    """Select a globally compatible subset of alignment candidates."""

    @abstractmethod
    def select(
        self,
        candidates: Sequence[AlignmentCandidate],
        regions: Sequence[AlignmentRegion],
    ) -> tuple[AlignmentCandidate, ...]:
        raise NotImplementedError
