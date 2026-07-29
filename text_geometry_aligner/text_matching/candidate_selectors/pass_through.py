"""Pass-through text candidate selection."""

from __future__ import annotations

from typing import Sequence

from ...models import AlignmentRegion
from ..candidate import AlignmentCandidate
from .base import CandidateSelector


class PassThroughCandidateSelector(CandidateSelector):
    """Return generated candidates unchanged."""

    def select(
        self,
        candidates: Sequence[AlignmentCandidate],
        regions: Sequence[AlignmentRegion],
    ) -> tuple[AlignmentCandidate, ...]:
        return tuple(candidates)
