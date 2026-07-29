"""Text candidate generation, global selection, and diagnostics."""

from .candidate import (
    CER_SCALE,
    SIMILARITY_SCALE,
    AlignmentCandidate,
)
from .candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
    OrderedAlignmentCandidateConfig,
    OrderedAlignmentCandidateGenerator,
)
from .candidate_selectors import (
    CPSATCandidateSelector,
    CandidateSelector,
    PassThroughCandidateSelector,
)

__all__ = [
    "AlignmentCandidate", "AnchoredFuzzyTextCandidateGenerator",
    "CER_SCALE", "CandidateGenerator", "CandidateSelector",
    "CompositeCandidateGenerator", "CPSATCandidateSelector",
    "ExactTextCandidateGenerator", "FuzzyCandidateConfig",
    "OrderedAlignmentCandidateConfig", "OrderedAlignmentCandidateGenerator",
    "PassThroughCandidateSelector", "SIMILARITY_SCALE",
]
