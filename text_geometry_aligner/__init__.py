"""Bidirectional text/geometry alignment against word-level ALTO OCR."""

from .io_alto import ALTOPage, ALTOReader, ALTOWord
from .base_aligner import BaseAligner
from .geometry_building import (
    GeometryBuilder,
    OrthogonalPolygonGeometryBuilder,
    UnionBoundingBoxGeometryBuilder,
    validate_geometry_format,
)
from .geometry_matching import (
    AllOverThresholdWordAssigner,
    BoundingBoxOverlapCalculator,
    GeometryOverlapCalculator,
    GeometryOverlapStrategy,
    GeometryWordOverlap,
    GeometryWordAssigner,
    GreatestCoverageWordAssigner,
    RegionWordAssignment,
    ShapelyOverlapCalculator,
    WordAssignmentStrategy,
    create_overlap_calculator,
    create_word_assigner,
)
from .io_json import (
    AlignmentJSONWriter,
    JSONGeometryReader,
    JSONTextReader,
)
from .io_label_studio import LabelStudioReader, LabelStudioWriter
from .label_mapping import LabelMapper
from .models import (
    AlignmentDocument,
    AlignmentMode,
    AlignmentPage,
    AlignmentRegion,
    AlignmentWord,
    BoundingBox,
    InputFormat,
    JSONPath,
    OutputGeometry,
    OutputGeometryFormat,
    OutputGeometrySource,
    OutputTextSource,
    Point,
    Polygon,
    ScalarText,
)
from .normalization import (
    DiacriticStrippingTextNormalizer,
    LowercaseTextNormalizer,
    PunctuationStrippingTextNormalizer,
    TextNormalizationPipeline,
    TextNormalizer,
    UnicodeTextNormalizer,
    WhitespaceTextNormalizer,
)
from .rendering import AlignmentRenderer, PillowAlignmentRenderer
from .text_building import SpaceSeparatedTextBuilder, TextBuilder
from .text_matching import (
    ALTOTextIndex,
    ALTOWordSpan,
    CER_SCALE,
    SIMILARITY_SCALE,
    AlignmentCandidate,
)
from .text_matching.candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
    OrderedAlignmentCandidateConfig,
    OrderedAlignmentCandidateGenerator,
)
from .text_matching.candidate_selectors import (
    CPSATCandidateSelector,
    CandidateSelector,
    PassThroughCandidateSelector,
)
from .io_yolo import YOLODetection, YOLOReader


def __getattr__(name: str):
    if name == "TextAligner":
        from .text_aligner import TextAligner

        return TextAligner
    if name == "GeometryAligner":
        from .geometry_aligner import GeometryAligner

        return GeometryAligner
    raise AttributeError(name)


__all__ = [
    "ALTOPage",
    "ALTOReader",
    "ALTOTextIndex",
    "ALTOWord",
    "AlignmentCandidate",
    "AlignmentDocument",
    "AlignmentJSONWriter",
    "AlignmentMode",
    "AlignmentPage",
    "AlignmentRegion",
    "AlignmentRenderer",
    "AlignmentWord",
    "AllOverThresholdWordAssigner",
    "AnchoredFuzzyTextCandidateGenerator",
    "BaseAligner",
    "BoundingBox",
    "BoundingBoxOverlapCalculator",
    "CER_SCALE",
    "CPSATCandidateSelector",
    "CandidateGenerator",
    "CandidateSelector",
    "CompositeCandidateGenerator",
    "DiacriticStrippingTextNormalizer",
    "ExactTextCandidateGenerator",
    "FuzzyCandidateConfig",
    "GeometryAligner",
    "GeometryBuilder",
    "GeometryOverlapCalculator",
    "GeometryOverlapStrategy",
    "GeometryWordOverlap",
    "GeometryWordAssigner",
    "GreatestCoverageWordAssigner",
    "InputFormat",
    "JSONGeometryReader",
    "JSONPath",
    "JSONTextReader",
    "LabelStudioReader",
    "LabelStudioWriter",
    "LabelMapper",
    "LowercaseTextNormalizer",
    "ALTOWordSpan",
    "OrthogonalPolygonGeometryBuilder",
    "OutputGeometry",
    "OutputGeometryFormat",
    "OutputGeometrySource",
    "OutputTextSource",
    "OrderedAlignmentCandidateConfig",
    "OrderedAlignmentCandidateGenerator",
    "PassThroughCandidateSelector",
    "PillowAlignmentRenderer",
    "Point",
    "Polygon",
    "PunctuationStrippingTextNormalizer",
    "RegionWordAssignment",
    "SIMILARITY_SCALE",
    "ScalarText",
    "ShapelyOverlapCalculator",
    "SpaceSeparatedTextBuilder",
    "TextAligner",
    "TextBuilder",
    "TextNormalizationPipeline",
    "TextNormalizer",
    "UnicodeTextNormalizer",
    "UnionBoundingBoxGeometryBuilder",
    "validate_geometry_format",
    "WhitespaceTextNormalizer",
    "WordAssignmentStrategy",
    "YOLODetection",
    "YOLOReader",
    "create_overlap_calculator",
    "create_word_assigner",
]
