"""Bidirectional text/geometry alignment against word-level ALTO OCR."""

from .alto_io import ALTOPage, ALTOReader, ALTOWord
from .alto_processing import ALTOTextIndex, ALTOWordSpan
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
    GeometryWordAssigner,
    GreatestCoverageWordAssigner,
    RegionWordAssignment,
    ShapelyOverlapCalculator,
    WordAssignmentStrategy,
    WordCoverage,
    create_overlap_calculator,
    create_word_assigner,
)
from .json_io import JSONReader, JSONWriter
from .json_processing import (
    AlignmentJSONExporter,
    JSONGeometryExtractor,
    JSONTextExtractor,
)
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
from .yolo_io import YOLODetection, YOLOReader
from .yolo_processing import YOLOGeometryExtractor


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
    "AlignmentJSONExporter",
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
    "GeometryWordAssigner",
    "GreatestCoverageWordAssigner",
    "InputFormat",
    "JSONGeometryExtractor",
    "JSONPath",
    "JSONReader",
    "JSONTextExtractor",
    "JSONWriter",
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
    "WordCoverage",
    "YOLODetection",
    "YOLOGeometryExtractor",
    "YOLOReader",
    "create_overlap_calculator",
    "create_word_assigner",
]
