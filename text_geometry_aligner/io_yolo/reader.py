"""YOLO detection reader."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..label_mapping import LabelMapper
from ..models import (
    AlignmentPage,
    AlignmentRegion,
    BoundingBox,
    InputFormat,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YOLODetection:
    """One absolute-coordinate YOLO detection."""

    category_id: int
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    class_name: str


@dataclass(frozen=True)
class LabelDeduplicationGroup:
    """Cross-class labels deduplicated at a mutual geometry coverage."""

    labels: frozenset[str]
    minimum_coverage: float

    def __post_init__(self) -> None:
        if len(self.labels) < 2:
            raise ValueError(
                "A label deduplication group must contain at least two "
                "distinct labels"
            )
        if any(
            not isinstance(label, str) or not label.strip()
            for label in self.labels
        ):
            raise ValueError(
                "Label deduplication group labels must be non-empty strings"
            )
        if (
            isinstance(self.minimum_coverage, bool)
            or not isinstance(self.minimum_coverage, (int, float))
            or not math.isfinite(self.minimum_coverage)
            or not 0.0 < self.minimum_coverage <= 1.0
        ):
            raise ValueError(
                "Label deduplication minimum_coverage must be within (0, 1]"
            )


@dataclass(frozen=True)
class _SuppressedRegion:
    suppressed: AlignmentRegion
    retained: AlignmentRegion
    mutual_coverage: float
    minimum_coverage: float


class YOLOReader:
    """Read absolute-coordinate YOLO detections into alignment pages."""

    def __init__(
        self,
        label_mapper: LabelMapper | None = None,
        label_deduplication_groups: (
            Sequence[LabelDeduplicationGroup] | None
        ) = None,
    ):
        self.label_mapper = label_mapper
        self.label_deduplication_groups = tuple(
            label_deduplication_groups or ()
        )
        self._group_by_label = _groups_by_label(
            self.label_deduplication_groups
        )

    def read(
        self,
        path: str | os.PathLike[str],
        *,
        page_key: str | None = None,
    ) -> AlignmentPage:
        """Read one YOLO file into an alignment page."""

        source_path = Path(path)
        return self.from_data(
            self._read_detections(source_path),
            page_key=source_path.stem if page_key is None else page_key,
            input_file_path=source_path,
        )

    def from_data(
        self,
        detections: Sequence[YOLODetection],
        *,
        page_key: str = "page",
        input_file_path: Path | None = None,
    ) -> AlignmentPage:
        """Convert in-memory YOLO detections into an alignment page."""

        logger.info(
            "Loading YOLO geometry page %r from %s",
            page_key,
            input_file_path or "in-memory data",
        )
        page = AlignmentPage(
            page_key=page_key,
            input_format=InputFormat.YOLO,
            regions=[
                AlignmentRegion(
                    region_id=region_id,
                    label=detection.class_name,
                    label_mapper=self.label_mapper,
                    input_geometry=BoundingBox(
                        x=detection.center_x - detection.width / 2,
                        y=detection.center_y - detection.height / 2,
                        width=detection.width,
                        height=detection.height,
                    ),
                    category_id=detection.category_id,
                    input_geometry_confidence=detection.confidence,
                )
                for region_id, detection in enumerate(detections)
            ],
            input_file_path=input_file_path,
        )
        if not self.label_deduplication_groups:
            return page

        original_count = len(page.regions)
        page.regions, suppressed = self._deduplicate(page.regions)
        logger.info(
            "YOLO cross-class deduplication: page=%r, regions=%d, "
            "retained=%d, suppressed=%d",
            page.page_key,
            original_count,
            len(page.regions),
            len(suppressed),
        )
        for item in suppressed:
            logger.debug(
                "Suppressed YOLO region: page=%r, region_id=%d, label=%r, "
                "confidence=%s; retained_region_id=%d, retained_label=%r, "
                "retained_confidence=%s, mutual_coverage=%.4f, "
                "minimum_coverage=%.4f",
                page.page_key,
                item.suppressed.region_id,
                item.suppressed.label,
                item.suppressed.input_geometry_confidence,
                item.retained.region_id,
                item.retained.label,
                item.retained.input_geometry_confidence,
                item.mutual_coverage,
                item.minimum_coverage,
            )
        return page

    def _deduplicate(
        self,
        regions: Sequence[AlignmentRegion],
    ) -> tuple[list[AlignmentRegion], list[_SuppressedRegion]]:
        ranked_indices = sorted(
            range(len(regions)),
            key=lambda index: (
                -(
                    regions[index].input_geometry_confidence
                    if regions[index].input_geometry_confidence is not None
                    else 0.0
                ),
                index,
            ),
        )
        retained_indices: list[int] = []
        suppressed_by_index: dict[int, _SuppressedRegion] = {}
        for index in ranked_indices:
            candidate = regions[index]
            candidate_group = self._group_by_label.get(candidate.label)
            if candidate_group is None or candidate.input_geometry is None:
                retained_indices.append(index)
                continue

            duplicate = None
            for retained_index in retained_indices:
                retained = regions[retained_index]
                if (
                    retained.label == candidate.label
                    or self._group_by_label.get(retained.label)
                    is not candidate_group
                    or retained.input_geometry is None
                ):
                    continue
                coverage = _mutual_geometry_coverage(candidate, retained)
                if coverage >= candidate_group.minimum_coverage:
                    duplicate = _SuppressedRegion(
                        suppressed=candidate,
                        retained=retained,
                        mutual_coverage=coverage,
                        minimum_coverage=candidate_group.minimum_coverage,
                    )
                    break
            if duplicate is None:
                retained_indices.append(index)
            else:
                suppressed_by_index[index] = duplicate

        retained_set = set(retained_indices)
        return (
            [
                region
                for index, region in enumerate(regions)
                if index in retained_set
            ],
            [
                suppressed_by_index[index]
                for index in sorted(suppressed_by_index)
            ],
        )

    def _read_detections(
        self,
        source_path: Path,
    ) -> tuple[YOLODetection, ...]:
        detections: list[YOLODetection] = []
        category_names: dict[int, str] = {}
        name_categories: dict[str, int] = {}

        with source_path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) < 7:
                    raise ValueError(
                        f"Invalid YOLO row at {source_path}:{line_number}: "
                        "expected at least seven fields"
                    )
                category_id = self._parse_category_id(
                    fields[0],
                    source_path,
                    line_number,
                )
                center_x, center_y, width, height, confidence = (
                    self._parse_number(
                        value,
                        source_path,
                        line_number,
                        label,
                    )
                    for value, label in zip(
                        fields[1:6],
                        (
                            "center_x",
                            "center_y",
                            "width",
                            "height",
                            "confidence",
                        ),
                    )
                )
                class_name = " ".join(fields[6:]).strip()
                if not class_name:
                    raise ValueError(
                        f"Invalid YOLO row at {source_path}:{line_number}: "
                        "class_name must not be empty"
                    )
                if width <= 0 or height <= 0:
                    raise ValueError(
                        f"Invalid YOLO row at {source_path}:{line_number}: "
                        "width and height must be positive"
                    )
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(
                        f"Invalid YOLO row at {source_path}:{line_number}: "
                        "confidence must be within [0, 1]"
                    )

                previous_name = category_names.setdefault(
                    category_id,
                    class_name,
                )
                previous_id = name_categories.setdefault(
                    class_name,
                    category_id,
                )
                if previous_name != class_name or previous_id != category_id:
                    raise ValueError(
                        f"Inconsistent YOLO class mapping at "
                        f"{source_path}:{line_number}"
                    )

                detections.append(
                    YOLODetection(
                        category_id=category_id,
                        center_x=center_x,
                        center_y=center_y,
                        width=width,
                        height=height,
                        confidence=confidence,
                        class_name=class_name,
                    )
                )
        return tuple(detections)

    @staticmethod
    def _parse_category_id(
        value: str,
        path: Path,
        line_number: int,
    ) -> int:
        try:
            category_id = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid YOLO category ID at {path}:{line_number}: {value!r}"
            ) from exc
        if category_id < 0:
            raise ValueError(
                f"Invalid YOLO category ID at {path}:{line_number}: "
                "must not be negative"
            )
        return category_id

    @staticmethod
    def _parse_number(
        value: str,
        path: Path,
        line_number: int,
        label: str,
    ) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid YOLO {label} at {path}:{line_number}: {value!r}"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(
                f"Invalid YOLO {label} at {path}:{line_number}: "
                "must be finite"
            )
        return number


def _groups_by_label(
    groups: Sequence[LabelDeduplicationGroup],
) -> dict[str, LabelDeduplicationGroup]:
    result: dict[str, LabelDeduplicationGroup] = {}
    for group in groups:
        if not isinstance(group, LabelDeduplicationGroup):
            raise TypeError(
                "label_deduplication_groups must contain only "
                "LabelDeduplicationGroup objects"
            )
        repeated = result.keys() & group.labels
        if repeated:
            raise ValueError(
                "A label cannot belong to multiple deduplication groups: "
                + ", ".join(sorted(repeated))
            )
        result.update((label, group) for label in group.labels)
    return result


def _mutual_geometry_coverage(
    first: AlignmentRegion,
    second: AlignmentRegion,
) -> float:
    first_box = first.input_geometry.bounds
    second_box = second.input_geometry.bounds
    intersection_width = max(
        0.0,
        min(first_box.x_max, second_box.x_max)
        - max(first_box.x, second_box.x),
    )
    intersection_height = max(
        0.0,
        min(first_box.y_max, second_box.y_max)
        - max(first_box.y, second_box.y),
    )
    intersection = intersection_width * intersection_height
    first_area = first_box.width * first_box.height
    second_area = second_box.width * second_box.height
    larger_area = max(first_area, second_area)
    return intersection / larger_area if larger_area > 0 else 0.0
