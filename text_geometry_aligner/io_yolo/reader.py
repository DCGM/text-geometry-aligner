"""YOLO detection reader."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..models import (
    AlignmentPage,
    AlignmentRegion,
    BoundingBox,
    InputFormat,
)


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


class YOLOReader:
    """Read absolute-coordinate YOLO detections into alignment pages."""

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

        return AlignmentPage(
            page_key=page_key,
            input_format=InputFormat.YOLO,
            regions=[
                AlignmentRegion(
                    region_id=region_id,
                    label=detection.class_name,
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
