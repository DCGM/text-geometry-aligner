from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..models import (
    AlignmentPage,
    AlignmentRegion,
    BoundingBox,
    InputFormat,
)
from ..yolo_io import YOLODetection, YOLOReader


class YOLOGeometryExtractor:
    """Convert detections to geometry-driven alignment regions."""

    def __init__(self, reader: YOLOReader | None = None):
        self.reader = reader or YOLOReader()

    def extract_alignment_region(
        self,
        detections: Sequence[YOLODetection],
    ) -> tuple[AlignmentRegion, ...]:
        return tuple(
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
        )

    def extract_alignment_page(
        self,
        path: str | Path,
        *,
        page_key: str,
    ) -> AlignmentPage:
        """Read a YOLO file and extract one geometry alignment page."""

        input_path = Path(path)
        return AlignmentPage(
            page_key=page_key,
            input_format=InputFormat.YOLO,
            regions=list(
                self.extract_alignment_region(self.reader.read(input_path))
            ),
            input_file_path=input_path,
        )
