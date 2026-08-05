"""Label Studio prediction writer."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from ..models import (
    AlignmentMode,
    AlignmentPage,
    AlignmentRegion,
    BoundingBox,
    OutputGeometry,
    OutputGeometrySource,
    OutputTextSource,
    ScalarText,
)

logger = logging.getLogger(__name__)


class LabelStudioWriter:
    """Convert aligned pages to Label Studio rectangle predictions."""

    def __init__(
        self,
        *,
        alignment_mode: AlignmentMode | str,
        image_prefix: str,
        output_text_source: OutputTextSource | str = OutputTextSource.JSON,
        output_geometry_source: OutputGeometrySource | str = (
            OutputGeometrySource.INPUT
        ),
    ):
        if not image_prefix.strip():
            raise ValueError("image_prefix must not be empty")
        self.alignment_mode = AlignmentMode(alignment_mode)
        self.image_prefix = image_prefix.rstrip("/")
        self.output_text_source = OutputTextSource(output_text_source)
        self.output_geometry_source = OutputGeometrySource(
            output_geometry_source
        )

    def to_data(self, page: AlignmentPage) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        skipped_count = 0
        for region in page.regions:
            geometry = self._selected_geometry(region)
            text = self._selected_text(region)
            if geometry is None or text is None:
                skipped_count += 1
                continue
            results.append(
                self._result(
                    page,
                    region,
                    geometry,
                    str(text),
                )
            )

        if skipped_count:
            logger.info(
                "Skipped %d incomplete Label Studio prediction results "
                "for page %r",
                skipped_count,
                page.page_key,
            )

        prediction: dict[str, Any] = {"result": results}
        scores = [
            result["score"]
            for result in results
            if "score" in result
        ]
        if scores:
            prediction["score"] = min(scores)

        return {
            "data": {
                "image": f"{self.image_prefix}/{page.page_key}.jpg",
            },
            "predictions": [prediction],
        }

    def write(
        self,
        page: AlignmentPage,
        output_path: str | os.PathLike[str],
    ) -> None:
        """Atomically write one page as UTF-8 Label Studio JSON."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as output:
                json.dump(
                    self.to_data(page),
                    output,
                    ensure_ascii=False,
                    indent=2,
                )
                output.write("\n")
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _result(
        self,
        page: AlignmentPage,
        region: AlignmentRegion,
        geometry: OutputGeometry,
        text: str,
    ) -> dict[str, Any]:
        percentage_bbox = _percentage_bbox(page, region, geometry)
        metadata: dict[str, Any] = {"text": [text]}
        raw_scores = {
            "input_geometry_confidence": region.input_geometry_confidence,
            "alignment_score": region.alignment_score,
        }
        metadata.update(
            (name, score)
            for name, score in raw_scores.items()
            if score is not None
        )

        result: dict[str, Any] = {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                **percentage_bbox.to_json(),
                "rectanglelabels": [region.label_for_export],
            },
            "meta": metadata,
        }
        available_scores = [
            score for score in raw_scores.values() if score is not None
        ]
        if available_scores:
            result["score"] = min(available_scores)
        return result

    def _selected_text(
        self,
        region: AlignmentRegion,
    ) -> ScalarText | None:
        if self.alignment_mode is AlignmentMode.GEOMETRY:
            return region.alto_text
        if self.output_text_source is OutputTextSource.JSON:
            return region.input_text
        return region.alto_text

    def _selected_geometry(
        self,
        region: AlignmentRegion,
    ) -> OutputGeometry | None:
        if self.alignment_mode is AlignmentMode.TEXT:
            return region.alto_geometry
        if self.output_geometry_source is OutputGeometrySource.INPUT:
            return region.input_geometry
        return region.alto_geometry


def _percentage_bbox(
    page: AlignmentPage,
    region: AlignmentRegion,
    geometry: OutputGeometry,
) -> BoundingBox:
    page_width = page.alto_width
    page_height = page.alto_height
    if (
        page_width is None
        or page_height is None
        or not math.isfinite(page_width)
        or not math.isfinite(page_height)
        or page_width <= 0
        or page_height <= 0
    ):
        raise ValueError(
            "Label Studio output requires positive finite ALTO dimensions "
            f"for page {page.page_key!r}"
        )

    bounds = geometry.bounds
    if (
        not all(
            math.isfinite(value)
            for value in (bounds.x, bounds.y, bounds.width, bounds.height)
        )
        or bounds.width <= 0
        or bounds.height <= 0
    ):
        raise ValueError(
            "Label Studio output requires positive finite geometry for "
            f"region {region.region_id} on page {page.page_key!r}"
        )
    return BoundingBox(
        x=bounds.x / page_width * 100,
        y=bounds.y / page_height * 100,
        width=bounds.width / page_width * 100,
        height=bounds.height / page_height * 100,
    )
