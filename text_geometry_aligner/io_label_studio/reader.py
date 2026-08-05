"""Label Studio rectangle annotation reader."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ..label_mapping import LabelMapper
from ..models import (
    AlignmentDocument,
    AlignmentMode,
    AlignmentPage,
    AlignmentRegion,
    BoundingBox,
    InputFormat,
    JSONPath,
    _format_json_path,
)

logger = logging.getLogger(__name__)


class LabelStudioReader:
    """Read one Label Studio project export containing rectangle labels."""

    def __init__(self, label_mapper: LabelMapper | None = None):
        self.label_mapper = label_mapper

    def read(
        self,
        path: str | os.PathLike[str],
    ) -> AlignmentDocument:
        """Read one UTF-8 Label Studio project export."""

        source_path = Path(path)
        with source_path.open("r", encoding="utf-8") as source:
            data = json.load(source)
        return self.from_data(data, input_file_path=source_path)

    def from_data(
        self,
        data: Any,
        *,
        input_file_path: Path | None = None,
    ) -> AlignmentDocument:
        """Convert an in-memory Label Studio project export."""

        if not isinstance(data, list):
            raise TypeError("Label Studio project export root must be an array")

        source = input_file_path or "in-memory data"
        logger.info(
            "Loading Label Studio geometry project from %s with %d tasks",
            source,
            len(data),
        )

        pages: list[AlignmentPage] = []
        page_keys: set[str] = set()
        ignored_result_count = 0
        for task_index, task in enumerate(data):
            task_path: JSONPath = (task_index,)
            task = _mapping(task, task_path, "task")
            page_key = _page_key(task, task_path)
            if page_key in page_keys:
                raise ValueError(
                    "Duplicate Label Studio page key "
                    f"{page_key!r} at {_format_json_path(task_path)}"
                )
            page_keys.add(page_key)

            annotations = _annotations(task, task_path)
            selected = _newest_active_annotation(annotations)
            task_id = task.get("id")
            selected_index = None if selected is None else selected[0]
            logger.info(
                "Loading Label Studio geometry task index=%d task_id=%r "
                "page=%r annotation_index=%r from %s at %s",
                task_index,
                task_id,
                page_key,
                selected_index,
                source,
                _format_json_path(task_path),
            )

            regions: list[AlignmentRegion] = []
            if selected is not None:
                annotation_index, annotation = selected
                result_path = task_path + (
                    "annotations",
                    annotation_index,
                    "result",
                )
                results = annotation.get("result", [])
                if not _is_sequence(results):
                    raise TypeError(
                        "Label Studio annotation result must be an array at "
                        f"{_format_json_path(result_path)}"
                    )
                for result_index, result in enumerate(results):
                    path = result_path + (result_index,)
                    result = _mapping(result, path, "annotation result")
                    if result.get("type") != "rectanglelabels":
                        ignored_result_count += 1
                        continue
                    regions.append(
                        _region_from_result(
                            result,
                            path,
                            region_id=len(regions),
                            label_mapper=self.label_mapper,
                        )
                    )

            pages.append(
                AlignmentPage(
                    page_key=page_key,
                    input_format=InputFormat.LABEL_STUDIO,
                    regions=regions,
                    input_file_path=input_file_path,
                )
            )

        if ignored_result_count:
            logger.info(
                "Ignored %d non-rectangle Label Studio annotation results "
                "from %s",
                ignored_result_count,
                source,
            )

        return AlignmentDocument(
            alignment_mode=AlignmentMode.GEOMETRY,
            pages=pages,
            input_path=input_file_path,
        )


def _annotations(
    task: Mapping[str, Any],
    task_path: JSONPath,
) -> tuple[Mapping[str, Any], ...]:
    path = task_path + ("annotations",)
    annotations = task.get("annotations", [])
    if not _is_sequence(annotations):
        raise TypeError(
            "Label Studio task annotations must be an array at "
            f"{_format_json_path(path)}"
        )
    return tuple(
        _mapping(annotation, path + (index,), "annotation")
        for index, annotation in enumerate(annotations)
    )


def _newest_active_annotation(
    annotations: Sequence[Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any]] | None:
    active = [
        (index, annotation)
        for index, annotation in enumerate(annotations)
        if not annotation.get("was_cancelled", False)
    ]
    if not active:
        return None
    return max(active, key=lambda item: _annotation_order_key(*item))


def _annotation_order_key(
    index: int,
    annotation: Mapping[str, Any],
) -> tuple[float, float, int]:
    created_at = _timestamp(annotation.get("created_at"))
    updated_at = _timestamp(annotation.get("updated_at"))
    effective_time = updated_at if updated_at is not None else created_at
    return (
        float("-inf") if effective_time is None else effective_time,
        float("-inf") if created_at is None else created_at,
        index,
    )


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _page_key(task: Mapping[str, Any], task_path: JSONPath) -> str:
    data_path = task_path + ("data",)
    task_data = _mapping(task.get("data"), data_path, "task data")
    image_path = data_path + ("image",)
    image_reference = task_data.get("image")
    if not isinstance(image_reference, str) or not image_reference.strip():
        raise ValueError(
            "Label Studio task image must be a non-empty string at "
            f"{_format_json_path(image_path)}"
        )

    parsed = urlparse(image_reference)
    local_file_values = parse_qs(parsed.query).get("d")
    path_value = (
        local_file_values[0]
        if local_file_values and local_file_values[0]
        else parsed.path
    )
    filename = Path(unquote(path_value)).name
    page_key = Path(filename).stem
    if not page_key:
        raise ValueError(
            "Could not derive a page key from Label Studio image at "
            f"{_format_json_path(image_path)}"
        )
    return page_key


def _region_from_result(
    result: Mapping[str, Any],
    result_path: JSONPath,
    *,
    region_id: int,
    label_mapper: LabelMapper | None,
) -> AlignmentRegion:
    value_path = result_path + ("value",)
    value = _mapping(result.get("value"), value_path, "rectangle value")

    labels_path = value_path + ("rectanglelabels",)
    labels = value.get("rectanglelabels")
    if (
        not _is_sequence(labels)
        or len(labels) != 1
        or not isinstance(labels[0], str)
        or not labels[0]
    ):
        raise ValueError(
            "Label Studio rectangle must contain exactly one non-empty "
            f"label at {_format_json_path(labels_path)}"
        )

    rotation = _finite_number(value.get("rotation", 0), value_path, "rotation")
    image_rotation = _finite_number(
        result.get("image_rotation", 0),
        result_path,
        "image_rotation",
    )
    if rotation != 0 or image_rotation != 0:
        raise ValueError(
            "Rotated Label Studio rectangles are not supported at "
            f"{_format_json_path(value_path)}"
        )

    original_width = _positive_number(
        result.get("original_width"),
        result_path,
        "original_width",
    )
    original_height = _positive_number(
        result.get("original_height"),
        result_path,
        "original_height",
    )
    x_percent = _finite_number(value.get("x"), value_path, "x")
    y_percent = _finite_number(value.get("y"), value_path, "y")
    width_percent = _positive_number(value.get("width"), value_path, "width")
    height_percent = _positive_number(
        value.get("height"),
        value_path,
        "height",
    )

    return AlignmentRegion(
        region_id=region_id,
        label=labels[0],
        label_mapper=label_mapper,
        input_geometry=BoundingBox(
            x=x_percent / 100 * original_width,
            y=y_percent / 100 * original_height,
            width=width_percent / 100 * original_width,
            height=height_percent / 100 * original_height,
        ),
        json_geometry_path=value_path,
    )


def _mapping(value: Any, path: JSONPath, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Label Studio {label} must be an object at "
            f"{_format_json_path(path)}"
        )
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _positive_number(value: Any, path: JSONPath, label: str) -> float:
    number = _finite_number(value, path, label)
    if number <= 0:
        raise ValueError(
            f"Invalid Label Studio {label} at {_format_json_path(path)}: "
            "must be positive"
        )
    return number


def _finite_number(value: Any, path: JSONPath, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"Invalid Label Studio {label} at {_format_json_path(path)}: "
            "expected a number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"Invalid Label Studio {label} at {_format_json_path(path)}: "
            "must be finite"
        )
    return number
