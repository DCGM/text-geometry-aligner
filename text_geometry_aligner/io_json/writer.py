"""Alignment JSON writer."""

from __future__ import annotations

import copy
import json
import os
from collections import OrderedDict
from collections.abc import MutableMapping, MutableSequence
from pathlib import Path
from typing import Any

from ..models import (
    AlignmentMode,
    AlignmentPage,
    AlignmentRegion,
    BoundingBox,
    InputFormat,
    JSONPath,
    OutputGeometry,
    OutputGeometryFormat,
    OutputGeometrySource,
    OutputTextSource,
    Polygon,
)
from ..utils import _format_json_path


class AlignmentJSONWriter:
    """Convert alignment pages to JSON data and atomically write them."""

    def __init__(
        self,
        *,
        alignment_mode: AlignmentMode | str,
        geometry_suffix: str,
        output_geometry_format: OutputGeometryFormat | str,
        output_text_source: OutputTextSource | str = OutputTextSource.JSON,
        output_geometry_source: OutputGeometrySource | str = (
            OutputGeometrySource.INPUT
        ),
    ):
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")
        self.alignment_mode = AlignmentMode(alignment_mode)
        self.geometry_suffix = geometry_suffix
        self.output_geometry_format = OutputGeometryFormat(
            output_geometry_format
        )
        self.output_text_source = OutputTextSource(output_text_source)
        self.output_geometry_source = OutputGeometrySource(
            output_geometry_source
        )

    def to_data(self, page: AlignmentPage) -> dict[str, Any]:
        if page.input_format is InputFormat.YOLO:
            return self._to_grouped_data(page)
        return self._to_original_data(page)

    def write(
        self,
        page: AlignmentPage,
        output_path: str | os.PathLike[str],
    ) -> None:
        """Atomically write one alignment page as UTF-8 JSON."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as output_stream:
                json.dump(
                    self.to_data(page),
                    output_stream,
                    ensure_ascii=False,
                    indent=2,
                )
                output_stream.write("\n")
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _to_original_data(
        self,
        page: AlignmentPage,
    ) -> dict[str, Any]:
        if page.json_source_data is None:
            raise ValueError(
                f"JSON source data is missing for page {page.page_key!r}"
            )
        output = copy.deepcopy(page.json_source_data)
        if self.alignment_mode is AlignmentMode.GEOMETRY:
            _create_missing_text_destinations(
                output,
                self.geometry_suffix,
            )
        else:
            _create_missing_geometry_destinations(output, page.regions)
        for region in page.regions:
            if self.alignment_mode is AlignmentMode.TEXT:
                self._update_text_alignment_region(output, region)
            else:
                self._update_geometry_alignment_region(output, region)
        return output

    def _update_text_alignment_region(
        self,
        output: dict[str, Any],
        region: AlignmentRegion,
    ) -> None:
        if region.json_geometry_path is None:
            raise ValueError(
                f"Region {region.region_id} has no JSON geometry path"
            )
        geometry = self._converted_geometry(region.alto_geometry)
        _set_or_create_path(
            output,
            region.json_geometry_path,
            None if geometry is None else geometry.to_json(),
        )
        if (
            self.output_text_source is OutputTextSource.ALTO
            and region.alto_text is not None
            and region.json_text_path is not None
        ):
            _set_or_create_path(
                output,
                region.json_text_path,
                region.alto_text,
            )

    def _update_geometry_alignment_region(
        self,
        output: dict[str, Any],
        region: AlignmentRegion,
    ) -> None:
        if region.json_text_path is None:
            raise ValueError(
                f"Region {region.region_id} has no JSON text path"
            )
        _set_or_create_path(
            output,
            region.json_text_path,
            region.alto_text,
        )
        if (
            self.output_geometry_source is OutputGeometrySource.ALTO
            and region.json_geometry_path is not None
        ):
            geometry = self._converted_geometry(region.alto_geometry)
            _set_or_create_path(
                output,
                region.json_geometry_path,
                None if geometry is None else geometry.to_json(),
            )

    def _to_grouped_data(self, page: AlignmentPage) -> dict[str, Any]:
        grouped: OrderedDict[str, list[AlignmentRegion]] = OrderedDict()
        for region in page.regions:
            grouped.setdefault(region.label, []).append(region)

        output: dict[str, Any] = {}
        for label, regions in grouped.items():
            geometry_key = (
                f"{label}_{self.output_geometry_format.value}"
            )
            conflicting_keys = {
                key for key in (label, geometry_key) if key in output
            }
            if conflicting_keys:
                raise ValueError(
                    "YOLO class names collide with generated JSON keys: "
                    f"{sorted(conflicting_keys)}"
                )
            output[label] = [region.alto_text for region in regions]
            output[geometry_key] = [
                (
                    None
                    if (geometry := self._selected_geometry(region)) is None
                    else geometry.to_json()
                )
                for region in regions
            ]
        return output

    def _selected_geometry(
        self,
        region: AlignmentRegion,
    ) -> OutputGeometry | None:
        geometry = (
            region.input_geometry
            if self.output_geometry_source is OutputGeometrySource.INPUT
            else region.alto_geometry
        )
        return self._converted_geometry(geometry)

    def _converted_geometry(
        self,
        geometry: OutputGeometry | None,
    ) -> OutputGeometry | None:
        if geometry is None:
            return None
        if self.output_geometry_format is OutputGeometryFormat.BBOX:
            return geometry.bounds
        if isinstance(geometry, Polygon):
            return geometry
        return _bbox_polygon(geometry)


def _bbox_polygon(bbox: BoundingBox) -> Polygon:
    return Polygon(
        (
            (bbox.x, bbox.y),
            (bbox.x_max, bbox.y),
            (bbox.x_max, bbox.y_max),
            (bbox.x, bbox.y_max),
            (bbox.x, bbox.y),
        )
    )


def _create_missing_geometry_destinations(
    output: dict[str, Any],
    regions: list[AlignmentRegion],
) -> None:
    """Create parallel null-filled list shapes for text-list geometries."""

    initialized_paths: set[JSONPath] = set()
    for region in regions:
        text_path = region.json_text_path
        geometry_path = region.json_geometry_path
        if text_path is None or geometry_path is None:
            continue

        trailing_indexes = 0
        for component in reversed(geometry_path):
            if not isinstance(component, int):
                break
            trailing_indexes += 1
        if not trailing_indexes:
            continue

        geometry_container_path = geometry_path[:-trailing_indexes]
        if geometry_container_path in initialized_paths:
            continue
        text_container_path = text_path[:-trailing_indexes]
        source_container = _value_at_path(output, text_container_path)
        if not isinstance(source_container, list):
            continue
        if _path_exists(output, geometry_container_path):
            initialized_paths.add(geometry_container_path)
            continue
        _set_or_create_path(
            output,
            geometry_container_path,
            _empty_geometry_shape(source_container),
        )
        initialized_paths.add(geometry_container_path)


def _empty_geometry_shape(value: Any) -> Any:
    if isinstance(value, list):
        return [_empty_geometry_shape(item) for item in value]
    return None


def _create_missing_text_destinations(
    node: Any,
    geometry_suffix: str,
) -> None:
    """Mirror geometry containers for destinations absent from source JSON."""

    if isinstance(node, MutableMapping):
        additions: dict[str, Any] = {}
        for key, value in tuple(node.items()):
            if isinstance(key, str) and key.endswith(geometry_suffix):
                destination_key = key[: -len(geometry_suffix)]
                if destination_key and destination_key not in node:
                    additions[destination_key] = _empty_text_shape(value)
            else:
                _create_missing_text_destinations(value, geometry_suffix)
        node.update(additions)
    elif isinstance(node, MutableSequence):
        for value in node:
            _create_missing_text_destinations(value, geometry_suffix)


def _empty_text_shape(geometry: Any) -> Any:
    if not isinstance(geometry, list) or _looks_like_polygon(geometry):
        return None
    return [_empty_text_shape(value) for value in geometry]


def _looks_like_polygon(value: list[Any]) -> bool:
    return (
        len(value) >= 4
        and all(
            isinstance(point, (list, tuple))
            and len(point) == 2
            for point in value
        )
    )


def _path_exists(root: Any, path: JSONPath) -> bool:
    try:
        _value_at_path(root, path)
    except (KeyError, IndexError, TypeError):
        return False
    return True


def _value_at_path(root: Any, path: JSONPath) -> Any:
    node = root
    for component in path:
        if isinstance(component, str):
            if not isinstance(node, MutableMapping):
                raise TypeError(component)
            node = node[component]
        elif isinstance(component, int):
            if not isinstance(node, MutableSequence):
                raise TypeError(component)
            node = node[component]
        else:
            raise TypeError(component)
    return node


def _set_or_create_path(
    root: Any,
    path: JSONPath,
    value: Any,
) -> None:
    if not path:
        raise ValueError("Cannot replace the JSON root value")

    node = root
    for index, component in enumerate(path):
        is_last = index == len(path) - 1
        next_component = None if is_last else path[index + 1]
        if isinstance(component, str):
            if not isinstance(node, MutableMapping):
                raise TypeError(
                    f"Cannot create {_format_json_path(path)} through "
                    f"{type(node).__name__}"
                )
            if is_last:
                node[component] = value
                return
            if component not in node:
                node[component] = (
                    [] if isinstance(next_component, int) else {}
                )
            child = node[component]
        elif isinstance(component, int):
            if not isinstance(node, MutableSequence):
                raise TypeError(
                    f"Cannot create {_format_json_path(path)} through "
                    f"{type(node).__name__}"
                )
            while len(node) <= component:
                node.append(None)
            if is_last:
                node[component] = value
                return
            if node[component] is None:
                node[component] = (
                    [] if isinstance(next_component, int) else {}
                )
            child = node[component]
        else:
            raise TypeError(
                f"Invalid path component at {_format_json_path(path)}"
            )

        expected = (
            MutableSequence
            if isinstance(next_component, int)
            else MutableMapping
        )
        if not isinstance(child, expected):
            raise TypeError(
                f"Incompatible container while creating "
                f"{_format_json_path(path)}: found "
                f"{type(child).__name__}"
            )
        node = child
