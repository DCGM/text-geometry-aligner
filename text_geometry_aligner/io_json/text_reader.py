"""JSON text reader."""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

from ..models import AlignmentPage, AlignmentRegion, InputFormat, JSONPath
from ..utils import _format_json_path

logger = logging.getLogger(__name__)


class JSONTextReader:
    """Read JSON text values into the shared alignment representation."""

    def __init__(
        self,
        geometry_suffix: str = "_bbox",
        overwrite_existing_geometry: bool = False,
        ignored_geometry_suffixes: Sequence[str] = ("_bbox", "_polygon"),
    ):
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")
        if any(not suffix for suffix in ignored_geometry_suffixes):
            raise ValueError("ignored geometry suffixes must not be empty")
        self.geometry_suffix = geometry_suffix
        self.overwrite_existing_geometry = overwrite_existing_geometry
        self.ignored_geometry_suffixes = frozenset(
            (*ignored_geometry_suffixes, geometry_suffix)
        )

    def _extract_regions(
        self,
        data: Any,
    ) -> tuple[AlignmentRegion, ...]:
        if not isinstance(data, dict):
            raise TypeError("Text-alignment JSON root must be an object")

        regions: list[AlignmentRegion] = []

        def append_value(
            value: str | int | float,
            path: JSONPath,
            key: str,
            geometry_path: JSONPath,
        ) -> None:
            regions.append(
                AlignmentRegion(
                    region_id=len(regions),
                    label=key,
                    input_text=value,
                    json_text_path=path,
                    json_geometry_path=geometry_path,
                )
            )

        def visit(
            node: Any,
            path: JSONPath,
            list_geometry_path: JSONPath | None = None,
            list_key: str | None = None,
        ) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if not isinstance(key, str):
                        logger.debug(
                            "Skipping non-string JSON object key at %s: %r",
                            path,
                            key,
                        )
                        continue

                    child_path = path + (key,)
                    if any(
                        key.endswith(suffix)
                        for suffix in self.ignored_geometry_suffixes
                    ):
                        # Existing geometry is not an alignable text value.
                        continue

                    geometry_key = f"{key}{self.geometry_suffix}"
                    if (
                        not self.overwrite_existing_geometry
                        and geometry_key in node
                    ):
                        logger.debug(
                            "Preserving existing geometry at %s",
                            _format_json_path(child_path),
                        )
                        continue

                    if _is_alignable_scalar(value):
                        append_value(
                            value=value,
                            path=child_path,
                            key=key,
                            geometry_path=path + (geometry_key,),
                        )
                    elif isinstance(value, list):
                        visit(
                            value,
                            child_path,
                            list_geometry_path=path + (geometry_key,),
                            list_key=key,
                        )
                    elif isinstance(value, dict):
                        visit(value, child_path)

            elif isinstance(node, list):
                for index, value in enumerate(node):
                    child_path = path + (index,)
                    child_geometry_path = (
                        None
                        if list_geometry_path is None
                        else list_geometry_path + (index,)
                    )
                    if isinstance(value, list):
                        visit(
                            value,
                            child_path,
                            list_geometry_path=child_geometry_path,
                            list_key=list_key,
                        )
                    elif isinstance(value, dict):
                        visit(value, child_path)
                    elif _is_alignable_scalar(value):
                        if child_geometry_path is None or list_key is None:
                            logger.debug(
                                "Skipping scalar list element at %s; its list has "
                                "no owning dictionary key",
                                _format_json_path(child_path),
                            )
                            continue
                        append_value(
                            value=value,
                            path=child_path,
                            key=list_key,
                            geometry_path=child_geometry_path,
                        )

        visit(data, ())
        return tuple(regions)

    def from_data(
        self,
        data: Any,
        *,
        page_key: str = "page",
        input_file_path: Path | None = None,
    ) -> AlignmentPage:
        """Convert in-memory JSON text data into an alignment page."""

        return AlignmentPage(
            page_key=page_key,
            input_format=InputFormat.JSON,
            regions=list(self._extract_regions(data)),
            input_file_path=input_file_path,
            json_source_data=copy.deepcopy(data),
        )

    def read(
        self,
        input_path: str | os.PathLike[str],
        *,
        page_key: str | None = None,
    ) -> AlignmentPage:
        """Read one UTF-8 JSON file into an alignment page."""

        path = Path(input_path)
        with path.open("r", encoding="utf-8") as input_stream:
            data = json.load(input_stream)
        return self.from_data(
            data,
            page_key=path.stem if page_key is None else page_key,
            input_file_path=path,
        )


def _is_alignable_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)
