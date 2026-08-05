"""Class-name mapping for exported alignment labels."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any


class LabelMapper:
    """Exact source-label to export-label mapping."""

    def __init__(self, mapping: Mapping[str, str]):
        self._mapping = MappingProxyType(_validated_mapping(mapping))

    @classmethod
    def from_data(cls, mapping: Mapping[str, str]) -> LabelMapper:
        """Create a mapper from an in-memory mapping."""

        return cls(mapping)

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
    ) -> LabelMapper:
        """Read an exact mapping from a UTF-8 JSON object."""

        source_path = Path(path)
        try:
            with source_path.open("r", encoding="utf-8") as source:
                mapping = json.load(
                    source,
                    object_pairs_hook=_unique_object,
                )
        except ValueError as exc:
            raise ValueError(
                f"Invalid class mapping in {source_path}: {exc}"
            ) from exc
        if not isinstance(mapping, dict):
            raise TypeError(
                f"Class mapping root must be an object: {source_path}"
            )
        try:
            return cls(mapping)
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"Invalid class mapping in {source_path}: {exc}"
            ) from exc

    def map(self, label: str) -> str | None:
        """Return the explicit export label, or ``None`` when unmapped."""

        return self._mapping.get(label)


def _validated_mapping(mapping: Mapping[Any, Any]) -> dict[str, str]:
    if not isinstance(mapping, Mapping):
        raise TypeError("Class mapping must be an object")
    validated: dict[str, str] = {}
    for source, target in mapping.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError("Class mapping names must be strings")
        if not source.strip() or not target.strip():
            raise ValueError("Class mapping names must not be empty")
        validated[source] = target
    return validated


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate class mapping key: {key!r}")
        output[key] = value
    return output
