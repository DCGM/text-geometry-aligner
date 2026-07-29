from __future__ import annotations

import argparse
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .alto_io import ALTOPage, ALTOReader
from .geometry_building import (
    GeometryBuilder,
    OrthogonalPolygonGeometryBuilder,
    UnionBoundingBoxGeometryBuilder,
)
from .json_io import JSONWriter
from .models import (
    AlignmentDocument,
    AlignmentMode,
    AlignmentPage,
    InputFormat,
    OutputGeometryFormat,
    OutputGeometrySource,
    OutputTextSource,
)
from .rendering import AlignmentRenderer, PillowAlignmentRenderer
from .text_building import SpaceSeparatedTextBuilder, TextBuilder

logger = logging.getLogger(__name__)

ALTO_TEXT_FORMAT_CHOICES = ("space-separated",)

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _build_text_builder(output_format: str) -> TextBuilder:
    if output_format == "space-separated":
        return SpaceSeparatedTextBuilder()
    raise ValueError(
        f"Unsupported output ALTO text format: {output_format}"
    )


def _build_geometry_builder(
    output_format: OutputGeometryFormat,
) -> GeometryBuilder:
    if output_format is OutputGeometryFormat.BBOX:
        return UnionBoundingBoxGeometryBuilder()
    if output_format is OutputGeometryFormat.POLYGON:
        return OrthogonalPolygonGeometryBuilder()
    raise ValueError(
        f"Unsupported output ALTO geometry format: {output_format}"
    )


class BaseAligner(ABC):
    """Shared document, page pairing, export, and rendering workflow."""

    alignment_mode: AlignmentMode
    supported_input_formats: tuple[InputFormat, ...]

    def __init__(
        self,
        *,
        alto_reader: ALTOReader | None = None,
        json_writer: JSONWriter | None = None,
        output_geometry_format: OutputGeometryFormat | str = (
            OutputGeometryFormat.BBOX
        ),
        geometry_builder: GeometryBuilder | None = None,
        text_builder: TextBuilder | None = None,
        renderer: AlignmentRenderer | None = None,
    ):
        self.alto_reader = alto_reader or ALTOReader()
        self.json_writer = json_writer or JSONWriter()
        self.output_geometry_format = OutputGeometryFormat(
            output_geometry_format
        )
        self.geometry_builder = geometry_builder or _build_geometry_builder(
            self.output_geometry_format
        )
        self.text_builder = text_builder or _build_text_builder(
            ALTO_TEXT_FORMAT_CHOICES[0]
        )
        self.renderer = renderer or PillowAlignmentRenderer()

    @abstractmethod
    def read_input_page(
        self,
        input_file: Path,
        input_format: InputFormat,
        page_key: str,
    ) -> AlignmentPage:
        raise NotImplementedError

    @abstractmethod
    def align_page(
        self,
        alto_page: ALTOPage,
        page: AlignmentPage,
    ) -> AlignmentPage:
        raise NotImplementedError

    @abstractmethod
    def export_page(self, page: AlignmentPage) -> dict[str, object]:
        raise NotImplementedError

    @property
    def render_text_source(self) -> OutputTextSource:
        return OutputTextSource.JSON

    @property
    def render_geometry_source(self) -> OutputGeometrySource:
        return OutputGeometrySource.ALTO

    @property
    def render_geometry_format(self) -> OutputGeometryFormat:
        return self.output_geometry_format

    def align_files(
        self,
        alto_file: str | os.PathLike[str],
        input_file: str | os.PathLike[str],
        json_output_file: str | os.PathLike[str],
        *,
        input_format: InputFormat | str = InputFormat.JSON,
        image_file: str | os.PathLike[str] | None = None,
        render_output_file: str | os.PathLike[str] | None = None,
    ) -> AlignmentDocument:
        """Align one input/ALTO pair and return a one-page document."""

        if (image_file is None) != (render_output_file is None):
            raise ValueError(
                "image_file and render_output_file must be provided together"
            )

        parsed_format = self._validate_input_format(input_format)
        input_path = Path(input_file)
        alto_path = Path(alto_file)
        page = self.read_input_page(
            input_path,
            parsed_format,
            _page_key(input_path),
        )
        parsed_alto = self.alto_reader.read(alto_path)
        self._attach_alto(page, parsed_alto, alto_path)
        self.align_page(parsed_alto, page)
        self.json_writer.write(
            self.export_page(page),
            Path(json_output_file),
        )

        if image_file is not None and render_output_file is not None:
            self.renderer.render(
                image_path=image_file,
                output_path=render_output_file,
                page=page,
                output_text_source=self.render_text_source,
                output_geometry_source=self.render_geometry_source,
                output_geometry_format=self.render_geometry_format,
            )

        return AlignmentDocument(
            alignment_mode=self.alignment_mode,
            pages=[page],
            input_path=input_path.parent,
            alto_path=alto_path.parent,
        )

    def process_directories(
        self,
        alto_input_dir: str | os.PathLike[str],
        input_dir: str | os.PathLike[str],
        json_output_dir: str | os.PathLike[str] | None = None,
        *,
        input_format: InputFormat | str = InputFormat.JSON,
        images_input_dir: str | os.PathLike[str] | None = None,
        render_output_dir: str | os.PathLike[str] | None = None,
        fail_on_missing_alto: bool = False,
    ) -> AlignmentDocument:
        """Align every top-level input page and optionally export JSON."""

        if (images_input_dir is None) != (render_output_dir is None):
            raise ValueError(
                "images_input_dir and render_output_dir must be provided together"
            )

        parsed_format = self._validate_input_format(input_format)
        alto_dir = Path(alto_input_dir)
        source_dir = Path(input_dir)
        output_dir = (
            None
            if json_output_dir is None
            else Path(json_output_dir)
        )
        images_dir = (
            None if images_input_dir is None else Path(images_input_dir)
        )
        render_dir = (
            None if render_output_dir is None else Path(render_output_dir)
        )
        _require_directory(alto_dir, "ALTO input")
        _require_directory(source_dir, "input")
        if images_dir is not None:
            _require_directory(images_dir, "images input")
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)

        alto_by_key = _files_by_key(
            alto_dir,
            allowed_suffixes={".xml"},
            label="ALTO",
        )
        images_by_key = (
            _files_by_key(
                images_dir,
                allowed_suffixes=IMAGE_EXTENSIONS,
                label="image",
            )
            if images_dir is not None
            else {}
        )
        input_by_key = _input_files(source_dir, parsed_format)
        document = AlignmentDocument(
            alignment_mode=self.alignment_mode,
            pages=[],
            input_path=source_dir,
            alto_path=alto_dir,
        )

        for index, (page_key, input_path) in enumerate(
            input_by_key.items(),
            start=1,
        ):
            alto_path = alto_by_key.get(page_key)
            if alto_path is None:
                message = (
                    f"No ALTO XML found for input file {input_path.name}"
                )
                if fail_on_missing_alto:
                    raise FileNotFoundError(message)
                logger.warning(message)
                continue

            logger.info(
                "Processing %d/%d: %s with %s",
                index,
                len(input_by_key),
                input_path.name,
                alto_path.name,
            )
            page = self.read_input_page(
                input_path,
                parsed_format,
                page_key,
            )
            parsed_alto = self.alto_reader.read(alto_path)
            self._attach_alto(page, parsed_alto, alto_path)
            self.align_page(parsed_alto, page)
            if output_dir is not None:
                self.json_writer.write(
                    self.export_page(page),
                    output_dir / f"{page_key}.json",
                )
            document.pages.append(page)

            if images_dir is not None and render_dir is not None:
                image_path = images_by_key.get(page_key)
                if image_path is None:
                    logger.warning(
                        "No source image found for input file %s; skipping "
                        "rendering",
                        input_path.name,
                    )
                else:
                    self.renderer.render(
                        image_path=image_path,
                        output_path=render_dir / image_path.name,
                        page=page,
                        output_text_source=self.render_text_source,
                        output_geometry_source=self.render_geometry_source,
                        output_geometry_format=self.render_geometry_format,
                    )

        _validate_category_mappings(document)
        logger.info(
            "Processed %d/%d pages: regions=%d matched=%d unmatched=%d",
            len(document.pages),
            len(input_by_key),
            sum(len(page.regions) for page in document.pages),
            document.matched_count,
            document.unmatched_count,
        )
        return document

    def _validate_input_format(
        self,
        input_format: InputFormat | str,
    ) -> InputFormat:
        parsed = InputFormat(input_format)
        if parsed not in self.supported_input_formats:
            choices = ", ".join(item.value for item in self.supported_input_formats)
            raise ValueError(
                f"{type(self).__name__} does not support {parsed.value!r} "
                f"input; expected one of: {choices}"
            )
        return parsed

    @staticmethod
    def _attach_alto(
        page: AlignmentPage,
        alto_page: ALTOPage,
        alto_path: Path,
    ) -> None:
        page.alto_file_path = alto_path
        page.alto_page_id = alto_page.page_id
        page.alto_width = alto_page.width
        page.alto_height = alto_page.height


def add_common_cli_arguments(
    parser: argparse.ArgumentParser,
    *,
    input_formats: tuple[InputFormat, ...],
) -> None:
    parser.add_argument(
        "--alto-dir",
        required=True,
        help="Directory containing ALTO XML files",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input files",
    )
    parser.add_argument(
        "--input-format",
        choices=tuple(item.value for item in input_formats),
        default=InputFormat.JSON.value,
        help="Input format (default: json)",
    )
    parser.add_argument(
        "--json-output-dir",
        required=True,
        help="Directory for aligned output JSON files",
    )
    parser.add_argument(
        "--output-alto-text-format",
        choices=ALTO_TEXT_FORMAT_CHOICES,
        default="space-separated",
        help="Format used to build alto_text (default: space-separated)",
    )
    parser.add_argument(
        "--output-alto-geometry-format",
        choices=tuple(item.value for item in OutputGeometryFormat),
        default=OutputGeometryFormat.BBOX.value,
        help="Format used to build alto_geometry (default: bbox)",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional source-image directory used for rendering",
    )
    parser.add_argument(
        "--render-dir",
        default=None,
        help="Optional directory for rendered alignment visualizations",
    )
    parser.add_argument(
        "--fail-on-missing-alto",
        action="store_true",
        help="Fail instead of skipping an input page without ALTO",
    )


def validate_common_cli_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if (args.images_dir is None) != (args.render_dir is None):
        parser.error("--images-dir and --render-dir must be provided together")


def _page_key(path: Path) -> str:
    return path.stem if path.suffix else path.name


def _input_files(
    directory: Path,
    input_format: InputFormat,
) -> dict[str, Path]:
    allowed_suffixes = {".json"} if input_format is InputFormat.JSON else None
    return _files_by_key(
        directory,
        allowed_suffixes=allowed_suffixes,
        label=f"{input_format.value} input",
    )


def _files_by_key(
    directory: Path,
    *,
    allowed_suffixes: set[str] | None,
    label: str,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if (
            allowed_suffixes is not None
            and path.suffix.lower() not in allowed_suffixes
        ):
            continue
        key = _page_key(path)
        if key in files:
            raise ValueError(
                f"Multiple {label} files resolve to page key {key!r}: "
                f"{files[key]} and {path}"
            )
        files[key] = path
    return files


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{label} directory not found: {path}")


def _validate_category_mappings(document: AlignmentDocument) -> None:
    names_by_id: dict[int, str] = {}
    ids_by_name: dict[str, int] = {}
    for page in document.pages:
        for region in page.regions:
            if region.category_id is None:
                continue
            previous_name = names_by_id.setdefault(
                region.category_id,
                region.label,
            )
            previous_id = ids_by_name.setdefault(
                region.label,
                region.category_id,
            )
            if (
                previous_name != region.label
                or previous_id != region.category_id
            ):
                raise ValueError(
                    "Inconsistent category ID/name mapping across input pages"
                )
