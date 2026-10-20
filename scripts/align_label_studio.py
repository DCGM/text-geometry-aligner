#!/usr/bin/env python3
"""Align a Label Studio rectangle-label export with ALTO and write the text of every box.

For every task of the export whose image has an ALTO file (<alto-dir>/<image stem>.xml), the
GeometryAligner (default settings unless overridden) assigns the ALTO words covered by each box and
writes the package's grouped JSON - per label a list of texts and a parallel list of geometries - to
<output-dir>/<image stem>.json.

The export is adapted before it is read, since LabelStudioReader accepts neither:
- several tasks of one image: the task whose annotation was updated last is kept;
- rotated rectangles: each is replaced by the axis-aligned box enclosing it (Label Studio rotates a
  rectangle clockwise around its top-left corner).

Run with the aligner's environment from the repository root, e.g.
    ~/python_env/text-geometry-aligner/bin/python scripts/align_label_studio.py EXPORT.json \
        --alto-dir ALTO_DIR --output-dir OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path

from text_geometry_aligner.geometry_aligner import GeometryAligner
from text_geometry_aligner.geometry_matching.assignment import WordAssignmentStrategy
from text_geometry_aligner.io_label_studio import LabelStudioReader

logger = logging.getLogger("align_label_studio")


def _updated_at(task: dict) -> str:
    return max((a.get("updated_at") or "" for a in task.get("annotations", [])), default="")


def deduplicate_tasks(tasks: list[dict]) -> list[dict]:
    """Keep one task per image: the one whose annotation was updated last."""
    kept: dict[str, dict] = {}
    for task in tasks:
        key = Path(task["data"]["image"]).stem
        if key in kept:
            logger.warning("image %s has several tasks; keeping the last updated one", key)
            if _updated_at(task) <= _updated_at(kept[key]):
                continue
        kept[key] = task
    return list(kept.values())


def unrotate(result: dict) -> bool:
    """Replace a rotated rectangle by its axis-aligned bounds; True when it was rotated."""
    value = result["value"]
    angle = value.get("rotation") or 0
    if not angle:
        return False
    page_w, page_h = result["original_width"], result["original_height"]
    x, y = value["x"] * page_w / 100, value["y"] * page_h / 100
    w, h = value["width"] * page_w / 100, value["height"] * page_h / 100
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    xs, ys = zip(*((x + dx * cos - dy * sin, y + dx * sin + dy * cos) for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))))
    left, top = max(min(xs), 0.0), max(min(ys), 0.0)
    right, bottom = min(max(xs), page_w), min(max(ys), page_h)
    value.update(x=left / page_w * 100, y=top / page_h * 100,
                 width=(right - left) / page_w * 100, height=(bottom - top) / page_h * 100, rotation=0)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path, help="Label Studio JSON project export")
    parser.add_argument("--alto-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-overlap-coverage", type=float, default=0.65)
    parser.add_argument("--word-assignment-strategy", choices=[s.value for s in WordAssignmentStrategy],
                        default=WordAssignmentStrategy.GREATEST_COVERAGE.value,
                        help="all-over-threshold lets a word belong to every box covering it, e.g. one firm "
                             "annotated both as publisher and as printer")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    tasks = deduplicate_tasks(copy.deepcopy(json.loads(args.export.read_text(encoding="utf-8"))))
    rotated = sum(unrotate(result) for task in tasks for annotation in task.get("annotations", [])
                  for result in annotation.get("result", []) if result.get("type") == "rectanglelabels")
    document = LabelStudioReader().from_data(tasks, input_file_path=args.export)

    aligner = GeometryAligner(minimum_overlap_coverage=args.minimum_overlap_coverage,
                              word_assignment_strategy=args.word_assignment_strategy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = missing_alto = regions = unmatched = 0
    for page in document.pages:
        alto_path = args.alto_dir / f"{page.page_key}.xml"
        if not alto_path.exists():
            missing_alto += 1
            continue
        alto_page = aligner.alto_reader.read(alto_path)
        aligner._attach_alto(page, alto_page, alto_path)
        aligner.align_page(alto_page, page)
        aligner.json_writer.write(page, args.output_dir / f"{page.page_key}.json")
        written += 1
        regions += len(page.regions)
        unmatched += sum(region.alto_text is None for region in page.regions)
    print(f"{args.export.name}: {len(tasks)} tasks, {written} pages written, {missing_alto} without ALTO, "
          f"{regions} boxes ({rotated} rotated, made axis-aligned), {unmatched} boxes without ALTO text")


if __name__ == "__main__":
    main()
