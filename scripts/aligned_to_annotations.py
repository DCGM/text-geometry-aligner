#!/usr/bin/env python3
"""Turn grouped geometry-alignment JSON into a reading-ordered list of class/value annotations.

Reads every <stem>.json in INPUT_DIR - the aligner's grouped output, per label a list of texts and a
parallel list of "<label>_bbox" geometries, as scripts/align_label_studio.py writes it - and writes
<stem>.json into OUTPUT_DIR:

    [{"class": "titleInfo.title", "value": "LÁSKA A SMRT"}, {"class": "agents.author", "value": null}, ...]

Classes are renamed by --class-mapping (a JSON object, source label -> class); a label missing from it
is an error, so no annotation is dropped or passed on under its source name unnoticed. A box with no
text keeps its entry with a null value. Geometry is left out; it only orders the entries: top to
bottom by rows - boxes whose vertical centre lies within the current row's span share it - and left to
right within a row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GEOMETRY_SUFFIX = "_bbox"


def entries(data: dict, mapping: dict[str, str]) -> list[tuple[dict, dict]]:
    """(bbox, annotation) of every box of one page."""
    result = []
    for label, values in data.items():
        if label.endswith(GEOMETRY_SUFFIX):
            continue
        if label not in mapping:
            raise KeyError(f"label {label!r} has no class in the mapping")
        boxes = data[label + GEOMETRY_SUFFIX]
        if not isinstance(values, list):
            values, boxes = [values], [boxes]
        for value, box in zip(values, boxes, strict=True):
            result.append((box, {"class": mapping[label], "value": value}))
    return result


def reading_order(items: list[tuple[dict, dict]]) -> list[dict]:
    rows: list[tuple[float, float, list[tuple[dict, dict]]]] = []     # (top, bottom, items)
    for box, annotation in sorted(items, key=lambda item: item[0]["y"]):
        centre = box["y"] + box["height"] / 2
        if rows and rows[-1][0] <= centre <= rows[-1][1]:
            top, bottom, members = rows[-1]
            rows[-1] = (top, max(bottom, box["y"] + box["height"]), members + [(box, annotation)])
        else:
            rows.append((box["y"], box["y"] + box["height"], [(box, annotation)]))
    return [annotation for _, _, members in rows for _, annotation in sorted(members, key=lambda m: m[0]["x"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--class-mapping", type=Path, required=True, help="JSON object: source label -> class")
    args = parser.parse_args()

    mapping = json.loads(args.class_mapping.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = annotations = empty = 0
    for path in sorted(args.input_dir.glob("*.json")):
        ordered = reading_order(entries(json.loads(path.read_text(encoding="utf-8")), mapping))
        (args.output_dir / path.name).write_text(json.dumps(ordered, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8")
        pages += 1
        annotations += len(ordered)
        empty += sum(a["value"] is None for a in ordered)
    print(f"{pages} pages, {annotations} annotations, {empty} without a value", file=sys.stderr)


if __name__ == "__main__":
    main()
