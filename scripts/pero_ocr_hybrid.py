#!/usr/bin/env python3
"""OCR page images with PERO: transformer transcription, CTC word positions.

A PERO transformer recognizes the text of each line, but its logits belong to
decoder steps, not to positions in the line image, so ALTO words built from
them are misplaced. This script therefore also runs a CTC model with the same
alphabet on the same line crops and keeps its logits; the ALTO export then
force-aligns the transformer transcription to the CTC logits, which places the
words where they are printed.

Writes <stem>.xml into --alto-dir and --page-xml-dir for every image. Pages
whose outputs already exist are skipped, so an interrupted run can be resumed.
Dense pages are CPU-bound on one core; to use more, start several runs on the
same folder with --shard 0/N, 1/N, ... (each loads its own engines, ~3 GB GPU
and ~2 GB RAM).

Run with the pero-ocr environment, e.g.
    ~/python_env/pero-ocr/bin/python scripts/pero_ocr_hybrid.py IMAGES_DIR \
        --alto-dir ALTO_DIR --page-xml-dir PAGE_XML_DIR --pair printed
"""

from __future__ import annotations

import argparse
import configparser
import contextlib
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

MODELS_DIR = Path.home() / "data" / "pero-worker_data"
LAYOUT_ENGINE_CONFIG = MODELS_DIR / "layout.2022-04-13.printed.transformer.nfkd.2025-12-05" / "config.ini"

# Transformer and CTC models of each pair share one alphabet (NFKD, 2200 characters).
PAIRS = {
    "universal": ("universal_transformer.nfkd.220k.2025-12-05", "universal_ctc.nfkd.2025-12-05"),
    "printed": ("printed.transformer.nfkd.2025-12-05", "printed.ctc.nfkd.2025-12-05"),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".jp2"}


def build_engines(pair: str, device):
    from pero_ocr.document_ocr.page_parser import PageOCR, PageParser

    transformer_dir, ctc_dir = (MODELS_DIR / name for name in PAIRS[pair])
    config = configparser.ConfigParser()
    config.read(LAYOUT_ENGINE_CONFIG)
    config["OCR"]["OCR_JSON"] = str(transformer_dir / "ocr_engine.json")
    parser = PageParser(config, device=device, config_path=str(LAYOUT_ENGINE_CONFIG.parent))

    ctc_config = configparser.ConfigParser()
    ctc_config["OCR"] = {"METHOD": "pytorch_ocr", "OCR_JSON": str(ctc_dir / "ocr_engine.json")}
    ctc = PageOCR(ctc_config["OCR"], device=device)
    return parser, ctc


def process_page(parser, ctc, image, name: str):
    from pero_ocr.core.layout import PageLayout

    layout = parser.process_page(image, PageLayout(id=name, page_size=(image.shape[0], image.shape[1])))
    transcriptions = {line.id: line.transcription for line in layout.lines_iterator()}
    layout = ctc.process_page(image, layout)    # replaces transcriptions and logits of the lines
    for line in layout.lines_iterator():
        line.transcription = transcriptions[line.id]
        line.calculate_confidences()            # CTC confidence of the transformer transcription
    layout.calculate_confidence()
    return layout


def load_images(paths: list[Path], out: queue.Queue) -> None:
    import cv2

    for path in paths:
        out.put((path, cv2.imread(str(path), cv2.IMREAD_COLOR)))
    out.put(None)


def main() -> None:
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument("images", type=Path, help="folder of page images")
    arg_parser.add_argument("--alto-dir", type=Path, required=True)
    arg_parser.add_argument("--page-xml-dir", type=Path, required=True)
    arg_parser.add_argument("--pair", choices=PAIRS, default="printed", help="transformer + CTC model pair")
    arg_parser.add_argument("--limit", type=int, help="process at most this many pages")
    arg_parser.add_argument("--shard", default="0/1", metavar="I/N",
                            help="process only every N-th image starting with the I-th (0-based), so N runs started "
                                 "with 0/N .. N-1/N share the folder without overlap")
    args = arg_parser.parse_args()
    shard, shards = (int(part) for part in args.shard.split("/"))
    if not 0 <= shard < shards:
        arg_parser.error(f"--shard {args.shard}: I must be in 0..N-1")

    logging.basicConfig(level=logging.WARNING)
    import torch

    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU available; the PERO layout model runs on the GPU only")

    args.alto_dir.mkdir(parents=True, exist_ok=True)
    args.page_xml_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[shard::shards]
    todo = [p for p in paths
            if not ((args.alto_dir / f"{p.stem}.xml").exists() and (args.page_xml_dir / f"{p.stem}.xml").exists())]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(paths)} images, {len(paths) - len(todo)} already done, {len(todo)} to process", file=sys.stderr)

    with contextlib.redirect_stdout(sys.stderr):    # PERO prints progress to stdout
        parser, ctc = build_engines(args.pair, torch.device("cuda"))

    images: queue.Queue = queue.Queue(maxsize=8)
    threading.Thread(target=load_images, args=(todo, images), daemon=True).start()
    start, failed = time.time(), []
    for count, (path, image) in enumerate(iter(images.get, None), start=1):
        try:
            if image is None:
                raise ValueError("unreadable image")
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                layout = process_page(parser, ctc, image, path.stem)
                # Write to temporary files first so that a killed run never leaves a truncated output behind.
                for directory, write in ((args.page_xml_dir, layout.to_pagexml), (args.alto_dir, layout.to_altoxml)):
                    tmp = directory / f".{path.stem}.xml.tmp"
                    write(str(tmp), output_nfc=True)
                    tmp.replace(directory / f"{path.stem}.xml")
        except Exception as e:
            failed.append(path.name)
            print(f"FAILED {path.name}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        if count % 50 == 0 or count == len(todo):
            elapsed = time.time() - start
            print(f"[{count}/{len(todo)}] {elapsed / count:.2f} s/page, "
                  f"ETA {(len(todo) - count) * elapsed / count / 60:.0f} min, {len(failed)} failed",
                  file=sys.stderr, flush=True)
    if failed:
        print(f"{len(failed)} pages failed: {' '.join(failed)}", file=sys.stderr)


if __name__ == "__main__":
    main()
