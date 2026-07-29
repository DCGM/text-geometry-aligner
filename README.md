# Text geometry aligner

This package aligns structured text or geometry with word-level ALTO OCR in
either direction:

- `TextAligner` finds JSON text in ALTO and writes the geometry of the matched
  ALTO words.
- `GeometryAligner` finds ALTO words covered by JSON or YOLO geometry and
  writes their text.

The aligner name describes the information used for matching, not the
information it produces. Both directions enrich the same document hierarchy.
Consequently, the **text alignment pipeline contains a geometry builder**,
while the **geometry alignment pipeline contains a text builder**.

## Shared alignment model

All input adapters create the same mutable hierarchy:

```text
AlignmentDocument
└── AlignmentPage
    └── AlignmentRegion
        └── AlignmentWord
```

The hierarchy is available before and after matching. Input fields are kept
separate from ALTO-derived fields:

- `AlignmentDocument` holds `input_path`, `alto_path`, and all pages.
- `AlignmentPage` holds its input and ALTO file paths, `alto_page_id`,
  `alto_width`, `alto_height`, and its regions.
- `AlignmentRegion` holds `input_text`, `input_text_normalized`,
  `input_geometry`, the selected text-alignment candidate, `alto_text`,
  `alto_text_normalized`, `alto_geometry`, assigned `words`, JSON paths, and
  optional YOLO category metadata.
- `AlignmentWord` records the selected ALTO word, its normalized comparison
  text, bbox, ALTO indexes, element ID, and optional word-coverage score.

Fields produced by matching are `None` before matching and remain `None` when
no match is found. JSON pages additionally retain a private working copy in
`json_source_data` so their original arbitrary nesting can be reconstructed
without reading the file again. YOLO pages set `json_source_data` to `None`
because their output JSON is constructed from the regions.

## Installation

Install the dependencies needed by this package from the repository root:

```bash
python -m pip install \
  -r requirements.txt
```

The dependency roles are:

| Dependency | Used for |
| --- | --- |
| Levenshtein | Fuzzy and ordered text alignment |
| OR-Tools | Global CP-SAT candidate selection |
| Pillow | Optional alignment rendering |
| Shapely | Polygon-to-word overlap; bbox overlap also has a dependency-free implementation |

ALTO XML and JSON are read with the Python standard library.

## The two directions

### Text alignment: text in, geometry out

`TextAligner` recursively extracts JSON scalar values, compares them with ALTO
text, chooses compatible matches, and builds geometry around the selected ALTO
words.

```mermaid
flowchart LR
    J["JSON text values"] --> JE["JSON input adapter<br/>regions + retained paths"]
    A["ALTO words"] --> AI["ALTO text index"]
    JE --> N["Same normalization<br/>on JSON and ALTO"]
    AI --> N
    N --> CG["Candidate generator<br/>exact / combined / ordered"]
    CG --> CS["Candidate selector<br/>CP-SAT / pass-through"]
    CS --> MW["Selected ALTO word spans"]
    MW --> TB["Text builder<br/>ALTO text"]
    MW --> GB["Geometry builder<br/>union bbox / orthogonal polygon"]
    TB --> M["Enriched regions"]
    GB --> M
    M --> JM["JSON exporter"]
    JM --> O["Output JSON<br/>*_bbox or *_polygon"]
    M --> R["Optional rendering<br/>text + similarity"]
```

The geometry builder is the final step because text matching identifies ALTO
words but does not itself define how their output geometry should be
represented.

Available geometry builders:

- `UnionBoundingBoxGeometryBuilder` returns one bounding box enclosing all
  matched word boxes.
- `OrthogonalPolygonGeometryBuilder` groups matched words by ALTO line and
  creates a closed, axis-aligned polygon that closely covers the words.

For example:

```json
{
  "title": "ROKY ZA STOLETÍ",
  "publisher": [
    "ŠOLC a ŠIMÁČEK",
    "společnost s r. o. v Praze."
  ]
}
```

With bbox output, the parallel result has the same list shape:

```json
{
  "title": "ROKY ZA STOLETÍ",
  "publisher": [
    "ŠOLC a ŠIMÁČEK",
    "společnost s r. o. v Praze."
  ],
  "title_bbox": {
    "x": 108,
    "y": 442,
    "width": 1262,
    "height": 168
  },
  "publisher_bbox": [
    {
      "x": 509,
      "y": 2257,
      "width": 478,
      "height": 52
    },
    {
      "x": 504,
      "y": 2307,
      "width": 486,
      "height": 54
    }
  ]
}
```

An unmatched value receives `null` geometry. String, integer, and float values
are alignable; booleans and existing geometry keys are ignored. Nested
dictionaries and lists are retained through their JSON paths. Scalar list
values must ultimately be owned by a dictionary key so that a parallel
geometry key can be created.

### Geometry alignment: geometry in, text out

`GeometryAligner` creates regions from suffixed JSON geometry or absolute
YOLO detections, measures how much of each ALTO word they cover, resolves
competing regions, and enriches the regions with ALTO text and geometry.

```mermaid
flowchart LR
    J["JSON bbox/polygon"] --> GE["JSON geometry adapter<br/>regions + retained paths"]
    Y["YOLO detections"] --> YE["YOLO geometry adapter<br/>regions + class metadata"]
    A["ALTO word boxes and text"] --> OC["Word-area overlap calculation"]
    GE --> OC
    YE --> OC
    OC --> WC["Eligible word coverages<br/>coverage >= threshold"]
    WC --> WA["Word assignment<br/>greatest coverage / retain all"]
    WA --> MW["Assigned ALTO words<br/>in document order"]
    MW --> TB["Text builder<br/>space-separated"]
    MW --> GB["Geometry builder<br/>union bbox / orthogonal polygon"]
    TB --> M["Enriched regions"]
    GB --> M
    M --> JM["JSON exporter"]
    JM --> O["Output JSON"]
    M --> R["Optional rendering<br/>text + average coverage"]
```

The text builder is the final step because geometry matching identifies the
covered ALTO words but does not define how their text should be reconstructed.
The current `SpaceSeparatedTextBuilder` joins words in ALTO document order
using one space.

For example:

```json
{
  "title_bbox": {
    "x": 108,
    "y": 442,
    "width": 1262,
    "height": 168
  },
  "publisher_bbox": [
    {
      "x": 509,
      "y": 2257,
      "width": 478,
      "height": 52
    },
    {
      "x": 504,
      "y": 2307,
      "width": 486,
      "height": 54
    }
  ]
}
```

Can produce:

```json
{
  "title": "ROKY ZA STOLETÍ",
  "title_bbox": {
    "x": 108,
    "y": 442,
    "width": 1262,
    "height": 168
  },
  "publisher": [
    "ŠOLC a ŠIMÁČEK",
    "společnost s r. o. v Praze."
  ],
  "publisher_bbox": [
    {
      "x": 509,
      "y": 2257,
      "width": 478,
      "height": 52
    },
    {
      "x": 504,
      "y": 2307,
      "width": 486,
      "height": 54
    }
  ]
}
```

The JSON root must be an object. Bboxes and polygons may occur at any nested
dictionary or list depth. Existing destination values are skipped before
overlap calculation unless `--overwrite-existing-text` is enabled. A processed
geometry with no eligible ALTO words produces `null`.

### YOLO geometry input

YOLO input uses one detection per line:

```text
<class_id> <center_x> <center_y> <width> <height> <confidence> <class_name>
```

Coordinates are absolute values in the original image/ALTO coordinate system,
not normalized fractions. The adapter converts center-based coordinates to the
package's top-left `BoundingBox`. `class_id`, `class_name`, and confidence are
retained in each `AlignmentRegion`; ID/name mappings must be consistent within
and across pages.

YOLO files may have any extension or no extension. Pairing removes one final
suffix when present, so `page.full.labels` pairs with `page.full.xml`;
extensionless `page` pairs with `page.xml`.

Because YOLO has no source JSON structure, output is grouped by class name.
Repeated detections remain parallel lists:

```json
{
  "PageNumber": ["12", null],
  "PageNumber_bbox": [
    {"x": 10, "y": 20, "width": 30, "height": 10},
    {"x": 10, "y": 180, "width": 30, "height": 10}
  ]
}
```

## Geometry representations

A bbox is an object in ALTO coordinates:

```json
{
  "x": 100,
  "y": 200,
  "width": 300,
  "height": 50
}
```

A polygon is a list of `[x, y]` points. It must contain at least three vertices
and be closed by repeating the first point:

```json
[
  [100, 200],
  [400, 200],
  [400, 250],
  [100, 250],
  [100, 200]
]
```

The default suffix is `_bbox`. Text alignment automatically uses `_polygon`
when polygon output is selected, unless `--geometry-suffix` overrides it.
Geometry alignment detects whichever suffix is supplied through
`--geometry-suffix`; it can parse both bbox and polygon values beneath that
suffix.

Word coverage is:

```text
area(JSON geometry ∩ ALTO word bbox) / area(ALTO word bbox)
```

Therefore, a threshold of `1` requires the complete word box to be covered,
while `0` accepts any positive intersection. With the default
`greatest-coverage` assignment, a word belongs to only the eligible region
covering the largest fraction of it. Ties are resolved by stable JSON traversal
order. `all-over-threshold` retains the word for every eligible region.

## Command-line usage

Both commands process top-level input files in a directory and pair them with
ALTO `.xml` files by page key. Output directories are created automatically.

### Text alignment CLI

```bash
python -m text_geometry_aligner.text_aligner \
  --alto-dir data/alto \
  --input-dir data/json \
  --input-format json \
  --json-output-dir output/json \
  --candidate-generator combined \
  --candidate-selector cp-sat \
  --output-alto-text-format space-separated \
  --output-alto-geometry-format polygon \
  --text-normalizer lowercase \
  --text-normalizer strip-diacritics \
  --text-normalizer strip-punctuation
```

Important text-alignment options:

| Option | Choices/default | Meaning |
| --- | --- | --- |
| `--candidate-generator` | `combined` | `exact`, exact plus bounded fuzzy (`combined`), or `ordered-alignment` |
| `--candidate-selector` | `cp-sat` | Globally select non-overlapping candidates, or use `pass-through` |
| `--output-alto-text-format` | `space-separated` | Build `alto_text` from matched ALTO words |
| `--output-alto-geometry-format` | `bbox` | Build `alto_geometry` as `bbox` or `polygon` |
| `--geometry-suffix` | automatic | Defaults to `_bbox` or `_polygon` from the output format |
| `--output-text-source` | `json` | Preserve JSON text or replace matched values with original `alto` text |
| `--text-normalizer` | optional | Normalize both JSON and ALTO comparison texts before alignment; repeat to compose lowercase, diacritic stripping, and punctuation stripping |
| `--overwrite-existing-geometry` | off | Process and replace existing geometry destinations; otherwise their text values are skipped before matching |

Unicode NFKC normalization and whitespace collapsing always run. Each
`--text-normalizer` optionally adds a transformation to both the JSON and ALTO
comparison texts before alignment. Repeated arguments stack the transformations
in the order supplied. The `ordered-alignment` generator assumes JSON values
are already in correct reading order and is normally paired with
`--candidate-selector pass-through`.

Normalization is visible in the in-memory alignment hierarchy without
changing the original text. `input_text_normalized` is populated before
candidate generation, so it is also available for unmatched regions and is
not replaced by candidate data. For a successful selection,
`alto_text_normalized` is read directly from the normalized ALTO index.
The complete selected candidate is retained separately in
`text_alignment_candidate`, including its comparison snapshots, word and
character spans, matching source, edit distance, CER, and similarity.
Each selected `AlignmentWord` additionally stores its independently normalized
ALTO token in `text_normalized`. These inspection fields are not added to the
exported JSON.

CP-SAT first maximizes match quality, exact-match count, and matched-value
count. If complete solutions remain equal, it prefers candidates whose
`start_word` positions are closer to the beginning of the ALTO document, then
uses candidate IDs as a deterministic final fallback.

Fuzzy acceptance defaults:

| Option | Default |
| --- | --- |
| `--fuzzy-query-length-boundary` | `6` non-whitespace characters |
| `--fuzzy-max-cer-at-or-above-boundary` | `0.20` |
| `--fuzzy-max-edit-distance-below-boundary` | `1` |
| `--fuzzy-max-candidates-per-value` | `5` |

Setting the boundary to `0` applies the CER rule to every query.

### Geometry alignment CLI

```bash
python -m text_geometry_aligner.geometry_aligner \
  --alto-dir data/alto \
  --input-dir data/yolo \
  --input-format yolo \
  --json-output-dir output/json \
  --minimum-word-coverage 0.65 \
  --word-assignment-strategy greatest-coverage \
  --output-alto-text-format space-separated \
  --output-geometry-source input \
  --output-alto-geometry-format bbox
```

Important geometry-alignment options:

| Option | Choices/default | Meaning |
| --- | --- | --- |
| `--input-format` | `json` | Read `json` or `yolo` geometry input |
| `--geometry-suffix` | `_bbox` | For JSON input, identify geometry keys and derive destination text keys |
| `--minimum-word-coverage` | `0.65` | Minimum covered fraction of each ALTO word |
| `--word-assignment-strategy` | `greatest-coverage` | Choose one winner or use `all-over-threshold` |
| `--output-alto-text-format` | `space-separated` | Build `alto_text` from assigned ALTO words |
| `--output-alto-geometry-format` | `bbox` | Build `alto_geometry` as `bbox` or `polygon` |
| `--output-geometry-source` | `input` | Export and render `input` or ALTO-derived (`alto`) geometry |
| `--overwrite-existing-text` | off | Process and replace destinations that already exist |

### Rendering

Add both common rendering arguments to either command:

```bash
--images-dir data/images --render-dir output/rendered
```

Images are paired by filename stem. Text-alignment labels show match
similarity; geometry-alignment labels show average word coverage. Geometry is
scaled from ALTO page coordinates when the ALTO page dimensions differ from
the source image dimensions.

Use `--fail-on-missing-alto` to fail rather than skip JSON files without a
paired ALTO XML file.

## Python API

### Text to geometry

The text aligner receives its candidate generator and selector explicitly:

```python
from text_geometry_aligner import (
    AnchoredFuzzyTextCandidateGenerator,
    CPSATCandidateSelector,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
    TextAligner,
)

candidate_generator = CompositeCandidateGenerator(
    (
        ExactTextCandidateGenerator(),
        AnchoredFuzzyTextCandidateGenerator(FuzzyCandidateConfig()),
    )
)

aligner = TextAligner(
    candidate_generator=candidate_generator,
    candidate_selector=CPSATCandidateSelector(),
    output_geometry_format="polygon",
)
result = aligner.align_files(
    "data/alto/page.xml",
    "data/json/page.json",
    "output/page.json",
)

print(result.matched_count, result.unmatched_count)
print(result.pages[0].regions[0].alto_geometry)
```

### Geometry to text

```python
from text_geometry_aligner import GeometryAligner

aligner = GeometryAligner(
    geometry_suffix="_bbox",
    minimum_word_coverage=0.65,
    word_assignment_strategy="greatest-coverage",
)
result = aligner.align_files(
    "data/alto/page.xml",
    "data/json/page.json",
    "output/page.json",
)

print(result.matched_count, result.unmatched_count)
print(result.pages[0].regions[0].alto_text)
```

For in-memory use, parse or construct an `ALTOPage` and call
`aligner.align_data(alto_page, input_data)`.

Directory processing can also be used without exporting JSON. The returned
document contains every enriched page and region:

```python
document = aligner.process_directories(
    alto_input_dir="data/alto",
    input_dir="data/json",
)
```

Pass `json_output_dir` only when JSON files should also be written.

The adapters and matcher can also be used separately when the intermediate
hierarchy is needed:

```python
from pathlib import Path
from text_geometry_aligner import ALTOReader, GeometryAligner, InputFormat

aligner = GeometryAligner()
page = aligner.read_input_page(
    Path("data/yolo/page.labels"),
    InputFormat.YOLO,
    page_key="page",
)

# Input fields are available; ALTO-derived fields are still None.
alto_page = ALTOReader().read("data/alto/page.xml")
aligner.align_page(alto_page, page)

# The same page and region objects now contain alto_text,
# alto_geometry, and assigned words.
output_data = aligner.export_page(page)
```

## Package structure and extension points

| Area | Responsibility and primary extension points |
| --- | --- |
| `alto_io`, `alto_processing` | Read ALTO and create its normalized text index |
| `json_io`, `json_processing` | Read/write JSON, create pages with retained paths, and export enriched pages |
| `yolo_io`, `yolo_processing` | Read absolute YOLO detections and create geometry regions |
| `text_matching` | `CandidateGenerator` and `CandidateSelector` implementations |
| `geometry_matching` | `GeometryOverlapCalculator` and `GeometryWordAssigner` implementations |
| `geometry_building` | `GeometryBuilder` implementations used directly by both aligners |
| `text_building` | `TextBuilder` implementations used directly by both aligners |
| `normalization.py` | Composable `TextNormalizer` stages applied equally to JSON and ALTO |
| `rendering.py` | Direction-neutral `AlignmentRenderer` implementations |

The abstract interfaces enforce each component contract. Custom components can
be injected into the aligner constructors without changing the orchestration
logic. Both aligners accept `text_builder` and `geometry_builder`. The selected
builders populate `alto_text` and `alto_geometry`, respectively. A custom
geometry builder must return the geometry type selected by
`output_geometry_format`; a custom text builder receives assigned `ALTOWord`
objects in ALTO document order and returns the final string or `None` for an
empty assignment.

`BaseAligner` provides the shared ALTO text/geometry builder configuration and
CLI options, file and directory processing, output writing, filename pairing,
category validation, and optional rendering workflow for both directions.
Input adapters and exporters are intentionally separate from the matching
algorithms so additional formats can reuse the same hierarchy.
