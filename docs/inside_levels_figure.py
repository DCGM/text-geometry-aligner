"""Draw docs/inside-levels.svg, the README figure of the alignment in levels.

The example is the one the figure shows: a chapter page whose JSON names
whole sections (the heading, the author statement) and the values printed
inside them (inside_paths). Run: python docs/inside_levels_figure.py
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).with_name("inside-levels.svg")

W = 1120
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"
CW = 7.83  # monospace character width at 13px
INK, MUTED, FAINT, LINE = "#1e293b", "#64748b", "#94a3b8", "#e2e8f0"

COLORS = {
    "heading": "#2563eb", "designation": "#7c3aed", "partNumber": "#c026d3", "title": "#0891b2",
    "statement": "#15803d", "name": "#b45309", "description": "#ea580c", "affiliation": "#dc2626",
    "pageNumber": "#475569",
}
LABELS = {"statement": "agents[0].statement", "name": "agents[0].name",
          "description": "agents[0].description", "affiliation": "agents[0].affiliation"}

OCR = "Kapitola 3 Lidová píseň v Čechách Jan Novák profesor Univerzlty Karlovy v Praze 45".split()
JSON_TEXT = {9: "Univerzity"}  # what the description's words read at a deeper level
LINES = [(0, 6), (6, 8), (8, 13), (13, 14)]  # word ranges of the printed lines

out = []


def add(s):
    out.append(s)


def text(x, y, s, size=13, color=INK, weight=400, anchor="start", family=FONT, extra=""):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}" {extra}>{escape(s)}</text>')


def label(key):
    return LABELS.get(key, key)


def word_width(s):
    return len(s) * CW + 14


def strip(x, y, words, highlight=()):
    """Word boxes in a row; returns the x extent of each word."""
    spans = []
    for index, word in words:
        w = word_width(word)
        fill, stroke = ("#fef3c7", "#d97706") if index in highlight else ("#f8fafc", "#cbd5e1")
        add(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="26" rx="5" fill="{fill}" stroke="{stroke}"/>')
        text(x + w / 2, y + 17.5, word, family=MONO, anchor="middle")
        spans.append((x, x + w))
        x += w + 6
    return spans


def bracket(spans, first, last, y, key, note=None):
    x1, x2 = spans[first][0] + 2, spans[last][1] - 2
    c = COLORS[key]
    add(f'<path d="M{x1:.1f},{y} v7 H{x2:.1f} v-7" fill="none" stroke="{c}" stroke-width="2.2" '
        f'stroke-linejoin="round"/>')
    text((x1 + x2) / 2, y + 23, label(key), size=12.5, color=c, weight=600, anchor="middle", family=MONO)
    if note:
        text((x1 + x2) / 2, y + 39, note, size=11.5, color=MUTED, anchor="middle")
    return (x1 + x2) / 2, y + 7


def arrow(x1, y1, x2, y2, color):
    my = (y1 + y2) / 2
    add(f'<path d="M{x1:.1f},{y1:.1f} C{x1:.1f},{my:.1f} {x2:.1f},{my:.1f} {x2:.1f},{y2 - 6:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#a-{color[1:]})"/>')


def panel(x, y, w, h, title, subtitle=None):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="{LINE}"/>')
    text(x + 18, y + 28, title, size=15, weight=700)
    if subtitle:
        text(x + 18, y + 47, subtitle, size=12.5, color=MUTED)


# ---------------------------------------------------------------- input row
def page_mock(x, y, boxes=False):
    """The printed page: lines of OCR words; with `boxes`, the nested output boxes."""
    add(f'<rect x="{x}" y="{y}" width="420" height="250" rx="4" fill="#fffdf7" stroke="#e7e1cf"/>')
    positions = {}
    line_y = [y + 40, y + 102, y + 132, y + 215]
    sizes = [16, 13, 13, 12]
    for (first, last), ly, size in zip(LINES, line_y, sizes):
        cw = size * 0.602
        widths = [len(OCR[i]) * cw for i in range(first, last)]
        total = sum(widths) + 14 * (last - first - 1)
        cx = x + (420 - total) / 2
        for i, wd in zip(range(first, last), widths):
            positions[i] = (cx, ly, wd, size)
            weight = 700 if first == 0 else 400
            color = "#b91c1c" if i == 9 and not boxes else INK
            text(cx, ly, OCR[i], size=size, weight=weight, family=MONO, color=color)
            cx += wd + 14
    if not boxes:
        px, py, pw, ps = positions[9]
        add(f'<path d="M{px:.1f},{py + 4:.1f} q{pw / 8:.1f},3 {pw / 4:.1f},0 t{pw / 4:.1f},0 t{pw / 4:.1f},0 '
            f't{pw / 4:.1f},0" fill="none" stroke="#dc2626" stroke-width="1.3"/>')
        text(px + pw / 2, py + 22, "OCR misread", size=11, color="#dc2626", anchor="middle")
        return positions

    def box(first, last, key, pad, label_side="top"):
        xs = [positions[i] for i in range(first, last + 1)]
        x1 = min(p[0] for p in xs) - pad
        x2 = max(p[0] + p[2] for p in xs) + pad
        y1 = min(p[1] - p[3] * 0.85 for p in xs) - pad
        y2 = max(p[1] + p[3] * 0.25 for p in xs) + pad
        c = COLORS[key]
        add(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" height="{y2 - y1:.1f}" rx="4" '
            f'fill="{c}" fill-opacity="0.06" stroke="{c}" stroke-width="1.8"/>')
        return x1, y1, x2, y2

    box(0, 5, "heading", 12)
    box(0, 1, "designation", 5)
    box(1, 1, "partNumber", 2)
    box(2, 5, "title", 5)
    box(6, 12, "statement", 12)
    box(6, 7, "name", 5)
    box(8, 12, "description", 5)
    box(9, 12, "affiliation", 2)
    box(13, 13, "pageNumber", 7)
    return positions


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="1298" viewBox="0 0 {W} 1298">')
add('<defs>')
for c in sorted(set(COLORS.values()) | {MUTED}):
    add(f'<marker id="a-{c[1:]}" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="1298" rx="16" fill="#f8fafc"/>')
text(32, 46, "Alignment in levels", size=24, weight=800)
text(32, 72, "Values printed inside other values are aligned one level below them, on the words "
             "their container matched.", size=14, color=MUTED)

panel(24, 96, 470, 368, "Input: the page", "ALTO words of a chapter's first page")
page_mock(49, 180)

panel(510, 96, 586, 368, "Input: the JSON and the inside rules")
json_lines = [
    ("{", None), ('  "heading": "Kapitola 3 Lidová píseň v Čechách",', "heading"),
    ('  "designation": "Kapitola 3",', "designation"), ('  "partNumber": "3",', "partNumber"),
    ('  "title": "Lidová píseň v Čechách",', "title"), ('  "agents": [{', None),
    ('    "statement": "Jan Novák profesor Univerzity Karlovy v Praze",', "statement"),
    ('    "name": "Jan Novák",', "name"),
    ('    "description": "profesor Univerzity Karlovy v Praze",', "description"),
    ('    "affiliation": "Univerzity Karlovy v Praze"', "affiliation"),
    ("  }],", None), ('  "pageNumber": "45"', "pageNumber"), ("}", None),
]
for n, (line, key) in enumerate(json_lines):
    ly = 150 + n * 16.5
    if key:
        add(f'<circle cx="531" cy="{ly - 4:.1f}" r="4" fill="{COLORS[key]}"/>')
    text(542, ly, line, size=12, family=MONO, color=INK if key else MUTED)
rules = [("designation", "heading"), ("title", "heading"), ("partNumber", "designation"),
         ("name", "statement"), ("description", "statement"), ("affiliation", "description")]
text(530, 375, "inside_paths", size=12.5, weight=700, color=MUTED)
text(625, 375, "a value is inside when its text is a substring of its container's text",
     size=11.5, color=MUTED)
for n, (value, container) in enumerate(rules):
    rx, ly = (530, 745)[n // 3], 398 + (n % 3) * 19
    gap = (106, 162)[n // 3]
    v = label(value).replace("agents[0]", "agents[a]")
    c = label(container).replace("agents[0]", "agents[a]")
    text(rx, ly, v, size=12, family=MONO, color=COLORS[value], weight=600)
    text(rx + gap, ly, "in", size=11.5, color=FAINT, anchor="middle")
    text(rx + gap + 13, ly, c, size=12, family=MONO, color=COLORS[container], weight=600)


# ---------------------------------------------------------------- levels
panel(24, 480, 1072, 460, "The recursion",
      "Every level runs the same alignment (candidates → word groups → selector) on its own page of words.")

LX = 210  # where the strips start


def level_label(y, n, what, reads):
    add(f'<rect x="44" y="{y - 4}" width="146" height="62" rx="8" fill="#f1f5f9"/>')
    text(58, y + 15, f"Level {n}", size=14, weight=700)
    text(58, y + 33, what, size=11.5, color=MUTED)
    text(58, y + 49, reads, size=11.5, color=MUTED)


# Level 0: the whole page
y0 = 548
level_label(y0, 0, "the whole page", "reads the OCR")
s0 = strip(LX, y0, list(enumerate(OCR)), highlight=())
add(f'<path d="M{s0[9][0] + 5:.1f},{y0 + 24} h{s0[9][1] - s0[9][0] - 10:.1f}" stroke="#dc2626" stroke-width="1.5"/>')
h_anchor = bracket(s0, 0, 5, y0 + 32, "heading", "exact")
st_anchor = bracket(s0, 6, 12, y0 + 32, "statement", "≈ 0.98 vs the OCR")
bracket(s0, 13, 13, y0 + 32, "pageNumber")

# Level 1: the words of heading and of statement
y1 = y0 + 150
level_label(y1, 1, "words of a matched value", "read its JSON text")
s1a = strip(LX, y1, [(i, OCR[i]) for i in range(0, 6)])
s1b_x = LX + 420
s1b = strip(s1b_x, y1, [(i, JSON_TEXT.get(i, OCR[i])) for i in range(6, 13)], highlight=(9,))
arrow(h_anchor[0], h_anchor[1] + 46, s1a[0][0] + 30, y1, COLORS["heading"])
arrow(st_anchor[0], st_anchor[1] + 46, s1b[0][0] + 30, y1, COLORS["statement"])
d_anchor = bracket(s1a, 0, 1, y1 + 32, "designation", "exact")
bracket(s1a, 2, 5, y1 + 32, "title", "exact")
bracket(s1b, 0, 1, y1 + 32, "name", "exact")
de_anchor = bracket(s1b, 2, 6, y1 + 32, "description", "exact")

# Level 2: the words of designation and of description
y2 = y1 + 150
level_label(y2, 2, "words of a value", "found on level 1")
s2a = strip(LX, y2, [(0, OCR[0]), (1, OCR[1])])
s2b = strip(s1b[2][0], y2, [(i, JSON_TEXT.get(i, OCR[i])) for i in range(8, 13)], highlight=(9,))
arrow(d_anchor[0], d_anchor[1] + 46, s2a[0][0] + 30, y2, COLORS["designation"])
arrow(de_anchor[0], de_anchor[1] + 46, s2b[0][0] + 30, y2, COLORS["description"])
bracket(s2a, 1, 1, y2 + 32, "partNumber", "exact")
bracket(s2b, 1, 4, y2 + 32, "affiliation", "exact, although the OCR misread it")

# the note on the JSON text
nx, ny = s1b[3][1] + 10, y1 - 30
add(f'<rect x="{s1b[3][0] - 4:.1f}" y="{ny - 15}" width="236" height="22" rx="11" fill="#fef3c7"/>')
text(s1b[3][0] + 8, ny, "Univerzlty reads Univerzity here", size=11.5, color="#92400e", weight=600)

# ---------------------------------------------------------------- output
panel(24, 956, 470, 322, "Output: nested boxes on the page", "each value's box, coloured as in the JSON")
page_mock(49, 1016, boxes=True)
panel(510, 956, 586, 322, "What the levels give")
points = [
    "A value takes words only within its container's words, so the part",
    "number, the name and the affiliation cannot lose them to the heading,",
    "the statement or the description that print them, nor take them.",
    "",
    "Deeper levels read the container's JSON text over its words: the",
    "longer container text matched the OCR, so a short value inside it",
    "matches the text the container gives (inside_location=json).",
    "",
    "A value whose container is not found, or which is not found among",
    "its container's words, stays unmatched.",
]
for n, line in enumerate(points):
    text(530, 1004 + n * 19, line, size=13, color=INK)

add('</svg>')
svg = "\n".join(out)
OUT.write_text(svg + "\n", encoding="utf-8")
