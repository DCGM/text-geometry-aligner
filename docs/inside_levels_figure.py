"""Draw docs/inside-levels.svg, the README figure of the alignment in levels.

The example is the one the figure shows: a chapter page whose JSON names
whole sections (the heading, the author statement) and the values printed
inside them (inside_paths). The page's running head repeats a part number
and an affiliation, so those short values, aligned on their own, take the
running head's words; aligned in levels they stay within their sections.
The results drawn are those of the aligner on this input.
Run: python docs/inside_levels_figure.py
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).with_name("inside-levels.svg")

W, H = 1120, 1360
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"
CW = 7.83  # monospace character width at 13px
INK, MUTED, FAINT, LINE = "#1e293b", "#64748b", "#94a3b8", "#e2e8f0"
GOOD, BAD = "#15803d", "#dc2626"

COLORS = {
    "heading": "#2563eb", "designation": "#7c3aed", "partNumber": "#c026d3", "title": "#0891b2",
    "statement": "#15803d", "name": "#b45309", "description": "#ea580c", "affiliation": "#e11d48",
    "running": "#94a3b8",
}
LABELS = {"statement": "agents[0].statement", "name": "agents[0].name",
          "description": "agents[0].description", "affiliation": "agents[0].affiliation"}

WORDS = "Sborník Univerzity Karlovy 3 Kapitola 3 Lidová píseň Jan Novák profesor Univerzity Karlovy".split()
RUNNING = range(0, 4)  # the running head, in no JSON value
LINES = [(0, 4), (4, 8), (8, 10), (10, 13)]  # word ranges of the printed lines

# The words each value takes (TextAligner on this input).
IN_LEVELS = {"heading": (4, 7), "designation": (4, 5), "partNumber": (5, 5), "title": (6, 7),
             "statement": (8, 12), "name": (8, 9), "description": (10, 12), "affiliation": (11, 12)}
ON_THEIR_OWN = dict(IN_LEVELS, partNumber=(3, 3), affiliation=(1, 2))

out = []


def add(s):
    out.append(s)


def text(x, y, s, size=13, color=INK, weight=400, anchor="start", family=FONT):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')


def label(key):
    return LABELS.get(key, key)


def panel(x, y, w, h, title, subtitle=None):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="{LINE}"/>')
    text(x + 18, y + 28, title, size=15, weight=700)
    if subtitle:
        text(x + 18, y + 47, subtitle, size=12.5, color=MUTED)


def mark(x, y, good):
    """A tick or a cross in a disc."""
    c = GOOD if good else BAD
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{c}"/>')
    if good:
        add(f'<path d="M{x - 3.6:.1f},{y:.1f} l2.6,2.8 l4.8,-5.4" fill="none" stroke="#fff" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        add(f'<path d="M{x - 3:.1f},{y - 3:.1f} l6,6 M{x + 3:.1f},{y - 3:.1f} l-6,6" stroke="#fff" '
            f'stroke-width="2" stroke-linecap="round"/>')


# ------------------------------------------------------------------ the page
def page(x, y, matches=None, judged=()):
    """The printed page; with `matches`, each value's box on it."""
    add(f'<rect x="{x}" y="{y}" width="420" height="232" rx="4" fill="#fffdf7" stroke="#e7e1cf"/>')
    positions = {}
    styles = [(y + 34, 11.5, 400, MUTED), (y + 98, 17, 700, INK), (y + 160, 13.5, 400, INK),
              (y + 190, 13.5, 400, INK)]
    for (first, last), (ly, size, weight, color) in zip(LINES, styles):
        cw = size * 0.602
        widths = [len(WORDS[i]) * cw for i in range(first, last)]
        gap = 14
        total = sum(widths) + gap * (last - first - 1)
        cx = x + (420 - total) / 2 if first else x + 420 - 28 - total
        for i, wd in zip(range(first, last), widths):
            positions[i] = (cx, ly, wd, size)
            text(cx, ly, WORDS[i], size=size, weight=weight, family=MONO, color=color)
            cx += wd + gap
    add(f'<path d="M{x + 28},{y + 46} H{x + 392}" stroke="#d6cfba" stroke-width="1"/>')
    text(x + 28, y + 34, "running head", size=10.5, color=FAINT, family=FONT)
    if matches is None:
        return

    pads = {"heading": 12, "statement": 12, "designation": 5, "title": 5, "name": 5, "description": 5,
            "partNumber": 2, "affiliation": 2}
    for key in ["heading", "statement", "designation", "title", "name", "description", "partNumber",
                "affiliation"]:
        first, last = matches[key]
        xs = [positions[i] for i in range(first, last + 1)]
        pad = pads[key]
        x1 = min(p[0] for p in xs) - pad
        x2 = max(p[0] + p[2] for p in xs) + pad
        y1 = min(p[1] - p[3] * 0.85 for p in xs) - pad
        y2 = max(p[1] + p[3] * 0.25 for p in xs) + pad
        c = COLORS[key]
        add(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" height="{y2 - y1:.1f}" rx="4" '
            f'fill="{c}" fill-opacity="0.07" stroke="{c}" stroke-width="1.8"/>')
        if key in judged:
            good = matches[key] == IN_LEVELS[key]
            mark((x1 + x2) / 2, y2 + 9, good)


# ------------------------------------------------------------------ strips
def word_width(s):
    return len(s) * CW + 14


def strip(x, y, indexes):
    """Word boxes in a row; returns each word's x extent by index."""
    spans = {}
    for index in indexes:
        word = WORDS[index]
        w = word_width(word)
        running = index in RUNNING
        add(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="26" rx="5" '
            f'fill="{"#f1f5f9" if running else "#f8fafc"}" stroke="{"#e2e8f0" if running else "#cbd5e1"}"/>')
        text(x + w / 2, y + 17.5, word, family=MONO, anchor="middle", color=FAINT if running else INK)
        spans[index] = (x, x + w)
        x += w + 6
    return spans


def bracket(spans, first, last, y, key, note=None, note_color=MUTED):
    x1, x2 = spans[first][0] + 2, spans[last][1] - 2
    c = COLORS[key]
    dash = ' stroke-dasharray="4 3"' if key == "running" else ""
    add(f'<path d="M{x1:.1f},{y} v7 H{x2:.1f} v-7" fill="none" stroke="{c}" stroke-width="2.2" '
        f'stroke-linejoin="round"{dash}/>')
    name = "running head" if key == "running" else label(key)
    text((x1 + x2) / 2, y + 23, name, size=12.5, color=c, weight=600, anchor="middle",
         family=FONT if key == "running" else MONO)
    if note:
        text((x1 + x2) / 2, y + 39, note, size=11.5, color=note_color, anchor="middle")
    return x1 + 14, y + 46


def arrow(x1, y1, x2, y2, color):
    add(f'<path d="M{x1:.1f},{y1:.1f} C{x1:.1f},{y1 + 30:.1f} {x2:.1f},{y2 - 34:.1f} {x2:.1f},{y2 - 6:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-dasharray="5 4" '
        f'marker-end="url(#a-{color[1:]})"/>')


def level_label(y, n, what, reads):
    add(f'<rect x="44" y="{y - 4}" width="146" height="62" rx="8" fill="#f1f5f9"/>')
    text(58, y + 15, f"Level {n}", size=14, weight=700)
    text(58, y + 33, what, size=11.5, color=MUTED)
    text(58, y + 49, reads, size=11.5, color=MUTED)


# ------------------------------------------------------------------ figure
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
for c in sorted(set(COLORS.values())):
    add(f'<marker id="a-{c[1:]}" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" rx="16" fill="#f8fafc"/>')
text(32, 46, "Alignment in levels", size=24, weight=800)
text(32, 72, "A long section matches its place on the page surely; the short values printed inside it "
             "are looked for only among its words.", size=14, color=MUTED)

# input
panel(24, 96, 470, 330, "Input: the page", "ALTO words of a chapter's first page")
page(49, 166)

panel(510, 96, 586, 330, "Input: the JSON and the inside rules")
json_lines = [
    ("{", None), ('  "heading": "Kapitola 3 Lidová píseň",', "heading"),
    ('  "designation": "Kapitola 3",', "designation"), ('  "partNumber": "3",', "partNumber"),
    ('  "title": "Lidová píseň",', "title"), ('  "agents": [{', None),
    ('    "statement": "Jan Novák profesor Univerzity Karlovy",', "statement"),
    ('    "name": "Jan Novák",', "name"),
    ('    "description": "profesor Univerzity Karlovy",', "description"),
    ('    "affiliation": "Univerzity Karlovy"', "affiliation"),
    ("  }]", None), ("}", None),
]
for n, (line, key) in enumerate(json_lines):
    ly = 146 + n * 16.5
    if key:
        add(f'<circle cx="531" cy="{ly - 4:.1f}" r="4" fill="{COLORS[key]}"/>')
    text(542, ly, line, size=12, family=MONO, color=INK if key else MUTED)
rules = [("designation", "heading"), ("title", "heading"), ("partNumber", "designation"),
         ("name", "statement"), ("description", "statement"), ("affiliation", "description")]
text(530, 360, "inside_paths", size=12.5, weight=700, color=MUTED)
text(625, 360, "a value is inside when its text is a substring of its container's text",
     size=11.5, color=MUTED)
for n, (value, container) in enumerate(rules):
    rx, ly = (530, 745)[n // 3], 383 + (n % 3) * 19
    gap = (106, 162)[n // 3]
    v = label(value).replace("agents[0]", "agents[a]")
    c = label(container).replace("agents[0]", "agents[a]")
    text(rx, ly, v, size=12, family=MONO, color=COLORS[value], weight=600)
    text(rx + gap, ly, "in", size=11.5, color=FAINT, anchor="middle")
    text(rx + gap + 13, ly, c, size=12, family=MONO, color=COLORS[container], weight=600)

# the recursion
panel(24, 442, 1072, 490, "The recursion",
      "Every level runs the same alignment (candidates → word groups → selector) on its own page of words.")
LX = 216

y0 = 512
level_label(y0, 0, "the whole page", "reads the OCR")
s0 = strip(LX, y0, range(13))
bracket(s0, 0, 3, y0 + 32, "running", "in no value: holds a 3 and", FAINT)
text((s0[0][0] + s0[3][1]) / 2, y0 + 87, "Univerzity Karlovy as well", size=11.5, color=FAINT, anchor="middle")
h_from = bracket(s0, 4, 7, y0 + 32, "heading", "long: one place on the page")
st_from = bracket(s0, 8, 12, y0 + 32, "statement", "long: one place on the page")

y1 = y0 + 158
level_label(y1, 1, "words of a matched value", "read its JSON text")
s1a = strip(s0[4][0], y1, range(4, 8))
s1b = strip(s0[8][0], y1, range(8, 13))
arrow(*h_from, s1a[4][0] + 24, y1, COLORS["heading"])
arrow(*st_from, s1b[8][0] + 24, y1, COLORS["statement"])
d_from = bracket(s1a, 4, 5, y1 + 32, "designation")
bracket(s1a, 6, 7, y1 + 32, "title")
bracket(s1b, 8, 9, y1 + 32, "name")
de_from = bracket(s1b, 10, 12, y1 + 32, "description")

y2 = y1 + 158
level_label(y2, 2, "words of a value", "found on level 1")
s2a = strip(s1a[4][0], y2, range(4, 6))
s2b = strip(s1b[10][0], y2, range(10, 13))
arrow(*d_from, s2a[4][0] + 24, y2, COLORS["designation"])
arrow(*de_from, s2b[10][0] + 24, y2, COLORS["description"])
bracket(s2a, 5, 5, y2 + 32, "partNumber", "the 3 of Kapitola 3", GOOD)
bracket(s2b, 11, 12, y2 + 32, "affiliation", "the author's, not the running head's", GOOD)

# the output, on their own and in levels
panel(24, 948, 528, 316, "Each value aligned on its own",
      "a short value takes the first place its text is printed")
page(78, 1010, ON_THEIR_OWN, judged=("partNumber", "affiliation"))
panel(568, 948, 528, 316, "Aligned in levels",
      "a value inside another takes words only among its container's")
page(622, 1010, IN_LEVELS, judged=("partNumber", "affiliation"))

panel(24, 1280, 1072, 56, "")
text(44, 1313, "A value whose container is not found, or which is not found among its container's words, "
               "stays unmatched: looking for it anywhere else could only find a wrong place.",
     size=13, color=INK)

add('</svg>')
OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
