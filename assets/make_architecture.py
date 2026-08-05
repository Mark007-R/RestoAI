"""Render assets/architecture.png.

Pillow, drawn at 2x and downsampled. Dark card with light text so it reads on
both the GitHub light and dark themes.

Run:  python assets/make_architecture.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 980 * S, 600 * S
OUT = Path(__file__).with_name("architecture.png")

BG, FG, MUTED, LINE = (13, 17, 23), (201, 209, 217), (139, 148, 158), (110, 118, 129)
ACCENT, GREEN, AMBER = (188, 140, 255), (63, 185, 80), (210, 153, 34)
FONTS = r"C:\Windows\Fonts"


def font(n, s):
    return ImageFont.truetype(f"{FONTS}\\{n}", s * S)


f_title, f_head = font("seguisb.ttf", 15), font("seguisb.ttf", 12)
f_small, f_lbl = font("segoeui.ttf", 10), font("segoeuii.ttf", 9)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def box(x, y, w, h, c=LINE, width=2):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=6 * S, outline=c, width=int(width * S))


def text(x, y, s, f=f_small, fill=MUTED, anchor="mm"):
    d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)


def _head(p0, p1, c, size=6):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** .5, 1e-6)
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    s = size * S
    d.polygon([(x1, y1),
               (x1 - ux * s + px * s * .5, y1 - uy * s + py * s * .5),
               (x1 - ux * s - px * s * .5, y1 - uy * s - py * s * .5)], fill=c)


def arrow(pts, c=LINE, w=1.5):
    pts = [(x * S, y * S) for x, y in pts]
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=c, width=int(w * S))
    _head(pts[-2], pts[-1], c)


text(490, 26, "Restaurant-Intelligence-Platform — three NLP components, each measured against its baseline",
     f_title, FG)

# ── data ────────────────────────────────────────────────────────────────────
box(30, 54, 380, 62)
text(220, 76, "5 public review CSVs", f_head, FG)
text(220, 95, "~248k rows -> ~23.8k unique after dedup")

arrow([(410, 85), (446, 85)])
box(448, 54, 350, 62, GREEN)
text(623, 76, "Gold eval sets, stratified across all five", f_head, FG)
text(623, 95, "built before any model was chosen")

# ── three components ────────────────────────────────────────────────────────
arrow([(220, 116), (220, 148)])
arrow([(623, 116), (623, 132), (490, 132), (490, 148)])

cols = [
    (30, "Sentiment", "VADER  ->  NLI zero-shot", "macro-F1 0.505 -> 0.607", ACCENT),
    (355, "Complaint classifier", "keyword  ->  TF-IDF + LightGBM", "macro-F1 0.8335 -> 0.8525", ACCENT),
    (680, "RAG chat", "chunking + FAISS + rerank", "composite 0.680 -> 0.663", AMBER),
]
for x, title, sub, met, c in cols:
    box(x, 150, 300, 84, c)
    text(x + 150, 172, title, f_head, FG)
    text(x + 150, 192, sub)
    text(x + 150, 214, met, f_small, FG)

text(830, 246, "the one that got worse", f_lbl, AMBER, anchor="mm")

# ── production union ────────────────────────────────────────────────────────
arrow([(180, 234), (180, 264)])
arrow([(505, 234), (505, 264)])
box(30, 266, 625, 56)
text(342, 287, "Production classifier unions the trained output with the keyword fallback", f_head, FG)
text(342, 306, "keeps keyword recall on narrow-lexicon categories the model misses")

# ── serving ─────────────────────────────────────────────────────────────────
arrow([(342, 322), (342, 352)])
box(30, 354, 440, 76, ACCENT)
text(250, 376, "FastAPI + Redis  :8000", f_head, FG)
text(250, 395, "/sentiment  /complaints  /rag  — async, Pydantic-validated")
text(250, 413, "/health  /health/cache  /metrics/ragas")

box(500, 354, 450, 76)
text(725, 376, "Flask  :5000", f_head, FG)
text(725, 395, "manager dashboard — analytics, triage, RAG chat")
text(725, 413, "customer app — discovery and booking")

arrow([(470, 392), (498, 392)])

# ── caveat ──────────────────────────────────────────────────────────────────
box(30, 456, 920, 62, AMBER)
text(490, 478, "Frontier-model rows were never run", f_head, FG)
text(490, 497, "Claude and GPT comparisons are recorded as skipped — no API key was present. A local NLI model stood in.")

text(30, 548, "Every figure above comes from a committed artifact under results/ · 88-test suite",
     f_lbl, MUTED, anchor="lm")
text(30, 568, "Docker Compose runs FastAPI + Redis; the Flask apps call the service",
     f_lbl, MUTED, anchor="lm")

img.resize((W // S, H // S), Image.LANCZOS).save(OUT, "PNG", optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
