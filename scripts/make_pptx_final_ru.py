"""
Build docs/vt_analysis_final_ru.pptx from docs/vt_analysis_final_ru.md.

White background, clean professional style.

Usage:
    python -X utf8 scripts/make_pptx_final_ru.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt

REPO     = Path(__file__).resolve().parents[1]
MD_FILE  = REPO / "docs" / "vt_analysis_final_ru.md"
OUT_FILE = REPO / "docs" / "Vt_frequency_analysis_final_ru.pptx"

# ── colour palette ─────────────────────────────────────────────────────────────
C_WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
C_TITLE    = RGBColor(0x1A, 0x27, 0x44)
C_BODY     = RGBColor(0x22, 0x22, 0x22)
C_ACCENT   = RGBColor(0xB3, 0x6A, 0x00)
C_BLUE     = RGBColor(0x1F, 0x5C, 0x99)
C_GREEN    = RGBColor(0x15, 0x6B, 0x3C)
C_RED      = RGBColor(0xAA, 0x22, 0x22)
C_GRAY     = RGBColor(0x55, 0x55, 0x55)
C_RULE     = RGBColor(0x1A, 0x27, 0x44)
C_TBL_HDR  = RGBColor(0x1A, 0x27, 0x44)
C_TBL_ALT  = RGBColor(0xEE, 0xF2, 0xF7)
C_TBL_TXT  = RGBColor(0xFF, 0xFF, 0xFF)
C_TBL_BODY = RGBColor(0x22, 0x22, 0x22)

# ── slide dimensions 16:9 ─────────────────────────────────────────────────────
W = Cm(33.87)
H = Cm(19.05)

MARGIN      = Cm(1.1)
TITLE_H     = Cm(1.9)
RULE_H      = Cm(0.07)
CONTENT_TOP = int(TITLE_H + RULE_H + Cm(0.3))
CONTENT_H   = int(H - CONTENT_TOP - MARGIN)


def _img_fit(img_path: Path, max_w: int, max_h: int) -> tuple[int, int, int, int]:
    try:
        with PILImage.open(img_path) as im:
            iw, ih = im.size
        scale = min(max_w / iw, max_h / ih)
        w, h = int(iw * scale), int(ih * scale)
    except Exception:
        w, h = max_w, max_h
    return (max_w - w) // 2, (max_h - h) // 2, w, h


def _place_image(slide, img_path: Path,
                 left: int, top: int, max_w: int, max_h: int) -> bool:
    if not img_path.exists():
        print(f"  [skip img] {img_path.name}")
        return False
    try:
        xoff, yoff, w, h = _img_fit(img_path, max_w, max_h)
        slide.shapes.add_picture(str(img_path), left + xoff, top + yoff, w, h)
        return True
    except Exception as e:
        print(f"  [warn] {img_path.name}: {e}")
        return False


def _white_bg(slide) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = C_WHITE


def _add_title(slide, title: str) -> None:
    txb = slide.shapes.add_textbox(
        int(MARGIN), int(Cm(0.25)),
        int(W - 2 * MARGIN), int(TITLE_H - Cm(0.25))
    )
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size  = Pt(22)
    r.font.bold  = True
    r.font.color.rgb = C_TITLE
    r.font.name  = "Calibri"

    rule = slide.shapes.add_shape(
        1, 0, int(TITLE_H), int(W), int(RULE_H)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = C_RULE
    rule.line.fill.background()


def _strip(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"`(.+?)`",        r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def _inline_runs(para, text: str, size: Pt, color: RGBColor) -> None:
    for seg in re.split(r"(\*\*[^*]+\*\*)", text):
        r = para.add_run()
        if seg.startswith("**") and seg.endswith("**"):
            inner = seg[2:-2]
            r.text = inner
            r.font.bold  = True
            r.font.color.rgb = C_ACCENT
        else:
            r.text = seg
            r.font.color.rgb = color
        r.font.size = size
        r.font.name = "Calibri"


def _add_body(slide, lines: list[str],
              top: int, left: int | None = None,
              width: int | None = None) -> None:
    if not lines:
        return
    left  = left  if left  is not None else int(MARGIN)
    width = width if width is not None else int(W - 2 * MARGIN)

    txb = slide.shapes.add_textbox(left, top, width, int(H - top - MARGIN))
    tf  = txb.text_frame
    tf.word_wrap = True

    first = True
    for raw in lines:
        line = raw.rstrip()
        if not line and first:
            continue
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_before = Pt(2)
        p.space_after  = Pt(1)

        if line.startswith("> "):
            inner = line[2:]
            p.level = 1
            r = p.add_run()
            r.text = _strip(inner)
            r.font.size   = Pt(14)
            r.font.italic = True
            r.font.color.rgb = C_ACCENT
            r.font.name   = "Calibri"
        elif line.startswith("### "):
            r = p.add_run()
            r.text = line[4:]
            r.font.size  = Pt(16)
            r.font.bold  = True
            r.font.color.rgb = C_BLUE
            r.font.name  = "Calibri"
        elif line.startswith("#### "):
            r = p.add_run()
            r.text = line[5:]
            r.font.size  = Pt(14)
            r.font.bold  = True
            r.font.color.rgb = C_GRAY
            r.font.name  = "Calibri"
        elif re.match(r"^[-*•] ", line):
            p.level = 1
            _inline_runs(p, "• " + _strip(line[2:]), Pt(14), C_BODY)
        elif re.match(r"^ {2,4}[-*] ", line):
            p.level = 2
            text = re.sub(r"^ {2,4}[-*] ", "", line)
            _inline_runs(p, "  ◦ " + _strip(text), Pt(12), C_GRAY)
        elif line.startswith("```"):
            pass
        else:
            _inline_runs(p, _strip(line), Pt(14), C_BODY)


def _add_table(slide, rows: list[list[str]],
               top: int, left: int | None = None,
               width: int | None = None) -> int:
    if not rows:
        return top
    left  = left  if left  is not None else int(MARGIN)
    width = width if width is not None else int(W - 2 * MARGIN)
    ncols = max(len(r) for r in rows)

    row_h    = int(Cm(0.65))
    avail_h  = int(H - top - MARGIN)
    max_rows = max(1, avail_h // row_h)
    rows     = rows[:max_rows]
    nrows    = len(rows)

    tbl = slide.shapes.add_table(
        nrows, ncols, left, top, width, row_h * nrows
    ).table

    col_w = width // ncols
    for ci in range(ncols):
        tbl.columns[ci].width = col_w

    for ri, row in enumerate(rows):
        is_hdr = (ri == 0)
        while len(row) < ncols:
            row.append("")
        for ci, cell_text in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = _strip(cell_text.strip())
            f = cell.fill
            f.solid()
            if is_hdr:
                f.fore_color.rgb = C_TBL_HDR
            elif ri % 2 == 0:
                f.fore_color.rgb = C_TBL_ALT
            else:
                f.fore_color.rgb = C_WHITE
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT
            run = para.runs[0] if para.runs else para.add_run()
            run.font.name  = "Calibri"
            run.font.size  = Pt(11)
            run.font.bold  = is_hdr
            run.font.color.rgb = C_TBL_TXT if is_hdr else C_TBL_BODY

    return top + row_h * nrows


# ── markdown parser ────────────────────────────────────────────────────────────

class Slide:
    def __init__(self):
        self.title: str = ""
        self.is_cover: bool = False
        self.body: list[str] = []
        self.tables: list[list[list[str]]] = []
        self.images: list[Path] = []


def _parse_table_block(lines: list[str]) -> list[list[str]]:
    rows = []
    for ln in lines:
        ln = ln.strip()
        if re.match(r"^\|[-| :]+\|$", ln):
            continue
        if ln.startswith("|") and ln.endswith("|"):
            rows.append([c.strip() for c in ln[1:-1].split("|")])
    return rows


def parse_slides(path: Path) -> list[Slide]:
    text   = path.read_text(encoding="utf-8")
    blocks = re.split(r"^---\s*$", text, flags=re.MULTILINE)
    slides = []

    for block in blocks:
        block = block.strip()
        if not block or re.match(r"^<!--.*-->$", block, re.DOTALL):
            continue

        s = Slide()
        lines = block.splitlines()
        tbl_buf: list[str] = []
        in_tbl = False

        if lines and lines[0].strip() == "cover":
            s.is_cover = True
            lines = lines[1:]

        for ln in lines:
            if ln.startswith("## "):
                s.title = ln[3:].strip()
                continue

            img_m = re.match(r"!\[[^\]]*\]\((.+?)\)", ln)
            if img_m:
                if in_tbl:
                    s.tables.append(_parse_table_block(tbl_buf))
                    tbl_buf, in_tbl = [], False
                s.images.append(REPO / img_m.group(1).strip())
                continue

            if ln.strip().startswith("|"):
                in_tbl = True
                tbl_buf.append(ln)
            else:
                if in_tbl:
                    s.tables.append(_parse_table_block(tbl_buf))
                    tbl_buf, in_tbl = [], False
                s.body.append(ln)

        if in_tbl:
            s.tables.append(_parse_table_block(tbl_buf))

        slides.append(s)
    return slides


# ── layout ────────────────────────────────────────────────────────────────────

def _render_content(slide, s: Slide, top: int, left: int, width: int) -> None:
    _add_body(slide, s.body, top, left, width)

    text_lines = [l for l in s.body if l.strip()]
    heading_lines = [l for l in text_lines if l.startswith(("#", ">"))]
    bullet_lines  = [l for l in text_lines if re.match(r"^[-*•] ", l) or re.match(r"^ {2}", l)]
    plain_lines   = [l for l in text_lines
                     if l not in heading_lines and l not in bullet_lines]

    est_h = (len(heading_lines) * int(Cm(0.75))
             + len(bullet_lines) * int(Cm(0.62))
             + len(plain_lines)  * int(Cm(0.57)))
    tbl_top = top + max(est_h, int(Cm(0.5))) + int(Cm(0.25))

    for tbl_data in s.tables:
        if not tbl_data:
            continue
        tbl_top = _add_table(slide, tbl_data, tbl_top, left, width) + int(Cm(0.3))


def _build_slide(prs: Presentation, s: Slide) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _white_bg(slide)
    _add_title(slide, s.title)

    n_img    = len(s.images)
    has_text = bool(s.body or s.tables)

    if n_img == 0:
        _render_content(slide, s, CONTENT_TOP, int(MARGIN), int(W - 2 * MARGIN))

    elif n_img == 1:
        img = s.images[0]
        if not has_text:
            _place_image(slide, img,
                         int(MARGIN), CONTENT_TOP,
                         int(W - 2 * MARGIN), CONTENT_H)
        else:
            text_w = int(W * 0.42)
            gap    = int(Cm(0.5))
            img_l  = int(MARGIN) + text_w + gap
            img_w  = int(W - img_l - MARGIN)
            _render_content(slide, s, CONTENT_TOP, int(MARGIN), text_w)
            _place_image(slide, img, img_l, CONTENT_TOP, img_w, CONTENT_H)

    elif n_img == 2:
        # Two images side by side in upper 55%, text below
        img_h = int(CONTENT_H * 0.54)
        img_w = int((W - 3 * MARGIN) / 2)
        _place_image(slide, s.images[0], int(MARGIN), CONTENT_TOP, img_w, img_h)
        _place_image(slide, s.images[1],
                     int(MARGIN) + img_w + int(MARGIN),
                     CONTENT_TOP, img_w, img_h)
        text_top = CONTENT_TOP + img_h + int(Cm(0.25))
        _render_content(slide, s, text_top, int(MARGIN), int(W - 2 * MARGIN))

    else:
        # 3+ images: first two side-by-side, rest discarded
        img_h = int(CONTENT_H * 0.54)
        img_w = int((W - 3 * MARGIN) / 2)
        _place_image(slide, s.images[0], int(MARGIN), CONTENT_TOP, img_w, img_h)
        _place_image(slide, s.images[1],
                     int(MARGIN) + img_w + int(MARGIN),
                     CONTENT_TOP, img_w, img_h)
        text_top = CONTENT_TOP + img_h + int(Cm(0.25))
        _render_content(slide, s, text_top, int(MARGIN), int(W - 2 * MARGIN))
        if n_img > 2:
            print(f"  [note] {s.title[:40]}: {n_img} images, only first 2 used")


def _build_cover(prs: Presentation, s: Slide) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _white_bg(slide)

    # Accent bar on left
    bar = slide.shapes.add_shape(1, 0, 0, int(Cm(1.2)), int(H))
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_TITLE
    bar.line.fill.background()

    # Main title — split on \n or "." in the title
    parts = s.title.split(".")
    title_text = s.title

    txb = slide.shapes.add_textbox(
        int(Cm(2.0)), int(Cm(3.5)),
        int(W - Cm(3.0)), int(Cm(5.5))
    )
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title_text
    r.font.size  = Pt(30)
    r.font.bold  = True
    r.font.color.rgb = C_TITLE
    r.font.name  = "Calibri"

    _add_body(slide, s.body, int(Cm(10.0)), int(Cm(2.0)), int(W - Cm(3.0)))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    slides = parse_slides(MD_FILE)
    print(f"Parsed {len(slides)} slides from {MD_FILE.name}")

    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    for i, s in enumerate(slides):
        if s.is_cover:
            _build_cover(prs, s)
        else:
            _build_slide(prs, s)

        found  = sum(1 for p in s.images if p.exists())
        n      = len(s.images)
        status = f"imgs {found}/{n}" if n else "text-only"
        print(f"  [{i:02d}] '{s.title[:55]}' {status}")

    prs.save(str(OUT_FILE))
    kb = OUT_FILE.stat().st_size // 1024
    print(f"\nSaved: {OUT_FILE} ({kb} KB, {len(slides)} slides)")


if __name__ == "__main__":
    main()
