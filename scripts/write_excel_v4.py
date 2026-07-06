"""
Write v4 computed parameters into ВТЛУ_Программа УВЧ.xlsx
  - Updates cols 12-26 (current + after-uplift RUL) with v4 values
  - Adds cols 27-33: rul_best, rul_max, d_rul_best, d_rul_max, hr_cox, band, RAG
Output: docs/uvch_claude/ВТЛУ_Программа УВЧ v4.xlsx
"""
import sys, os, io, shutil, math
sys.stdout.reconfigure(encoding='utf-8')

import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

REPO     = r"d:\GitHub\Pump2"
ANALYSIS = os.path.join(REPO, "docs", "uvch_claude")
SRC_XLS  = os.path.join(ANALYSIS, "ВТЛУ_Программа УВЧ.xlsx")
DST_XLS  = os.path.join(ANALYSIS, "ВТЛУ_Программа УВЧ v4.xlsx")

# ── Run analysis.py to get v4 results ─────────────────────────────────────────
src = open(os.path.join(ANALYSIS, "analysis.py"), encoding="utf-8").read()
src = src.replace('"01_screening.csv"', '"01_screening_v4.csv"') \
         .replace('"02_detailed.csv"',  '"02_detailed_v4.csv"')

import builtins
orig_open = builtins.open
def redirect(path, mode="r", *a, **kw):
    if isinstance(path, str) and "_v4.csv" in path and "w" in str(mode):
        return io.StringIO()
    return orig_open(path, mode, *a, **kw)
builtins.open = redirect
ns = {}
try:
    exec(src, ns)
finally:
    builtins.open = orig_open

v4_results = {r["well"]: r for r in ns["results"]}

# ── Color palettes ─────────────────────────────────────────────────────────────
FILL_GREEN  = PatternFill("solid", fgColor="C6EFCE")
FILL_YELLOW = PatternFill("solid", fgColor="FFEB9C")
FILL_RED    = PatternFill("solid", fgColor="FFC7CE")
FILL_HEADER = PatternFill("solid", fgColor="D9E1F2")
FILL_V4_HDR = PatternFill("solid", fgColor="FCE4D6")  # orange-ish for new v4 block
FILL_BEST   = PatternFill("solid", fgColor="E2EFDA")  # light green for rul_best
FILL_MAX    = PatternFill("solid", fgColor="DDEBF7")  # light blue for rul_max

RAG_FILL = {"Green": FILL_GREEN, "Yellow": FILL_YELLOW, "Red": FILL_RED}

def thin_border():
    s = Side(style="thin")
    return Border(left=s, right=s, top=s, bottom=s)

CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
RIGHT  = Alignment(horizontal="right", vertical="center")

# ── Load source workbook ───────────────────────────────────────────────────────
shutil.copy2(SRC_XLS, DST_XLS)
wb = openpyxl.load_workbook(DST_XLS)
ws = wb["СВОД"]

# ── Map well name → row number ─────────────────────────────────────────────────
well_row = {}
for row in ws.iter_rows(min_row=5, max_row=61):
    cell = row[2]  # col C (index 2)
    if cell.value and str(cell.value).startswith(("Vt_", "Bt_")):
        well_row[cell.value] = cell.row

print(f"Found {len(well_row)} wells in Excel")

# ── Helpers ────────────────────────────────────────────────────────────────────
def set_cell(ws, row, col, value, fill=None, font=None, alignment=RIGHT, num_fmt=None):
    c = ws.cell(row=row, column=col, value=value)
    if fill:      c.fill = fill
    if font:      c.font = font
    if alignment: c.alignment = alignment
    if num_fmt:   c.number_format = num_fmt
    return c

def header(ws, row, col, text, fill=FILL_HEADER):
    c = set_cell(ws, row, col, text, fill=fill, alignment=CENTER)
    c.font = Font(bold=True, size=9)
    return c

# ── Update section header row 3 for new v4 block ──────────────────────────────
header(ws, 3, 27, "v4 — Анализ рисков (Компонент 2 + Cox)", fill=FILL_V4_HDR)
ws.merge_cells(start_row=3, start_column=27, end_row=3, end_column=33)

# ── Update col 4 header row 3 (ensure it spans correctly) ─────────────────────
# (these already exist in the template, just ensure v4 context is clear)

# ── New column headers in row 4 ───────────────────────────────────────────────
new_col_headers = {
    27: "ОР лучш. процесс P50, сут",
    28: "ОР теор. макс. P50, сут",
    29: "Δ подрядчик, сут",
    30: "Δ H2S пенальти, сут",
    31: "HR Cox (хим.)",
    32: "Полоса УВЧ",
    33: "RAG",
}
for col, text in new_col_headers.items():
    header(ws, 4, col, text, fill=FILL_V4_HDR)

# ── Remove old legend in col 28 rows 5-9 ──────────────────────────────────────
for r in range(5, 10):
    for c in [28, 29]:
        ws.cell(row=r, column=c).value = None

# ── Write per-well v4 data ─────────────────────────────────────────────────────
fmt_int  = "0"
fmt_pct  = "0.0"
fmt_hr   = "0.000"

for well, row in well_row.items():
    r = v4_results.get(well)
    if r is None:
        print(f"  WARNING: no v4 data for {well}")
        continue

    nno = r["nno"]

    # ── Current RUL (cols 12-15) — v4 values
    set_cell(ws, row, 12, round(r["rul_p10_c"]),  num_fmt=fmt_int)
    set_cell(ws, row, 13, round(r["rul_med_c"]),  num_fmt=fmt_int)
    set_cell(ws, row, 14, round(r["rul_p90_c"]),  num_fmt=fmt_int)
    set_cell(ws, row, 15, round(r["proj_cur"]),   num_fmt=fmt_int)

    # ── After uplift RUL (cols 21-26) — v4 values
    set_cell(ws, row, 21, round(r["rul_p10_a"]),     num_fmt=fmt_int)
    set_cell(ws, row, 22, round(r["rul_med_a"]),     num_fmt=fmt_int)
    set_cell(ws, row, 23, round(r["rul_p90_a"]),     num_fmt=fmt_int)
    set_cell(ws, row, 24, round(r["proj_aft"]),      num_fmt=fmt_int)
    set_cell(ws, row, 25, round(r["delta_rul_pct"], 1), num_fmt=fmt_pct)
    # ΔННО% = (proj_aft - proj_cur) / proj_cur * 100
    dnno_pct = round((r["proj_aft"] - r["proj_cur"]) / r["proj_cur"] * 100, 1) if r["proj_cur"] > 0 else 0.0
    set_cell(ws, row, 26, dnno_pct, num_fmt=fmt_pct)

    # ── v4 new columns ────────────────────────────────────────────────────────
    rul_best = round(r["rul_best"])
    rul_max  = round(r["rul_max"])
    d_best   = round(r["d_rul_best"])
    d_max    = round(r["d_rul_max"])
    hr_cox   = round(float(r["hr_cox"]), 3)
    band     = r["band"]
    rag      = r["rag_n"]   # without motor proxy

    set_cell(ws, row, 27, rul_best, fill=FILL_BEST if d_best > 0 else None, num_fmt=fmt_int)
    set_cell(ws, row, 28, rul_max,  fill=FILL_MAX  if d_max  > 0 else None, num_fmt=fmt_int)
    set_cell(ws, row, 29, d_best,   num_fmt=fmt_int)
    set_cell(ws, row, 30, d_max,    num_fmt=fmt_int)
    set_cell(ws, row, 31, hr_cox,   num_fmt=fmt_hr)
    set_cell(ws, row, 32, band,     alignment=CENTER)
    # RAG with color fill
    c_rag = set_cell(ws, row, 33, rag, fill=RAG_FILL.get(rag), alignment=CENTER)
    c_rag.font = Font(bold=True, size=9)

    # Color-code current P50 RUL by RAG
    fill_cur = RAG_FILL.get(rag)
    ws.cell(row=row, column=13).fill = fill_cur or PatternFill()

# ── Column widths ──────────────────────────────────────────────────────────────
col_widths = {
    27: 14, 28: 14, 29: 12, 30: 12, 31: 8, 32: 7, 33: 8
}
for col, width in col_widths.items():
    ws.column_dimensions[get_column_letter(col)].width = width

# ── Add legend block (cols 35-36, rows 5-12) ──────────────────────────────────
legend = [
    ("ОР", "Остаточный ресурс"),
    ("P10", "90% скважин переживают этот порог"),
    ("P50", "Медиана (50% переживают)"),
    ("P90", "10% скважин переживают этот порог"),
    ("rul_best", "Лучший процесс: Компонент 2 × Cox(H2S)"),
    ("rul_max",  "Теор. макс: Компонент 2, без рисков"),
    ("Δ подрядчик", "Прирост ОР от смены подрядчика (без НТ)"),
    ("Δ H2S", "Геол. потеря ОР от H2S (неустранима)"),
]
header(ws, 4, 35, "Обозначение", fill=FILL_HEADER)
header(ws, 4, 36, "Описание",    fill=FILL_HEADER)
ws.column_dimensions[get_column_letter(35)].width = 14
ws.column_dimensions[get_column_letter(36)].width = 36
for i, (abbr, desc) in enumerate(legend, 5):
    ws.cell(row=i, column=35, value=abbr)
    ws.cell(row=i, column=36, value=desc)

# ── Portfolio stats block (row 63-70) ─────────────────────────────────────────
all_r = list(v4_results.values())
def median(vals):
    s = sorted(vals)
    n = len(s)
    return (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]

stats = [
    ("Медиана ОР P50 текущий (v4)", f"{median([round(r['rul_med_c']) for r in all_r]):.0f} сут"),
    ("Медиана ОР P50 после УВЧ (v4)", f"{median([round(r['rul_med_a']) for r in all_r]):.0f} сут"),
    ("Медиана rul_best (лучш. процесс)", f"{median([round(r['rul_best']) for r in all_r]):.0f} сут"),
    ("Медиана rul_max (теор. макс.)",   f"{median([round(r['rul_max']) for r in all_r]):.0f} сут"),
    ("Ср. Δ подрядчик (d_rul_best)",   f"+{sum(float(r['d_rul_best']) for r in all_r)/len(all_r):.1f} сут"),
    ("Ср. Δ H2S пенальти (d_rul_max)", f"+{sum(float(r['d_rul_max']) for r in all_r)/len(all_r):.1f} сут"),
    ("RAG Green (без proxy)",  str(sum(1 for r in all_r if r['rag_n']=='Green'))),
    ("RAG Yellow",             str(sum(1 for r in all_r if r['rag_n']=='Yellow'))),
    ("RAG Red",                str(sum(1 for r in all_r if r['rag_n']=='Red'))),
]
ws.cell(row=62, column=3, value="Сводная статистика v4").font = Font(bold=True)
for i, (lbl, val) in enumerate(stats, 63):
    ws.cell(row=i, column=3, value=lbl)
    ws.cell(row=i, column=4, value=val)

# ── Save ───────────────────────────────────────────────────────────────────────
wb.save(DST_XLS)
print(f"\nSaved: {DST_XLS}")
print(f"Wells updated: {len(well_row)}")
print(f"New columns: 27-33 (rul_best, rul_max, d_best, d_max, hr_cox, band, RAG)")
