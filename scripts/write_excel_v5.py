"""
Write v5 computed parameters into ВТЛУ_Программа УВЧ v5.xlsx

Column layout:
  12: rul_med_c      — ОР P50 текущий
  13: rul_best       — ОР P50 текущий, лучш. процесс
  14: d_contr_c      — Δ подрядчик (текущий)
  15: proj_cur       — ННО прогноз текущий

  21: rul_med_a      — ОР P50 после УВЧ
  22: rul_best_a     — ОР P50 лучш. процесс (после)
  23: rul_max        — ОР теор. макс (Component 2, hr=1)
  24: proj_aft       — ННО прогноз после УВЧ
  25: d_uvch         — Δ УВЧ, сут
  26: d_contr_a      — Δ подрядчик (после), сут
  27: d_h2s          — Δ H2S, сут
  28: d_chem         — Δ хим. суммарно (H2S+Ca+SO4), сут
  29: hr_cox         — HR Cox полный
  30: hr_geo         — HR Cox геол. (без НТ)
  31: band           — Полоса УВЧ
  32: RAG            — RAG (без proxy)
"""
import sys, os, io, builtins, shutil
sys.stdout.reconfigure(encoding='utf-8')

import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

ARCHIVE  = r"d:\GitHub\Pump2\archive\uvch_claude"
SRC_XLS  = os.path.join(ARCHIVE, "ВТЛУ_Программа УВЧ.xlsx")
DST_XLS  = os.path.join(ARCHIVE, "ВТЛУ_Программа УВЧ v5.xlsx")

# ── Run analysis.py ────────────────────────────────────────────────────────────
ana_path = os.path.join(ARCHIVE, "analysis.py")
src = open(ana_path, encoding="utf-8").read()
src = src.replace('"01_screening.csv"', '"01_screening_v5.csv"') \
         .replace('"02_detailed.csv"',  '"02_detailed_v5.csv"')

orig_open = builtins.open
def _redirect(path, mode="r", *a, **kw):
    if isinstance(path, str) and "_v5.csv" in path and "w" in str(mode):
        return io.StringIO()
    return orig_open(path, mode, *a, **kw)

builtins.open = _redirect
ns = {"__file__": ana_path}
try:
    exec(src, ns)
finally:
    builtins.open = orig_open

results = {r["well"]: r for r in ns["results"]}

# ── Styles ─────────────────────────────────────────────────────────────────────
FILL_GREEN  = PatternFill("solid", fgColor="C6EFCE")
FILL_YELLOW = PatternFill("solid", fgColor="FFEB9C")
FILL_RED    = PatternFill("solid", fgColor="FFC7CE")
FILL_HDR    = PatternFill("solid", fgColor="D9E1F2")   # blue — existing
FILL_CUR    = PatternFill("solid", fgColor="EBF5FB")   # light blue — текущий
FILL_AFT    = PatternFill("solid", fgColor="FCE4D6")   # orange — после УВЧ
FILL_REF    = PatternFill("solid", fgColor="F2F3F4")   # gray — справочно
FILL_DELTA  = PatternFill("solid", fgColor="FEF9E7")   # cream — delta block
FILL_POS    = PatternFill("solid", fgColor="E2EFDA")   # light green — positive delta
FILL_NEG    = PatternFill("solid", fgColor="FADBD8")   # light red — negative delta

RAG_FILL = {"Green": FILL_GREEN, "Yellow": FILL_YELLOW, "Red": FILL_RED}

CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
RIGHT  = Alignment(horizontal="right",  vertical="center")

def sc(ws, row, col, value, fill=None, fmt=None, align=RIGHT, bold=False, size=9):
    c = ws.cell(row=row, column=col, value=value)
    if fill:  c.fill = fill
    if fmt:   c.number_format = fmt
    if align: c.alignment = align
    c.font = Font(bold=bold, size=size)
    return c

def hdr(ws, row, col, text, fill=FILL_HDR, span=1):
    c = sc(ws, row, col, text, fill=fill, align=CENTER, bold=True)
    if span > 1:
        ws.merge_cells(start_row=row, start_column=col,
                       end_row=row,   end_column=col + span - 1)
    return c

def delta_fill(val):
    if val is None: return None
    return FILL_POS if float(val) > 0 else (FILL_NEG if float(val) < 0 else None)

# ── Load workbook ──────────────────────────────────────────────────────────────
shutil.copy2(SRC_XLS, DST_XLS)
wb = openpyxl.load_workbook(DST_XLS)
ws = wb["СВОД"]

# ── Map well → row ─────────────────────────────────────────────────────────────
well_row = {}
for row in ws.iter_rows(min_row=5, max_row=61):
    cell = row[2]
    if cell.value and str(cell.value).startswith(("Vt_", "Bt_")):
        well_row[cell.value] = cell.row
print(f"Found {len(well_row)} wells in Excel")

# ── Unmerge any existing merged cells in our write area ───────────────────────
to_unmerge = [rng for rng in ws.merged_cells.ranges
              if rng.min_col <= 33 and rng.max_col >= 12
              and rng.min_row <= 4  and rng.max_row >= 3]
for rng in to_unmerge:
    ws.unmerge_cells(str(rng))

# ── Section headers row 3 ──────────────────────────────────────────────────────
hdr(ws, 3, 12, "Текущий режим",                    fill=FILL_CUR, span=4)
hdr(ws, 3, 21, "После УВЧ — ОР и декомпозиция",   fill=FILL_AFT, span=8)
hdr(ws, 3, 29, "Справочно",                        fill=FILL_REF, span=4)

# ── Column headers row 4 ──────────────────────────────────────────────────────
cur_headers = {
    12: "ОР P50 тек., сут",
    13: "ОР P50 лучш. проц., сут",
    14: "Δ подрядчик тек., сут",
    15: "ННО прогноз тек., сут",
}
aft_headers = {
    21: "ОР P50 после УВЧ, сут",
    22: "ОР P50 лучш. проц. (после), сут",
    23: "ОР теор. макс., сут",
    24: "ННО прогноз (после), сут",
    25: "Δ УВЧ, сут",
    26: "Δ подрядчик (после), сут",
    27: "Δ H2S, сут",
    28: "Δ хим. суммарно, сут",
}
ref_headers = {
    29: "HR Cox полный",
    30: "HR Cox геол.",
    31: "Полоса",
    32: "RAG",
}
for col, text in cur_headers.items():
    hdr(ws, 4, col, text, fill=FILL_CUR)
for col, text in aft_headers.items():
    hdr(ws, 4, col, text, fill=FILL_AFT)
for col, text in ref_headers.items():
    hdr(ws, 4, col, text, fill=FILL_REF)

# ── Clear stale old-block cells (rows 5-61, cols 12-33) ───────────────────────
for r_idx in range(5, 62):
    for c_idx in range(12, 34):
        ws.cell(row=r_idx, column=c_idx).value = None

# ── Per-well data ──────────────────────────────────────────────────────────────
I = "0"      # integer
H = "0.000"  # HR

for well, row in well_row.items():
    r = results.get(well)
    if r is None:
        print(f"  WARNING: no v5 data for {well}")
        continue

    rag = r["rag_n"]
    rag_fill = RAG_FILL.get(rag)

    # ── Текущий режим (12-15)
    sc(ws, row, 12, round(r["rul_med_c"]),   fill=rag_fill, fmt=I)
    sc(ws, row, 13, round(r["rul_best"]),    fmt=I)
    sc(ws, row, 14, round(r["d_contr_c"]),   fill=delta_fill(r["d_contr_c"]), fmt=I)
    sc(ws, row, 15, round(r["proj_cur"]),    fmt=I)

    # ── После УВЧ — ОР (21-24)
    sc(ws, row, 21, round(r["rul_med_a"]),   fmt=I)
    sc(ws, row, 22, round(r["rul_best_a"]),  fmt=I)
    sc(ws, row, 23, round(r["rul_max"]),     fmt=I)
    sc(ws, row, 24, round(r["proj_aft"]),    fmt=I)

    # ── Дельты (25-28)
    sc(ws, row, 25, round(r["d_uvch"]),      fill=delta_fill(r["d_uvch"]),    fmt=I)
    sc(ws, row, 26, round(r["d_contr_a"]),   fill=delta_fill(r["d_contr_a"]), fmt=I)
    sc(ws, row, 27, round(r["d_h2s"]),       fill=delta_fill(r["d_h2s"]),     fmt=I)
    sc(ws, row, 28, round(r["d_chem"]),      fill=delta_fill(r["d_chem"]),    fmt=I)

    # ── Справочно (29-32)
    sc(ws, row, 29, round(float(r["hr_cox"]), 3), fmt=H)
    sc(ws, row, 30, round(float(r["hr_geo"]), 3), fmt=H)
    sc(ws, row, 31, r["band"],  align=CENTER)
    c_rag = sc(ws, row, 32, rag, fill=rag_fill, align=CENTER, bold=True)

# ── Column widths ──────────────────────────────────────────────────────────────
widths = {
    12: 10, 13: 14, 14: 13, 15: 12,
    21: 12, 22: 15, 23: 13, 24: 12,
    25: 10, 26: 14, 27: 10, 28: 14,
    29:  9, 30:  9, 31:  7, 32:  7,
}
for col, w in widths.items():
    ws.column_dimensions[get_column_letter(col)].width = w

# ── Legend (cols 34-35, rows 4-16) ────────────────────────────────────────────
hdr(ws, 4, 34, "Обозначение", fill=FILL_HDR)
hdr(ws, 4, 35, "Описание",    fill=FILL_HDR)
ws.column_dimensions[get_column_letter(34)].width = 16
ws.column_dimensions[get_column_letter(35)].width = 48
legend = [
    ("ОР P50 текущий",       "Медиана ОР: 2-компонентный Weibull × Cox(H2S+Ca+SO4+НТ)"),
    ("ОР P50 лучш. процесс", "Медиана ОР: Компонент 2 × Cox(H2S+Ca+SO4) — убран штраф НТ-подрядчика"),
    ("ОР теор. макс.",       "Медиана ОР: Компонент 2 без рисков (hr=1) — потолок"),
    ("Δ подрядчик (тек.)",   "rul_best − rul_med_c: выигрыш от смены НТ-подрядчика сейчас"),
    ("Δ УВЧ",                "rul_med_a − rul_med_c: изменение ОР после перевода на УВЧ (часто <0)"),
    ("Δ подрядчик (после)",  "rul_best_a − rul_med_a: выигрыш от смены НТ-подрядчика после УВЧ"),
    ("Δ H2S",                "штраф от H2S: ОР без H2S − ОР лучш. процесс (после УВЧ)"),
    ("Δ хим. суммарно",      "rul_max − rul_best_a: суммарный хим. штраф (H2S+Ca+SO4) после УВЧ"),
    ("HR Cox полный",        "Множитель риска Cox: H2S + Ca + SO4 + НТ"),
    ("HR Cox геол.",         "Множитель риска Cox: только геология (H2S + Ca + SO4, без НТ)"),
    ("Δ < 0",                "Отрицательная дельта = УВЧ снижает надёжность (но ↑ добычу)"),
]
for i, (abbr, desc) in enumerate(legend, 5):
    ws.cell(row=i, column=34, value=abbr).font = Font(bold=True, size=9)
    ws.cell(row=i, column=35, value=desc).font = Font(size=9)

# ── Portfolio stats (rows 63+) ─────────────────────────────────────────────────
all_r = list(results.values())

def med(vals):
    s = sorted(vals); n = len(s)
    return (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]

def avg(key):
    return sum(float(r[key]) for r in all_r) / len(all_r)

ws.cell(row=62, column=3, value="Сводная статистика v5").font = Font(bold=True, size=10)
stats = [
    ("Медиана ОР P50 текущий",          f"{med([round(r['rul_med_c']) for r in all_r]):.0f} сут"),
    ("Медиана ОР P50 лучш. процесс",    f"{med([round(r['rul_best'])   for r in all_r]):.0f} сут"),
    ("Медиана ОР P50 после УВЧ",        f"{med([round(r['rul_med_a']) for r in all_r]):.0f} сут"),
    ("Медиана ОР теор. макс.",          f"{med([round(r['rul_max'])   for r in all_r]):.0f} сут"),
    ("Ср. Δ УВЧ (d_uvch)",             f"{avg('d_uvch'):+.1f} сут"),
    ("Ср. Δ подрядчик текущий",        f"{avg('d_contr_c'):+.1f} сут"),
    ("Ср. Δ подрядчик после УВЧ",      f"{avg('d_contr_a'):+.1f} сут"),
    ("Ср. Δ H2S (штраф)",              f"{avg('d_h2s'):+.1f} сут"),
    ("Ср. Δ хим. суммарно (штраф)",    f"{avg('d_chem'):+.1f} сут"),
    ("RAG Green (без proxy)",           str(sum(1 for r in all_r if r['rag_n']=='Green'))),
    ("RAG Yellow",                      str(sum(1 for r in all_r if r['rag_n']=='Yellow'))),
    ("RAG Red",                         str(sum(1 for r in all_r if r['rag_n']=='Red'))),
]
for i, (lbl, val) in enumerate(stats, 63):
    ws.cell(row=i, column=3, value=lbl).font = Font(size=9)
    ws.cell(row=i, column=4, value=val).font = Font(size=9)

# ── Save ───────────────────────────────────────────────────────────────────────
wb.save(DST_XLS)
print(f"\nSaved: {DST_XLS}")
print(f"Wells updated: {len(well_row)}")
print(f"Columns: 12-15 (текущий) | 21-28 (после УВЧ + дельты) | 29-32 (справочно)")
