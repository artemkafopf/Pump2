"""
Enrich UVCh candidate wells with chemical and contractor factors from mart.

Joins mart__vt_freq55 for each of 56 UVCh wells to extract:
  - cl_daily   : cumulative chloride load / run_days (kg/d)
  - ca_daily   : cumulative calcium load / run_days (kg/d)
  - so4_daily  : cumulative sulfate load / run_days (kg/d)
  - h2s_mg_l   : H2S proxy concentration (mg/L)
  - h2s_flag   : 1 if h2s_proxy_mg_l >= 100 (threshold from weibull_external_factors_report)
  - is_nt      : 1 if contractor is НТ (Новые технологии), else 0
  - cl_log     : log1p(cl_daily) — continuous Cox covariate for chloride

Strategy: prefer the current (ongoing) run per well; fall back to most recent run.

Output: docs/uvch_claude/well_chemical_profile.csv
"""
import math, csv, os, sqlite3, re, ast

DB_PATH  = r"d:\GitHub\Pump2\data\warehouse\pump2.db"
RAW_PATH = r"d:\GitHub\Pump2\docs\uvch_claude\analysis.py"
OUT_PATH = r"d:\GitHub\Pump2\docs\uvch_claude\well_chemical_profile.csv"

# ── Load UVCh well list from analysis.py RAW ─────────────────────────────────
with open(RAW_PATH, encoding="utf-8") as f:
    code = f.read()

m = re.search(r"^RAW = \[(.*?)^\]", code, re.DOTALL | re.MULTILINE)
if not m:
    raise RuntimeError("RAW list not found in analysis.py")
raw_text = "[" + m.group(1) + "]"
RAW = ast.literal_eval(raw_text)
uvch_wells = [r[0] for r in RAW]
print(f"UVCh wells: {len(uvch_wells)}")

# ── Check distinct contractors in mart ───────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

contractors = [r[0] for r in conn.execute(
    "SELECT DISTINCT contractor FROM mart__vt_freq55 ORDER BY contractor"
)]
print("Contractors in mart:", contractors)

# ── Extract per-well chemical profile ────────────────────────────────────────
NT_KEYWORDS = ("нов", "новые", "new tech", "нт")

rows = []
missing = []

for well in uvch_wells:
    # Prefer ongoing run (event=0), then most recent by stop_date
    cur = conn.execute("""
        SELECT well, contractor,
               h2s_proxy_mg_l,
               cum_chloride_load_kg,
               cum_calcium_load_kg,
               cum_sulfate_load_kg,
               run_days, event,
               freq_w_mean,
               stop_date, install_date
        FROM mart__vt_freq55
        WHERE well = ?
        ORDER BY
            CASE WHEN event = 0 THEN 0 ELSE 1 END,
            stop_date DESC
        LIMIT 1
    """, (well,))
    row = cur.fetchone()

    if row is None:
        missing.append(well)
        continue

    d = dict(row)
    rd = max(d["run_days"] or 1, 1)

    cl_daily  = (d["cum_chloride_load_kg"]  or 0.0) / rd
    ca_daily  = (d["cum_calcium_load_kg"]   or 0.0) / rd
    so4_daily = (d["cum_sulfate_load_kg"]   or 0.0) / rd
    h2s       = d["h2s_proxy_mg_l"] or 0.0
    h2s_flag  = 1 if h2s >= 100.0 else 0

    contractor = (d["contractor"] or "").strip()
    is_nt      = 1 if any(kw in contractor.lower() for kw in NT_KEYWORDS) else 0

    rows.append({
        "well":         well,
        "contractor":   contractor,
        "h2s_mg_l":     round(h2s, 2),
        "h2s_flag":     h2s_flag,
        "cl_daily_kg":  round(cl_daily, 1),
        "ca_daily_kg":  round(ca_daily, 1),
        "so4_daily_kg": round(so4_daily, 1),
        "cl_log":       round(math.log1p(cl_daily), 4),
        "is_nt":        is_nt,
        "run_days_mart": d["run_days"],
        "event_mart":    d["event"],
        "freq_w_mean":   round(d["freq_w_mean"] or 0.0, 2),
        "install_date":  d["install_date"] or "",
        "stop_date":     d["stop_date"] or "",
    })

conn.close()

# ── Write output ─────────────────────────────────────────────────────────────
cols = [
    "well", "contractor", "h2s_mg_l", "h2s_flag",
    "cl_daily_kg", "ca_daily_kg", "so4_daily_kg", "cl_log",
    "is_nt", "run_days_mart", "event_mart", "freq_w_mean",
    "install_date", "stop_date",
]
with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} wells to {OUT_PATH}")
if missing:
    print(f"Missing from mart ({len(missing)}): {', '.join(missing)}")

# ── Quick summary ─────────────────────────────────────────────────────────────
h2s_high  = sum(1 for r in rows if r["h2s_flag"])
nt_count  = sum(1 for r in rows if r["is_nt"])
cl_vals   = [r["cl_daily_kg"] for r in rows]
print(f"\nProfile summary ({len(rows)} wells):")
print(f"  H2S >= 100 mg/L : {h2s_high} wells")
print(f"  НТ contractor   : {nt_count} wells")
print(f"  Cl_daily  range : {min(cl_vals):.0f} – {max(cl_vals):.0f} kg/d  "
      f"(median {sorted(cl_vals)[len(cl_vals)//2]:.0f})")
