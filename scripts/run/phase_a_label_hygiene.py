"""Phase A T5 — label-hygiene report (reporting only; no stratum changes).

Produces, under ``results/phase_a_label_hygiene/<date>/``:

  tables/failed_node_trim.csv   — raw vs trimmed failure-node counts (НКТ / 'НКТ ' merge)
  tables/field_alias_map.csv    — every raw field value → resolved code (or blank)
  tables/needs_review.csv       — unmapped field names for the user (NOT guessed)
  tables/nonvt_sour_count.csv   — non-Vt runs that would flip to sour at 10 mg/l

Run:
    python scripts/run/phase_a_label_hygiene.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir, WAREHOUSE_DIR
from analysis.data.label_hygiene import (
    trim_failed_node,
    field_alias_report,
    count_nonvt_sour,
    H2S_SOUR_THRESHOLD,
)


def main() -> None:
    out = results_dir("phase_a_label_hygiene")
    tbl = out / "tables"
    con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    runs = pd.read_sql(
        "SELECT field, failed_node, h2s_mg_l FROM raw__v03_runs", con)
    mart = pd.read_sql(
        "SELECT field, h2s_proxy_mg_l FROM mart__weibull_input", con)
    con.close()

    # ── T5.1 failed_node TRIM ─────────────────────────────────────────────────
    raw_counts = runs["failed_node"].value_counts(dropna=False)
    runs["failed_node_trim"] = runs["failed_node"].map(trim_failed_node)
    trim_counts = runs["failed_node_trim"].value_counts(dropna=False)
    fn_rows = []
    for label, n in raw_counts.items():
        trimmed = trim_failed_node(label)
        fn_rows.append({"raw_label": label, "n_raw": int(n),
                        "trimmed_label": trimmed,
                        "changed": (str(label) != str(trimmed))})
    fn = pd.DataFrame(fn_rows)
    fn.to_csv(tbl / "failed_node_trim.csv", index=False, encoding="utf-8-sig")
    n_merged = int(fn["changed"].sum())
    print(f"[T5.1] failed_node: {len(raw_counts)} raw → {len(trim_counts)} trimmed "
          f"categories; {n_merged} labels changed by TRIM/case-fold")
    # Show the НКТ merge explicitly
    nkt = fn[fn["raw_label"].astype(str).str.strip() == "НКТ"]
    if not nkt.empty:
        print("       'НКТ ' (trailing space) now merges into 'НКТ'")

    # ── T5.2 field alias mapping ──────────────────────────────────────────────
    applied, needs_review = field_alias_report(mart["field"])
    applied.to_csv(tbl / "field_alias_map.csv", index=False, encoding="utf-8-sig")
    needs_review.to_csv(tbl / "needs_review.csv", index=False, encoding="utf-8-sig")
    print(f"\n[T5.2] field aliases: {len(applied)} distinct values; "
          f"{len(needs_review)} UNMAPPED → needs_review.csv (not guessed):")
    for _, r in needs_review.iterrows():
        print(f"       {r['field_alias']!r}  (n={r['n_runs']})")

    # ── T5.3 non-Vt sour count (report only) ──────────────────────────────────
    sour = count_nonvt_sour(mart, threshold=H2S_SOUR_THRESHOLD)
    sour.to_csv(tbl / "nonvt_sour_count.csv", index=False, encoding="utf-8-sig")
    n_flip = int(sour["would_flip_to_sour"].sum())
    print(f"\n[T5.3] Applying the {H2S_SOUR_THRESHOLD:.0f} mg/l threshold to ALL fields, "
          f"{n_flip} non-Vt runs would flip to sour.")
    print("       Active behaviour stays Vt-only until the user reviews this count.")
    flips = sour[(sour["would_flip_to_sour"] > 0)]
    if not flips.empty:
        print(flips[["field", "n_runs", "would_flip_to_sour"]].to_string(index=False))

    print(f"\n[T5] Outputs written to: {out}")


if __name__ == "__main__":
    main()
