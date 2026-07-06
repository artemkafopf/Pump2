"""Mixture fits restricted to Борец + Шлюмберже only.

Re-runs Phase 2 (stratum-level) and field-level EM on the subset of
runs belonging to these two contractors.  Results go to separate slugs
so existing full-population outputs are not overwritten.
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.workflows.esp_survival.data import load_failures_df
from analysis.workflows.esp_survival import phase2_mixture
from analysis.workflows.esp_survival.field_mixture import run as run_field

TOP2 = {"Борец", "Шлюмберже"}

t0 = time.monotonic()
print("\n" + "=" * 70)
print("ESP SURVIVAL — TOP-2 CONTRACTORS (Борец + Шлюмберже) ONLY")
print("=" * 70)

df_all = load_failures_df()
df = df_all[df_all["contractor"].isin(TOP2)].copy()

n_all = int(df_all["event"].sum())
n_sub = int(df["event"].sum())
print(f"\n[Filter] {len(df):,} runs  |  {n_sub:,} failures  "
      f"({n_sub/n_all*100:.1f}% of total {n_all:,})")
print(f"[Filter] Contractors retained: {sorted(df['contractor'].unique())}")

print("\n" + "-" * 60)
print("[Stratum-level] K=2 mixture — Борец + Шлюмберже")
out2 = results_dir("esp_survival_phase2_top2")
r2 = phase2_mixture.run(df, out2 / "figures")

print("\n" + "-" * 60)
print("[Field-level] K=2 mixture — Борец + Шлюмберже")
out_field = results_dir("esp_survival_field_top2")
r_field = run_field(df, out_field / "figures")

print(f"\n{'=' * 70}")
print(f"COMPLETE — {time.monotonic() - t0:.0f}s")
print("=" * 70)
