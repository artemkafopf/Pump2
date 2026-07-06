"""Field-level ESP survival mixture — no sour/non-sour split.

Usage:
    python scripts/run/esp_survival_field.py
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
from analysis.workflows.esp_survival.field_mixture import run

t0 = time.monotonic()
print("\n" + "=" * 70)
print("ESP SURVIVAL — FIELD-LEVEL MIXTURE (no sour/non-sour split)")
print("=" * 70)

df = load_failures_df()
print(f"\n[Data] {len(df):,} runs  |  {int(df['event'].sum()):,} failures  "
      f"|  {df['field_clean'].nunique()} fields")

out = results_dir("esp_survival_field_mixture")
result = run(df, out / "figures")

print(f"\n{'=' * 70}")
print(f"COMPLETE — {time.monotonic() - t0:.0f}s")
print("=" * 70)
