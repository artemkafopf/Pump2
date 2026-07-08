"""Export the versioned VBA model bundle (ESP prediction system v2).

Thin CLI wrapper around ``analysis.workflows.esp_survival.vba_bundle.build_bundle``.
Regenerates every artifact from the SQLite mart on the operating-time clock
(``ttf_mix``), full population, post-label-hygiene — the single source of truth
for the Excel VBA registry.  See ``agents/analyses/vba_model_v2.md``.

Writes, under ``results_dir("esp_survival_vba_models")/<date>/``:
    esp_models.csv        registry rows (params, model_kind, B50 CIs, uptime, clock)
    esp_mode_mix.csv      per field-level stratum x mode group incidence (Phase B)
    mode_group_map.csv    frozen node -> mode map (audit)
    bundle_manifest.txt   row counts, git commit, clock, date
    vba/mdlModelSeed.bas  generated CreateModelSheet literal block

The VBA ``ImportModelCSV`` macro reads esp_models.csv; ``ImportModeMixCSV`` reads
esp_mode_mix.csv + mode_group_map.csv.

Usage:
    python scripts/run/export_model_csv_for_vba.py [n_boot]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.paths import results_dir
from analysis.workflows.esp_survival.vba_bundle import build_bundle


def main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out_dir = results_dir("esp_survival_vba_models")
    build_bundle(out_dir, n_boot=n_boot)


if __name__ == "__main__":
    main()
