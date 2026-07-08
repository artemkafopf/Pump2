"""Build the operational + completion hazard layer for the VBA workbook (Prompt A).

Thin CLI over ``analysis.workflows.esp_survival.hazard_layer.build``. Fits the
subset θ (GLF, frequency, chronic underload, load, curvature-missingness, pump
size), attributes each hazard to its failure mode, runs the out-of-sample temporal-
holdout gate, and exports the coefficient card + per-run covariates for the VBA side.

Usage:
    python scripts/run/hazard_layer.py
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
from analysis.workflows.esp_survival.hazard_layer import build


def main() -> None:
    out_dir = results_dir("esp_survival_vba_models")   # sits beside the v2 bundle
    card_dir = results_dir("vba_model_v2_hazards")
    build(out_dir, card_dir)


if __name__ == "__main__":
    main()
