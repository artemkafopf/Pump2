"""Acceptance tests for the ESP VBA model bundle (agents/analyses/vba_model_v2.md).

Reads the most recent bundle written by
``scripts/run/export_model_csv_for_vba.py`` and asserts the contract the VBA
importer relies on:

* schema (all v2 columns present, in order);
* ``clock == "ttf_mix"`` on every row of esp_models.csv and esp_mode_mix.csv;
* every stratum has a usable row (no NaN in the params the VBA math touches);
* B50 recomputed from (w1, beta, eta) by an INDEPENDENT bisection matches the
  exported ``b50`` column within 1% (catches param/column drift in one shot);
* the generated seed .bas and node->mode map exist.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "results" / "esp_survival_vba_models"

from analysis.workflows.esp_survival.vba_bundle import MODEL_COLUMNS, MODEMIX_COLUMNS


def _latest_bundle() -> Path:
    if not BUNDLE_ROOT.exists():
        pytest.skip("no VBA bundle exported yet (run export_model_csv_for_vba.py)")
    dates = sorted(BUNDLE_ROOT.glob("????-??-??"), reverse=True)
    for d in dates:
        if (d / "esp_models.csv").exists():
            return d
    pytest.skip("no esp_models.csv in any dated bundle folder")


@pytest.fixture(scope="module")
def bundle() -> dict:
    d = _latest_bundle()
    return {
        "dir": d,
        "models": pd.read_csv(d / "esp_models.csv", encoding="utf-8-sig"),
        "modemix": pd.read_csv(d / "esp_mode_mix.csv", encoding="utf-8-sig"),
    }


# ── independent mixture bisection (mirrors the VBA math, not the exporter's) ──

def _mixture_sf(t: float, w1: float, b1: float, e1: float, b2: float, e2: float) -> float:
    return w1 * math.exp(-((t / e1) ** b1)) + (1.0 - w1) * math.exp(-((t / e2) ** b2))


def _bisect_b50(w1: float, b1: float, e1: float, b2: float, e2: float) -> float:
    target = 0.5  # S(t) = 0.5
    lo, hi = 0.0, 15.0 * max(e1, e2)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _mixture_sf(mid, w1, b1, e1, b2, e2) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── schema / clock ───────────────────────────────────────────────────────────

def test_models_schema(bundle):
    assert list(bundle["models"].columns) == list(MODEL_COLUMNS)


def test_modemix_schema(bundle):
    assert list(bundle["modemix"].columns) == list(MODEMIX_COLUMNS)


def test_clock_is_ttf_mix_everywhere(bundle):
    assert (bundle["models"]["clock"] == "ttf_mix").all()
    assert (bundle["modemix"]["clock"] == "ttf_mix").all()


def test_global_pooled_present(bundle):
    assert "Global_Pooled" in set(bundle["models"]["stratum"])


def test_model_kind_values(bundle):
    allowed = {"k2", "k1_aic", "k1_degenerate"}
    assert set(bundle["models"]["model_kind"]) <= allowed


# ── usable params (no NaN in the columns the VBA math reads) ──────────────────

def test_no_nan_in_math_params(bundle):
    m = bundle["models"]
    for col in ("w1", "beta1", "eta1", "beta2", "eta2"):
        assert m[col].notna().all(), f"NaN in {col}"
    # scale/shape must be strictly positive for the Weibull math
    assert (m["beta1"] > 0).all() and (m["eta1"] > 0).all()
    assert (m["beta2"] > 0).all() and (m["eta2"] > 0).all()
    assert ((m["w1"] >= 0) & (m["w1"] <= 1)).all()


def test_k1_rows_have_zero_weight(bundle):
    m = bundle["models"]
    k1 = m[m["model_kind"].isin(["k1_aic", "k1_degenerate"])]
    assert (k1["w1"] == 0).all(), "single-Weibull rows must store w1=0 (§0.3)"


# ── the core cross-check: exported b50 == independent bisection within 1% ─────

def test_b50_matches_bisection(bundle):
    m = bundle["models"]
    for _, r in m.iterrows():
        b50 = pd.to_numeric(r["b50"], errors="coerce")
        if not np.isfinite(b50) or b50 <= 0:
            continue
        recomputed = _bisect_b50(
            float(r["w1"]), float(r["beta1"]), float(r["eta1"]),
            float(r["beta2"]), float(r["eta2"]),
        )
        rel = abs(recomputed - b50) / b50
        assert rel <= 0.01, (
            f"{r['stratum']}: exported b50={b50} vs bisection={recomputed:.1f} "
            f"(rel {rel:.3f})"
        )


# ── companion artifacts ──────────────────────────────────────────────────────

def test_seed_and_map_exist(bundle):
    d = bundle["dir"]
    assert (d / "vba" / "mdlModelSeed.bas").exists()
    assert (d / "mode_group_map.csv").exists()
    assert (d / "bundle_manifest.txt").exists()


def test_b20_b50_b80_ordering(bundle):
    m = bundle["models"]
    b20 = pd.to_numeric(m["b20"], errors="coerce")
    b50 = pd.to_numeric(m["b50"], errors="coerce")
    b80 = pd.to_numeric(m["b80"], errors="coerce")
    ok = b20.notna() & b50.notna() & b80.notna()
    assert ((b20[ok] <= b50[ok]) & (b50[ok] <= b80[ok])).all(), "quantiles must order B20<=B50<=B80"


def test_uptime_factor_in_range(bundle):
    m = bundle["models"]
    u = pd.to_numeric(m["uptime_factor"], errors="coerce").dropna()
    # operating time <= calendar time, so 0 < u_s <= 1
    assert ((u > 0) & (u <= 1.0001)).all()
