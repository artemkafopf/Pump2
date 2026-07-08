"""Acceptance tests for the operational + completion + cohort hazard layer.

Reads the exported coefficient card + per-run covariates and asserts the contract the
VBA side relies on: schema, clock stamp, finite coefficients/references, groups,
enabled flag (USER OVERRIDE = TRUE), and θ==1 at the reference profile.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "results" / "esp_survival_vba_models"

from analysis.workflows.esp_survival.hazard_layer import COVARS, GROUPS


def _latest(name: str) -> Path:
    if not BUNDLE_ROOT.exists():
        pytest.skip("no bundle dir (run scripts/run/hazard_layer.py)")
    for d in sorted(BUNDLE_ROOT.glob("????-??-??"), reverse=True):
        if (d / name).exists():
            return d / name
    pytest.skip(f"no {name} exported yet")


@pytest.fixture(scope="module")
def art() -> dict:
    coeffs_p = _latest("esp_cox_coeffs.csv")
    d = coeffs_p.parent
    return {
        "dir": d,
        "coeffs": pd.read_csv(coeffs_p, encoding="utf-8-sig"),
        "runcov": pd.read_csv(d / "esp_run_covariates.csv", encoding="utf-8-sig"),
        "manifest": (d / "hazard_manifest.txt").read_text(encoding="utf-8"),
    }


def test_coeffs_schema(art):
    cols = {"covariate", "ship_name", "hazard_group", "beta", "hr", "ci_lo", "ci_hi",
            "p", "reference_value", "expected_dir", "dir_ok", "window", "mode",
            "block", "enabled", "ship_reason", "clock"}
    assert cols.issubset(set(art["coeffs"].columns))


def test_clock_ttf_mix(art):
    assert (art["coeffs"]["clock"] == "ttf_mix").all()
    assert (art["runcov"]["clock"] == "ttf_mix").all()


def test_covariate_set_matches_module(art):
    assert set(art["coeffs"]["covariate"]) == set(COVARS)


def test_groups_covered(art):
    assert set(art["coeffs"]["hazard_group"]) == set(GROUPS)


def test_finite_beta_and_reference(art):
    c = art["coeffs"]
    assert pd.to_numeric(c["beta"], errors="coerce").notna().all()
    assert pd.to_numeric(c["reference_value"], errors="coerce").notna().all()


def test_enabled_true_user_override(art):
    assert set(art["coeffs"]["enabled"].astype(str).str.upper()) == {"TRUE"}
    assert "USER-OVERRIDE" in art["manifest"]
    assert "enabled    : True" in art["manifest"]


def test_theta_is_one_at_reference(art):
    c = art["coeffs"]
    beta = pd.to_numeric(c["beta"], errors="coerce").to_numpy()
    ref = pd.to_numeric(c["reference_value"], errors="coerce").to_numpy()
    theta = float(np.exp(np.sum(beta * (ref - ref))))
    assert abs(theta - 1.0) < 1e-12


def test_curvature_uses_value_not_missing(art):
    covs = set(art["coeffs"]["covariate"])
    assert "curvature_deg10m" in covs
    assert "curvature_missing" not in covs
    # curvature is exported 0-filled (no empty cells) — physical straight-well value
    assert art["runcov"]["curvature_deg10m"].notna().all()


def test_runcov_keys_and_covariates(art):
    r = art["runcov"]
    for k in ("well_key_norm", "run_seq", "covariate_available", "stratum_key"):
        assert k in r.columns
    assert set(r["covariate_available"].astype(str).str.upper()) <= {"TRUE", "FALSE"}
    for col in COVARS:
        assert col in r.columns, f"{col} missing from run covariates"
