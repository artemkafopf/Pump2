"""Focused tests for the Workstream-C hazard refit transforms and fit/serve identity."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import hazard_refit_c as HR
from analysis.workflows.production_risk.survival import HazardLayer, ql_hazard_theta


# ── Ql z-transform ────────────────────────────────────────────────────────────

def test_ql_z_transform_matches_clip_formula():
    ref = C.QL_HAZARD_FIELD_REF_LOG["Vt"]
    ql = 120.0
    expected = np.clip(math.log1p(ql) - ref, -HR.QL_CAP, HR.QL_CAP)
    assert HR.ql_z_transform(ql, ref) == pytest.approx(expected)


def test_ql_z_transform_clips_at_log5():
    ref = 4.0
    # huge Ql -> clipped to +log(5); tiny Ql -> clipped to -log(5)
    assert HR.ql_z_transform(1e9, ref) == pytest.approx(HR.QL_CAP)
    assert HR.ql_z_transform(0.0, ref) == pytest.approx(-HR.QL_CAP)


def test_ql_z_transform_nan_on_bad_input():
    assert math.isnan(HR.ql_z_transform(float("nan"), 4.0))
    assert math.isnan(HR.ql_z_transform(-1.0, 4.0))


def test_ql_z_transform_agrees_with_served_z(monkeypatch):
    """The fit-side z must equal the z the serve path uses inside ql_hazard_theta.

    ql_hazard_theta at age=1 gives theta = exp(z * beta); invert to recover z.
    """
    monkeypatch.setattr(C, "QL_HAZARD_ENABLED", True)  # dial off by default; test the mechanism
    ref = C.QL_HAZARD_FIELD_REF_LOG["Ya"]
    ql = 90.0
    z_fit = HR.ql_z_transform(ql, ref)
    theta = float(ql_hazard_theta(math.log1p(ql), "Ya", 1.0))
    z_served = math.log(theta) / C.QL_HAZARD_BETA  # coef at age 1 == beta (log1=0)
    assert z_fit == pytest.approx(z_served, rel=1e-6)


# ── fit link == serve mechanic (guardrail 1) ─────────────────────────────────

def _synthetic_train(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "code": [f"W_{i}" for i in range(n)],
        "month": "2024-06",
        "model_field": "Vt",
        "age_start": rng.uniform(30, 900, n),
        "op_days": rng.uniform(5, 30, n),
        "mu_baseline": rng.uniform(0.01, 0.2, n),
        "cov_log_run_seq": rng.normal(1.2, 0.5, n),
        "cov_load_std_early": rng.normal(6.7, 1.0, n),
        "ql_z_fit": 0.0,
        "ql_z_logage": 0.0,
        "ql_available": False,
    })
    refs = {"log_run_seq": 1.2, "load_std_early": 6.7}
    theta_true = np.exp(0.4 * (df["cov_log_run_seq"] - 1.2) + 0.02 * (df["cov_load_std_early"] - 6.7))
    df["event"] = (rng.random(n) < np.clip(df["mu_baseline"] * theta_true, 0, 1)).astype(float)
    df.attrs["refs"] = refs
    return df


def test_poisson_multiplier_reproduces_hazardlayer_theta():
    train = _synthetic_train()
    refs = train.attrs["refs"]
    fit = HR.fit_poisson_offset(train, refs, covariates=["log_run_seq", "load_std_early"], use_ql=False)

    # served theta via HazardLayer.theta with the refit coefficients + shipped references
    coeffs = pd.DataFrame({
        "covariate": fit.columns,
        "beta": fit.beta,
        "reference_value": [refs[c] for c in fit.columns],
        "enabled": "TRUE",
    })
    hz = HazardLayer.__new__(HazardLayer)
    hz.coeffs = coeffs
    theta_serve = np.array([
        hz.theta({"log_run_seq": r["cov_log_run_seq"], "load_std_early": r["cov_load_std_early"]})
        for _, r in train.iterrows()
    ])
    theta_fit = HR.predict_mu(train, fit) / train["mu_baseline"].to_numpy()
    assert np.allclose(theta_fit, theta_serve, atol=1e-9)


def test_ql_hazard_bundle_override_applied():
    """The externalized esp_ql_hazard.csv must be read at import and applied to config,
    so Ql can be tuned without rebuilding the EXE."""
    import csv
    path = C.ql_hazard_path()
    assert path.exists(), "bundle is missing esp_ql_hazard.csv"
    vals: dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            vals[row["param"].strip()] = row["value"].strip()
    assert C.QL_HAZARD_BETA == float(vals["beta"])
    assert C.QL_HAZARD_GAMMA == float(vals["gamma"])
    assert C.QL_HAZARD_ENABLED == (vals["enabled"].upper() == "TRUE")
    assert C.QL_HAZARD_FIELD_REF_LOG["Vt"] == float(vals["field_ref.Vt"])
    assert "esp_ql_hazard.csv" in C.QL_HAZARD_SOURCE


def test_kpod_hazard_bundle_override_applied():
    """The externalized esp_kpod_hazard.csv must be read at import and applied to config."""
    import csv
    path = C.kpod_hazard_path()
    assert path.exists(), "bundle is missing esp_kpod_hazard.csv"
    vals: dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            vals[row["param"].strip()] = row["value"].strip()
    assert C.KPOD_HAZARD_ENABLED == (vals["enabled"].upper() == "TRUE")
    assert C.KPOD_HAZARD_K_LO == float(vals["k_lo"])
    assert C.KPOD_HAZARD_K_HI == float(vals["k_hi"])
    assert C.KPOD_HAZARD_BETA_UNDER == float(vals["beta_under"])
    assert C.KPOD_HAZARD_BETA_OVER == float(vals["beta_over"])
    assert C.KPOD_HAZARD_GAMMA_UNDER == float(vals["gamma_under"])
    assert C.KPOD_HAZARD_GAMMA_OVER == float(vals["gamma_over"])
    assert C.KPOD_HAZARD_CAP_UNDER == float(vals["cap_under"])
    assert C.KPOD_HAZARD_CAP_OVER == float(vals["cap_over"])
    assert C.KPOD_HAZARD_FIELD_PARAMS["Bt"]["k_hi"] == float(vals["field.Bt.k_hi"])
    assert C.KPOD_HAZARD_FIELD_PARAMS["Vt"]["beta_over"] == float(vals["field.Vt.beta_over"])
    assert "esp_kpod_hazard.csv" in C.KPOD_HAZARD_SOURCE


def test_uncertainty_hazard_bundle_override_applied():
    """The externalized esp_uncertainty_hazard.csv must be read and applied to config."""
    import csv
    path = C.uncertainty_hazard_path()
    assert path.exists(), "bundle is missing esp_uncertainty_hazard.csv"
    vals: dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            vals[row["param"].strip()] = row["value"].strip()
    assert C.UNCERTAINTY_HAZARD_ENABLED == (vals["enabled"].upper() == "TRUE")
    assert C.UNCERTAINTY_HAZARD_U0 == float(vals["u0"])
    assert C.UNCERTAINTY_HAZARD_TAU_DAYS == float(vals["tau_days"])
    assert C.UNCERTAINTY_HAZARD_CAP == float(vals["cap"])
    assert C.UNCERTAINTY_HAZARD_APPLY_TO == set(vals["apply_to"].split(","))
    assert C.UNCERTAINTY_HAZARD_FIELD_PARAMS["Bt"]["u0"] == float(vals["field.Bt.u0"])
    assert C.UNCERTAINTY_HAZARD_FIELD_PARAMS["Bt"]["cap"] == float(vals["field.Bt.cap"])
    assert "esp_uncertainty_hazard.csv" in C.UNCERTAINTY_HAZARD_SOURCE


def test_missing_covariate_is_reference_neutral():
    """A missing covariate must contribute theta=1 (exactly as HazardLayer.theta skips it)."""
    train = _synthetic_train(n=50)
    refs = train.attrs["refs"]
    fit = HR.fit_poisson_offset(train, refs, covariates=["log_run_seq", "load_std_early"], use_ql=False)
    row = train.iloc[[0]].copy()
    row["cov_log_run_seq"] = np.nan  # missing -> filled to reference -> centered 0
    theta = HR.predict_mu(row, fit) / row["mu_baseline"].to_numpy()
    # only load_std_early contributes now
    expected = math.exp(fit.beta[fit.columns.index("load_std_early")]
                        * (float(row["cov_load_std_early"].iloc[0]) - refs["load_std_early"]))
    assert theta[0] == pytest.approx(expected, rel=1e-9)
