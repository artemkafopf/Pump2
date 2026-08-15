from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "run" / "production_risk_joint_validation.py"
    spec = importlib.util.spec_from_file_location("production_risk_joint_validation", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_script(name: str):
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "run" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_idle_reattribution_moves_idle_failure_to_last_producing_month():
    mod = _load_module()
    run_months = pd.DataFrame(
        [
            {
                "run_idx": 1,
                "well_code": "MC_001",
                "month": "2024-01",
                "plan_producing": True,
                "is_failure_month": False,
            },
            {
                "run_idx": 1,
                "well_code": "MC_001",
                "month": "2024-02",
                "plan_producing": False,
                "is_failure_month": True,
            },
            {
                "run_idx": 2,
                "well_code": "YA_001",
                "month": "2024-01",
                "plan_producing": True,
                "is_failure_month": True,
            },
        ]
    )

    summary, counts = mod.idle_reattribution(run_months, {"MC_001": "Мирнинский УН", "YA_001": "Ярактинский УН"})

    mc = summary[(summary["field"].eq("Мирнинский УН")) & (summary["observed_mode"].eq("reattributed"))].iloc[0]
    assert int(mc["idle_failures"]) == 0
    assert int(mc["producing_failures"]) == 1
    assert int(mc["moved_failures"]) == 1

    rec = counts[
        counts["observed_mode"].eq("reattributed")
        & counts["field"].eq("Мирнинский УН")
        & counts["month"].eq("2024-01")
    ]
    assert float(rec["observed_failures"].iloc[0]) == 1.0


def test_mc_variant_model_registry_replaces_only_mc_pooled_row():
    mod = _load_script("production_risk_mc_variant_validation")
    base = pd.DataFrame(
        [
            {
                "stratum": "Mc_nonsour_Pooled",
                "model_kind": "old",
                "w1": 0.1,
                "beta1": 1.0,
                "eta1": 100.0,
                "beta2": 1.0,
                "eta2": 100.0,
                "b20": 20.0,
                "b50": 50.0,
                "b80": 80.0,
                "field": "Mc",
                "h2s_class": "nonsour",
                "contractor_group": "Pooled",
                "mc_window": "old",
            },
            {
                "stratum": "Ya_nonsour_Pooled",
                "model_kind": "keep",
                "w1": 0.2,
                "beta1": 2.0,
                "eta1": 200.0,
                "beta2": 2.0,
                "eta2": 200.0,
                "b20": 40.0,
                "b50": 100.0,
                "b80": 160.0,
                "field": "Ya",
                "h2s_class": "nonsour",
                "contractor_group": "Pooled",
                "mc_window": "",
            },
        ]
    )
    variant = pd.Series(
        {
            "label": "install_2024plus",
            "model_kind": "k1_aic",
            "w1": 0.0,
            "beta1": 1.17,
            "eta1": 765.92,
            "beta2": 1.17,
            "eta2": 765.92,
            "b20": 213.0,
            "b50": 560.2,
            "b80": 1149.5,
            "n_runs": 157,
            "n_failures": 39,
        }
    )

    out = mod.mc_variant_model_registry(base, variant)

    assert int(out["stratum"].eq("Mc_nonsour_Pooled").sum()) == 1
    mc = out[out["stratum"].eq("Mc_nonsour_Pooled")].iloc[0]
    assert mc["model_kind"] == "k1_aic"
    assert float(mc["b50"]) == 560.2
    assert mc["mc_window"] == "install_2024plus"
    assert "Ya_nonsour_Pooled" in set(out["stratum"])


def test_bundle_observed_failures_preferred_when_present(tmp_path, monkeypatch):
    from analysis.workflows.production_risk import failure_rate

    path = tmp_path / "esp_observed_failures.csv"
    pd.DataFrame(
        [
            {"field": "Мирнинский УН", "month": "2024-01", "observed_failures": "2"},
            {"field": "Мирнинский УН", "month": "2024-01", "observed_failures": "3"},
            {"field": "Мирнинский УН", "month": "2024-02", "observed_failures": "7"},
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(failure_rate.C, "observed_failures_path", lambda bundle_date: path)

    out = failure_rate._bundle_observed_failures("candidate", {"2024-01"})

    assert out is not None
    assert len(out) == 1
    assert out.iloc[0]["field"] == "Мирнинский УН"
    assert out.iloc[0]["month"] == "2024-01"
    assert float(out.iloc[0]["observed_failures"]) == 5.0
