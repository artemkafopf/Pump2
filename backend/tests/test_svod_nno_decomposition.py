"""Unit tests for the Свод ННО decomposition — synthetic panels, no workbook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import svod_nno_decomposition as D


def _panel(n: int = 600, seed: int = 0, peak: float | None = 52.0) -> pd.DataFrame:
    """Panel where ННО depends on field, log Ql, and (optionally) a freq peak."""
    rng = np.random.default_rng(seed)
    field = rng.choice(["Ya", "Vt"], n)
    ql = np.exp(rng.normal(5.0, 0.5, n))
    qnom = rng.choice([100.0, 250.0, 450.0], n)   # varying pump size
    freq = rng.uniform(40.0, 62.0, n)
    nno = 700 + 120 * (field == "Ya") - 40 * np.log(ql) + rng.normal(0, 20, n)
    if peak is not None:
        nno = nno - 1.5 * (freq - peak) ** 2
    return pd.DataFrame({
        "field": field, "ql": ql, "log_ql": np.log(ql),
        "qnom": qnom, "log_qnom": np.log(qnom), "freq": freq,
        "kpod": ql / qnom, "delta": ql - qnom,
        "nno": nno, "event": 1,
        "install": pd.Timestamp("2024-06-01"),
    })


class TestResiduals:
    def test_residuals_are_in_days_not_log_days(self):
        p = _panel()
        days = D.field_ql_residuals(p, log_response=False)
        logs = D.field_ql_residuals(p, log_response=True)
        # Days residuals scale with ННО itself; log residuals are O(0.1).
        assert days.abs().mean() > 5.0
        assert logs.abs().mean() < 1.0

    def test_residuals_are_centred_and_orthogonal_to_controls(self):
        p = _panel()
        r = D.field_ql_residuals(p)
        assert r.mean() == pytest.approx(0.0, abs=1e-8)
        assert np.corrcoef(r, p["log_ql"])[0, 1] == pytest.approx(0.0, abs=1e-8)

    def test_field_level_is_removed(self):
        """The whole point of the control: a pure field offset must not survive."""
        p = _panel(peak=None)
        r = D.field_ql_residuals(p)
        means = r.groupby(p["field"]).mean()
        assert means.abs().max() < 5.0

    def test_control_set_selects_which_covariate_is_held_fixed(self):
        p = _panel(peak=None)
        rq = D.field_residuals(p, controls=("log_qnom",))
        # Qnom control leaves the residual orthogonal to log_qnom, not log_ql.
        assert np.corrcoef(rq, p["log_qnom"])[0, 1] == pytest.approx(0.0, abs=1e-8)
        assert abs(np.corrcoef(rq, p["log_ql"])[0, 1]) > 0.05
        assert D.CONTROL_SETS["qnom"] == ("log_qnom",)

    def test_log_response_rejects_non_positive_nno(self):
        p = _panel(peak=None)
        p.loc[p.index[0], "nno"] = 0.0
        with pytest.raises(ValueError, match="positive"):
            D.field_ql_residuals(p, log_response=True)


class TestQuadFit:
    def test_recovers_a_planted_peak(self):
        p = _panel(peak=52.0)
        r = D.field_ql_residuals(p)
        fit = D.quad_fit(p["freq"], r, n_boot=200)
        assert fit.beta2 < 0
        assert fit.p_beta2 < 0.01
        assert fit.vertex == pytest.approx(52.0, abs=1.5)
        assert fit.p_inverted_u > 0.95
        assert fit.vertex_lo < 52.0 < fit.vertex_hi

    def test_flat_truth_is_not_reported_as_a_peak(self):
        p = _panel(peak=None, seed=3)
        fit = D.quad_fit(p["freq"], D.field_ql_residuals(p), n_boot=200)
        assert fit.p_beta2 > 0.05
        assert 0.05 < fit.p_inverted_u < 0.95

    def test_too_few_rows_returns_none(self):
        p = _panel(n=20)
        assert D.quad_fit(p["freq"], p["nno"], n_boot=50) is None


class TestDecompose:
    def test_shapes_and_coverage(self):
        p = _panel()
        decomp, coverage = D.decompose(p, bins=10)
        assert set(decomp) == {"kpod", "delta", "freq"}
        assert len(decomp["freq"].marginal) == 10
        assert decomp["freq"].marginal["n"].sum() == len(p)
        assert coverage.set_index("x").loc["freq", "n_dropped_out_of_bounds"] == 0

    def test_out_of_range_frequency_is_dropped_and_counted(self):
        p = _panel()
        p.loc[p.index[:5], "freq"] = 236.0  # the real Свод defect
        p.loc[p.index[5:8], "freq"] = 2.0
        decomp, coverage = D.decompose(p, bins=10)
        row = coverage.set_index("x").loc["freq"]
        assert row["n_dropped_out_of_bounds"] == 8
        assert decomp["freq"].marginal["xmid"].max() <= 70.0
        # Kpod keeps those rows — the bound is per-axis, not a panel filter.
        assert coverage.set_index("x").loc["kpod", "n_in_bounds"] == len(p)

    def test_residual_fit_does_not_depend_on_the_frequency_bound(self):
        """Dropping a bad freq must not move the Kpod residual."""
        p = _panel()
        base, _ = D.decompose(p, bins=10)
        p2 = p.copy()
        p2.loc[p2.index[:5], "freq"] = 236.0
        after, _ = D.decompose(p2, bins=10)
        pd.testing.assert_frame_equal(base["kpod"].residual, after["kpod"].residual)


class TestSelectionFunnel:
    """The funnel is an audit trail, so its only real failure mode is drifting from the
    filter it claims to describe.  Every assertion here ties one to the other."""

    def test_last_stage_is_the_panel_itself(self, svod_workbook):
        wb = svod_workbook
        panel = D.load_panel(wb.path)
        fleet = D.selection_funnel(wb.path).groupby("step")["kept"].sum()
        assert int(fleet.iloc[0]) == wb.n_rows         # nothing hidden before stage 0
        assert int(fleet.iloc[-1]) == len(panel)

    def test_each_stage_drops_exactly_the_row_planted_for_it(self, svod_workbook):
        wb = svod_workbook
        fleet = D.selection_funnel(wb.path).groupby("step")["kept"].sum()
        assert list(fleet.diff().dropna()) == [-1.0] * len(wb.planted)
        assert set(D.load_panel(wb.path)["code"]) == set(wb.survivors)

    def test_all_closed_keeps_the_workover_but_not_the_other_stages(self, svod_workbook):
        wb = svod_workbook
        fleet = D.selection_funnel(wb.path, variant="all_closed")
        panel = D.load_panel(wb.path, variant="all_closed")
        assert int(fleet.groupby("step")["kept"].sum().iloc[-1]) == len(panel)
        assert "YA_8" in set(panel["code"])           # ГТМ is a closed run, not a failure
        assert "YA_5" not in set(panel["code"])       # …but a running pump still has no ННО

    def test_mc_cohort_stage_appears_only_when_the_rule_is_on(self, svod_workbook):
        wb = svod_workbook
        on = D.selection_funnel(wb.path)["stage"].unique()
        off = D.selection_funnel(wb.path, mc_cohort=False)["stage"].unique()
        assert any("Мирнинский" in s for s in on)
        assert not any("Мирнинский" in s for s in off)
        assert "MC_7" in set(D.load_panel(wb.path, mc_cohort=False)["code"])

    def test_counts_are_split_by_field_and_sour_class(self, svod_workbook):
        f = D.selection_funnel(svod_workbook.path)
        last = f[f["step"] == f["step"].max()]
        assert last.groupby("field")["kept"].sum().to_dict() == {"Ya": 2, "Vt": 1}
        assert last.set_index(["field", "h2s_class"])["kept"].loc[("Vt", "sour")] == 1

    def test_unmapped_wells_are_named_rather_than_vanishing(self, svod_workbook):
        f = D.selection_funnel(svod_workbook.path)
        assert D.NO_FIELD in set(f[f["step"] == 0]["field"])


class TestByField:
    def test_splits_and_drops_thin_fields(self):
        p = _panel(n=600)
        p.loc[p.index[:30], "field"] = "Za"  # below min_n
        per_field = D.by_field(p, var="freq", min_n=80)
        assert "Za" not in per_field
        assert {"Ya", "Vt"} <= set(per_field)

    def test_ordered_by_sample_size(self):
        p = _panel(n=600)
        p["field"] = np.where(np.arange(len(p)) < 450, "Ya", "Vt")
        per_field = D.by_field(p, var="freq", min_n=80)
        assert list(per_field) == ["Ya", "Vt"]

    def test_fit_table_columns(self):
        p = _panel()
        table = D.fit_table(D.by_field(p, var="freq", min_n=80))
        assert {"key", "n", "beta2", "p_beta2", "vertex", "p_inverted_u"} <= set(table.columns)
        assert len(table) == 2


class TestLoadPanel:
    def test_rejects_unknown_variant(self):
        with pytest.raises(ValueError, match="variant"):
            D.load_panel(variant="everything")
