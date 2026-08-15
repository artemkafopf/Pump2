import numpy as np
import pandas as pd

from analysis.features.restart_events import _window_first_op_days, reconcile_restarts
from analysis.models.ml.regression_ttf import (
    RULRegressor,
    TTFRegressor,
    assert_no_perfect_tte_leakage,
    conditional_ipcw_weights,
    ipcw_weights,
)


def _synthetic_survival_df():
    return pd.DataFrame(
        {
            "tte": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "event": [1, 0, 1, 0, 0, 0, 0],
            "stratum_key": ["A", "B", "A", "B", "B", "B", "A"],
            "well_key": ["w1", "w2", "w1", "w3", "w4", "w5", "w6"],
        }
    )


def _minimal_regression_df():
    return pd.DataFrame(
        {
            "row_id": [1, 2],
            "tte": [200.0, 90.0],
            "event": [1, 0],
            "run_days": [400.0, 180.0],
            "stratum_key": ["S", "S"],
            "well_key": ["w1", "w2"],
            "install_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "is_sour_flagged": [0, 1],
            "log_glf_mean_opdays": [1.0, 2.0],
            "freq_above_55hz_pct_early": [10.0, 20.0],
            "n_freq_steps_per_100d": [1.0, 2.0],
            "freq_std_early": [0.5, 0.7],
            "frac_kpod_below_0p7": [0.1, 0.2],
            "kpod_freq_mean": [0.8, 0.9],
            "load_std_early": [3.0, 4.0],
            "load_mean": [70.0, 80.0],
            "n_restarts_early30": [1.0, 0.0],
            "log_motor_power_kw": [4.0, 5.0],
            "vg_m": [1000.0, 1200.0],
            "nominal_freq_hz": [50.0, 50.0],
            "pbubble_atm": [100.0, 110.0],
            "curvature_deg10m": [0.0, 0.3],
            "log_run_seq": [0.69, 1.1],
            "log_days_since_prev_failure": [0.0, 2.0],
            "field": ["F", "F"],
            "contractor": ["C", "C"],
            "h2s_class": ["nonsour", "nonsour"],
            "pump_gabarit": ["5", "5"],
            "install_period": ["2020-2022", "2020-2022"],
        }
    )


def test_ipcw_weights_global_fallback_known_km():
    df = _synthetic_survival_df()
    w = ipcw_weights(df, truncate=0.0)
    assert w.iloc[0] == 1.0
    assert np.isclose(w.iloc[2], 1.2)  # global censoring KM G(30)=5/6
    assert (w[df["event"].eq(0)] == 0).all()


def test_conditional_landmark_weight_known_km():
    df = _synthetic_survival_df()
    ldf = df.loc[[2]].copy()
    ldf["age_op_d"] = 25.0
    w = conditional_ipcw_weights(ldf, df, truncate=0.0)
    assert w.iloc[0] == 1.0  # no censoring event in (25, 30]


def test_early_window_restarts_excludes_late_restart():
    days = pd.date_range("2020-01-01", periods=210, freq="D")
    qliq = np.ones(len(days))
    qliq[8] = 0.0
    qliq[9] = 1.0
    qliq[198] = 0.0
    qliq[199] = 1.0
    frame = pd.DataFrame({"dt": days, "qliq": qliq})
    whole = reconcile_restarts(frame)
    early = reconcile_restarts(_window_first_op_days(frame, 30))
    assert whole["n_restarts_reconciled"] == 2
    assert early["n_restarts_reconciled"] == 1


def test_ttf_quantiles_are_sorted_after_prediction():
    class Dummy:
        def __init__(self, values):
            self.values = values

        def predict(self, _x):
            return np.asarray(self.values)

    model = TTFRegressor(include_equipment=False)
    model.features_ = ["field"]
    model.models_ = {
        0.1: Dummy([30.0, 5.0]),
        0.5: Dummy([10.0, 15.0]),
        0.9: Dummy([20.0, 1.0]),
    }
    pred = model.predict_ttf(pd.DataFrame({"field": ["F", "F"]}))
    assert (pred["b10_op_d"] <= pred["b50_op_d"]).all()
    assert (pred["b50_op_d"] <= pred["b90_op_d"]).all()


def test_landmark_target_and_censor_flag():
    df = _minimal_regression_df()
    ldf = RULRegressor(include_equipment=False).make_landmark_frame(df, landmarks=(0, 90, 180, 200))
    fail_rows = ldf[ldf["row_id"].eq(1)]
    assert fail_rows["age_op_d"].tolist() == [0.0, 90.0, 180.0]
    assert fail_rows["rul_target_op_d"].tolist() == [200.0, 110.0, 20.0]
    cens_rows = ldf[ldf["row_id"].eq(2)]
    assert cens_rows["event"].eq(0).all()
    assert 90.0 not in cens_rows["age_op_d"].tolist()


def test_feature_leakage_guard_no_perfect_tte_feature():
    df = pd.DataFrame({"tte": [10.0, 20.0, 30.0, 40.0], "leaky": [1.0, 2.0, 3.0, 4.0], "ok": [1.0, 1.0, 2.0, 1.0]})
    try:
        assert_no_perfect_tte_leakage(df, ["leaky", "ok"])
    except AssertionError as exc:
        assert "leaky" in str(exc)
    else:
        raise AssertionError("leakage guard did not flag a perfect TTE correlate")


def test_to_calendar_adds_calendar_columns():
    rul = pd.DataFrame({"rul_b10_op_d": [50.0], "rul_b50_op_d": [100.0], "rul_b90_op_d": [150.0]})
    model = RULRegressor(include_equipment=False)
    model.uptime_factors_ = pd.Series({"S": 0.5})
    out = model.to_calendar(rul, pd.Series(["S"]))
    assert out["rul_b50_cal_d"].iloc[0] == 200.0
    assert "rul_b50_op_d" in out.columns
