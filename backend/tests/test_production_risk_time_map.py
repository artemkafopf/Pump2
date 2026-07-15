from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis.workflows.production_risk.time_map import TimeMap, interval_slices


def _map(rows: list[dict]) -> TimeMap:
    return TimeMap(pd.DataFrame(rows))


def test_time_map_predicts_field_age_season_before_global():
    tmap = _map(
        [
            {
                "row_type": "uptime",
                "field": "GLOBAL",
                "age_band": "ALL",
                "calendar_month": 0,
                "uptime": 0.4,
                "n_run_months": 100,
                "idle_hazard_fraction": np.nan,
                "source": "test",
            },
            {
                "row_type": "uptime",
                "field": "Ya",
                "age_band": "000_030",
                "calendar_month": 1,
                "uptime": 0.7,
                "n_run_months": 25,
                "idle_hazard_fraction": np.nan,
                "source": "test",
            },
        ]
    )

    value, source = tmap.predict_uptime("Ya", 10.0, 1, fallback=0.9)

    assert value == 0.7
    assert source == "map:Ya:000_030:1"


def test_interval_slices_scale_profile_to_known_total_op_days():
    tmap = _map(
        [
            {
                "row_type": "uptime",
                "field": "Ya",
                "age_band": "ALL",
                "calendar_month": 0,
                "uptime": 0.5,
                "n_run_months": 100,
                "idle_hazard_fraction": np.nan,
                "source": "test",
            }
        ]
    )

    slices = interval_slices(
        months=["2026-01", "2026-02"],
        start=datetime(2026, 1, 1),
        end=datetime(2026, 3, 1),
        code="YA_001",
        reporting_field="Ярактинский УН",
        model_field="Ya",
        total_op=20.0,
        fallback_uptime=1.0,
        time_map=tmap,
    )

    assert len(slices) == 2
    assert abs(sum(s.op_days for s in slices) - 20.0) < 1e-9
    assert slices[1].age_start == slices[0].op_days


def test_interval_slices_add_idle_fraction_for_nonproducing_plan_month():
    tmap = _map(
        [
            {
                "row_type": "uptime",
                "field": "Ya",
                "age_band": "ALL",
                "calendar_month": 0,
                "uptime": 0.5,
                "n_run_months": 100,
                "idle_hazard_fraction": np.nan,
                "source": "test",
            },
            {
                "row_type": "idle_hazard",
                "field": "Ярактинский УН",
                "age_band": "ALL",
                "calendar_month": 0,
                "uptime": np.nan,
                "n_run_months": 100,
                "idle_hazard_fraction": 0.2,
                "source": "test",
            },
        ]
    )
    plan = SimpleNamespace(
        months=["2026-01"],
        op_days=pd.DataFrame([[0.0]], index=["YA_001"], columns=["2026-01"]),
        op_days_raw=pd.DataFrame([[0.0]], index=["YA_001"], columns=["2026-01"]),
        cal_days=pd.Series({"2026-01": 31.0}),
    )
    active = pd.DataFrame([[False]], index=["YA_001"], columns=["2026-01"])

    slices = interval_slices(
        months=["2026-01"],
        start=datetime(2026, 1, 1),
        end=datetime(2026, 2, 1),
        code="YA_001",
        reporting_field="Ярактинский УН",
        model_field="Ya",
        total_op=None,
        fallback_uptime=1.0,
        time_map=tmap,
        plan=plan,
        active=active,
    )

    assert len(slices) == 1
    assert abs(slices[0].op_days - 3.1) < 1e-9
    assert slices[0].producing is False
