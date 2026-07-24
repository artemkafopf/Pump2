import pandas as pd

from analysis.workflows.production_risk import esp_population as P


def _svod(rows):
    return pd.DataFrame(rows, columns=[
        "code", "install", "end", "tte_full", "pull_reason",
        "contractor_group", "has_failed_unit", "field", "h2s_class", "source",
    ])


def _big(rows):
    return pd.DataFrame(rows, columns=[
        "code", "install", "end", "tte_full", "pull_reason",
        "contractor_group", "has_failed_unit", "field", "h2s_class", "source",
    ])


SVOD = _svod([
    ["Vt_001", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01"), 151.0,
     "клин", "brt", True, "Vt", "sour", "svod"],
    ["Vt_002", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01"), 151.0,
     "клин", "brt", True, "Vt", "nonsour", "svod"],
])

BIG = _big([
    # running pump on the sour well — Big carries no «Кислый/Некислый»
    ["Vt_001", pd.Timestamp("2025-01-01"), pd.NaT, float("nan"), None,
     "brt", False, "Vt", "nonsour", "big_open"],
    ["Vt_002", pd.Timestamp("2025-01-01"), pd.NaT, float("nan"), None,
     "brt", False, "Vt", "nonsour", "big_open"],
    # well absent from Свод — no sour evidence, stays nonsour
    ["Vt_999", pd.Timestamp("2025-01-01"), pd.NaT, float("nan"), None,
     "brt", False, "Vt", "nonsour", "big_open"],
    # a CLOSED Big-only run: a real failure Свод never recorded
    ["Vt_003", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-01"), 182.0,
     "клин", "brt", True, "Vt", "nonsour", "big_closed"],
])


def test_big_run_inherits_sour_from_its_wells_svod_history():
    pop = P.build("2026-01-01", svod=SVOD, big_runs=BIG)
    running = pop[pop["source"] == "big_open"].set_index("code")["h2s_class"]
    assert running["Vt_001"] == "sour"
    assert running["Vt_002"] == "nonsour"


def test_big_run_of_unknown_well_defaults_nonsour():
    pop = P.build("2026-01-01", svod=SVOD, big_runs=BIG)
    running = pop[pop["source"] == "big_open"].set_index("code")["h2s_class"]
    assert running["Vt_999"] == "nonsour"


def test_sour_stratum_keeps_its_censored_runs():
    """The bug this guards: every running sour pump landing in the nonsour stratum
    strips sour of its survivors and inflates its hazard."""
    pop = P.build("2026-01-01", svod=SVOD, big_runs=BIG)
    sour = pop[pop["h2s_class"] == "sour"]
    assert (sour["event"] == 0).sum() == 1
    assert set(sour["stratum"]) == {"Vt_sour_brt"}


def test_big_only_closed_run_contributes_its_failure():
    """Свод-only populations lose these entirely — 619 runs for Ya, a third of its
    events — which biases the fitted hazard down."""
    pop = P.build("2026-01-01", svod=SVOD, big_runs=BIG)
    closed = pop[pop["source"] == "big_closed"].set_index("code")
    assert closed.loc["Vt_003", "event"] == 1
    assert closed.loc["Vt_003", "tte"] == 182.0


def test_v32_flag_relabels_svod_running_pump_on_a_sour_well():
    """v3.2: the Свод sour flag is written only from the failure/workover DB, so a
    running (OPEN) Свод pump on a sour well never gets it.  With the flag on, the
    well-level roll-up recovers it; with the flag off (default) it stays nonsour."""
    from analysis.workflows.production_risk import config as C

    svod = _svod([
        # failed run: flagged sour by the workover DB
        ["Vt_050", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01"), 151.0,
         "клин", "brt", True, "Vt", "sour", "svod"],
        # running run on the same well: no flag written, arrives nonsour
        ["Vt_050", pd.Timestamp("2025-06-01"), pd.NaT, float("nan"), None,
         "brt", False, "Vt", "nonsour", "svod"],
    ])
    big = _big([])
    big["install"] = pd.to_datetime(big["install"])
    big["end"] = pd.to_datetime(big["end"])

    C.SOUR_WELL_LEVEL_ALL_RUNS = False
    off = P.build("2026-01-01", svod=svod, big_runs=big)
    assert off.loc[off["end"].isna(), "h2s_class"].iloc[0] == "nonsour"

    try:
        C.SOUR_WELL_LEVEL_ALL_RUNS = True
        on = P.build("2026-01-01", svod=svod, big_runs=big)
        assert on.loc[on["end"].isna(), "h2s_class"].iloc[0] == "sour"
        assert (on["h2s_class"] == "sour").sum() == 2
    finally:
        C.SOUR_WELL_LEVEL_ALL_RUNS = False


def test_within_tol_matches_only_inside_the_window():
    """The dedup rule: a Big row within ±7 d of a Свод install is the SAME physical
    run.  Taking every Big open row double-counted 171 runs as free censored exposure."""
    target = pd.Timestamp("2024-01-10")
    assert P._within_tol(target, [pd.Timestamp("2024-01-05")], 7)
    assert P._within_tol(target, [pd.Timestamp("2024-01-17")], 7)
    assert not P._within_tol(target, [pd.Timestamp("2024-01-18")], 7)
    assert not P._within_tol(target, [pd.NaT], 7)
    assert not P._within_tol(target, [], 7)


def test_select_applies_the_mc_installs_2024plus_cohort():
    """Every Mc statistic is built from installs 2024+ — a cohort filter on the install
    date, not left truncation of exposure."""
    pop = pd.DataFrame({
        "code": ["Mc_1", "Mc_2", "Ya_1"],
        "install": [pd.Timestamp("2023-06-01"), pd.Timestamp("2024-06-01"),
                    pd.Timestamp("2019-01-01")],
        "field": ["Mc", "Mc", "Ya"],
        "h2s_class": ["nonsour"] * 3,
        "contractor_group": ["brt"] * 3,
    })
    mc = P.select(pop, "Mc")
    assert list(mc["code"]) == ["Mc_2"]
    # opt-out still available for an explicitly labelled all-history contrast
    assert len(P.select(pop, "Mc", mc_cohort=False)) == 2
    # other fields keep their full history
    assert len(P.select(pop, "Ya")) == 1


def test_well_sour_map_is_sour_if_any_run_is_sour():
    m = P.well_sour_map(_svod([
        ["Vt_001", pd.Timestamp("2023-01-01"), pd.NaT, 1.0, None, "brt", False, "Vt", "nonsour", "svod"],
        ["Vt_001", pd.Timestamp("2024-01-01"), pd.NaT, 1.0, None, "brt", False, "Vt", "sour", "svod"],
    ]))
    assert m["Vt_001"] == "sour"
