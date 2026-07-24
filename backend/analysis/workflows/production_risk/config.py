"""Configuration and defaults for the production-risk workflow."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[4]


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return Path(__file__).resolve().parents[4]


REPO_ROOT = _app_root()
RESOURCE_ROOT = _resource_root()
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SLUG = "production_risk_forecast"

BUNDLE_DATE = "2026-07-15-mc2023plus"
BUNDLE_ROOT = RESOURCE_ROOT / "results" / "esp_survival_vba_models"

PLAN_FIRST_MONTH = date(2024, 1, 1)
PLAN_LAST_MONTH = date(2027, 12, 1)
FORECAST_START = date(2026, 7, 1)
HORIZON_END = date(2027, 12, 1)

PRIMARY_SCENARIO_ID = "base_p50"
STRESS_SCENARIO_ID = "stress_p75"

ESP_SCOPE_CONSERVATIVE = "conservative"

GLOBAL_DOWNTIME_FALLBACK = {"p25": 7, "p50": 16, "p75": 44}
MIN_FIELD_DOWNTIME_N = 30
MAX_REPAIR_GAP_DAYS = 730

CHANGEOUT_P90_THRESHOLD = 0.20
IMPUTED_AGE_PMF_MAX_POINTS = 25
MIN_STATE_MASS = 1e-12
HAZARD_SHIP_REASON = "physical_sensitivity_not_oos_validated"

# Monthly liquid-rate (IOR) hazard, deployed only in the stress/hazard scenario as a
# directional sensitivity dial (NOT an OOS-validated forecast term).  Covariate is the
# field-centered capped log liquid-rate ratio:
#   z = clip(log1p(Ql_m3d) - log1p(Ql_ref_field_m3d), +/- log(5))
#   theta = exp(z * (beta + gamma * log(max(age, 1))))
#
# Sign correction (Workstream C, 2026-07-15, results/production_risk_hazard_refit_c):
# the previous Extended-Cox fit (beta=-0.587, gamma=+0.093) served a physically BACKWARDS
# effect (higher liquid rate => *lower* young-pump hazard).  A serve-consistent monthly
# refit shows the empirical association is positive (higher Ql -> more failures) and that
# the negative sign was purely an artifact of the collinear z*log(age) interaction.  A
# lagged/lead causal check (lead t+3 beta=+0.84 vs lag t-3 beta=+0.09) shows most of the
# raw association is reverse causation (wells act up around the pull), leaving only a small
# forward-causal effect.  So Ql ships as a plain proportional-hazard dial with NO age
# interaction (gamma=0) at the conservative forward magnitude beta=+0.07: higher liquid
# rate -> modestly higher hazard at every age, capped at ~+12% at 5x field Ql.  It is a
# what-if lever, not promoted into the base_p50 forecast (it does not improve OOS).
QL_HAZARD_ENABLED = False  # retired 2026-07-15: superseded by the raw-Kpod U-shape dial
QL_HAZARD_COVARIATE = "log_mean_qliq_m3d_field_ref_clip5"
QL_HAZARD_BETA = 0.070
QL_HAZARD_GAMMA = 0.0
QL_HAZARD_CAP_LOG_RATIO = 1.6094379124341003  # log(5)
QL_HAZARD_GLOBAL_REF_LOG = 4.050671
QL_HAZARD_SOURCE = "production_risk_hazard_refit_c/2026-07-15 (sign-corrected plain-PH sensitivity)"
QL_HAZARD_FIELD_REF_LOG = {
    "Az": 3.946800,
    "Da": 3.689805,
    "Ic": 3.889624,
    "Mc": 3.782201,
    "Vt": 4.016012,
    "Ya": 4.253127,
    "Za": 3.832684,
}

FIELD_PREFIX_MAP = {
    "YA": "Ya",
    "VT": "Vt",
    "VTI": "Vt",
    "VTB": "Vt",
    "VTBB": "Vt",
    "IC": "Ic",
    "MC": "Mc",
    "MR": "Mc",   # Мирнинский Мр pad — same field/stratum as Mc, the driver of this refit.
    "AZ": "Az",
    "AZA": "Az",
    "AU": "Za",
    "AUY": "Za",
    "AUZ": "Za",
    "DA": "Da",
    "DAD": "Da",
}

# Prefixes intentionally routed to the registry's Global_Pooled fallback.  They
# are kept out of FIELD_PREFIX_MAP so code can distinguish an explicit decision
# from an accidental unmapped prefix and report fallback share diagnostics.
EXPLICIT_GLOBAL_FALLBACK = {
    "BT",   # Большетирский: material Big history, but no defensible existing field stratum.
    "KI",   # Кийский: no existing Ki stratum in the shipped Weibull registry.
    "MSH",  # Мышевского / Кийский area: no existing Msh/Ki stratum.
    # Minor / unclear prefixes: too few wells to justify borrowing another field's
    # Weibull, so pooled rather than force-mapped (see failure-rate review).
    "NE",   # Непский — geologically distinct from Мирнинский; do not model as Mc.
    "AM",
    "ZYI",
    "YAY",  # single well; not clearly the Ya stratum.
}

# --- Infant-mortality hazard ------------------------------------------------
# MEASURED, unlike the new-launch uncertainty dial below.  The shipped Weibull is
# calibrated past ~45 days (empirical/fitted hazard ratio hugs 1.0 in every stratum)
# but delivers only ~65% of the real day-8 hazard — a defect that is identical
# fleet-wide because a single Weibull cannot express "infant spike then flat":
#
#   empirical/fitted at 8 d:  Ya 1.53 | Global 1.52 | Vt 1.78 | Az 1.58 | Za 1.36 | Mc 4.37
#   ... and at 45-950 d:      ~1.0 everywhere
#
# So the registry keeps the long-run shape it earns on the thick strata, and this
# dial supplies the early-life correction it structurally cannot hold.  Fitted as
# theta(age) = 1 + u0*exp(-age/tau) to the pooled ratio, weighted by events.
#
# Unlike UNCERTAINTY_HAZARD this is NOT a stress sensitivity and NOT restricted to
# new wells: every pump re-enters its infant window after every renewal.
INFANT_HAZARD_ENABLED = False
INFANT_HAZARD_U0 = 1.184
INFANT_HAZARD_TAU_DAYS = 11.0
INFANT_HAZARD_CAP = 2.50
INFANT_HAZARD_FIELD_PARAMS: dict[str, dict[str, float]] = {}
INFANT_HAZARD_SOURCE = "measured 2026-07-17: empirical/fitted hazard ratio, pooled over strata"


def infant_hazard_params(field: str | None = None) -> dict[str, float]:
    params = {
        "u0": float(INFANT_HAZARD_U0),
        "tau_days": float(INFANT_HAZARD_TAU_DAYS),
        "cap": float(INFANT_HAZARD_CAP),
    }
    if field:
        params.update(INFANT_HAZARD_FIELD_PARAMS.get(str(field), {}))
    return params


# --- ГТМ pump-age reset -----------------------------------------------------
# A ГТМ pull replaces the pump: tracked by serial in WellsArtificialLiftBig, the
# SAME pump comes back after a ГТМ in only 4.0% of cases (35 of 872 consecutive
# strict-ESP run pairs, fleet-wide; 2 of 45 for Mc).  So a planned ГТМ resets the
# pump's operating age to 0 for ~96% of the mass.
#
# This matters because the survival fit treats every install→pull as its own unit
# starting at age 0 and censors at ГТМ.  Without the reset the projection renews
# only on FAILURE, so pumps in ГТМ-heavy fields (Мирнинский: 84 of 133 pulls) age
# far past anything observed — Mc has zero failures beyond 663 days, so the hazard
# out there is pure parametric extrapolation.
GTM_AGE_RESET_ENABLED = False

# --- Мирнинский (Mc/Mr) cohort ----------------------------------------------
# EVERY Mc statistic — population, fit, calibration, per-УН table — is built from runs
# INSTALLED on or after this date.
#
# This is a COHORT filter on the install date, NOT left truncation of exposure.  The
# difference matters: left truncation (`esp_population.add_entry_age`) keeps a pre-2024
# run's post-2024 exposure, which hides that recently installed runs are shorter.  That
# was tried and is wrong.
#
# Pre-2023 Mc is the anomaly, not 2024+: 2 failures against 11.8 expected on 21 runs —
# luck or incomplete early records.  Other fields use all history.
MC_INSTALL_COHORT_START = date(2024, 1, 1)
# Мирнинский is the Mc + Mr pads.  `esp_population` already maps the MR prefix to "Mc"
# via FIELD_PREFIX_MAP, so "Mc" alone covers it there; the survival mart keeps the raw
# field, hence both names.
MC_COHORT_FIELDS = ("Mc", "Mr")
GTM_PUMP_REPLACED_SHARE = 0.96

# v3.2 sour re-labeling.  The Свод «Кислый/Некислый» flag is written only from the
# failure/workover DB, so a well's *running* (OPEN) pump never receives it and defaults
# nonsour — 30 Vt running pumps land in Vt_nonsour though their well is sour.  When True,
# the well-level sour class ("sour" if ANY run of the well is sour) is inherited by ALL
# runs, not just the Big rows; when False the shipped v3.1 behaviour is kept (Свод runs
# keep their per-run flag).  Caveat: the OR roll-up propagates any FALSE-positive sour
# flag on one workover row to the whole well — prefer a lab-H2S crosscheck if the failure
# DB flags are unverified.  Default False (flipping it moves every Vt sour statistic).
SOUR_WELL_LEVEL_ALL_RUNS = False

# Factual failure-rate display through the last available partly observed month.
# Earlier builds stopped at 2026-04 because May/June were incomplete; keep them
# visible now so the workbook shows every factual data point in the source.
DEFAULT_FACT_THROUGH_MONTH = "2026-06"

# The reworked Свод's OPEN (running, censored) rows carry no «Наработка (сут)»; their
# techregime snapshot is this date, so a running pump's age = SVOD_OPEN_ASOF - монтаж
# (user decision 2026-07-22).  Bump when a new Свод drop changes the snapshot.
SVOD_OPEN_ASOF = date(2026, 6, 30)
# Dates beyond this bound are typos (e.g. Vt_7805: демонтаж 2026-11-04 against
# остановка 2026-04-14); rows carrying them are excluded as weird.
SVOD_FUTURE_BOUND = date(2026, 8, 15)


def bundle_dir(bundle_date: str = BUNDLE_DATE) -> Path:
    return BUNDLE_ROOT / bundle_date


def model_registry_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_models.csv"


def hazard_coeffs_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_cox_coeffs.csv"


def run_covariates_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_run_covariates.csv"


def time_map_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_time_map.csv"


def observed_failures_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_observed_failures.csv"


def ql_hazard_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_ql_hazard.csv"


def kpod_hazard_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_kpod_hazard.csv"


def uncertainty_hazard_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_uncertainty_hazard.csv"


def _load_ql_hazard_overrides(bundle_date: str = BUNDLE_DATE) -> None:
    """Override the in-code Ql hazard defaults from an optional bundle file.

    Lets Ql (enabled/beta/gamma/cap/field refs) be tuned by editing
    ``esp_ql_hazard.csv`` in the bundle — no rebuild of the frozen EXE required,
    exactly like ``esp_models.csv`` (strata) and ``esp_cox_coeffs.csv`` (static Cox),
    which are already read at runtime.  Absent/invalid file => the hardcoded defaults
    above stay in force.  Loaded once at import for the default bundle date; a run that
    points at a different bundle keeps these values (Ql is a module-level constant).
    """
    global QL_HAZARD_ENABLED, QL_HAZARD_BETA, QL_HAZARD_GAMMA
    global QL_HAZARD_CAP_LOG_RATIO, QL_HAZARD_GLOBAL_REF_LOG
    global QL_HAZARD_FIELD_REF_LOG, QL_HAZARD_SOURCE
    path = ql_hazard_path(bundle_date)
    if not path.exists():
        return
    import csv
    import warnings
    try:
        refs = dict(QL_HAZARD_FIELD_REF_LOG)
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                key = str(row.get("param", "")).strip()
                val = str(row.get("value", "")).strip()
                if not key or not val:
                    continue
                if key == "enabled":
                    QL_HAZARD_ENABLED = val.upper() in ("TRUE", "1", "YES")
                elif key == "beta":
                    QL_HAZARD_BETA = float(val)
                elif key == "gamma":
                    QL_HAZARD_GAMMA = float(val)
                elif key == "cap_log_ratio":
                    QL_HAZARD_CAP_LOG_RATIO = float(val)
                elif key == "global_ref_log":
                    QL_HAZARD_GLOBAL_REF_LOG = float(val)
                elif key.startswith("field_ref."):
                    refs[key.split(".", 1)[1]] = float(val)
        QL_HAZARD_FIELD_REF_LOG = refs
        QL_HAZARD_SOURCE = f"{QL_HAZARD_SOURCE} + bundle override esp_ql_hazard.csv"
    except Exception as exc:  # never let a bad override crash the workflow
        warnings.warn(f"esp_ql_hazard.csv override ignored ({exc}); using in-code Ql defaults")


# Raw Kpod = Ql / Qnominal load-mismatch hazard, stress-only.  This is intentionally
# a sensitivity dial rather than an OOS-validated base forecast term.  Static
# freq-normalized Kpod rows in esp_cox_coeffs.csv are disabled in the shipped bundle
# to avoid counting the same load-mismatch idea twice.
KPOD_HAZARD_ENABLED = True
KPOD_HAZARD_K_LO = 0.7
KPOD_HAZARD_K_HI = 1.2
KPOD_HAZARD_BETA_UNDER = 0.0
KPOD_HAZARD_BETA_OVER = 0.8
KPOD_HAZARD_GAMMA_UNDER = 0.0
KPOD_HAZARD_GAMMA_OVER = 0.0
KPOD_HAZARD_CAP_UNDER = 0.7
KPOD_HAZARD_CAP_OVER = 0.8
KPOD_HAZARD_FIELD_PARAMS: dict[str, dict[str, float]] = {}
KPOD_HAZARD_SOURCE = "kpod_ushape_v1 (stress sensitivity, not OOS-validated)"


def kpod_hazard_params(field: str | None = None) -> dict[str, float]:
    """Resolve Kpod U-shape parameters, with optional field-specific overrides."""
    params = {
        "k_lo": float(KPOD_HAZARD_K_LO),
        "k_hi": float(KPOD_HAZARD_K_HI),
        "beta_under": float(KPOD_HAZARD_BETA_UNDER),
        "beta_over": float(KPOD_HAZARD_BETA_OVER),
        "gamma_under": float(KPOD_HAZARD_GAMMA_UNDER),
        "gamma_over": float(KPOD_HAZARD_GAMMA_OVER),
        "cap_under": float(KPOD_HAZARD_CAP_UNDER),
        "cap_over": float(KPOD_HAZARD_CAP_OVER),
    }
    if field is not None:
        params.update(KPOD_HAZARD_FIELD_PARAMS.get(str(field), {}))
    return params


def _load_kpod_hazard_overrides(bundle_date: str = BUNDLE_DATE) -> None:
    """Override Kpod U-shape hazard defaults and field values from an optional bundle CSV."""
    global KPOD_HAZARD_ENABLED, KPOD_HAZARD_K_LO, KPOD_HAZARD_K_HI
    global KPOD_HAZARD_BETA_UNDER, KPOD_HAZARD_BETA_OVER
    global KPOD_HAZARD_GAMMA_UNDER, KPOD_HAZARD_GAMMA_OVER
    global KPOD_HAZARD_CAP_UNDER, KPOD_HAZARD_CAP_OVER, KPOD_HAZARD_FIELD_PARAMS, KPOD_HAZARD_SOURCE
    path = kpod_hazard_path(bundle_date)
    if not path.exists():
        return
    import csv
    import warnings
    field_key_map = {
        "k_lo": "k_lo",
        "k_hi": "k_hi",
        "beta_under": "beta_under",
        "beta_over": "beta_over",
        "gamma_under": "gamma_under",
        "gamma_over": "gamma_over",
        "cap_under": "cap_under",
        "cap_over": "cap_over",
    }
    try:
        field_params = {str(k): dict(v) for k, v in KPOD_HAZARD_FIELD_PARAMS.items()}
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                key = str(row.get("param", "")).strip()
                val = str(row.get("value", "")).strip()
                if not key or not val:
                    continue
                if key.startswith("field."):
                    parts = key.split(".")
                    if len(parts) == 3 and parts[2] in field_key_map:
                        dest = field_params.setdefault(parts[1], {})
                        value = float(val)
                        if parts[2] in {"beta_under", "beta_over", "cap_under", "cap_over"}:
                            value = max(0.0, value)
                        dest[field_key_map[parts[2]]] = value
                    continue
                if key == "enabled":
                    KPOD_HAZARD_ENABLED = val.upper() in ("TRUE", "1", "YES")
                elif key == "k_lo":
                    KPOD_HAZARD_K_LO = float(val)
                elif key == "k_hi":
                    KPOD_HAZARD_K_HI = float(val)
                elif key == "beta_under":
                    KPOD_HAZARD_BETA_UNDER = max(0.0, float(val))
                elif key == "beta_over":
                    KPOD_HAZARD_BETA_OVER = max(0.0, float(val))
                elif key == "gamma_under":
                    KPOD_HAZARD_GAMMA_UNDER = float(val)
                elif key == "gamma_over":
                    KPOD_HAZARD_GAMMA_OVER = float(val)
                elif key == "cap_under":
                    KPOD_HAZARD_CAP_UNDER = max(0.0, float(val))
                elif key == "cap_over":
                    KPOD_HAZARD_CAP_OVER = max(0.0, float(val))
        KPOD_HAZARD_FIELD_PARAMS = field_params
        KPOD_HAZARD_SOURCE = f"{KPOD_HAZARD_SOURCE} + bundle override esp_kpod_hazard.csv"
    except Exception as exc:
        warnings.warn(f"esp_kpod_hazard.csv override ignored ({exc}); using in-code Kpod defaults")


UNCERTAINTY_HAZARD_ENABLED = False  # off by default 2026-07-15: no OOS/causal validation yet
UNCERTAINTY_HAZARD_U0 = 0.25
UNCERTAINTY_HAZARD_TAU_DAYS = 60.0
UNCERTAINTY_HAZARD_CAP = 1.30
UNCERTAINTY_HAZARD_APPLY_TO = {"new_plan_only", "new_df_esp", "new_tr_esp"}
UNCERTAINTY_HAZARD_FIELD_PARAMS: dict[str, dict[str, float]] = {}
UNCERTAINTY_HAZARD_SOURCE = "new_launch_uncertainty_v1 (stress sensitivity, not OOS-validated)"


def uncertainty_hazard_params(field: str | None = None) -> dict[str, float]:
    params = {
        "u0": float(UNCERTAINTY_HAZARD_U0),
        "tau_days": float(UNCERTAINTY_HAZARD_TAU_DAYS),
        "cap": float(UNCERTAINTY_HAZARD_CAP),
    }
    if field is not None:
        params.update(UNCERTAINTY_HAZARD_FIELD_PARAMS.get(str(field), {}))
    return params


def _load_uncertainty_hazard_overrides(bundle_date: str = BUNDLE_DATE) -> None:
    """Override new-launch uncertainty hazard defaults and field values from a bundle CSV."""
    global UNCERTAINTY_HAZARD_ENABLED, UNCERTAINTY_HAZARD_U0, UNCERTAINTY_HAZARD_TAU_DAYS
    global UNCERTAINTY_HAZARD_CAP, UNCERTAINTY_HAZARD_APPLY_TO
    global UNCERTAINTY_HAZARD_FIELD_PARAMS, UNCERTAINTY_HAZARD_SOURCE
    path = uncertainty_hazard_path(bundle_date)
    if not path.exists():
        return
    import csv
    import warnings
    field_key_map = {"u0": "u0", "tau_days": "tau_days", "cap": "cap"}
    try:
        field_params = {str(k): dict(v) for k, v in UNCERTAINTY_HAZARD_FIELD_PARAMS.items()}
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                key = str(row.get("param", "")).strip()
                val = str(row.get("value", "")).strip()
                if not key or not val:
                    continue
                if key.startswith("field."):
                    parts = key.split(".")
                    if len(parts) == 3 and parts[2] in field_key_map:
                        value = float(val)
                        if parts[2] in {"u0", "tau_days"}:
                            value = max(0.0, value)
                        elif parts[2] == "cap":
                            value = max(1.0, value)
                        field_params.setdefault(parts[1], {})[field_key_map[parts[2]]] = value
                    continue
                if key == "enabled":
                    UNCERTAINTY_HAZARD_ENABLED = val.upper() in ("TRUE", "1", "YES")
                elif key == "u0":
                    UNCERTAINTY_HAZARD_U0 = max(0.0, float(val))
                elif key == "tau_days":
                    UNCERTAINTY_HAZARD_TAU_DAYS = max(0.0, float(val))
                elif key == "cap":
                    UNCERTAINTY_HAZARD_CAP = max(1.0, float(val))
                elif key == "apply_to":
                    UNCERTAINTY_HAZARD_APPLY_TO = {item.strip() for item in val.split(",") if item.strip()}
        UNCERTAINTY_HAZARD_FIELD_PARAMS = field_params
        UNCERTAINTY_HAZARD_SOURCE = f"{UNCERTAINTY_HAZARD_SOURCE} + bundle override esp_uncertainty_hazard.csv"
    except Exception as exc:
        warnings.warn(
            f"esp_uncertainty_hazard.csv override ignored ({exc}); using in-code uncertainty defaults"
        )


_load_ql_hazard_overrides()
_load_kpod_hazard_overrides()
_load_uncertainty_hazard_overrides()


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    label_ru: str
    downtime_key: str
    hazard_mode: str
    primary: bool = False


DEFAULT_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("base_p25", "База / П25 ремонт", "p25", "baseline"),
    ScenarioSpec("base_p50", "База / П50 ремонт", "p50", "baseline", primary=True),
    ScenarioSpec("base_p75", "База / П75 ремонт", "p75", "baseline"),
    ScenarioSpec("stress_p75", "Чувствительность / П75 ремонт", "p75", "stress"),
)


@dataclass(frozen=True)
class RunConfig:
    forecast_start: date = FORECAST_START
    horizon_end: date = HORIZON_END
    bundle_date: str = BUNDLE_DATE
    pp_master_path: Path | None = None
    gtm_schedule_path: Path | None = None
    prediction_workbook_path: Path | None = None
    techregime_workbook_path: Path | None = None
    equipment_big_path: Path | None = None
    fact_through_month: str = DEFAULT_FACT_THROUGH_MONTH
    downtime_override_days: int | None = None
    esp_scope_policy: str = ESP_SCOPE_CONSERVATIVE
    changeout_p90: float = CHANGEOUT_P90_THRESHOLD
    write_excel: bool = True
    # By default only the three deliverable workbooks are written (flat in /results);
    # set True (CLI --full-tables) to also emit the detailed CSV/parquet/json tree.
    full_tables: bool = False
    # CatBoost-vs-Weibull failure-rate comparison (heavy, Python/CLI only — never the
    # frozen EXE).  Off by default so the shipped Weibull deliverable is byte-identical.
    enable_catboost_compare: bool = False
    output_slug: str = SLUG
    scenarios: tuple[ScenarioSpec, ...] = DEFAULT_SCENARIOS
    gtm_age_reset: bool = GTM_AGE_RESET_ENABLED
