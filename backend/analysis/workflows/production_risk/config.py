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

BUNDLE_DATE = "2026-07-08"
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

FIELD_PREFIX_MAP = {
    "YA": "Ya",
    "VT": "Vt",
    "VTI": "Vt",
    "VTB": "Vt",
    "VTBB": "Vt",
    "IC": "Ic",
    "MC": "Mc",
    "AZ": "Az",
    "AZA": "Az",
    "AU": "Za",
    "AUY": "Za",
    "AUZ": "Za",
    "DA": "Da",
    "DAD": "Da",
}


def bundle_dir(bundle_date: str = BUNDLE_DATE) -> Path:
    return BUNDLE_ROOT / bundle_date


def model_registry_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_models.csv"


def hazard_coeffs_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_cox_coeffs.csv"


def run_covariates_path(bundle_date: str = BUNDLE_DATE) -> Path:
    return bundle_dir(bundle_date) / "esp_run_covariates.csv"


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
    downtime_override_days: int | None = None
    esp_scope_policy: str = ESP_SCOPE_CONSERVATIVE
    changeout_p90: float = CHANGEOUT_P90_THRESHOLD
    write_excel: bool = True
    # By default only the three deliverable workbooks are written (flat in /results);
    # set True (CLI --full-tables) to also emit the detailed CSV/parquet/json tree.
    full_tables: bool = False
    output_slug: str = SLUG
    scenarios: tuple[ScenarioSpec, ...] = DEFAULT_SCENARIOS
