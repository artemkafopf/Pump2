"""Unified path registry for the Pump2 project.

Single source of truth for:
  - data layer roots (raw / interim / marts / warehouse)
  - input file resolution (env var → local copy → external default)
  - sqlite database resolution
  - results directory factory
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date as _date
from pathlib import Path


# ── repo roots ────────────────────────────────────────────────────────────────

def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return Path(__file__).resolve().parents[2]


REPO_ROOT: Path = _app_root()
RESOURCE_ROOT: Path = _resource_root()

# ── data layer ────────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # source xlsx/sqlite, read-only inputs
INTERIM_DIR = DATA_DIR / "interim"  # cleaned / enriched intermediates
MARTS_DIR = DATA_DIR / "marts"      # modelling-ready datasets
WAREHOUSE_DIR = DATA_DIR / "warehouse"  # final db / mart layers

LOCAL_INPUT_DIR = DATA_DIR / "inputs"   # legacy xlsx mirror location
LOCAL_SQLITE_DIR = DATA_DIR / "sqlite"  # legacy sqlite mirror location

# ── results layer ─────────────────────────────────────────────────────────────

RESULTS_ROOT = REPO_ROOT / "results"

_RESULT_SUBDIRS = ("tables", "figures", "reports", "models", "logs")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=3,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def results_dir(slug: str, run_date: str | _date | None = None) -> Path:
    """Return (and create) ``results/<slug>/<YYYY-MM-DD>/``.

    Creates standard subdirs (tables, figures, reports, models, logs) and
    writes a skeleton manifest.json on first call.  Subsequent calls with the
    same slug+date are idempotent.
    """
    if run_date is None:
        run_date = _date.today()
    date_str = run_date.isoformat() if isinstance(run_date, _date) else str(run_date)
    run_dir = RESULTS_ROOT / slug / date_str
    for sub in _RESULT_SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "analysis": slug,
            "date": date_str,
            "git_commit": _git_commit(),
            "script": None,
            "inputs": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


# ── shared resolver ───────────────────────────────────────────────────────────

def _resolve(*, env_var: str, local_dir: Path, local_name: str, external_default: Path) -> Path:
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return Path(env_value)
    local_path = local_dir / local_name
    if local_path.exists():
        return local_path
    return external_default


# ── input xlsx ────────────────────────────────────────────────────────────────

DEFAULT_V03_ALL_EXTERNAL = Path(r"D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_all.xlsx")
DEFAULT_V03_FAILURES_EXTERNAL = Path(r"D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_failures.xlsx")
DEFAULT_PRESENTATION_EXTERNAL = Path(r"C:\Users\alexe\Downloads\Отказность Аналитика(1).pptx")
DEFAULT_EQUIPMENT_BIG_EXTERNAL = Path(r"D:\Projects\Pumps\data\big\WellsArtificialLiftBig.xlsx")


def resolve_equipment_big_path() -> Path:
    """Equipment-passport workbook (WellsArtificialLiftBig) — one row per спуск."""
    return _resolve(
        env_var="PUMP2_EQUIPMENT_BIG_PATH",
        local_dir=LOCAL_INPUT_DIR,
        local_name="WellsArtificialLiftBig.xlsx",
        external_default=DEFAULT_EQUIPMENT_BIG_EXTERNAL,
    )


def resolve_v03_all_path() -> Path:
    return _resolve(
        env_var="PUMP2_V03_ALL_PATH",
        local_dir=LOCAL_INPUT_DIR,
        local_name="Отказы свод с анализом_БДА_V03_all.xlsx",
        external_default=DEFAULT_V03_ALL_EXTERNAL,
    )


def resolve_v03_failures_path() -> Path:
    return _resolve(
        env_var="PUMP2_V03_FAILURES_PATH",
        local_dir=LOCAL_INPUT_DIR,
        local_name="Отказы свод с анализом_БДА_V03_failures.xlsx",
        external_default=DEFAULT_V03_FAILURES_EXTERNAL,
    )


def resolve_presentation_path() -> Path:
    return _resolve(
        env_var="PUMP2_PRESENTATION_PATH",
        local_dir=LOCAL_INPUT_DIR,
        local_name="Отказность Аналитика(1).pptx",
        external_default=DEFAULT_PRESENTATION_EXTERNAL,
    )


# ── production / development plan (ПП) inputs ─────────────────────────────────

DEFAULT_PP_MASTER_EXTERNAL = Path(
    r"D:\Projects\Pumps\data\pp\ТМ-06_2026_2027_Р50_Базовый_Мастер файл.xlsx"
)
DEFAULT_GTM_SCHEDULE_EXTERNAL = Path(
    r"D:\Projects\Pumps\data\pp\М08 2025 (СД 2026 - 2027)\ДФ_04.xlsx"
)
DEFAULT_PREDICTION_TARGET_DIR = Path(r"D:\Projects\Pumps\data\target")
DEFAULT_TECHREGIME_WORKBOOK_EXTERNAL = Path(r"D:\Projects\Pumps\data\tr\ТР_НЕФТЬ 30.06.2026.xlsm")


def resolve_pp_master_path() -> Path:
    """Production-plan master file (Реестр: monthly per-well rates, MAP registry)."""
    return _resolve(
        env_var="PUMP2_PP_MASTER_PATH",
        local_dir=LOCAL_INPUT_DIR,
        local_name="ТМ-06_2026_2027_Р50_Базовый_Мастер файл.xlsx",
        external_default=DEFAULT_PP_MASTER_EXTERNAL,
    )


def resolve_gtm_schedule_path() -> Path:
    """ГТМ intervention schedule (ДФ_04: planned jobs, dates, ЭЦН flag, КРС durations)."""
    return _resolve(
        env_var="PUMP2_GTM_SCHEDULE_PATH",
        local_dir=LOCAL_INPUT_DIR,
        local_name="ДФ_04.xlsx",
        external_default=DEFAULT_GTM_SCHEDULE_EXTERNAL,
    )


def resolve_prediction_workbook_path() -> Path:
    """Production-risk source workbook with the ``Свод`` sheet.

    Preference order:
      1. explicit env var
      2. repo-local ``data/inputs/Отказы свод с анализом.xlsx``
      3. legacy local simple_prediction mirror
      4. newest non-backup simple_prediction workbook under the target folder
      5. V03_all workbook
    """
    env_value = os.environ.get("PUMP2_PREDICTION_WORKBOOK_PATH", "").strip()
    if env_value:
        return Path(env_value)
    local_source = LOCAL_INPUT_DIR / "Отказы свод с анализом.xlsx"
    if local_source.exists():
        return local_source
    local_path = LOCAL_INPUT_DIR / "Отказы свод с анализом_БДА_V03_simple_prediction.xlsm"
    if local_path.exists():
        return local_path
    if DEFAULT_PREDICTION_TARGET_DIR.exists():
        candidates = list(DEFAULT_PREDICTION_TARGET_DIR.glob("*simple_prediction*.xlsm"))
        if candidates:
            def _rank(p: Path) -> tuple[int, int, int, float]:
                name = p.name.lower()
                is_backup = 1 if ".backup" in name else 0
                is_30d = 1 if "30d" in name else 0
                not_90d = 1 if ("90d" not in name and "simple_prediction" in name) else 0
                return (is_backup, is_30d, not_90d, -p.stat().st_mtime)

            candidates.sort(key=_rank)
            return candidates[0]
    return resolve_v03_all_path()


def resolve_techregime_workbook_path() -> Path:
    """Current techregime workbook with fund status as of the report date."""
    return _resolve(
        env_var="PUMP2_TECHREGIME_WORKBOOK_PATH",
        local_dir=LOCAL_INPUT_DIR,
        local_name="ТР_НЕФТЬ 30.06.2026.xlsm",
        external_default=DEFAULT_TECHREGIME_WORKBOOK_EXTERNAL,
    )


# ── sqlite databases ──────────────────────────────────────────────────────────

DEFAULT_TELEMETRY_EXTERNAL = Path(r"D:\Projects\Pumps\data\telemetry\telemetry.sqlite")
DEFAULT_TECHREGIME_EXTERNAL = Path(r"D:\Projects\Pumps\data\techregime\techregime.sqlite")
DEFAULT_LAB_EXTERNAL = Path(r"D:\Projects\Pumps\data\lab\lab.sqlite")


def resolve_telemetry_db_path() -> Path:
    return _resolve(
        env_var="PUMP2_TELEMETRY_DB_PATH",
        local_dir=LOCAL_SQLITE_DIR,
        local_name="telemetry.sqlite",
        external_default=DEFAULT_TELEMETRY_EXTERNAL,
    )


def resolve_techregime_db_path() -> Path:
    return _resolve(
        env_var="PUMP2_TECHREGIME_DB_PATH",
        local_dir=LOCAL_SQLITE_DIR,
        local_name="techregime.sqlite",
        external_default=DEFAULT_TECHREGIME_EXTERNAL,
    )


def resolve_lab_db_path() -> Path:
    return _resolve(
        env_var="PUMP2_LAB_DB_PATH",
        local_dir=LOCAL_SQLITE_DIR,
        local_name="lab.sqlite",
        external_default=DEFAULT_LAB_EXTERNAL,
    )


__all__ = [
    # roots
    "REPO_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "MARTS_DIR",
    "WAREHOUSE_DIR",
    "LOCAL_INPUT_DIR",
    "LOCAL_SQLITE_DIR",
    "RESULTS_ROOT",
    # results
    "results_dir",
    # xlsx defaults & resolvers
    "DEFAULT_V03_ALL_EXTERNAL",
    "DEFAULT_V03_FAILURES_EXTERNAL",
    "DEFAULT_PRESENTATION_EXTERNAL",
    "resolve_v03_all_path",
    "resolve_v03_failures_path",
    "resolve_equipment_big_path",
    "resolve_presentation_path",
    "DEFAULT_PP_MASTER_EXTERNAL",
    "DEFAULT_GTM_SCHEDULE_EXTERNAL",
    "DEFAULT_TECHREGIME_WORKBOOK_EXTERNAL",
    "resolve_pp_master_path",
    "resolve_gtm_schedule_path",
    "resolve_prediction_workbook_path",
    "resolve_techregime_workbook_path",
    # sqlite defaults & resolvers
    "DEFAULT_TELEMETRY_EXTERNAL",
    "DEFAULT_TECHREGIME_EXTERNAL",
    "DEFAULT_LAB_EXTERNAL",
    "resolve_telemetry_db_path",
    "resolve_techregime_db_path",
    "resolve_lab_db_path",
]
