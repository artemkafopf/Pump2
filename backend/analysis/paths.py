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
from datetime import date as _date
from pathlib import Path


# ── repo roots ────────────────────────────────────────────────────────────────

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

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
    # sqlite defaults & resolvers
    "DEFAULT_TELEMETRY_EXTERNAL",
    "DEFAULT_TECHREGIME_EXTERNAL",
    "DEFAULT_LAB_EXTERNAL",
    "resolve_telemetry_db_path",
    "resolve_techregime_db_path",
    "resolve_lab_db_path",
]
