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
import re
import subprocess
import sys
from datetime import date as _date
from datetime import datetime as _datetime
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


# ── dated document versions ───────────────────────────────────────────────────
#
# Конвенция источников: у каждой версии документа в имени стоит дата актуализации,
# ``<имя> YYYYMMDD.<ext>`` — например ``WellsArtificialLiftBig 20260815.xlsx``.
# Резолверы обязаны брать САМУЮ СВЕЖУЮ такую версию, иначе датированный файл просто
# не находится и сборка молча идёт на позапрошлой выгрузке. Недатированное имя
# остаётся рабочим запасным вариантом.

#: ``<stem><разделитель>YYYYMMDD`` — разделителем служит пробел, ``_`` или ``-``.
_DATED_STEM_RE = re.compile(r"^(?P<stem>.+?)[ _-](?P<date>\d{8})$")


def parse_version_date(path: Path) -> _date | None:
    """Дата актуализации из имени файла, или ``None`` если её там нет."""
    match = _DATED_STEM_RE.match(Path(path).stem)
    if not match:
        return None
    try:
        return _datetime.strptime(match.group("date"), "%Y%m%d").date()
    except ValueError:
        # Восемь цифр, но не дата (номер, диапазон лет) — версией не считаем.
        return None


def _version_stem(path: Path) -> str:
    match = _DATED_STEM_RE.match(Path(path).stem)
    return match.group("stem") if match and parse_version_date(path) else Path(path).stem


def resolve_latest_version(directory: Path, filename: str) -> Path | None:
    """Самая свежая версия документа ``filename`` в папке ``directory``.

    Из ``WellsArtificialLiftBig.xlsx`` и ``WellsArtificialLiftBig 20260815.xlsx``
    выбирается вторая. При равных датах побеждает более позднее время файла;
    недатированный вариант проигрывает любому датированному, потому что про него
    неизвестно, к какому числу он актуален.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    wanted_stem = _version_stem(Path(filename))
    wanted_suffix = Path(filename).suffix.lower()

    candidates: list[tuple[int, _date | None, float, Path]] = []
    for child in directory.iterdir():
        if not child.is_file() or child.name.startswith("~$"):
            continue
        if wanted_suffix and child.suffix.lower() != wanted_suffix:
            continue
        if _version_stem(child) != wanted_stem:
            continue
        version = parse_version_date(child)
        try:
            mtime = child.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((1 if version else 0, version, mtime, child))

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1] or _date.min, item[2]))[3]


# ── shared resolver ───────────────────────────────────────────────────────────

#: Каноническое имя регистра ЭЦН. Версия дописывается датой АКТУАЛЬНОСТИ —
#: горизонтом наблюдения ПДК, а не днём сборки: регистр не может сказать ничего
#: правее горизонта, и две сборки в разные дни из одной выгрузки ПДК содержат одно
#: и то же. Дата сборки лежит на листе «Метаданные сборки».
SVOD_DOCUMENT_NAME = "Свод ЭЦН.xlsx"
#: Прежнее имя. Остаётся запасным вариантом, пока живут регистры старой сборки.
SVOD_LEGACY_NAME = "Отказы свод с анализом.xlsx"


def _resolve_any(*, env_var: str, local_dir: Path, local_names, external_default: Path) -> Path:
    """Как :func:`_resolve`, но пробует несколько имён документа по порядку.

    Нужно на время переименования: ``Свод ЭЦН`` ищется первым, прежнее
    ``Отказы свод с анализом`` — запасным. Без этого регистр под новым именем
    просто не находится, и расчёт молча продолжает читать позапрошлый файл.
    """
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return Path(env_value)
    for name in local_names:
        latest = resolve_latest_version(local_dir, name)
        if latest is not None:
            return latest
    for name in local_names:
        latest = resolve_latest_version(external_default.parent, name)
        if latest is not None:
            return latest
    return external_default


def _resolve(*, env_var: str, local_dir: Path, local_name: str, external_default: Path) -> Path:
    """Env var → newest local version → newest external version → the plain default.

    «Newest version» учитывает дату в имени файла (см. :func:`resolve_latest_version`),
    поэтому положенная рядом свежая выгрузка подхватывается сама, а не требует
    переименования в старое имя.
    """
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return Path(env_value)

    local_latest = resolve_latest_version(local_dir, local_name)
    if local_latest is not None:
        return local_latest

    external_latest = resolve_latest_version(external_default.parent, external_default.name)
    if external_latest is not None:
        return external_latest
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


def resolve_svod_main_path() -> Path:
    """Основной Свод — полный регистр, а не отфильтрованная выгрузка отказов.

    ``_БДА_V03_failures`` содержит ТОЛЬКО строки с заполненным отказавшим узлом (1573 из
    1573), поэтому по нему нельзя отличить «узел неизвестен» от «пуска нет в выгрузке».
    Основной регистр покрывает 1577 наших отказов против 1452 и закрывает 18 из тех, что
    иначе становятся категорией «не определено».
    """
    return _resolve_any(
        env_var="PUMP2_SVOD_MAIN_PATH",
        local_dir=LOCAL_INPUT_DIR,
        local_names=(SVOD_DOCUMENT_NAME, SVOD_LEGACY_NAME),
        external_default=DEFAULT_V03_ALL_EXTERNAL,
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
    for name in (SVOD_DOCUMENT_NAME, SVOD_LEGACY_NAME):
        local_source = resolve_latest_version(LOCAL_INPUT_DIR, name)
        if local_source is not None:
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


# ── raw source folders (inputs of the Свод builder) ───────────────────────────
#
# ``analysis.ingest`` rebuilds the Свод register from these folders. They are the
# operator's own export drops, so they live outside the repo; each is overridable
# by env var and falls back to a repo-local mirror before the external default.

EXTERNAL_SOURCE_ROOT = Path(r"D:\Projects\Pumps\data")

SOURCE_DIR_DEFAULTS = {
    "pdk": EXTERNAL_SOURCE_ROOT / "pdk",
    "artificial_lift": EXTERNAL_SOURCE_ROOT / "artificial_lift",
    "opz": EXTERNAL_SOURCE_ROOT / "opz-clean",
    "telemetry": EXTERNAL_SOURCE_ROOT / "telemetry",
    "techregime": EXTERNAL_SOURCE_ROOT / "techregime",
    "lab": EXTERNAL_SOURCE_ROOT / "lab",
    "frac": EXTERNAL_SOURCE_ROOT / "frac",
}

_SOURCE_ENV_VARS = {
    "pdk": "PUMP2_PDK_DIR",
    "artificial_lift": "PUMP2_ARTIFICIAL_LIFT_DIR",
    "opz": "PUMP2_OPZ_DIR",
    "telemetry": "PUMP2_TELEMETRY_DIR",
    "techregime": "PUMP2_TECHREGIME_DIR",
    "lab": "PUMP2_LAB_DIR",
    "frac": "PUMP2_FRAC_DIR",
}


def resolve_source_dir(kind: str) -> Path:
    """Folder holding the raw exports for one ingest source.

    ``kind`` is one of :data:`SOURCE_DIR_DEFAULTS`. Resolution order is env var →
    repo-local ``data/raw/<kind>/`` → the external default drop folder.
    """
    if kind not in SOURCE_DIR_DEFAULTS:
        raise KeyError(f"unknown ingest source: {kind!r} (known: {sorted(SOURCE_DIR_DEFAULTS)})")
    env_value = os.environ.get(_SOURCE_ENV_VARS[kind], "").strip()
    if env_value:
        return Path(env_value)
    local_path = RAW_DIR / kind
    if local_path.is_dir():
        return local_path
    return SOURCE_DIR_DEFAULTS[kind]


def resolve_telemetry_source_dir() -> Path:
    """Newest telemetry export drop.

    The telemetry folder accumulates one dated sub-folder per export
    (``20261506``, ``20261508`` …) beside the built ``telemetry.sqlite``. Each
    drop is a **full-history** dump, so the newest one supersedes its
    predecessors entirely and is the only one that should be read — merging them
    would just make the loader re-read millions of rows that dedup then discards.
    Falls back to the folder itself when it holds the exports directly.
    """
    root = resolve_source_dir("telemetry")
    if not root.is_dir():
        return root
    dated = [child for child in root.iterdir() if child.is_dir() and child.name.isdigit()]
    if not dated:
        return root
    # ⚠ Имена папок выгрузок непоследовательны: ``20261508`` это YYYY-DD-MM, а
    # файлы внутри — YYYY-MM-DD. Настоящая дата в имени папки надёжнее строкового
    # сравнения, поэтому она в приоритете; иначе — лексикографический максимум,
    # который для одной и той же (пусть и странной) конвенции работает верно.
    parsed = [(parse_version_date(child.name + ".x"), child) for child in dated]
    real_dates = [(version, child) for version, child in parsed if version]
    if real_dates:
        return max(real_dates, key=lambda item: item[0])[1]
    return max(dated, key=lambda path: path.name)


def resolve_frac_path() -> Path:
    """Newest ``Свод ГРП`` workbook (fracturing register).

    Follows the same dated-version convention as every other input:
    ``Свод ГРП 20260815.xlsx`` wins over an undated file of the same name.
    """
    root = resolve_source_dir("frac")
    if root.is_file():
        return root
    latest = resolve_latest_version(root, "Свод ГРП.xlsx")
    if latest is not None:
        return latest
    candidates = [
        path for path in root.glob("*.xlsx")
        if path.is_file() and not path.name.startswith("~$")
    ]
    if not candidates:
        return root
    return max(candidates, key=lambda path: (parse_version_date(path) or _date.min, path.stat().st_mtime))


# ── справочники ───────────────────────────────────────────────────────────────
#
# Небольшие таблицы соответствий, которые не приходят выгрузкой, а ведутся руками:
# плотность нефти по (месторождение, ЛУ), паспорт сепараторов. Живут в
# ``data/reference/``; внешняя копия — запасной вариант.

REFERENCE_DIR = DATA_DIR / "reference"

DEFAULT_SEPARATOR_CATALOG_EXTERNAL = Path(
    r"D:\Projects\Pumps\data\unsorted\Сепараторы mapped.xlsx"
)


def resolve_oil_density_path() -> Path:
    """Плотность нефти, т/м³, по ПАРЕ (месторождение, ЛУ).

    ⚠ Ключ — именно пара: у ``Bt``, ``Pg``, ``Ya`` и группы «Без месторождения»
    плотность различается по участкам. Нужна для точного перевода между газовыми
    осями: ГЖФ [м³/м³] = ГФ [м³/т] × ρ × (1 − обводнённость).
    """
    return _resolve(
        env_var="PUMP2_OIL_DENSITY_PATH",
        local_dir=REFERENCE_DIR,
        local_name="oil_density.csv",
        external_default=REFERENCE_DIR / "oil_density.csv",
    )


def resolve_separator_catalog_path() -> Path:
    """Паспорт газосепараторов (номинал по газу).

    ⚠ Был захардкожен в ``field_v53.SEP_XLSX`` на ``C:\\Users\\alexe\\Downloads`` —
    папку, которой на машине уже нет, из-за чего кадр модели не собирался вовсе.
    """
    return _resolve(
        env_var="PUMP2_SEPARATOR_CATALOG_PATH",
        local_dir=REFERENCE_DIR,
        local_name="Сепараторы mapped.xlsx",
        external_default=DEFAULT_SEPARATOR_CATALOG_EXTERNAL,
    )


# ── sqlite databases ──────────────────────────────────────────────────────────

DEFAULT_TELEMETRY_EXTERNAL = Path(r"D:\Projects\Pumps\data\telemetry\telemetry.sqlite")
DEFAULT_TECHREGIME_EXTERNAL = Path(r"D:\Projects\Pumps\data\techregime\techregime.sqlite")
DEFAULT_LAB_EXTERNAL = Path(r"D:\Projects\Pumps\data\lab\lab.sqlite")
DEFAULT_FRAC_EXTERNAL = Path(r"D:\Projects\Pumps\data\frac\frac.sqlite")


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


def resolve_frac_db_path() -> Path:
    return _resolve(
        env_var="PUMP2_FRAC_DB_PATH",
        local_dir=LOCAL_SQLITE_DIR,
        local_name="frac.sqlite",
        external_default=DEFAULT_FRAC_EXTERNAL,
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
    "resolve_svod_main_path",
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
    # dated document versions
    "parse_version_date",
    "resolve_latest_version",
    "SVOD_DOCUMENT_NAME",
    "SVOD_LEGACY_NAME",
    # raw ingest sources
    "EXTERNAL_SOURCE_ROOT",
    "SOURCE_DIR_DEFAULTS",
    "resolve_source_dir",
    "resolve_telemetry_source_dir",
    "resolve_frac_path",
    # справочники
    "REFERENCE_DIR",
    "resolve_oil_density_path",
    "resolve_separator_catalog_path",
    # sqlite defaults & resolvers
    "DEFAULT_TELEMETRY_EXTERNAL",
    "DEFAULT_TECHREGIME_EXTERNAL",
    "DEFAULT_LAB_EXTERNAL",
    "DEFAULT_FRAC_EXTERNAL",
    "resolve_telemetry_db_path",
    "resolve_techregime_db_path",
    "resolve_lab_db_path",
    "resolve_frac_db_path",
]
