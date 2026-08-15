from __future__ import annotations

import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis.sqlite_paths import resolve_lab_db_path, resolve_techregime_db_path, resolve_telemetry_db_path


TELEMETRY_DB_PATH = resolve_telemetry_db_path()
TECHREGIME_DB_PATH = resolve_techregime_db_path()
LAB_DB_PATH = resolve_lab_db_path()

#: Значения меток провенанса. ``absent`` — не «пусто по недосмотру», а «ни один
#: источник этого не дал»: без такого явного значения пропуск неотличим от того,
#: что колонку просто не запросили.
SOURCE_TELEMETRY = "telemetry"
SOURCE_TECHREGIME = "techregime"
SOURCE_ABSENT = "absent"
#: Суффикс колонки-провенанса: ``freq`` -> ``freq_src``.
SOURCE_SUFFIX = "_src"

#: ⚠⚠ Единица — В ИМЕНИ, а не в комментарии. Вся история с газом произошла потому,
#: что две разные величины назывались одинаково: ГФ [м³/т НЕФТИ] из техрежима и ГЖФ
#: [м³/м³ ЖИДКОСТИ] из телеметрии склеивались в одну колонку ``gas_factor``.
#: Обе оси теперь существуют ЯВНО и по отдельности; выбор базы — решение модели по
#: кросс-проверке, данные лишь обязаны дать возможность выбрать.
GAS_FACTOR_COLUMN = "gas_factor_m3t"          # газ на тонну нефти
GAS_LIQUID_RATIO_COLUMN = "gas_liquid_ratio_m3m3"   # газ на куб жидкости
OIL_DENSITY_COLUMN = "oil_density_t_m3"

CANONICAL_COLUMNS = [
    "freq",
    "load",
    "rpl",
    "rpump_intake",
    "rzab",
    "qliq",
    "watercut",
    # ⚠ Историческое имя. Оставлено синонимом ``gas_factor_m3t``, чтобы не рвать
    # потребителей, но под ним лежит ГФ на ТОННУ НЕФТИ и ничто иное.
    "gas_factor",
    # ⚠ ГЖФ — ОТДЕЛЬНАЯ величина, не синоним ГФ. Телеметрия отдаёт её измеренной;
    # техрежим не даёт вовсе, там она достраивается пересчётом через плотность.
    "gas_liquid_ratio_m3m3",
    "qgas",
    "kprod",
]

LAB_CHEMISTRY_COLUMNS = [
    "chloride_mg_l",
    "sulfate_mg_l",
    "calcium_mg_l",
    "bicarbonate_mg_l",
    "magnesium_mg_l",
    "sodium_potassium_mg_l",
    "total_mineralization_g_l",
    "ph",
]

#: ⚠⚠ Смысл колонок техрежима лежит НЕ в их именах. В ``techregime_records`` они
#: называются ``col_0001 … col_0093``, а что под ними — в отдельной таблице
#: ``techregime_column_map``. Захардкоженный ``col_0067`` МОЛЧА прочитает другую
#: величину, если следующая выгрузка сдвинет порядок столбцов: запрос не упадёт,
#: данные просто станут другими.
#:
#: Это ровно тот класс ошибки, который уже стоил проекту дорого: позиционный ``run``
#: вместо ключа ``(скважина, монтаж)`` портил 669 пусков из 2308, модель считалась и
#: сходилась, а починка дала слоям +99.6.
#:
#: Поэтому здесь объявлены ИМЕНА, а номера разрешаются по карте в
#: :func:`resolve_techregime_columns`. Нет имени в карте — падаем с внятной ошибкой,
#: а не подставляем NULL.
TECHREGIME_SOURCE_NAMES = {
    "freq": "Текущий режим работы скважины | Частота",
    "load": "Текущий режим работы скважины | Загр. Двиг.",
    "rpl": "Текущий режим работы скважины | Рпл.",
    "rpump_intake": "Текущий режим работы скважины | Рпр. насоса",
    "rzab": "Текущий режим работы скважины | Рзаб",
    "qliq": "Текущий режим работы скважины | Дебит жидк.",
    "watercut": "Текущий режим работы скважины | Обводненность",
    "gas_factor": "Текущий режим работы скважины | Газовый фактор",
    "qgas": "Текущий режим работы скважины | Дебит газа",
    "kprod": "Текущий режим работы скважины | Кпрод.",
}

#: Величины, которых у техрежима нет в принципе. Перечислены явно, чтобы «нет в
#: карте» осталось ошибкой, а не молчаливым NULL.
TECHREGIME_ABSENT_COLUMNS = ("gas_liquid_ratio_m3m3",)


@lru_cache(maxsize=1)
def resolve_techregime_columns() -> dict[str, str]:
    """``alias -> storage_name`` по ``techregime_column_map``, а не по номеру.

    ⚠ Падает, если ожидаемого ``original_name`` в карте нет: молчаливый NULL здесь
    хуже остановки — колонка просто окажется пустой, а слой на ней «не подтвердится».
    """
    with sqlite3.connect(TECHREGIME_DB_PATH) as connection:
        mapping = pd.read_sql_query(
            "SELECT original_name, storage_name FROM techregime_column_map", connection
        )
    by_name = dict(zip(mapping["original_name"].astype(str).str.strip(), mapping["storage_name"]))

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for alias, original_name in TECHREGIME_SOURCE_NAMES.items():
        storage = by_name.get(original_name)
        if storage is None:
            missing.append(f"{alias} ← {original_name!r}")
        else:
            resolved[alias] = str(storage)
    if missing:
        raise KeyError(
            "techregime_column_map: не найдены колонки " + "; ".join(missing)
            + ". Выгрузка сменила заголовки — сверьте имена, номера столбцов "
              "использовать нельзя."
        )
    for alias in TECHREGIME_ABSENT_COLUMNS:
        resolved[alias] = "NULL"
    return resolved

TELEMETRY_DAILY_COLUMN_CANDIDATES = {
    "freq": ("frequency_hz",),
    "load": ("motor_load_percent",),
    "rpl": (),
    "rpump_intake": ("P_intake_atm",),
    "rzab": ("P_bhp_atm",),
    "qliq": ("Qliq_m3d",),
    "watercut": ("watercut_percent",),
    # ⚠⚠ ГФ, а НЕ ГЖФ. Раньше здесь стоял ``GLF_m3m3`` — газ на куб ЖИДКОСТИ, — тогда
    # как техрежим отдаёт газ на тонну НЕФТИ. Одна колонка несла две величины,
    # расходящиеся в 2.5–3 раза, и обновление телеметрии молча подменило смысл
    # `gas_factor` на 50–60 % строк 2021–2024. ``gas_factor_m3t`` появляется в
    # хранилище после пересборки телеметрии; до неё колонка берётся из техрежима,
    # то есть определение остаётся ЕДИНЫМ в любом случае.
    "gas_factor": ("gas_factor_m3t",),
    "gas_liquid_ratio_m3m3": ("GLF_m3m3",),
    "qgas": ("Qgas_m3d",),
    "kprod": (),
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalize_well_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["well_id", "dt", *CANONICAL_COLUMNS, "source"])


def _build_date_clause(date_from: pd.Timestamp | None, date_to: pd.Timestamp | None, date_column: str) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if date_from is not None:
        clauses.append(f"{date_column} >= ?")
        params.append(pd.Timestamp(date_from).strftime("%Y-%m-%d"))
    if date_to is not None:
        clauses.append(f"{date_column} <= ?")
        params.append(pd.Timestamp(date_to).strftime("%Y-%m-%d"))
    if not clauses:
        return "", params
    return " AND " + " AND ".join(clauses), params


def _load_telemetry_daily(wells: list[str], date_from: pd.Timestamp | None = None, date_to: pd.Timestamp | None = None) -> pd.DataFrame:
    well_map = {normalize_well_key(well): str(well).strip() for well in wells if normalize_well_key(well)}
    normalized_wells = sorted(key for key in well_map if key)
    if not normalized_wells:
        return _empty_daily_frame()

    with sqlite3.connect(TELEMETRY_DB_PATH) as connection:
        schema = pd.read_sql_query("PRAGMA table_info(telemetry_daily)", connection)
        available = set(schema["name"].astype(str))
        select_parts = ["_meta_normalized_well as well_key", "_meta_record_date as dt"]
        for alias in CANONICAL_COLUMNS:
            source_column = next((candidate for candidate in TELEMETRY_DAILY_COLUMN_CANDIDATES[alias] if candidate in available), None)
            if source_column is None:
                select_parts.append(f"NULL as {alias}")
            else:
                select_parts.append(f"{source_column} as {alias}")
        query_select = ", ".join(select_parts)

        frames: list[pd.DataFrame] = []
        date_clause, date_params = _build_date_clause(date_from, date_to, "_meta_record_date")
        for start in range(0, len(normalized_wells), 400):
            chunk = normalized_wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT {query_select}
                FROM telemetry_daily
                WHERE _meta_normalized_well IN ({placeholders}){date_clause}
            """
            frames.append(pd.read_sql_query(query, connection, params=[*chunk, *date_params]))

    df = pd.concat(frames, ignore_index=True) if frames else _empty_daily_frame()
    if df.empty:
        return _empty_daily_frame()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    for column in CANONICAL_COLUMNS:
        df[column] = _numeric(df[column])
    df = df.loc[df["well_key"].notna() & df["dt"].notna()].copy()
    df["well_id"] = df["well_key"].map(well_map).fillna(df["well_key"])
    df["source"] = "telemetry"
    result = (
        df.groupby(["well_id", "dt", "source"], as_index=False)[CANONICAL_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["well_id", "dt"])
        .reset_index(drop=True)
    )
    return result


def _load_techregime_daily(wells: list[str], date_from: pd.Timestamp | None = None, date_to: pd.Timestamp | None = None) -> pd.DataFrame:
    """Daily techregime rows for the given wells, aggregated to one row per (well, day).

    ⚠⚠ Фильтрация идёт по **нормализованному** ключу ``_meta_normalized_well_id`` и по
    **ISO-дате** ``_meta_report_date``, а не по сырым ``col_0003`` / ``col_0010``.
    Причины разные, но обе кусаются:

    * сырой ключ (``Ya_403``) совпадёт с фондом только пока обе стороны пишут скважину
      одинаково. Телеметрия хранит ``ya_403``, и стоит списку скважин прийти оттуда —
      пересечение станет ПУСТЫМ, а не частичным, то есть техрежим молча исчезнет
      целиком (на сегодняшнем фонде совпадают 931 скважина из 995 при обоих ключах,
      так что потерь пока нет — это защита, а не починка);
    * сырая дата ``01.04.2018`` сравнивается с границей ``2020-01-01`` КАК СТРОКА:
      ``'01.04.2018' >= '2020-01-01'`` ложно, потому что ``'0' < '2'``. Любой вызов с
      ``date_from`` отсекал почти всё. Витрина зовёт загрузчик без дат, поэтому дефект
      был латентным.
    """
    normalized_wells = sorted({normalize_well_key(well) for well in wells if normalize_well_key(well)})
    if not normalized_wells:
        return _empty_daily_frame()

    columns = resolve_techregime_columns()
    query_select = ", ".join(
        f"{columns[alias]} as {alias}" for alias in CANONICAL_COLUMNS if alias in columns
    )
    date_clause, date_params = _build_date_clause(date_from, date_to, "_meta_report_date")
    frames: list[pd.DataFrame] = []
    with sqlite3.connect(TECHREGIME_DB_PATH) as connection:
        for start in range(0, len(normalized_wells), 400):
            chunk = normalized_wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT _meta_normalized_well_id as well_id, _meta_report_date as dt, {query_select}
                FROM techregime_records
                WHERE _meta_normalized_well_id IN ({placeholders}){date_clause}
            """
            frames.append(pd.read_sql_query(query, connection, params=[*chunk, *date_params]))
    df = pd.concat(frames, ignore_index=True) if frames else _empty_daily_frame()
    if df.empty:
        return _empty_daily_frame()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["well_id"] = df["well_id"].astype("string").str.strip()
    for column in CANONICAL_COLUMNS:
        df[column] = _numeric(df[column])
    df = df.loc[df["well_id"].notna() & df["dt"].notna()].copy()
    df["source"] = "techregime"
    result = (
        df.groupby(["well_id", "dt", "source"], as_index=False)[CANONICAL_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["well_id", "dt"])
        .reset_index(drop=True)
    )
    return result


def _empty_lab_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["well_id", "sample_date", *LAB_CHEMISTRY_COLUMNS, "source_file"])


def _load_lab_samples(wells: list[str], date_from: pd.Timestamp | None = None, date_to: pd.Timestamp | None = None) -> pd.DataFrame:
    if not LAB_DB_PATH.exists():
        return _empty_lab_frame()
    well_map = {normalize_well_key(well): str(well).strip() for well in wells if normalize_well_key(well)}
    normalized_wells = sorted(key for key in well_map if key)
    if not normalized_wells:
        return _empty_lab_frame()

    with sqlite3.connect(LAB_DB_PATH) as connection:
        schema = pd.read_sql_query("PRAGMA table_info(lab_samples)", connection)
        if schema.empty:
            return _empty_lab_frame()
        available = set(schema["name"].astype(str))
        select_parts = ["well_key", "sample_date", "source_file"]
        for column in LAB_CHEMISTRY_COLUMNS:
            if column in available:
                select_parts.append(column)
            else:
                select_parts.append(f"NULL as {column}")
        query_select = ", ".join(select_parts)
        date_clause, date_params = _build_date_clause(date_from, date_to, "sample_date")
        frames: list[pd.DataFrame] = []
        for start in range(0, len(normalized_wells), 400):
            chunk = normalized_wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT {query_select}
                FROM lab_samples
                WHERE well_key IN ({placeholders}){date_clause}
            """
            frames.append(pd.read_sql_query(query, connection, params=[*chunk, *date_params]))
    df = pd.concat(frames, ignore_index=True) if frames else _empty_lab_frame()
    if df.empty:
        return _empty_lab_frame()
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    for column in LAB_CHEMISTRY_COLUMNS:
        df[column] = _numeric(df[column])
    df = df.loc[df["well_key"].notna() & df["sample_date"].notna()].copy()
    df["well_id"] = df["well_key"].map(well_map).fillna(df["well_key"])
    return (
        df.groupby(["well_id", "sample_date", "source_file"], as_index=False)[LAB_CHEMISTRY_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["well_id", "sample_date", "source_file"])
        .reset_index(drop=True)
    )


def attach_lab_chemistry(
    daily_df: pd.DataFrame,
    *,
    fill_mode: str = "step",
    extend_backward: bool = True,
) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df.copy()
    if fill_mode != "step":
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")

    wells = sorted(daily_df["well_id"].dropna().astype(str).unique().tolist())
    lab = _load_lab_samples(wells)
    result = daily_df.copy().sort_values(["well_id", "dt"]).reset_index(drop=True)

    if lab.empty:
        for column in LAB_CHEMISTRY_COLUMNS:
            result[column] = np.nan
        result["lab_sample_date"] = pd.NaT
        result["lab_source_file"] = pd.Series(pd.NA, index=result.index, dtype="string")
        return result

    frames: list[pd.DataFrame] = []
    for well_id, group in result.groupby("well_id", sort=False):
        group = group.sort_values("dt").reset_index(drop=True)
        chemistry = lab.loc[lab["well_id"].astype(str) == str(well_id)].copy()
        if chemistry.empty:
            frames.append(group)
            continue
        chemistry = chemistry.sort_values("sample_date").reset_index(drop=True)

        backward = pd.merge_asof(
            group,
            chemistry[["sample_date", "source_file", *LAB_CHEMISTRY_COLUMNS]],
            left_on="dt",
            right_on="sample_date",
            direction="backward",
        )
        if extend_backward:
            forward = pd.merge_asof(
                group,
                chemistry[["sample_date", "source_file", *LAB_CHEMISTRY_COLUMNS]],
                left_on="dt",
                right_on="sample_date",
                direction="forward",
            )
            for column in LAB_CHEMISTRY_COLUMNS:
                backward[column] = backward[column].combine_first(forward[column])
            backward["sample_date"] = backward["sample_date"].combine_first(forward["sample_date"])
            backward["source_file"] = backward["source_file"].combine_first(forward["source_file"])

        backward = backward.rename(columns={"sample_date": "lab_sample_date", "source_file": "lab_source_file"})
        frames.append(backward)

    filled = pd.concat(frames, ignore_index=True).sort_values(["well_id", "dt"]).reset_index(drop=True)
    if "lab_source_file" in filled.columns:
        filled["lab_source_file"] = filled["lab_source_file"].astype("string")
    return filled


def add_dynamic_salt_proxies(
    daily_df: pd.DataFrame,
    *,
    qliq_column: str = "qliq",
    fill_mode: str = "step",
    extend_backward: bool = True,
) -> pd.DataFrame:
    working = attach_lab_chemistry(daily_df, fill_mode=fill_mode, extend_backward=extend_backward)
    qliq = _numeric(working.get(qliq_column, pd.Series(np.nan, index=working.index, dtype=float))).clip(lower=0)
    calcium = _numeric(working.get("calcium_mg_l", pd.Series(np.nan, index=working.index, dtype=float)))
    chloride = _numeric(working.get("chloride_mg_l", pd.Series(np.nan, index=working.index, dtype=float)))
    sulfate = _numeric(working.get("sulfate_mg_l", pd.Series(np.nan, index=working.index, dtype=float)))

    # mg/L * m3/day / 1000 -> kg/day
    working["daily_calcium_load_kg"] = (calcium * qliq) / 1000.0
    working["daily_chloride_load_kg"] = (chloride * qliq) / 1000.0
    working["daily_sulfate_load_kg"] = (sulfate * qliq) / 1000.0
    working["daily_salt_load_kg"] = ((calcium + chloride + sulfate) * qliq) / 1000.0
    # Exposure proxy, not direct precipitated mass.
    working["daily_gypsum_scale_proxy"] = (calcium * sulfate * qliq) / 1_000_000.0

    grouped = working.groupby("well_id", sort=False)
    for daily_col, cum_col in [
        ("daily_calcium_load_kg", "cum_calcium_load_kg_dynamic"),
        ("daily_chloride_load_kg", "cum_chloride_load_kg_dynamic"),
        ("daily_sulfate_load_kg", "cum_sulfate_load_kg_dynamic"),
        ("daily_salt_load_kg", "cum_salt_load_kg_dynamic"),
        ("daily_gypsum_scale_proxy", "cum_gypsum_scale_proxy_dynamic"),
    ]:
        working[cum_col] = grouped[daily_col].cumsum()
    return working


#: ``col_0001`` (Месторождение) → код месторождения в справочнике плотностей.
#:
#: ⚠⚠ Справочник ведётся КОДАМИ (``Bt_Vt``), а телеметрия — полными названиями
#: («Большетирское НМ» / «Верхнетирский участок»). Без этого перевода каскад
#: ``DensityLookup.get`` вытягивал только те строки, где префикс ключа скважины
#: СЛУЧАЙНО совпадал с кодом месторождения (``ya``→``Ya``, ``da``→``Da``): плотность
#: находилась у 609 скважин из 1158, и дыра приходилась ровно на Vt (276 скважин),
#: Az (96), Au (98) и Ki (16). В витрине это давало ρ на 66.5 % строк и ГЖФ на 76.0 %
#: вместо 93.9 % и 82.1 %, причём у Большетирского плотность стояла на 8 % строк,
#: у Западно-Аянского — на 0 %.
#:
#: ⚠⚠ Шесть пар, выглядевших «отсутствующими в справочнике» (Ya/Az, Ic/Vt, Bt/Vt,
#: Za/Au, Ya/Au, Ya/Ki — 206 401 строка), в нём ЕСТЬ. Запрашивать их у заказчика не
#: нужно; запрашивать нужно Tk/Zy, где справочные 0.816 расходятся с наблюдёнными
#: 0.864 на 5.9 % — это расхождение величины, а не ключа.
FIELD_CODE = {
    "Ярактинское НГКМ": "Ya",
    "Большетирское НМ": "Bt",
    "Ичёдинское НМ": "Ic",
    "Маччобинское НГКМ": "Mc",
    "Западно-Аянское НГКМ": "Za",
    "Даниловское НГКМ": "Da",
    "Марковское НГКМ": "Ma",
    "Месторождение им. Н.В.Мышевского": "Msh",
    "Токминское НГКМ": "Tk",
    "Нелбинское НГКМ": "Nlt",
    "Бариктинское НГКМ": "Br",
    "Бурское НМ": "Bur",
    "Без месторождения": "Без месторождения",
}

#: ``col_0002`` (ЛУ) → код лицензионного участка в справочнике плотностей.
LU_CODE = {
    "Ярактинский участок": "Ya",
    "Верхнетирский участок": "Vt",
    "Западно-Ярактинский участок": "Zy",
    "Мирнинский участок": "Mr",
    "Аянский (Западный) участок": "Az",
    "Аянский участок": "Au",
    "Большетирский участок": "Bt",
    "Даниловский участок": "Da",
    "Марковский участок": "Ma",
    "Кийский участок": "Ki",
}


@lru_cache(maxsize=1)
def load_well_licence_map() -> pd.DataFrame:
    """``well_key -> (месторождение, ЛУ)`` из сырого слоя телеметрии, кодами справочника.

    Отдаёт и коды (``field`` / ``lu`` — ключ в справочник плотностей), и исходные
    полные названия (``field_name`` / ``lu_name``), чтобы читаемое имя не терялось.

    ⚠⚠ Принадлежность участку берётся из КОЛОНКИ ``col_0002``, а не из имени файла
    выгрузки. Имя файла — это то, как выгрузку назвали, а не сами данные: смена
    соглашения об именовании даёт ``NaN`` в ``lu`` → ``NaN`` в плотности → тихую
    потерю газовой оси, без единого исключения по дороге. Тот же класс ошибки, что
    позиционный ``run`` вместо ключа ``(скважина, монтаж)``. Имя файла оставлено
    ВТОРЫМ МНЕНИЕМ: расхождение печатается в лог, но ключом не служит.

    ⚠⚠ Соответствие НЕ выводится из отношения ГФ/ГЖФ. Отношение служит контролем
    стыковки (``analysis.data.oil_density.validate_against_observed``), и выводить из
    него же ключ означало бы проверять величину ею самой.

    ⚠ Ключ — ПАРА, не месторождение: у ``Bt`` плотность различается по участкам
    (``Bt_Vt`` 0.811 против ``Bt_Bt`` 0.821), у ``Ya`` — тоже (0.829 / 0.833).

    Незнакомое название падает с ошибкой: молчаливый ``NaN`` здесь хуже остановки —
    колонка просто окажется пустой, а слой на ней «не подтвердится».
    """
    import sqlite3

    with sqlite3.connect(TELEMETRY_DB_PATH) as connection:
        mapping = pd.read_sql_query(
            "SELECT original_name, storage_name FROM telemetry_column_map", connection
        )
        columns = dict(zip(mapping["original_name"], mapping["storage_name"]))
        needed = {"Месторождение", "ЛУ", "Скважина"}
        missing = needed - set(columns)
        if missing:
            raise KeyError(f"telemetry_column_map: нет колонок {sorted(missing)}")
        raw = pd.read_sql_query(
            f'SELECT DISTINCT _meta_normalized_well well_key, {columns["Месторождение"]} field_name, '
            f'{columns["ЛУ"]} lu_name, _meta_source_file source_file FROM telemetry_raw',
            connection,
        )

    raw = raw.dropna(subset=["well_key"]).drop_duplicates(subset=["well_key"], keep="first")
    for column in ("field_name", "lu_name"):
        raw[column] = raw[column].astype("string").str.strip()

    unknown_fields = sorted(set(raw["field_name"].dropna()) - set(FIELD_CODE))
    unknown_lus = sorted(set(raw["lu_name"].dropna()) - set(LU_CODE))
    if unknown_fields or unknown_lus:
        raise KeyError(
            "load_well_licence_map: незнакомые названия в телеметрии — "
            f"месторождения {unknown_fields}, участки {unknown_lus}. "
            "Добавьте их в FIELD_CODE/LU_CODE вместе с кодом справочника плотностей; "
            "оставлять их без кода нельзя — плотность молча станет пустой."
        )

    raw["field"] = raw["field_name"].map(FIELD_CODE)
    raw["lu"] = raw["lu_name"].map(LU_CODE)

    # ⚠ Имя файла — второе мнение, не источник. Телеметрия выгружается по одному файлу
    # на участок, поэтому расхождение с ``col_0002`` означает, что выгрузку собрали
    # иначе, чем раньше, и это стоит увидеть — но ключ всё равно берётся из колонки.
    from_file = raw["source_file"].astype("string").str.extract(
        r"телеметрии\s+([A-Za-zА-Яа-яЁё]+)\s", expand=False
    )
    disagreement = raw["lu"].notna() & from_file.notna() & (raw["lu"] != from_file)
    if disagreement.any():
        pairs = sorted(set(zip(raw.loc[disagreement, "lu"], from_file[disagreement])))
        print(
            f"[load_well_licence_map] ⚠ имя файла расходится с колонкой ЛУ у "
            f"{int(disagreement.sum())} скважин: {pairs[:6]} — ключом взята колонка"
        )

    return raw[["well_key", "field", "field_name", "lu", "lu_name"]].reset_index(drop=True)


#: Приоритет источников при слиянии. ⚠⚠ «Телеметрия первая» принято ПО УМОЛЧАНИЮ,
#: а не по проверке: на пересечении 2022–23 источники расходятся (дебит r = 0.883,
#: совпадает 67 % строк; обводнённость r = 0.604, 68 %), и какой ближе к истине —
#: неизвестно. Переключатель нужен, чтобы обе витрины можно было сравнить НА МОДЕЛИ,
#: а не спорить о них умозрительно.
SOURCE_PRIORITIES = ("telemetry", "techregime")


def load_daily_merged(
    wells: list[str],
    date_from: pd.Timestamp | None = None,
    date_to: pd.Timestamp | None = None,
    prefer: str = "telemetry",
) -> pd.DataFrame:
    if prefer not in SOURCE_PRIORITIES:
        raise ValueError(f"prefer must be one of {SOURCE_PRIORITIES}, got {prefer!r}")
    telemetry = _load_telemetry_daily(wells, date_from=date_from, date_to=date_to)
    techregime = _load_techregime_daily(wells, date_from=date_from, date_to=date_to)

    telemetry = telemetry.copy()
    techregime = techregime.copy()
    telemetry["well_key"] = telemetry["well_id"].map(normalize_well_key)
    techregime["well_key"] = techregime["well_id"].map(normalize_well_key)

    # ⚠⚠ Контроль стыковки печатается СРАЗУ, а не восстанавливается потом по витрине.
    # Ноль совпавших строк означает не «нет общих данных», а сломанный ключ или формат
    # даты — и различить это постфактум невозможно.
    control = daily_merge_control(telemetry, techregime, None)
    if control["telemetry_rows"] and control["techregime_rows"]:
        share = control["matched_share"]
        marker = "⚠⚠ " if share == 0 else ("⚠ " if share < 0.5 else "")
        print(
            f"  {marker}стыковка телеметрия↔техрежим: {control['matched_rows']} из "
            f"{control['telemetry_rows']} строк телеметрии нашли пару ({share:.1%})"
        )

    merged = telemetry.merge(
        techregime,
        on=["well_key", "dt"],
        how="outer",
        suffixes=("_tel", "_tr"),
    )

    well_id_tel = merged.get("well_id_tel", pd.Series(index=merged.index, dtype="string")).astype("string")
    well_id_tr = merged.get("well_id_tr", pd.Series(index=merged.index, dtype="string")).astype("string")
    result = pd.DataFrame(
        {
            "well_id": well_id_tel.fillna(well_id_tr),
            "dt": merged["dt"],
        }
    )
    # ── провенанс НА КАЖДУЮ КОЛОНКУ ───────────────────────────────────────────
    # ⚠⚠ Одной метки на строку недостаточно, и это не придирка. Строка помечается
    # «telemetry», если ХОТЬ ОДНА колонка пришла из телеметрии — обычно это дебит.
    # Частота при этом сплошь техрежимная: в telemetry_daily её до 2025 года ровно
    # 0 %. Из-за построчной метки в отчёте выходило «99.5 % частоты в
    # telemetry-строках» при нулевой частоте в самой телеметрии, и вопрос «откуда
    # взято ЭТО значение частоты» не имел ответа вообще.
    any_tel = pd.Series(False, index=merged.index)
    for column in CANONICAL_COLUMNS:
        tel_series = _numeric(merged.get(f"{column}_tel", pd.Series(np.nan, index=merged.index, dtype=float)))
        tr_series = _numeric(merged.get(f"{column}_tr", pd.Series(np.nan, index=merged.index, dtype=float)))
        provenance = pd.Series(SOURCE_ABSENT, index=merged.index, dtype="string")
        if prefer == SOURCE_TECHREGIME:
            result[column] = tr_series.combine_first(tel_series)
            provenance.loc[tel_series.notna()] = SOURCE_TELEMETRY
            provenance.loc[tr_series.notna()] = SOURCE_TECHREGIME
        else:
            result[column] = tel_series.combine_first(tr_series)
            provenance.loc[tr_series.notna()] = SOURCE_TECHREGIME
            provenance.loc[tel_series.notna()] = SOURCE_TELEMETRY
        result[f"{column}{SOURCE_SUFFIX}"] = provenance
        any_tel = any_tel | tel_series.notna()

    # ⚠ Построчная метка СОХРАНЕНА, но названа честно: `row_source` значит «в этой
    # строке хоть что-то от телеметрии», а НЕ «эти значения из телеметрии».
    # `source` оставлен синонимом, чтобы не рвать существующих потребителей.
    row_source = pd.Series(SOURCE_TECHREGIME, index=merged.index, dtype="string")
    row_source.loc[any_tel] = SOURCE_TELEMETRY
    result["row_source"] = row_source
    result["source"] = row_source

    result = result.loc[result["well_id"].notna() & result["dt"].notna()].copy()
    provenance_columns = [f"{column}{SOURCE_SUFFIX}" for column in CANONICAL_COLUMNS]
    aggregation = {column: "mean" for column in CANONICAL_COLUMNS}
    aggregation.update({column: "first" for column in [*provenance_columns, "row_source", "source"]})
    result = (
        result.groupby(["well_id", "dt"], as_index=False)
        .agg(aggregation)
        .sort_values(["well_id", "dt"])
        .reset_index(drop=True)
    )
    return _attach_gas_axes(result)


def _attach_gas_axes(daily: pd.DataFrame) -> pd.DataFrame:
    """Обе газовые оси явными колонками плюс плотность и её провенанс.

    ⚠⚠ Решение о базе здесь НЕ принимается — его примет модель по кросс-проверке.
    От данных требуется только возможность выбора, поэтому обе оси лежат рядом:

    ``gas_factor_m3t``          газ на ТОННУ НЕФТИ (синоним историческому ``gas_factor``);
    ``gas_liquid_ratio_m3m3``   газ на КУБ ЖИДКОСТИ;
    ``oil_density_t_m3``        плотность из СПРАВОЧНИКА;
    ``gas_liquid_ratio_m3m3_src``  ``measured`` / ``converted`` / ``absent``.

    ⚠ ГЖФ берётся ИЗМЕРЕННЫМ там, где источник его даёт (телеметрия отдаёт
    «Газожидкостный фактор, м3/м3» напрямую), и пересчитывается из ГФ только там, где
    измерения нет. Пересчёт без плотности не делается: пропуск остаётся пропуском, а
    не заполняется соседним участком — неверная плотность молча масштабирует всю ось
    на несколько процентов, и отличить это потом от физики нельзя.
    """
    from analysis.data.oil_density import attach_density, gas_liquid_ratio_from_gor

    result = daily.copy()
    result[GAS_FACTOR_COLUMN] = result["gas_factor"]
    result[f"{GAS_FACTOR_COLUMN}{SOURCE_SUFFIX}"] = result.get(f"gas_factor{SOURCE_SUFFIX}")

    licences = load_well_licence_map()
    result["well_key"] = result["well_id"].map(normalize_well_key)
    # ⚠ ``field``/``lu`` — КОДЫ справочника, читаемые названия едут рядом отдельно.
    result = result.merge(
        licences[["well_key", "field", "field_name", "lu", "lu_name"]], on="well_key", how="left"
    )
    result = attach_density(result, field_col="field", lu_col="lu", well_col="well_id",
                            out_col=OIL_DENSITY_COLUMN)

    measured = _numeric(result.get(GAS_LIQUID_RATIO_COLUMN, pd.Series(np.nan, index=result.index)))
    converted = gas_liquid_ratio_from_gor(
        result[GAS_FACTOR_COLUMN], result[OIL_DENSITY_COLUMN], result["watercut"]
    )
    result[GAS_LIQUID_RATIO_COLUMN] = measured.combine_first(converted)
    provenance = pd.Series(SOURCE_ABSENT, index=result.index, dtype="string")
    provenance.loc[converted.notna()] = "converted"
    provenance.loc[measured.notna()] = "measured"
    result[f"{GAS_LIQUID_RATIO_COLUMN}{SOURCE_SUFFIX}"] = provenance
    return result


def column_coverage_by_year(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    date_col: str = "dt",
) -> pd.DataFrame:
    """Заполненность колонок по годам, в процентах строк.

    ⚠⚠ Смотреть ДО того, как ставить фильтр по нескольким колонкам разреженного
    источника. Условие «И» на колонке с нулевым покрытием обнуляет выборку целиком и
    выглядит содержательным результатом: ``WHERE frequency_hz > 0 AND Qliq_m3d > 0``
    по телеметрии даёт НОЛЬ строк во все годы до 2025 — не потому, что фонд стоял, а
    потому что частоты там нет вовсе, а дебит есть. На этом однажды был построен и
    попал в отчёт вывод «слои опираются на 2021+».
    """
    if frame.empty or date_col not in frame.columns:
        return pd.DataFrame()
    columns = list(columns) if columns is not None else [
        c for c in frame.columns if c not in (date_col,) and not c.endswith(SOURCE_SUFFIX)
    ]
    working = frame.copy()
    working["_год"] = pd.to_datetime(working[date_col], errors="coerce").dt.year
    grouped = working.groupby("_год")
    report = pd.DataFrame({"строк": grouped.size()})
    for column in columns:
        if column in working.columns:
            report[column] = (grouped[column].count() / grouped.size() * 100).round(1)
    return report.reset_index()


def daily_merge_control(telemetry: pd.DataFrame, techregime: pd.DataFrame, merged: pd.DataFrame) -> dict:
    """Контроль стыковки: сколько строк телеметрии нашли пару в техрежиме.

    ⚠⚠ Ожидание — заметная доля, а НОЛЬ означает не «нет общих данных», а сломанный
    ключ или формат даты. Отличить одно от другого постфактум невозможно, поэтому
    число выводится в лог сборки, а не восстанавливается потом по витрине.
    """
    if telemetry.empty or techregime.empty:
        return {"telemetry_rows": len(telemetry), "techregime_rows": len(techregime), "matched_rows": 0, "matched_share": 0.0}
    tel_keys = set(zip(telemetry["well_id"].map(normalize_well_key), telemetry["dt"]))
    tr_keys = set(zip(techregime["well_id"].map(normalize_well_key), techregime["dt"]))
    matched = len(tel_keys & tr_keys)
    return {
        "telemetry_rows": len(telemetry),
        "techregime_rows": len(techregime),
        "matched_rows": matched,
        "matched_share": matched / max(len(tel_keys), 1),
    }


def column_provenance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Откуда взято значение каждой канонной колонки — в долях строк.

    Отвечает ровно на тот вопрос, который до сих пор не имел ответа: «частота в
    этом окне — измерена телеметрией или взята из уставки техрежима?»
    """
    rows = []
    for column in CANONICAL_COLUMNS:
        source_column = f"{column}{SOURCE_SUFFIX}"
        if source_column not in df.columns:
            continue
        counts = df[source_column].value_counts(dropna=False)
        total = int(counts.sum()) or 1
        rows.append(
            {
                "колонка": column,
                "телеметрия": int(counts.get(SOURCE_TELEMETRY, 0)),
                "техрежим": int(counts.get(SOURCE_TECHREGIME, 0)),
                "нет": int(counts.get(SOURCE_ABSENT, 0)),
                "доля_телеметрии": round(int(counts.get(SOURCE_TELEMETRY, 0)) / total, 4),
            }
        )
    return pd.DataFrame(rows)


def source_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    # ⚠ Это ПОСТРОЧНАЯ картина, а не «откуда взяты значения». Поколоночный ответ
    # даёт `column_provenance_report`.
    label = "row_source" if "row_source" in df.columns else "source"
    if df.empty or label not in df.columns:
        return pd.DataFrame(columns=["well_id", "telemetry_rows", "techregime_rows", "total_rows", "telemetry_fraction"])
    working = df.copy()
    rows: list[dict[str, object]] = []
    for well_id, frame in working.groupby("well_id", dropna=False):
        telemetry_rows = int(frame[label].eq(SOURCE_TELEMETRY).sum())
        techregime_rows = int(frame[label].eq(SOURCE_TECHREGIME).sum())
        total_rows = int(len(frame))
        rows.append(
            {
                "well_id": str(well_id),
                "telemetry_rows": telemetry_rows,
                "techregime_rows": techregime_rows,
                "total_rows": total_rows,
                "telemetry_fraction": float(telemetry_rows / max(total_rows, 1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["telemetry_fraction", "well_id"], ascending=[False, True]).reset_index(drop=True)


def split_by_run_id(
    df: pd.DataFrame,
    run_id_column: str = "row_id",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if run_id_column not in df.columns:
        raise ValueError(f"Run id column '{run_id_column}' not found.")
    from analysis.modeling_config import stable_hash_test_mask

    test_mask = stable_hash_test_mask(df[run_id_column], test_fraction=test_fraction)
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()
    return train_df, test_df


__all__ = [
    "CANONICAL_COLUMNS",
    "LAB_CHEMISTRY_COLUMNS",
    "LAB_DB_PATH",
    "TECHREGIME_DB_PATH",
    "TELEMETRY_DB_PATH",
    "_numeric",
    "add_dynamic_salt_proxies",
    "attach_lab_chemistry",
    "load_daily_merged",
    "normalize_well_key",
    "source_coverage_report",
    "split_by_run_id",
]
