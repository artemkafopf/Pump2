"""Well artificial lift dataset processing."""

from typing import Optional

import pandas as pd

from collections import Counter

from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta

from ..config import BIG_COLUMN_RULES, SVOD_COMMENT_COLUMN
from ..io import discover_source_files
from ..normalize import normalize_well, parse_date, parse_number
from .enrich import (
    build_big_columns_from_header_rows,
    derive_acid_type,
    derive_field_code_from_well,
    derive_runtime_group,
    get_first_matching_column,
    get_first_matching_value,
    is_esp_oil_run,
    is_mech_production_purpose,
    normalize_big_header,
)
from .well_identity import is_rassol_bore, non_oil_reason, well_number_key


def load_big_dataframe(file_path: str, sheet_name: str = "Скважинное оборудование") -> Optional[pd.DataFrame]:
    """Load the artificial lift workbook with a normalized two-row section header."""
    try:
        raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception:
        return None
    if raw.shape[0] < 3:
        return None
    header_frame = raw.iloc[[0, 1]].copy()
    columns = build_big_columns_from_header_rows(header_frame)
    if not columns:
        return None
    df = raw.iloc[2:].reset_index(drop=True)
    if df.empty:
        return None
    df.columns = pd.MultiIndex.from_tuples(columns, names=["section", "field"])
    return df.sort_index(axis=1)


def load_big_sources(artificial_lift_source, manual_source: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load and concatenate one or more artificial lift workbooks."""
    frames = []
    for source in [artificial_lift_source, manual_source]:
        if not source:
            continue
        for file_path in discover_source_files(source, "big"):
            try:
                df = load_big_dataframe(str(file_path))
            except Exception as exc:
                print(f"  Skipped malformed artificial lift file {file_path.name}: {exc}")
                df = None
            if df is None:
                print(f"  Skipped malformed artificial lift file {file_path.name}")
                continue
            df = df.copy()
            df[("_meta", "_source_file")] = file_path.name
            frames.append(df)
            print(f"  Loaded artificial lift file {file_path.name} | rows={len(df)}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def _find_big_column_by_field(df: pd.DataFrame, *field_names: str):
    normalized_candidates = {normalize_big_header(field_name) for field_name in field_names if field_name}
    for column in df.columns:
        if not isinstance(column, tuple) or len(column) < 2:
            continue
        if normalize_big_header(column[1]) in normalized_candidates:
            return column
    return None


def _build_big_record_payload(row: pd.Series) -> dict:
    source_aliases = {
        "esp_type": ("насос (50гц)", "модель гно"),
        "contractor": ("насос (50гц)", "собственник оборудования"),
        "installation_date": ("", "дата монтажа"),
        "launch_date": ("", "дата запуска"),
        "failure_date_source": ("", "дата отказа"),
        "dismantling_date": ("", "дата демонтажа"),
    }
    payload: dict = {}
    for alias, source_col in source_aliases.items():
        if source_col not in row.index:
            payload[alias] = None
            continue
        value = get_first_matching_value(row, source_col)
        payload[alias] = value if pd.notna(value) else None
    for target_col, source_col in BIG_COLUMN_RULES.items():
        if source_col not in row.index:
            payload[target_col] = None
            continue
        value = get_first_matching_value(row, source_col)
        payload[target_col] = value if pd.notna(value) else None
    return payload


def _closed_run_reference_dates(
    bore_key: pd.Series,
    installation_date: pd.Series,
    failure_date: pd.Series,
    dismantling_date: pd.Series,
    esp_oil_mask: pd.Series,
    as_of: Optional[pd.Timestamp] = None,
) -> dict:
    """Return, per **bore identity**, the latest date the bore's AL oil run closed.

    Keyed by the suffix-preserving bore identity (``normalize_well``), NOT the
    physical well number: a different bore of the same well (a side-track ``ш`` /
    ``вз``) has an independent run history and must never block an oil bore's
    live run. (Brine ``рс`` bores are handled separately, per physical well.)

    A run closes the bore's artificial-lift oil life when:
      * it is an ESP oil run (``Мех. добыча``, or ``Фонтанная`` with an ESP) that
        has ended -> its end is the dismantling date (the physical pull) when
        present, else the failure date; a ``fail > dismantling`` typo resolves to
        the real pull date. An *open* ESP oil run is not a closure.
      * it is a conversion **away** from ESP oil -- ``консервация`` /
        ``ликвидирована`` / ``Пьезометр`` / ``Нагнетательная`` / a non-ESP
        ``Фонтанная`` etc. -- in which case the AL oil run ended at that run's
        **mount** date (when the well left mechanized production).
    Dates after ``as_of`` are ignored as future-date typos.
    """
    reference: dict = {}
    for i in bore_key.index:
        bore = bore_key[i]
        if not bore:
            continue
        if esp_oil_mask[i]:
            if pd.isna(failure_date[i]) and pd.isna(dismantling_date[i]):
                continue  # a live ESP oil run is not a closure
            end = dismantling_date[i] if pd.notna(dismantling_date[i]) else failure_date[i]
        else:
            end = installation_date[i]  # conversion away from ESP oil -> closure at its mount
        if pd.isna(end) or (as_of is not None and end > as_of):
            continue
        current = reference.get(bore)
        if current is None or end > current:
            reference[bore] = end
    return reference


def _brine_bore_reference_dates(
    well_number: pd.Series,
    normalized_well: pd.Series,
    failure_date: pd.Series,
    dismantling_date: pd.Series,
    as_of: Optional[pd.Timestamp] = None,
) -> dict:
    """Latest closed-run end date of any **brine (``рс``) bore**, per physical well.

    A physical well whose brine bore has closed runs is (partly) a brine well; a
    suffixless open *oil* row of that number that predates the brine activity is a
    phantom -- the brine bore recorded without its suffix -- and must be dropped
    (the T2/T5 bore-identity-collapse guard). Non-brine side-tracks (``ш``,
    ``вз`` ...) are genuine separate oil bores and never appear here.
    """
    reference: dict = {}
    closed_mask = failure_date.notna() | dismantling_date.notna()
    for i in failure_date.index[closed_mask]:
        if not is_rassol_bore(normalized_well[i]):
            continue
        physical = well_number[i]
        if not physical:
            continue
        end = dismantling_date[i] if pd.notna(dismantling_date[i]) else failure_date[i]
        if pd.isna(end) or (as_of is not None and end > as_of):
            continue
        current = reference.get(physical)
        if current is None or end > current:
            reference[physical] = end
    return reference


def runtime_within_calendar(
    runtime_days,
    installation_date,
    as_of_date,
    tolerance_days: float = 2.0,
):
    """Return whether a live run's runtime fits the calendar age.

    A censored run's runtime must not exceed ``as_of_date - installation_date``
    (plus a small tolerance for day-rounding). Returns ``True``/``False`` when it
    can be evaluated, or ``None`` when inputs are missing.
    """
    runtime = parse_number(runtime_days)
    install = parse_date(installation_date)
    as_of = parse_date(as_of_date)
    if runtime is None or install is None or as_of is None:
        return None
    calendar_days = (as_of.normalize() - install.normalize()).days
    return runtime <= calendar_days + tolerance_days


def _describe_dropped(prepared, index, well_series, installation_date, purpose=None) -> pd.DataFrame:
    """Minimal identity of a dropped active run, for the exclusion ledger."""
    if not len(index):
        return pd.DataFrame()
    payload = {
        "Скв.": [well_series[i] for i in index],
        "well": [well_series[i] for i in index],
        "Дата монтажа": [installation_date[i] for i in index],
        "installation_date": [installation_date[i] for i in index],
        "raw_source": "BIG_RUNNING",
    }
    if purpose is not None:
        payload["Цель спуска"] = [purpose[i] for i in index]
    return pd.DataFrame(payload)


def extract_running_wells_from_artificial_lift(
    artificial_lift_path,
    manual_path: Optional[str] = None,
    as_of_date=None,
    *,
    excluded: Optional[dict] = None,
) -> pd.DataFrame:
    """Extract active (censored) pump runs from the artificial lift register.

    A run is considered active when both `Дата отказа` and `Дата демонтажа`
    are empty. These rows are returned as non-failure (right-censored) events
    for the `include all` workflow path -- the living half of the survival
    population that a failures-only register would otherwise miss.

    Reliability invariants enforced here:

    * Only mechanized (ESP) oil-production runs are kept -- rows where
      `Цель спуска == Мех. добыча` -- and brine (``рс``) bores are excluded,
      mirroring the PDK oil-well (`Тип скважины == НФ`) filter.
    * **One live run per well** -- the latest open run wins.
    * **Monotonic age** -- a live run may not start before the same well's last
      *closed* run ended; stale open rows (e.g. a 2016 mount whose run already
      closed in 2017) are dropped rather than reported as multi-year-old pumps.
    * **Runtime <= calendar** -- runtimes exceeding the age from mount to
      ``as_of_date`` are logged, never silently kept as if valid.

    Args:
        as_of_date: Build/cut-off date used for the runtime<=calendar check.
            Defaults to today when omitted.
        excluded: Optional dict filled with ``{reason_key: DataFrame}`` for the
            active runs this function drops. Without it those rows vanish into a
            print counter and nobody can check which well was dropped, or why.
    """
    as_of_ts = parse_date(as_of_date) if as_of_date is not None else pd.Timestamp.now().normalize()
    artificial_lift_df = load_big_sources(artificial_lift_path, manual_path)
    if artificial_lift_df is None or artificial_lift_df.empty:
        return pd.DataFrame()

    well_source_col = ("", "скважина")
    if well_source_col not in artificial_lift_df.columns:
        return pd.DataFrame()

    installation_col = _find_big_column_by_field(artificial_lift_df, "дата монтажа")
    launch_col = _find_big_column_by_field(artificial_lift_df, "дата запуска")
    failure_col = _find_big_column_by_field(artificial_lift_df, "дата отказа")
    dismantling_col = _find_big_column_by_field(artificial_lift_df, "дата демонтажа")
    purpose_col = _find_big_column_by_field(artificial_lift_df, "цель спуска")
    gno_type_col = _find_big_column_by_field(artificial_lift_df, "тип гно")
    runtime_col = _find_big_column_by_field(
        artificial_lift_df,
        "нно",
        "наработка (сут)",
        "наработка сут",
        "наработка, сут",
    )
    if installation_col is None:
        return pd.DataFrame()

    # The artificial-lift frame carries a `(section, field)` MultiIndex on its
    # columns, so all derived filtering fields are kept as index-aligned Series
    # (rather than scalar-keyed columns) and rows are selected by index label.
    prepared = artificial_lift_df.copy()
    empty_dates = pd.Series(pd.NaT, index=prepared.index)
    well_series = get_first_matching_column(prepared, well_source_col)
    normalized_well = well_series.apply(normalize_well)
    well_number = well_series.apply(well_number_key)
    installation_date = get_first_matching_column(prepared, installation_col).apply(parse_date)
    failure_date = (
        get_first_matching_column(prepared, failure_col).apply(parse_date)
        if failure_col is not None else empty_dates
    )
    dismantling_date = (
        get_first_matching_column(prepared, dismantling_col).apply(parse_date)
        if dismantling_col is not None else empty_dates
    )
    if runtime_col is not None:
        prepared["_runtime_nno"] = get_first_matching_column(prepared, runtime_col).apply(parse_number)
    else:
        prepared["_runtime_nno"] = None

    gno_type = (
        get_first_matching_column(prepared, gno_type_col)
        if gno_type_col is not None
        else pd.Series([None] * len(prepared), index=prepared.index)
    )
    if purpose_col is not None:
        purpose = get_first_matching_column(prepared, purpose_col)
        # An artificial-lift oil run is Мех. добыча, or a Фонтанная completion that
        # still carries an ESP (checked via Насос (50Гц) / Тип ГНО).
        esp_oil_mask = pd.Series(
            [is_esp_oil_run(purpose[i], gno_type[i]) for i in prepared.index],
            index=prepared.index,
        )
    else:
        # Older artificial-lift exports without a purpose column: keep prior behaviour.
        purpose = pd.Series([None] * len(prepared), index=prepared.index)
        esp_oil_mask = pd.Series(True, index=prepared.index)

    # A run is "active" when it has a mount date and no closure the register can
    # see. ⚠⚠ "Can see" means **at or before ``as_of_ts``**, not "ever": the
    # passport is refreshed independently of ПДК, so it routinely records a
    # closure in a window ПДК has not reached yet. Such a run is neither a ПДК
    # failure nor — under a plain ``isna()`` test — a live run, so it used to
    # vanish from the register entirely, taking its whole observed life with it.
    # Censoring it at the horizon keeps that life and claims nothing about an
    # outcome we cannot observe.
    #
    # Oil-production identity is decided by bore suffix and purpose -- never by
    # the bare well number: a brine (``рс``) bore keeps a separate identity and is
    # excluded even when its purpose reads ``Мех. добыча``.
    unclosed_at_horizon = (
        failure_date.isna() | (failure_date > as_of_ts)
    ) & (
        dismantling_date.isna() | (dismantling_date > as_of_ts)
    )
    active_mask = normalized_well.ne("") & installation_date.notna() & unclosed_at_horizon

    closed_beyond_horizon = int(
        (normalized_well.ne("") & installation_date.notna() & unclosed_at_horizon
         & (failure_date.notna() | dismantling_date.notna())).sum()
    )
    if closed_beyond_horizon:
        print(
            f"  Running-well horizon rule: {closed_beyond_horizon} run(s) close after "
            f"{as_of_ts.date()} — kept as censored at the horizon (ПДК does not reach them)"
        )
    rassol_mask = normalized_well.apply(is_rassol_bore)
    running_mask = active_mask & esp_oil_mask & ~rassol_mask

    # Log why active runs were dropped as non-oil, broken down by назначение,
    # so a regression in the excluded population is never silent (T6).
    dropped_index = prepared.index[active_mask & ~running_mask]
    if len(dropped_index):
        drop_reasons: Counter = Counter()
        for i in dropped_index:
            reason = non_oil_reason(normalized_well[i], purpose[i]) or "не мех. добыча"
            drop_reasons[reason] += 1
        summary = ", ".join(f"{reason}: {count}" for reason, count in sorted(drop_reasons.items()))
        print(f"  Running-well non-oil filter dropped {len(dropped_index)} active run(s) ({summary})")
        if excluded is not None:
            brine = [i for i in dropped_index if is_rassol_bore(normalized_well[i])]
            other = [i for i in dropped_index if i not in set(brine)]
            excluded["brine"] = _describe_dropped(prepared, brine, well_series, installation_date)
            excluded["non_oil"] = _describe_dropped(prepared, other, well_series, installation_date, purpose)

    running_index = list(prepared.index[running_mask])
    if not running_index:
        return pd.DataFrame()

    # Monotonic age: drop live runs that start before the same **bore's** last
    # oil closed run ended -- these are stale open rows, not live pumps. Keyed by
    # bore identity (not physical well number) so a brine/side-track/пьезометр
    # closure never blocks a genuine oil run.
    closed_reference = _closed_run_reference_dates(
        normalized_well, installation_date, failure_date, dismantling_date, esp_oil_mask, as_of_ts
    )
    brine_reference = _brine_bore_reference_dates(
        well_number, normalized_well, failure_date, dismantling_date, as_of_ts
    )
    kept_index: list = []
    stale_count = 0
    for i in running_index:
        bore_ref = closed_reference.get(normalized_well[i])
        brine_ref = brine_reference.get(well_number[i])
        candidates = [ref for ref in (bore_ref, brine_ref) if ref is not None]
        reference_date = max(candidates) if candidates else None
        if reference_date is not None and installation_date[i] < reference_date:
            stale_count += 1
        else:
            kept_index.append(i)
    if excluded is not None and stale_count:
        stale_index = [i for i in running_index if i not in set(kept_index)]
        excluded["stale"] = _describe_dropped(prepared, stale_index, well_series, installation_date)
    if stale_count:
        print(
            f"  Running-well monotonicity filter dropped {stale_count} stale open run(s) "
            "(mount predates the well's last closed run)"
        )
    if not kept_index:
        return pd.DataFrame()

    # One live run per physical well: keep the most recently mounted open run.
    best_by_well: dict = {}
    for i in kept_index:
        well_key = well_number[i]
        incumbent = best_by_well.get(well_key)
        if incumbent is None or installation_date[i] > installation_date[incumbent]:
            best_by_well[well_key] = i
    final_index = [i for i in kept_index if best_by_well.get(well_number[i]) == i]
    if excluded is not None and len(final_index) != len(kept_index):
        superseded = [i for i in kept_index if i not in set(final_index)]
        excluded["superseded"] = _describe_dropped(prepared, superseded, well_series, installation_date)
    if len(final_index) != len(kept_index):
        print(
            f"  Running-well one-per-well filter collapsed {len(kept_index)} open run(s) "
            f"to {len(final_index)} (latest mount kept)"
        )

    runtime_violations = 0
    calendar_runtime_count = 0
    records: list[dict] = []
    for i in final_index:
        row = prepared.loc[i]
        well_value = get_first_matching_value(row, well_source_col)
        payload = _build_big_record_payload(row)
        # Distinct names: ``installation_date`` above is the index-aligned Series
        # the monotonicity/one-per-well filters read. Rebinding it to a scalar here
        # (as the original did) leaves those filters reading a Timestamp on any
        # future edit that moves them below this loop.
        run_installation_date = parse_date(payload.get("installation_date"))
        run_launch_date = parse_date(payload.get("launch_date"))
        runtime_nno = parse_number(get_first_matching_value(row, ("_runtime_nno", "")))
        if runtime_within_calendar(runtime_nno, run_installation_date, as_of_ts) is False:
            runtime_violations += 1
        # An open (running) row must still carry «Наработка (сут)». When the
        # artificial-lift export has no ННО, fall back to the calendar age at the
        # build date, measured from the LAUNCH date when available (mount->launch
        # is the ~49-day gap that otherwise skews a mount-based age).
        runtime_from_calendar = False
        if runtime_nno is None:
            age_anchor = run_launch_date or run_installation_date
            if age_anchor is not None:
                runtime_nno = max((as_of_ts.normalize() - age_anchor.normalize()).days, 0)
                calendar_runtime_count += 1
                runtime_from_calendar = True
        enrichment_date = None
        if run_installation_date is not None and runtime_nno is not None and runtime_nno >= 0:
            try:
                enrichment_date = run_installation_date + pd.Timedelta(days=float(runtime_nno))
            except (OverflowError, OutOfBoundsTimedelta, OutOfBoundsDatetime):
                # A garbage ННО (already flagged by the runtime<=calendar check)
                # must not crash the build -- fall back to the mount date.
                enrichment_date = run_installation_date
        elif run_installation_date is not None:
            enrichment_date = run_installation_date

        field_code = derive_field_code_from_well(well_value)
        record = {
            "raw_source": "BIG_RUNNING",
            "well": well_value,
            "Скв.": well_value,
            "normalized_well": normalize_well(well_value),
            "field": field_code,
            "Месторождение": field_code,
            "acid_type": derive_acid_type(field_code),
            "Кислый/Некислый": derive_acid_type(field_code),
            "failure_date": None,
            "Дата остановки": None,
            "installation_date": run_installation_date,
            "Дата монтажа": run_installation_date,
            "launch_date": run_launch_date,
            "Дата запуска": run_launch_date,
            "dismantling_date": None,
            "Дата демонтажа": None,
            "runtime_nno": runtime_nno,
            "Наработка (сут)": runtime_nno,
            "failure_flag": 0,
            "Флаг отказа": 0,
            "failure_reason": None,
            "Причина остановки": None,
            "failed_node": None,
            "Отказавший узел": None,
            "failed_element": None,
            "Отказавший элемент": None,
            "failure_marker": None,
            "Признак отказа": None,
            "techregime_query_date": enrichment_date,
            "runtime_group": derive_runtime_group(runtime_nno),
            "группа наработок": derive_runtime_group(runtime_nno),
        }
        for key, value in payload.items():
            if key in {"installation_date", "launch_date", "failure_date_source", "dismantling_date"}:
                continue
            record[key] = value
        if runtime_from_calendar:
            anchor_label = "запуска" if run_launch_date is not None else "монтажа"
            record[SVOD_COMMENT_COLUMN] = (
                f"Наработка (сут) рассчитана по календарю от даты {anchor_label} "
                f"на {as_of_ts.date()} (нет ННО в источнике)"
            )
        records.append(record)

    if runtime_violations:
        print(
            f"  Running-well runtime check: {runtime_violations} live run(s) report "
            f"runtime > calendar age at {as_of_ts.date()} (kept, flagged for review)"
        )
    if calendar_runtime_count:
        print(
            f"  Running-well runtime: filled «Наработка (сут)» for {calendar_runtime_count} open row(s) "
            f"from calendar age at {as_of_ts.date()} (no ННО in the artificial-lift export)"
        )

    return pd.DataFrame(records)


def enrich_from_artificial_lift(
    records: pd.DataFrame,
    artificial_lift_path,
    manual_path: Optional[str] = None,
    well_col: str = "well",
    failure_date_col: str = "failure_date",
) -> pd.DataFrame:
    """Enrich records from the well artificial lift dataset."""
    records = records.copy()
    artificial_lift_df = load_big_sources(artificial_lift_path, manual_path)
    if artificial_lift_df is None:
        return records

    well_source_col = ("", "скважина")
    if well_source_col not in artificial_lift_df.columns:
        return records

    artificial_lift_df = artificial_lift_df.copy()
    artificial_lift_df["normalized_well"] = get_first_matching_column(artificial_lift_df, well_source_col).apply(normalize_well)

    source_aliases = {
        "esp_type": ("насос (50гц)", "модель гно"),
        "contractor": ("насос (50гц)", "собственник оборудования"),
        "installation_date": ("", "дата монтажа"),
        "launch_date": ("", "дата запуска"),
        "failure_date_source": ("", "дата отказа"),
        "dismantling_date": ("", "дата демонтажа"),
    }

    # These columns are filled cell-by-cell below with whatever the passport
    # holds — dates, model strings, numbers. An empty column that arrived as
    # float64 (an all-NaN date column collapses to that) rejects a Timestamp on
    # assignment under pandas' incompatible-dtype deprecation, so every target is
    # forced to object *before* the loop rather than after the first surprise.
    for alias in ["esp_type", "contractor", "launch_date", "installation_date", "dismantling_date"]:
        if alias not in records.columns:
            records[alias] = None
        elif records[alias].dtype != object:
            records[alias] = records[alias].astype(object)
    for target_col in BIG_COLUMN_RULES:
        if target_col not in records.columns:
            records[target_col] = None
        elif records[target_col].dtype != object:
            records[target_col] = records[target_col].astype(object)
    if "Дельта Дебита Ж и номинала, м3/сут" not in records.columns:
        records["Дельта Дебита Ж и номинала, м3/сут"] = None

    for idx, record in records.iterrows():
        if well_col not in record or pd.isna(record[well_col]):
            continue
        well_norm = normalize_well(record[well_col])
        matching = artificial_lift_df[artificial_lift_df["normalized_well"].eq(well_norm)].copy()
        if matching.empty:
            continue

        fail_date = parse_date(record.get(failure_date_col))
        installation_date = parse_date(record.get("installation_date", record.get("Дата монтажа")))
        failure_source_col = source_aliases["failure_date_source"]
        if fail_date is not None and failure_source_col in matching.columns:
            matching["_failure_date_source"] = get_first_matching_column(matching, failure_source_col).apply(parse_date)
            dated = matching[matching["_failure_date_source"].notna()].copy()
            if not dated.empty:
                dated["_date_diff"] = dated["_failure_date_source"].apply(lambda x: abs((x - fail_date).days))
                best_match = dated.loc[dated["_date_diff"].idxmin()]
            else:
                best_match = matching.iloc[0]
        elif installation_date is not None and source_aliases["installation_date"] in matching.columns:
            matching["_installation_date_source"] = get_first_matching_column(
                matching,
                source_aliases["installation_date"],
            ).apply(parse_date)
            dated = matching[matching["_installation_date_source"].notna()].copy()
            if not dated.empty:
                dated["_date_diff"] = dated["_installation_date_source"].apply(lambda x: abs((x - installation_date).days))
                best_match = dated.loc[dated["_date_diff"].idxmin()]
            else:
                best_match = matching.iloc[0]
        else:
            best_match = matching.iloc[0]

        for alias, source_col in source_aliases.items():
            if alias == "failure_date_source" or source_col not in artificial_lift_df.columns:
                continue
            value = get_first_matching_value(best_match, source_col)
            if pd.notna(value):
                records.loc[idx, alias] = value

        for target_col, source_col in BIG_COLUMN_RULES.items():
            normalized_source = (
                normalize_big_header(source_col[0]),
                normalize_big_header(source_col[1]),
            )
            if normalized_source not in artificial_lift_df.columns:
                continue
            value = get_first_matching_value(best_match, normalized_source)
            if pd.notna(value):
                records.loc[idx, target_col] = value

        if "big_match_confidence" not in records.columns:
            records["big_match_confidence"] = None
        records.loc[idx, "big_match_confidence"] = 0.70

    return records


def enrich_from_big(*args, **kwargs):
    """Backward-compatible alias for artificial lift enrichment."""
    return enrich_from_artificial_lift(*args, **kwargs)


def enrich_big_derived_fields(records: pd.DataFrame) -> pd.DataFrame:
    """Fill computed target fields derived from artificial lift plus techregime data."""
    records = records.copy()
    if "Дельта Дебита Ж и номинала, м3/сут" not in records.columns:
        records["Дельта Дебита Ж и номинала, м3/сут"] = None
    for idx, record in records.iterrows():
        liquid_rate = parse_number(record.get("Дебит жидк.", record.get("liquid_rate")))
        nominal_rate = parse_number(record.get("Ном. Произв. м₃/сут"))
        if liquid_rate is None or nominal_rate is None:
            continue
        records.loc[idx, "Дельта Дебита Ж и номинала, м3/сут"] = liquid_rate - nominal_rate
    return records


__all__ = [
    "extract_running_wells_from_artificial_lift",
    "runtime_within_calendar",
    "load_big_dataframe",
    "load_big_sources",
    "enrich_from_artificial_lift",
    "enrich_from_big",
    "enrich_big_derived_fields",
]
