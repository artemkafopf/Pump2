"""Tests for the unified column resolver and its ingestion wiring."""

# Перенесено из db_builder/tests/test_column_resolution.py вместе с кодом сборщика: тесты
# обязаны переезжать с тем, что они держат, иначе перенос молча теряет именно
# те инварианты, ради которых код и писался.

from pathlib import Path

import pandas as pd
import pytest

from analysis.ingest.column_resolution import (
    MissingRequiredColumnError,
    ResolutionMethod,
    build_specs,
    resolve_columns,
)
from analysis.ingest.svod.pdk_processor import extract_new_failures_from_pdk


SPECS = build_specs(
    {
        "Скв.": ["скважина", "скв", "скв.", "№ скважины", "well"],
        "Дата остановки": ["дата остановки", "дата отказа", "failure_date"],
        "Габарит": ["габарит", "gabarit"],
    },
    required=["Скв."],
)


def test_exact_header_resolves_as_exact():
    report = resolve_columns(["Скв.", "Дата остановки"], SPECS)
    methods = {r.source_header: r.method for r in report.resolutions}
    assert methods["Скв."] is ResolutionMethod.EXACT
    assert methods["Дата остановки"] is ResolutionMethod.EXACT
    assert report.mapping == {"Скв.": "Скв.", "Дата остановки": "Дата остановки"}


def test_renamed_with_dot_header_resolves_via_fuzzy():
    report = resolve_columns(["Скв .", "Дата остановки", "Габарит"], SPECS)
    scv = next(r for r in report.resolutions if r.source_header == "Скв .")
    assert scv.method is ResolutionMethod.FUZZY
    assert scv.canonical == "Скв."
    assert scv.score >= 0.8
    # Only the renamed header is in the rename map; exact ones are no-ops.
    assert report.rename_map() == {"Скв .": "Скв."}


def test_synonym_header_resolves_as_synonym():
    report = resolve_columns(["Скважина"], SPECS)
    res = report.resolutions[0]
    assert res.method is ResolutionMethod.SYNONYM
    assert res.canonical == "Скв."


def test_reordered_columns_resolve_identically():
    forward = resolve_columns(["Скв.", "Дата остановки", "Габарит"], SPECS)
    reordered = resolve_columns(["Габарит", "Скв.", "Дата остановки"], SPECS)
    assert forward.mapping == reordered.mapping


def test_missing_required_column_raises_with_candidates():
    report = resolve_columns(["Дата остановки", "Габарит"], SPECS, source_file="bad.xlsx")
    with pytest.raises(MissingRequiredColumnError) as excinfo:
        report.raise_if_required_missing()
    message = str(excinfo.value)
    assert "bad.xlsx" in message
    assert "Скв." in message
    # The nearest-miss candidates are named.
    assert "nearest" in message


def test_report_lists_unmapped_extra_columns():
    report = resolve_columns(["Скв.", "необычная колонка"], SPECS)
    unmapped = report.unmapped_headers()
    assert [r.source_header for r in unmapped] == ["необычная колонка"]
    records = {rec["source_header"]: rec["canonical"] for rec in report.to_records()}
    assert records["необычная колонка"] == "UNMAPPED"


def test_ambiguous_fuzzy_match_is_left_unmapped():
    specs = build_specs({"Дата монтажа": ["дата монтажа"], "Дата остановки": ["дата остановки"]})
    # A bare "Дата" is equally similar to both date canonicals -> ambiguous.
    report = resolve_columns(["Дата"], specs)
    assert report.resolutions[0].canonical is None


def _patch_loader(frames):
    original = extract_new_failures_from_pdk.__globals__["load_dataset_frames"]
    extract_new_failures_from_pdk.__globals__["load_dataset_frames"] = lambda path, dataset_type: frames
    return original


def _restore_loader(original):
    extract_new_failures_from_pdk.__globals__["load_dataset_frames"] = original


def test_pdk_extract_resolves_renamed_well_header():
    # The well column arrives renamed with a stray space; it must still be
    # recognized so repair/normalization downstream keeps working.
    source_df = pd.DataFrame(
        [
            {
                "Месторождение": "ЯНГКМ",
                "Куст": "14",
                "Скв .": "Ya_160",
                "Дата остановки": "2025-09-10",
                "Причина остановки": "Отказ",
                "Отказавший узел": "Насос",
                "Тип скважины": "НФ",
            }
        ]
    )
    original = _patch_loader([(Path("pdk.xlsx"), source_df, "Sheet1", 0)])
    try:
        extracted = extract_new_failures_from_pdk("dummy.xlsx", None)
    finally:
        _restore_loader(original)

    assert extracted["Скв."].tolist() == ["Ya_160"]
    assert extracted["normalized_well"].tolist() == ["ya_160"]


def test_pdk_extract_raises_when_well_column_unresolvable():
    source_df = pd.DataFrame(
        [
            {
                "Месторождение": "ЯНГКМ",
                "Дата остановки": "2025-09-10",
                "Причина остановки": "Отказ",
                "Тип скважины": "НФ",
            }
        ]
    )
    original = _patch_loader([(Path("pdk.xlsx"), source_df, "Sheet1", 0)])
    try:
        with pytest.raises(MissingRequiredColumnError):
            extract_new_failures_from_pdk("renamed.xlsx", None)
    finally:
        _restore_loader(original)


def test_pdk_extract_records_mapping_report_in_audit():
    from analysis.ingest.svod.audit import AuditReport

    source_df = pd.DataFrame(
        [
            {
                "Скв .": "Ya_160",
                "Дата остановки": "2025-09-10",
                "Причина остановки": "Отказ",
                "Отказавший узел": "Насос",
                "Тип скважины": "НФ",
            }
        ]
    )
    audit = AuditReport()
    original = _patch_loader([(Path("pdk.xlsx"), source_df, "Sheet1", 0)])
    try:
        extract_new_failures_from_pdk("dummy.xlsx", None, audit=audit)
    finally:
        _restore_loader(original)

    headers = {rec["source_header"]: rec for rec in audit.column_mappings}
    assert headers["Скв ."]["canonical"] == "Скв."
    assert headers["Скв ."]["method"] == "fuzzy"
