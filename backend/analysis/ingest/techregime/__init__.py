"""Merging TechRegime exports, splitting them into partitions, and querying them.

⚠ The db_builder original re-exported ``.cli.main`` from here, which dragged the
whole CLI (and its click dependency) into every import of the package. The CLI
is not ported — thin wrappers live in ``scripts/ingest/`` — so this package now
exposes data helpers only.
"""

from .extract import (
    DEFAULT_QUERY_INTERVAL,
    TechRegimeExtractResult,
    extract_techregime_values,
    extract_techregime_values_from_frame,
    extract_techregime_values_from_sqlite,
)
from .processor import (
    DEFAULT_SOURCE_DIR,
    DateQualityStat,
    FIELD_COLUMN,
    MergeByYearResult,
    PartitionOutput,
    PartitionResult,
    YearlyOutput,
    flatten_export_column_label,
    load_techregime_export,
    merge_techregime_exports,
    merge_techregime_exports_by_year,
    merge_techregime_exports_partitioned,
    resolve_techregime_files,
)
from .sqlite_store import (
    SQLITE_DEFAULT_PATH,
    SQLITE_DEFAULT_TABLE,
    SQLiteBuildResult,
    build_techregime_sqlite,
    lookup_latest_techregime_sqlite_row,
    query_techregime_sqlite,
)

__all__ = [
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_QUERY_INTERVAL",
    "DateQualityStat",
    "FIELD_COLUMN",
    "MergeByYearResult",
    "PartitionOutput",
    "PartitionResult",
    "SQLITE_DEFAULT_PATH",
    "SQLITE_DEFAULT_TABLE",
    "SQLiteBuildResult",
    "TechRegimeExtractResult",
    "YearlyOutput",
    "build_techregime_sqlite",
    "extract_techregime_values",
    "extract_techregime_values_from_frame",
    "extract_techregime_values_from_sqlite",
    "flatten_export_column_label",
    "load_techregime_export",
    "lookup_latest_techregime_sqlite_row",
    "merge_techregime_exports",
    "merge_techregime_exports_by_year",
    "merge_techregime_exports_partitioned",
    "query_techregime_sqlite",
    "resolve_techregime_files",
]
