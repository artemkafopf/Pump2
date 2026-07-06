"""Backward-compatible shim — import from analysis.paths directly."""
from analysis.paths import (
    DEFAULT_LAB_EXTERNAL,
    DEFAULT_TECHREGIME_EXTERNAL,
    DEFAULT_TELEMETRY_EXTERNAL,
    LOCAL_SQLITE_DIR,
    resolve_lab_db_path,
    resolve_techregime_db_path,
    resolve_telemetry_db_path,
)

__all__ = [
    "DEFAULT_LAB_EXTERNAL",
    "DEFAULT_TECHREGIME_EXTERNAL",
    "DEFAULT_TELEMETRY_EXTERNAL",
    "LOCAL_SQLITE_DIR",
    "resolve_lab_db_path",
    "resolve_techregime_db_path",
    "resolve_telemetry_db_path",
]
