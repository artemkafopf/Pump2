from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SQLITE_DIR = REPO_ROOT / "data" / "sqlite"

DEFAULT_TELEMETRY_EXTERNAL = Path(r"D:\Projects\Pumps\data\telemetry\telemetry.sqlite")
DEFAULT_TECHREGIME_EXTERNAL = Path(r"D:\Projects\Pumps\data\techregime\techregime.sqlite")
DEFAULT_LAB_EXTERNAL = Path(r"D:\Projects\Pumps\data\lab\lab.sqlite")


def _resolve_path(*, env_var: str, local_name: str, external_default: Path) -> Path:
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return Path(env_value)

    local_path = LOCAL_SQLITE_DIR / local_name
    if local_path.exists():
        return local_path

    return external_default


def resolve_telemetry_db_path() -> Path:
    return _resolve_path(
        env_var="PUMP2_TELEMETRY_DB_PATH",
        local_name="telemetry.sqlite",
        external_default=DEFAULT_TELEMETRY_EXTERNAL,
    )


def resolve_techregime_db_path() -> Path:
    return _resolve_path(
        env_var="PUMP2_TECHREGIME_DB_PATH",
        local_name="techregime.sqlite",
        external_default=DEFAULT_TECHREGIME_EXTERNAL,
    )


def resolve_lab_db_path() -> Path:
    return _resolve_path(
        env_var="PUMP2_LAB_DB_PATH",
        local_name="lab.sqlite",
        external_default=DEFAULT_LAB_EXTERNAL,
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
