from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis.presentation_schema import normalize_name
from analysis.sqlite_paths import LOCAL_SQLITE_DIR, resolve_lab_db_path


DEFAULT_LAB_SOURCE_DIR = Path(r"D:\Projects\Pumps\data\lab")

LAB_COLUMN_INDEX_MAP = {
    0: "well",
    1: "kp",
    2: "sample_date",
    3: "analysis_date",
    6: "watercut_percent",
    8: "water_density_kg_m3",
    46: "chloride_mg_l",
    47: "sulfate_mg_l",
    48: "bicarbonate_mg_l",
    49: "calcium_mg_l",
    50: "magnesium_mg_l",
    51: "sodium_potassium_mg_l",
    52: "total_mineralization_g_l",
    53: "cation_coefficient",
    54: "svb_h2s_indicator",
    55: "ph",
    59: "water_type",
    60: "mechanical_impurities_mg_l",
    61: "petroleum_products_mg_l",
    63: "comment",
}

NUMERIC_COLUMNS = [
    "watercut_percent",
    "water_density_kg_m3",
    "chloride_mg_l",
    "sulfate_mg_l",
    "bicarbonate_mg_l",
    "calcium_mg_l",
    "magnesium_mg_l",
    "sodium_potassium_mg_l",
    "total_mineralization_g_l",
    "cation_coefficient",
    "ph",
    "mechanical_impurities_mg_l",
    "petroleum_products_mg_l",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized lab.sqlite from lab Excel workbooks.")
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_LAB_SOURCE_DIR),
        help="Folder containing lab Excel files.",
    )
    parser.add_argument(
        "--target-db",
        default=str(LOCAL_SQLITE_DIR / "lab.sqlite"),
        help="Output SQLite path.",
    )
    return parser.parse_args()


def infer_field_code(path: Path) -> str:
    normalized = normalize_name(path.stem)
    for code in ("vt", "ya", "az", "za", "au", "ma", "ki", "da", "am", "zy"):
        if code in normalized:
            return code.upper()
    return ""


def parse_lab_workbook(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    if raw.empty:
        return pd.DataFrame()

    body = raw.iloc[3:].copy()
    body = body.rename(columns={index: column for index, column in LAB_COLUMN_INDEX_MAP.items() if index in body.columns})
    keep = [column for column in LAB_COLUMN_INDEX_MAP.values() if column in body.columns]
    body = body[keep].copy()
    body["source_file"] = path.name
    body["source_path"] = str(path)
    body["field_code_from_file"] = infer_field_code(path)

    for column in ("well", "kp", "water_type", "comment", "svb_h2s_indicator"):
        if column in body.columns:
            body[column] = body[column].astype("string").str.strip()
            body[column] = body[column].replace({"": pd.NA, "nan": pd.NA, "NaT": pd.NA, "–": pd.NA})

    for column in ("sample_date", "analysis_date"):
        if column in body.columns:
            body[column] = pd.to_datetime(body[column], errors="coerce", dayfirst=True, format="mixed")

    for column in NUMERIC_COLUMNS:
        if column in body.columns:
            body[column] = pd.to_numeric(body[column], errors="coerce")

    body = body.loc[body["well"].notna() & body["sample_date"].notna()].copy()
    body["well"] = body["well"].astype("string").str.strip()
    body["well_key"] = body["well"].str.casefold()

    chemistry_columns = ["chloride_mg_l", "sulfate_mg_l", "calcium_mg_l"]
    chemistry_present = pd.Series(False, index=body.index)
    for column in chemistry_columns:
        if column in body.columns:
            chemistry_present = chemistry_present | body[column].notna()
    body = body.loc[chemistry_present].copy()
    return body.reset_index(drop=True)


def build_lab_dataset(source_dir: Path) -> pd.DataFrame:
    files = sorted(source_dir.glob("*.xlsx"))
    frames = [parse_lab_workbook(path) for path in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values(["well_key", "sample_date", "analysis_date", "source_file"], kind="stable")
        .drop_duplicates(subset=["well_key", "sample_date", "chloride_mg_l", "sulfate_mg_l", "calcium_mg_l"], keep="last")
        .reset_index(drop=True)
    )
    combined["sample_date"] = combined["sample_date"].dt.strftime("%Y-%m-%d")
    combined["analysis_date"] = pd.to_datetime(combined["analysis_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return combined


def write_lab_sqlite(df: pd.DataFrame, target_db: Path) -> None:
    target_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target_db) as connection:
        df.to_sql("lab_samples", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_lab_samples_well_date ON lab_samples(well_key, sample_date)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_lab_samples_well ON lab_samples(well_key)")
        connection.commit()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    target_db = Path(args.target_db)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source lab folder was not found: {source_dir}")

    df = build_lab_dataset(source_dir)
    if df.empty:
        raise RuntimeError(f"No lab chemistry rows were parsed from {source_dir}")
    write_lab_sqlite(df, target_db)
    default_path = resolve_lab_db_path()
    print(f"rows={len(df)}")
    print(f"wells={df['well_key'].nunique()}")
    print(f"target_db={target_db}")
    print(f"default_resolver_path={default_path}")


if __name__ == "__main__":
    main()
