"""ESP type to specification helpers."""

from collections import Counter
from typing import Dict, Optional

import pandas as pd


def normalize_esp_type(value) -> Optional[str]:
    """Normalize ESP type text for stable dictionary lookups."""
    if pd.isna(value) or value is None:
        return None
    text = str(value).strip().replace("\t", " ")
    text = " ".join(text.split())
    return text or None


def normalize_gabarit(value) -> Optional[str]:
    """Normalize gabarit values from the target register."""
    if pd.isna(value) or value is None:
        return None
    text = str(value).strip().upper().replace("A", "А")
    text = " ".join(text.split())
    if text in {"", "-", "0", "NAN"}:
        return None
    return text


def normalize_productivity(value) -> Optional[str]:
    """Normalize productivity values from the target register."""
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip().upper().replace("М", "M")
    text = " ".join(text.split())
    if text in {"", "-", "0", "NAN"}:
        return None
    return text


def normalize_nominal_head(value) -> Optional[str]:
    """Normalize nominal head values from the target register."""
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip().upper()
    text = " ".join(text.split())
    if text in {"", "-", "0", "NAN"}:
        return None
    return text


def build_esp_specs_from_target(target_df: pd.DataFrame) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Build a reusable ESP spec map from the current register.

    For each ESP type, the most common non-empty gabarit and productivity
    values are selected.
    """
    required = ["Тип УЭЦН", "Габарит УЭЦН", "Ном. Произв. м₃/сут", "Ном.напор (50Гц)"]
    if any(col not in target_df.columns for col in required):
        return {}

    source = target_df[required].dropna(subset=["Тип УЭЦН"]).copy()
    source["esp_norm"] = source["Тип УЭЦН"].apply(normalize_esp_type)
    source["gab_norm"] = source["Габарит УЭЦН"].apply(normalize_gabarit)
    source["prod_norm"] = source["Ном. Произв. м₃/сут"].apply(normalize_productivity)
    source["head_norm"] = source["Ном.напор (50Гц)"].apply(normalize_nominal_head)

    specs: Dict[str, Dict[str, Optional[str]]] = {}
    for esp_type, group in source.groupby("esp_norm"):
        if not esp_type:
            continue
        gabarit_values = [value for value in group["gab_norm"] if value is not None]
        productivity_values = [value for value in group["prod_norm"] if value is not None]
        head_values = [value for value in group["head_norm"] if value is not None]

        gabarit = Counter(gabarit_values).most_common(1)[0][0] if gabarit_values else None
        productivity = Counter(productivity_values).most_common(1)[0][0] if productivity_values else None
        nominal_head = Counter(head_values).most_common(1)[0][0] if head_values else None

        if gabarit is not None or productivity is not None or nominal_head is not None:
            specs[esp_type] = {
                "gabarit": gabarit,
                "productivity": productivity,
                "nominal_head": nominal_head,
            }

    return specs


def fill_specs_from_esp_type(records: pd.DataFrame, esp_specs: Dict[str, Dict[str, Optional[str]]]) -> pd.DataFrame:
    """Fill missing gabarit and productivity fields using ESP type."""
    if not esp_specs:
        return records

    records = records.copy()
    if "Габарит УЭЦН" not in records.columns:
        records["Габарит УЭЦН"] = None
    if "Ном. Произв. м₃/сут" not in records.columns:
        records["Ном. Произв. м₃/сут"] = None
    if "gabarit" not in records.columns:
        records["gabarit"] = None
    if "productivity" not in records.columns:
        records["productivity"] = None
    if "Ном.напор (50Гц)" not in records.columns:
        records["Ном.напор (50Гц)"] = None
    if "nominal_head" not in records.columns:
        records["nominal_head"] = None

    for idx, record in records.iterrows():
        esp_type = normalize_esp_type(record.get("esp_type", record.get("Тип УЭЦН")))
        if not esp_type or esp_type not in esp_specs:
            continue
        spec = esp_specs[esp_type]

        if pd.isna(record.get("Габарит УЭЦН")) or record.get("Габарит УЭЦН") in [None, "", "-"]:
            if spec.get("gabarit") is not None:
                records.loc[idx, "Габарит УЭЦН"] = spec["gabarit"]
                records.loc[idx, "gabarit"] = spec["gabarit"]

        if pd.isna(record.get("Ном. Произв. м₃/сут")) or record.get("Ном. Произв. м₃/сут") in [None, "", "-"]:
            if spec.get("productivity") is not None:
                records.loc[idx, "Ном. Произв. м₃/сут"] = spec["productivity"]
                records.loc[idx, "productivity"] = spec["productivity"]

        if pd.isna(record.get("Ном.напор (50Гц)")) or record.get("Ном.напор (50Гц)") in [None, "", "-"]:
            if spec.get("nominal_head") is not None:
                records.loc[idx, "Ном.напор (50Гц)"] = spec["nominal_head"]
                records.loc[idx, "nominal_head"] = spec["nominal_head"]

    return records
