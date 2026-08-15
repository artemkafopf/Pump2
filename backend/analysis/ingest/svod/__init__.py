"""Сборка Свода ЭЦН из сырых источников.

Перенесено из ``db_builder.failure_update``. Точки входа:

``build_svod_from_scratch``  собрать регистр с нуля (штатный режим);
``FailureUpdateWorkflow``    дополнить существующий регистр.
"""

from .artificial_lift_processor import (
    enrich_from_artificial_lift,
    extract_running_wells_from_artificial_lift,
)
from .causes import attach_cause_columns, classify_cause_group, classify_node_category
from .lab_processor import enrich_from_lab
from .opz_processor import enrich_from_opz
from .pdk_processor import derive_failure_flag, extract_new_failures_from_pdk
from .techregime_processor import enrich_from_techregime, refresh_records_from_techregime

__all__ = [
    "attach_cause_columns",
    "classify_cause_group",
    "classify_node_category",
    "derive_failure_flag",
    "enrich_from_artificial_lift",
    "enrich_from_lab",
    "enrich_from_opz",
    "enrich_from_techregime",
    "extract_new_failures_from_pdk",
    "extract_running_wells_from_artificial_lift",
    "refresh_records_from_techregime",
]
