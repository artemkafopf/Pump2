"""Read-only Excel reads, accelerated with python-calamine when available.

``calamine`` reads .xlsx/.xls several times faster than openpyxl/xlrd, but it is
read-only. Use :func:`read_excel` for any read that only consumes data; keep
openpyxl for workbooks opened to write cells or styles.

If ``python-calamine`` is not installed (or a specific read is unsupported by it),
this transparently falls back to pandas' default engine, so behaviour is
unchanged — only the speed differs.
"""

from __future__ import annotations

import pandas as pd

try:  # optional acceleration
    import python_calamine  # noqa: F401

    _HAS_CALAMINE = True
except Exception:  # pragma: no cover - depends on the environment
    _HAS_CALAMINE = False


def read_excel(*args, **kwargs):
    """``pandas.read_excel`` with the calamine engine when available.

    Honors an explicit ``engine=`` if the caller passes one. On any calamine
    failure it retries with the default engine, so a workbook calamine can't
    parse still loads.
    """
    if _HAS_CALAMINE and "engine" not in kwargs:
        try:
            return pd.read_excel(*args, engine="calamine", **kwargs)
        except Exception:
            pass
    return pd.read_excel(*args, **kwargs)


__all__ = ["read_excel"]
