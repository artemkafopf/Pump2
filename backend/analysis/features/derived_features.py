from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


COLUMN_PATTERN = re.compile(r"\[([^\]]+)\]")
ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "clip": np.clip,
    "exp": np.exp,
    "log": np.log,
    "max": np.maximum,
    "maximum": np.maximum,
    "min": np.minimum,
    "minimum": np.minimum,
    "pow": np.power,
    "sqrt": np.sqrt,
}
ALLOWED_CONSTANTS = {
    "e": float(np.e),
    "pi": float(np.pi),
}


@dataclass(slots=True)
class DerivedColumnSpec:
    name: str
    formula: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DerivedColumnSpec":
        return cls(
            name=str(payload.get("name", "")).strip(),
            formula=str(payload.get("formula", "")).strip(),
        )


def _replace_column_tokens(formula: str, columns: list[str]) -> tuple[str, dict[str, str]]:
    symbol_map: dict[str, str] = {}
    column_lookup = {str(column): str(column) for column in columns}

    def replacer(match: re.Match[str]) -> str:
        raw_name = match.group(1).strip()
        if raw_name not in column_lookup:
            raise ValueError(f"Column '{raw_name}' referenced in formula was not found.")
        symbol = f"__col_{len(symbol_map)}"
        symbol_map[symbol] = column_lookup[raw_name]
        return symbol

    normalized_formula = COLUMN_PATTERN.sub(replacer, formula)
    return normalized_formula, symbol_map


def _evaluate_ast(node: ast.AST, names: dict[str, pd.Series | float]) -> pd.Series | float:
    if isinstance(node, ast.Expression):
        return _evaluate_ast(node.body, names)
    if isinstance(node, ast.Constant):
        return float(node.value) if isinstance(node.value, (int, float)) else node.value
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"Unknown symbol '{node.id}' in derived formula.")
        return names[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast(node.operand, names)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Only unary plus and minus are allowed in derived formulas.")
    if isinstance(node, ast.BinOp):
        left = _evaluate_ast(node.left, names)
        right = _evaluate_ast(node.right, names)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError("Unsupported arithmetic operator in derived formula.")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed in derived formulas.")
        function_name = node.func.id
        if function_name not in ALLOWED_FUNCTIONS:
            raise ValueError(
                f"Function '{function_name}' is not allowed. Supported functions: {', '.join(sorted(ALLOWED_FUNCTIONS))}."
            )
        args = [_evaluate_ast(argument, names) for argument in node.args]
        return ALLOWED_FUNCTIONS[function_name](*args)
    raise ValueError("Derived formula contains unsupported syntax.")


def evaluate_derived_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    if not str(formula).strip():
        raise ValueError("Derived formula cannot be empty.")

    normalized_formula, symbol_map = _replace_column_tokens(str(formula), df.columns.tolist())
    expression = ast.parse(normalized_formula, mode="eval")
    names: dict[str, pd.Series | float] = {symbol: pd.to_numeric(df[column], errors="coerce") for symbol, column in symbol_map.items()}
    names.update(ALLOWED_CONSTANTS)

    result = _evaluate_ast(expression, names)
    if isinstance(result, pd.Series):
        return pd.to_numeric(result, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if np.isscalar(result):
        scalar_value = float(result)
        if np.isinf(scalar_value):
            scalar_value = np.nan
        return pd.Series(np.full(len(df), scalar_value), index=df.index, dtype=float)
    return pd.Series(result, index=df.index, dtype=float).replace([np.inf, -np.inf], np.nan)


def apply_derived_columns(
    df: pd.DataFrame,
    specs: list[DerivedColumnSpec] | list[dict[str, Any]] | None,
) -> tuple[pd.DataFrame, list[str]]:
    processed = df.copy()
    notes: list[str] = []

    parsed_specs = [
        spec if isinstance(spec, DerivedColumnSpec) else DerivedColumnSpec.from_dict(spec)
        for spec in (specs or [])
    ]
    seen_names: set[str] = set()

    for spec in parsed_specs:
        if not spec.name:
            continue
        if spec.name in seen_names:
            raise ValueError(f"Derived column '{spec.name}' is defined more than once.")
        if not spec.formula:
            raise ValueError(f"Derived column '{spec.name}' is missing a formula.")
        processed[spec.name] = evaluate_derived_formula(processed, spec.formula)
        notes.append(f"Derived column '{spec.name}' calculated from formula: {spec.formula}")
        seen_names.add(spec.name)

    return processed, notes
