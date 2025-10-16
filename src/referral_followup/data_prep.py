"""Load and type the predictor/outcome columns for one variable group."""
from __future__ import annotations

import pandas as pd
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "variable_groups.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def group_columns(config: dict, group: str) -> list[str]:
    """Return the predictor column names belonging to a variable group."""
    entries = config[group]
    return [e["name"] for e in entries]


def is_ordinal(config: dict, group: str, name: str) -> bool:
    for e in config[group]:
        if e["name"] == name:
            return bool(e.get("ordinal", False))
    return False


def ordinal_levels(config: dict, group: str, name: str) -> list[str] | None:
    for e in config[group]:
        if e["name"] == name:
            return e.get("levels")
    return None


def prepare_group(
    df: pd.DataFrame,
    config: dict,
    group: str,
    outcome: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Subset to one variable group + one outcome, drop rows missing the
    outcome, impute (mode for categorical, median for continuous), and
    one-hot encode categoricals (dropping the configured reference
    level where one is given).

    Returns (X, y, feature_names) with y coded 0/1 (1 = positive label
    from config["outcomes"][outcome]["positive_label"]).
    """
    cols = group_columns(config, group)
    cols = [c for c in cols if c in df.columns]

    pos_label = config["outcomes"][outcome]["positive_label"]
    neg_label = config["outcomes"][outcome]["negative_label"]

    sub = df[cols + [outcome]].copy()
    sub = sub[sub[outcome].isin([pos_label, neg_label])]
    y = (sub[outcome] == pos_label).astype(int)
    X_raw = sub[cols]

    frames = []
    for col in cols:
        series = X_raw[col]
        if pd.api.types.is_numeric_dtype(series):
            filled = series.fillna(series.median())
            frames.append(filled.rename(col))
            continue

        filled = series.fillna(series.mode(dropna=True).iloc[0] if series.notna().any() else "Unknown")

        if group == "sociodemographic" and is_ordinal(config, group, col):
            levels = ordinal_levels(config, group, col)
            rank = pd.Categorical(filled, categories=levels, ordered=True).codes
            frames.append(pd.Series(rank, index=filled.index, name=col).astype(float))
            continue

        reference = _reference_for(config, group, col)
        dummies = pd.get_dummies(filled, prefix=col)
        drop_col = f"{col}_{reference}" if reference and f"{col}_{reference}" in dummies.columns else dummies.columns[0]
        dummies = dummies.drop(columns=[drop_col])
        frames.append(dummies)

    X = pd.concat(frames, axis=1)
    # Drop any column that ended up constant (no variation) -- same
    # rationale as the constant-column guard in the legacy R pipeline.
    nunique = X.nunique()
    X = X.loc[:, nunique[nunique > 1].index]

    return X.reset_index(drop=True), y.reset_index(drop=True), list(X.columns)


def _reference_for(config: dict, group: str, name: str) -> str | None:
    for e in config[group]:
        if e["name"] == name:
            return e.get("reference")
    return None
