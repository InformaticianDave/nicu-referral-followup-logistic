"""Bootstrapped multivariable logistic regression.

Fits one logistic regression on the full training data for the point
estimate (odds ratio = exp(coefficient)), then re-fits on `n_boot`
random resamples (with replacement, same size as the training data) to
build a non-parametric distribution for each coefficient. The 95% CI
is the 2.5th/97.5th percentile of the resulting odds-ratio
distribution, and the p-value is twice the smaller tail proportion
relative to OR = 1 (i.e. how often the resampled estimate disagrees in
direction with "no effect"), capped at 1.0.

This avoids the large-sample / approximate-normality assumptions
behind the classical Wald test, which is unreliable with modest sample
sizes and sparse categorical cells -- see docs/methods.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

DEFAULT_N_BOOT = 200


def fit_bootstrapped_logistic(
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = DEFAULT_N_BOOT,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """Fit the point-estimate model on a train split, bootstrap the
    training data n_boot times, and evaluate performance on the held-
    out test split.

    Returns a dict with:
      - results: DataFrame indexed by feature with columns
        [Effect, CI_Lower, CI_Upper, PValue_Bootstrap]
      - performance: dict of accuracy/precision/recall/roc_auc on the
        test split
      - model: the fitted sklearn LogisticRegression (point estimate)
      - X_test, y_test: held-out split, for ROC plotting
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    feature_names = list(X_train.columns)
    point_or = np.exp(model.coef_[0])

    rng = np.random.default_rng(random_state)
    n = len(X_train)
    boot_or = np.empty((n_boot, len(feature_names)))

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X_train.iloc[idx]
        yb = y_train.iloc[idx]
        if yb.nunique() < 2:
            boot_or[b, :] = np.nan
            continue
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(Xb, yb)
        boot_or[b, :] = np.exp(m.coef_[0])

    ci_lower = np.nanpercentile(boot_or, 2.5, axis=0)
    ci_upper = np.nanpercentile(boot_or, 97.5, axis=0)

    pvalues = np.empty(len(feature_names))
    for j in range(len(feature_names)):
        col = boot_or[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            pvalues[j] = np.nan
            continue
        prop_below_1 = np.mean(col < 1)
        prop_above_1 = np.mean(col >= 1)
        pvalues[j] = min(1.0, 2 * min(prop_below_1, prop_above_1))

    results = pd.DataFrame(
        {
            "Effect": point_or,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper,
            "PValue_Bootstrap": pvalues,
        },
        index=feature_names,
    ).sort_values("PValue_Bootstrap")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    performance = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "n_boot": n_boot,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return {
        "results": results,
        "performance": performance,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba,
    }
