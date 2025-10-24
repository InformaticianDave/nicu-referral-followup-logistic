"""Format bootstrap results as clinician-facing markdown tables.

Column layout matches what was used in the write-ups shared with
collaborators:
    Feature | Effect | 95% CI Lower | 95% CI Upper | P-value (Bootstrap)

"Feature" here is Base Variable + Feature Level combined (e.g.
"Insurance_type_Private"); docs/methods.md explains how to read the
two apart and why a reference/baseline category is needed for each
categorical predictor.
"""
from __future__ import annotations

import pandas as pd


def format_results_table(results: pd.DataFrame, decimals: int = 3) -> str:
    """results: DataFrame indexed by feature with columns
    [Effect, CI_Lower, CI_Upper, PValue_Bootstrap], as returned by
    bootstrap_logistic.fit_bootstrapped_logistic()['results'].
    """
    table = results.copy()
    table.index.name = "Feature"
    table = table.rename(columns={
        "CI_Lower": "95% CI Lower",
        "CI_Upper": "95% CI Upper",
        "PValue_Bootstrap": "P-value (Bootstrap)",
    })
    table = table.round(decimals)
    return table.reset_index().to_markdown(index=False)


def format_performance_block(performance: dict) -> str:
    lines = [
        "| Metric | Value | What it means |",
        "|---|---|---|",
        f"| ROC AUC | {performance['roc_auc']:.3f} | Probability the model ranks a randomly chosen positive case above a randomly chosen negative one; 0.5 = chance, 1.0 = perfect separation. |",
        f"| Accuracy | {performance['accuracy']:.3f} | Fraction of held-out infants classified correctly overall. |",
        f"| Precision | {performance['precision']:.3f} | Of infants predicted positive, the fraction that truly were. |",
        f"| Recall | {performance['recall']:.3f} | Of infants who were truly positive, the fraction the model caught. |",
        f"| n (train / test / bootstrap resamples) | {performance['n_train']} / {performance['n_test']} / {performance['n_boot']} | |",
    ]
    return "\n".join(lines)


def write_group_report(
    outcome: str,
    group: str,
    results: pd.DataFrame,
    performance: dict,
    out_path,
) -> None:
    title = f"# {outcome} — {group.replace('_', ' ').title()}\n\n"
    body = (
        title
        + "## Performance (held-out test split)\n\n"
        + format_performance_block(performance)
        + "\n\n## Bootstrapped logistic regression results\n\n"
        + format_results_table(results)
        + "\n"
    )
    with open(out_path, "w") as f:
        f.write(body)
