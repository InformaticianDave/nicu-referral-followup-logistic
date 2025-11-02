import pandas as pd

from referral_followup import reporting


def test_format_results_table_has_expected_headers():
    results = pd.DataFrame(
        {
            "Effect": [1.5, 0.8],
            "CI_Lower": [1.1, 0.5],
            "CI_Upper": [2.0, 1.1],
            "PValue_Bootstrap": [0.02, 0.3],
        },
        index=["coi", "Insurance_type_Private"],
    )
    md = reporting.format_results_table(results)
    assert "Feature" in md
    assert "Effect" in md
    assert "95% CI Lower" in md
    assert "95% CI Upper" in md
    assert "P-value (Bootstrap)" in md
    assert "coi" in md


def test_format_performance_block_contains_metrics():
    performance = {
        "roc_auc": 0.9,
        "accuracy": 0.8,
        "precision": 0.7,
        "recall": 0.6,
        "n_boot": 200,
        "n_train": 300,
        "n_test": 120,
    }
    block = reporting.format_performance_block(performance)
    assert "ROC AUC" in block
    assert "0.900" in block
    assert "200" in block
