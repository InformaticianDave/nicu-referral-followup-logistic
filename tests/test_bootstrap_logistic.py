import numpy as np
import pandas as pd
import pytest

from referral_followup import bootstrap_logistic


@pytest.fixture
def toy_xy():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 0.5 + 1.5 * x1 - 0.5 * x2
    p = 1 / (1 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, p))
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def test_results_have_expected_columns(toy_xy):
    X, y = toy_xy
    out = bootstrap_logistic.fit_bootstrapped_logistic(X, y, n_boot=30)
    assert list(out["results"].columns) == ["Effect", "CI_Lower", "CI_Upper", "PValue_Bootstrap"]
    assert set(out["results"].index) == {"x1", "x2"}


def test_ci_bounds_bracket_effect_or_are_sane(toy_xy):
    X, y = toy_xy
    out = bootstrap_logistic.fit_bootstrapped_logistic(X, y, n_boot=30)
    results = out["results"]
    assert (results["CI_Lower"] <= results["CI_Upper"]).all()
    assert (results["Effect"] > 0).all()


def test_strong_predictor_gets_low_pvalue(toy_xy):
    X, y = toy_xy
    out = bootstrap_logistic.fit_bootstrapped_logistic(X, y, n_boot=100, random_state=1)
    # x1 has a much larger true coefficient (1.5 vs -0.5) and should
    # separate from OR = 1 more reliably across bootstrap resamples.
    assert out["results"].loc["x1", "PValue_Bootstrap"] <= out["results"].loc["x2", "PValue_Bootstrap"] + 0.5


def test_performance_metrics_present(toy_xy):
    X, y = toy_xy
    out = bootstrap_logistic.fit_bootstrapped_logistic(X, y, n_boot=20)
    for key in ["roc_auc", "accuracy", "precision", "recall", "n_boot", "n_train", "n_test"]:
        assert key in out["performance"]
    assert 0 <= out["performance"]["roc_auc"] <= 1
