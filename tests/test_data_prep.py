import pandas as pd
import pytest

from referral_followup import data_prep

ROOT_CONFIG = data_prep.CONFIG_PATH


@pytest.fixture
def synthetic_df():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "referral_followup_synthetic.csv"
    return pd.read_csv(path)


def test_config_loads():
    config = data_prep.load_config()
    assert "outcomes" in config
    assert "clinical_continuous" in config
    assert "sociodemographic" in config


@pytest.mark.parametrize("group", ["clinical_continuous", "clinical_categorical", "sociodemographic"])
def test_prepare_group_no_missing_values(synthetic_df, group):
    config = data_prep.load_config()
    X, y, feats = data_prep.prepare_group(synthetic_df, config, group, "Referred_vs_NotReferred")
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_coi_encoded_as_ordinal_rank(synthetic_df):
    config = data_prep.load_config()
    X, y, feats = data_prep.prepare_group(synthetic_df, config, "sociodemographic", "Referred_vs_NotReferred")
    assert "coi" in X.columns
    assert X["coi"].min() >= 0
    assert X["coi"].max() <= 4


def test_reference_category_dropped(synthetic_df):
    config = data_prep.load_config()
    X, y, feats = data_prep.prepare_group(synthetic_df, config, "sociodemographic", "Referred_vs_NotReferred")
    # Insurance_type reference is "Public" -- only the Private dummy should remain.
    insurance_cols = [c for c in X.columns if c.startswith("Insurance_type_")]
    assert "Insurance_type_Public" not in insurance_cols


def test_followup_outcome_drops_not_referred_rows(synthetic_df):
    config = data_prep.load_config()
    X, y, feats = data_prep.prepare_group(synthetic_df, config, "clinical_continuous", "Followup_vs_LTFU")
    # Followup_vs_LTFU is only defined for referred infants.
    assert len(y) < len(synthetic_df)
    assert len(y) == synthetic_df["Followup_vs_LTFU"].isin(["Followup", "LTFU"]).sum()
