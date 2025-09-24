"""Generate a fully synthetic fixture matching the shape of the real
discharge-referral extract, for development and tests only.

No row in this file corresponds to a real infant. Values are drawn
from independent distributions chosen to look plausible, with a small
amount of engineered signal so the model has something non-trivial to
fit. Subject keys are sequential synthetic IDs (SYN0001, ...), never
MRNs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 4071
N = 420

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "referral_followup_synthetic.csv"


def make_synthetic(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ga = rng.normal(32, 4, n).clip(23, 42).round(1)
    bw = (ga * 100 + rng.normal(0, 300, n)).clip(400, 4500).round(0)
    los = (42 - ga) * rng.uniform(2.5, 4.5, n) + rng.normal(0, 5, n)
    los = los.clip(1, 180).round(0)
    age_at_dc = (los + rng.normal(0, 3, n)).clip(1, 200).round(0)
    ei_at_dc = rng.poisson(1.2, n)
    pt_ot_at_dc = rng.poisson(0.8, n)

    sex = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    inborn = rng.choice(["Inborn", "Outborn"], n, p=[0.8, 0.2])
    multiple = rng.choice(["Singleton", "Multiple"], n, p=[0.75, 0.25])
    dispo = rng.choice(["Home", "Transfer", "Other"], n, p=[0.85, 0.1, 0.05])
    pmd_in_system = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    wc_connect = rng.choice(["Yes", "No"], n, p=[0.55, 0.45])
    visit_mode = rng.choice(["In-person", "Video"], n, p=[0.7, 0.3])
    dd_flag = rng.choice(["Yes", "No"], n, p=[0.2, 0.8])
    graduate = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])

    coi = rng.choice(
        ["Very low", "Low", "Moderate", "High", "Very high"], n,
        p=[0.15, 0.2, 0.3, 0.2, 0.15],
    )
    insurance = rng.choice(["Public", "Private"], n, p=[0.55, 0.45])
    interp_need = rng.choice(["Yes", "No"], n, p=[0.25, 0.75])
    mat_birthplace = rng.choice(["US", "Non-US"], n, p=[0.7, 0.3])
    pat_birthplace = rng.choice(["US", "Non-US"], n, p=[0.7, 0.3])
    mat_edu = rng.choice(
        ["High school or less", "Some college", "College graduate", "Graduate degree"], n,
        p=[0.3, 0.25, 0.3, 0.15],
    )
    mat_employed = rng.choice(["Yes", "No"], n, p=[0.55, 0.45])

    race_choices = [
        "American Indian or Alaska Native", "Asian", "Black or African American",
        "Native Hawaiian or Other Pacific Islander", "White", "Other",
    ]
    race_p = [0.01, 0.1, 0.25, 0.02, 0.5, 0.12]
    race_patient = rng.choice(race_choices, n, p=race_p)
    race_maternal = rng.choice(race_choices, n, p=race_p)
    race_paternal = rng.choice(race_choices, n, p=race_p)
    eth_maternal = rng.choice(["Hispanic/Latino", "Not Hispanic/Latino"], n, p=[0.25, 0.75])
    eth_paternal = rng.choice(["Hispanic/Latino", "Not Hispanic/Latino"], n, p=[0.25, 0.75])

    # --- Engineer a little real signal so the pipeline has something to find ---
    coi_rank = pd.Series(coi).map(
        {"Very low": 0, "Low": 1, "Moderate": 2, "High": 3, "Very high": 4}
    ).to_numpy()

    referral_logit = (
        -1.0
        + 0.35 * (ga < 30)
        + 0.5 * (dd_flag == "Yes")
        + 0.3 * (ei_at_dc > 1)
        - 0.2 * coi_rank
        + rng.normal(0, 0.6, n)
    )
    referral_prob = 1 / (1 + np.exp(-referral_logit))
    referred = rng.binomial(1, referral_prob)

    followup_logit = (
        0.8
        + 0.6 * (pmd_in_system == "Yes")
        + 0.5 * (wc_connect == "Yes")
        - 0.6 * (interp_need == "Yes")
        + 0.25 * coi_rank
        + rng.normal(0, 0.7, n)
    )
    followup_prob = 1 / (1 + np.exp(-followup_logit))
    followup_draw = rng.binomial(1, followup_prob).astype(float)
    followup = np.where(referred == 1, followup_draw, np.nan)

    df = pd.DataFrame({
        "subject_id": [f"SYN{i:04d}" for i in range(1, n + 1)],
        "GA": ga,
        "delivery_bw": bw,
        "Length_of_stay": los,
        "ageatdc": age_at_dc,
        "dc_eiatdc": ei_at_dc,
        "dc_ptotatdc": pt_ot_at_dc,
        "patientinfo_sex": sex,
        "delivery_inborn_outborn": inborn,
        "delivery_singlevsmultiple": multiple,
        "dc_dispo": dispo,
        "pmd_in_system": pmd_in_system,
        "dc_wcconnect": wc_connect,
        "outpt_vv_vs_inperson_fup": visit_mode,
        "outpt_dd": dd_flag,
        "graduate": graduate,
        "coi": coi,
        "Insurance_type": insurance,
        "patientinfo_interpneed": interp_need,
        "mat_birthplace": mat_birthplace,
        "pat_place_of_birth": pat_birthplace,
        "maternal_education": mat_edu,
        "mat_employed_during_pregnancy": mat_employed,
        "Race_patient": race_patient,
        "Race_maternal": race_maternal,
        "Race_paternal": race_paternal,
        "Ethnicity_maternal": eth_maternal,
        "Ethnicity_paternal": eth_paternal,
        "Referred_vs_NotReferred": np.where(referred == 1, "Referred", "NotReferred"),
        "Followup_vs_LTFU": [
            None if pd.isna(v) else ("Followup" if v == 1 else "LTFU")
            for v in followup
        ],
    })
    return df


if __name__ == "__main__":
    data = make_synthetic()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(data)} synthetic rows to {OUT_PATH}")
