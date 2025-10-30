"""End-to-end: fit bootstrapped logistic regression for both outcomes
across all three variable groups, writing one markdown report per
(outcome, group) pair to outputs/tables/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from referral_followup import data_prep, bootstrap_logistic, reporting, roc

DATA_PATH = ROOT / "data" / "synthetic" / "referral_followup_synthetic.csv"
OUT_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"

GROUPS = ["clinical_continuous", "clinical_categorical", "sociodemographic"]
OUTCOMES = ["Referred_vs_NotReferred", "Followup_vs_LTFU"]


def main(n_boot: int = 200) -> None:
    df = pd.read_csv(DATA_PATH)
    config = data_prep.load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for outcome in OUTCOMES:
        for group in GROUPS:
            X, y, _ = data_prep.prepare_group(df, config, group, outcome)
            if y.nunique() < 2 or len(y) < 20:
                print(f"skip {outcome} / {group}: insufficient data ({len(y)} rows, {y.nunique()} classes)")
                continue

            result = bootstrap_logistic.fit_bootstrapped_logistic(X, y, n_boot=n_boot)
            out_path = OUT_DIR / f"{outcome}__{group}.md"
            reporting.write_group_report(
                outcome, group, result["results"], result["performance"], out_path
            )

            fig_path = FIG_DIR / f"ROC_{outcome}__{group}.png"
            title = f"ROC Curve — {outcome.replace('_', ' ')} ({group.replace('_', ' ')})"
            fig, roc_auc = roc.plot_roc(result["y_test"], result["y_proba"], title, out_path=fig_path)
            import matplotlib.pyplot as plt
            plt.close(fig)

            print(f"wrote {out_path} and {fig_path} (AUC={roc_auc:.3f})")


if __name__ == "__main__":
    main()
