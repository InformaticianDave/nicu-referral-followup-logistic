# Changelog

## Phase 2 (Oct 2025 – present)
- Rebuilt the pipeline in Python: `src/referral_followup/` (data prep,
  bootstrapped logistic regression, markdown reporting, ROC plotting).
- Bootstrapped inference (200 resamples) replaces the classical Wald
  CI/p-value used in phase 1, given the sample size and sparse
  categorical cells.
- `scripts/run_analysis.py` runs both outcomes (`Referred_vs_NotReferred`,
  `Followup_vs_LTFU`) across all three variable groups
  (`clinical_continuous`, `clinical_categorical`, `sociodemographic`).
- Added ROC curve plotting, a pytest suite, a static PHI scanner, and
  a CI workflow.

## Phase 1 (Jul – Oct 2025)
- Initial R/caret exploration of a binary logistic regression for
  `Referred_vs_NotReferred` (`legacy_R/`).
- Diagnosed and fixed an unused-factor-levels bug causing
  `contrasts<-` errors.
- Built a full caret dummy-encoding + class-weighted `glm()` pipeline
  with an OR/CI/p-value table; fixed a non-syntactic column-name bug
  in the table construction.
- Compared a full-predictor-set model against a reduced "core
  predictor" model; decided to report the core set for interpretability
  and move to bootstrapped inference in phase 2.
