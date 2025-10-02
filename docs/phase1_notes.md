# Phase 1 notes: R exploration (Jul-Oct 2025)

Summary of the exploratory work in `legacy_R/` before the pipeline was
rebuilt in Python for Phase 2.

## What this phase established

- A full predictor set (clinical + sociodemographic + the raw
  multi-select referral-indication/additional-indication columns)
  produces a high-dimensional model: after dummy-encoding, the number
  of columns approaches the number of informative rows, which risks
  overfitting and unstable p-values, especially for sparse categorical
  cells (e.g. a race/ethnicity combination with only a few infants).
- Two candidate result sets were compared:
  - **Full model** (all raw predictors, including the 19
    `referral_indications___*` and 16 `additional___*` multi-select
    columns) -- large odds ratios, several implausibly large
    coefficients, consistent with overfitting on sparse dummy columns.
  - **Reduced/core model** (the smaller set used in a parallel
    multinomial analysis -- GA, insurance, interpreter need, maternal
    education, discharge disposition, and similar core clinical/
    sociodemographic predictors) -- more stable estimates, easier for
    non-statistician clinicians to interpret, better suited for
    publication.
- Conclusion carried into Phase 2: run the full model to characterize
  AUC and rank predictors, but report the **core predictor set**
  results to clinical collaborators, and move to **bootstrapped**
  rather than classical (Wald) confidence intervals and p-values given
  the modest sample size and sparse categorical cells (see
  `docs/methods.md`).

## Known limitations carried forward

- Mode/median imputation is simple and may bias results if missingness
  isn't random; a sensitivity analysis with multiple imputation was
  identified as a future improvement, not implemented here.
- Several predictors are collinear (maternal vs. paternal education,
  insurance type vs. the raw private/public/other flags, GA vs. age at
  discharge vs. birth weight). Multicollinearity doesn't crash `glm()`
  but does inflate standard errors -- another reason to prefer
  bootstrap-based inference over the classical normal approximation.
