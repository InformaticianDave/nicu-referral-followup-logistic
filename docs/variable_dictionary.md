# Variable dictionary

All variable names match the coded columns in the department's
cleaned discharge extract. No free-text, name, MRN, or date-of-service
field is read by this pipeline.

## Outcomes

| Variable | Levels | Notes |
|---|---|---|
| `Referred_vs_NotReferred` | Referred / NotReferred | Referred to outpatient developmental/coordination services at discharge |
| `Followup_vs_LTFU` | Followup / LTFU | Among referred infants, attended follow-up vs. lost to follow-up |

## Clinical — continuous

GA, delivery_bw, Length_of_stay, ageatdc, dc_eiatdc, dc_ptotatdc — see
`config/variable_groups.yml` for labels.

## Clinical — categorical

delivery_inborn_outborn, delivery_singlevsmultiple, dc_dispo,
patientinfo_sex, pmd_in_system, dc_wcconnect,
outpt_vv_vs_inperson_fup, outpt_dd, graduate.

## Sociodemographic

coi (ordinal Child Opportunity Index, Very low -> Very high),
Insurance_type, patientinfo_interpneed, mat_birthplace,
maternal_education, mat_employed_during_pregnancy, Race_maternal,
Ethnicity_maternal, pat_place_of_birth, Race_paternal,
Ethnicity_paternal, Race_patient.

## Reference categories

Every categorical predictor needs a baseline/reference level dropped
during dummy-encoding for the model to be identifiable. `effect`/`OR`
values in result tables are always relative to that reference (listed
per-variable in `config/variable_groups.yml`) -- e.g. an OR of 2.0 for
`Insurance_type_Private` means the odds of the outcome are twice as
high for infants with private insurance **compared to the reference
category** (public), holding other predictors constant.
