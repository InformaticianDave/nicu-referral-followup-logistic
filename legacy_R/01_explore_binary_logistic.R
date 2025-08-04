# 01_explore_binary_logistic.R
#
# First pass at a binary logistic regression for Referred_vs_NotReferred.
# Loads a cleaned extract (not committed -- see data-raw/README.md) and
# fits glm() directly on the raw predictor set.
#
# NOTE: this script reproduces an early bug (unused factor levels) left
# in place intentionally -- see 02_droplevels_fix.R / git history for
# the fix and writeup of why it happened.

library(dplyr)

df <- read.csv(file.path("data-raw", "local_extract.csv"), stringsAsFactors = TRUE)

predictor_vars <- c(
  "coi", "maternal_education", "Race_maternal", "Ethnicity_maternal",
  "pat_place_of_birth", "Race_paternal", "Race_patient",
  "GA", "Length_of_stay", "mat_years_in_us", "pat_time_in_us",
  "delivery_bw", "ageatdc"
)

formula_string <- paste("Referred_vs_NotReferred ~", paste(predictor_vars, collapse = " + "))
fml <- as.formula(formula_string)

fit_bin <- glm(fml, data = df, family = binomial(link = "logit"))
summary(fit_bin)
