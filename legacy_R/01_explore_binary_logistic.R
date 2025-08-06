# 01_explore_binary_logistic.R
#
# First pass at a binary logistic regression for Referred_vs_NotReferred.
# Loads a cleaned extract (not committed -- see data-raw/README.md) and
# fits glm() on the raw predictor set.
#
# FIX (2025-08-06): several categorical predictors -- coi,
# maternal_education, Race_maternal, Ethnicity_maternal,
# pat_place_of_birth, Race_paternal, Race_patient -- had factor levels
# that were defined but represented by zero rows in this cohort (e.g.
# a race category with no infants in it). glm() tries to build dummy
# columns for every defined level, so it throws:
#   Error in contrasts<-: contrasts can be applied only to factors with
#   2 or more levels
# Dropping unused levels before fitting resolves it. Continuous
# predictors (GA, Length_of_stay, mat_years_in_us, pat_time_in_us,
# delivery_bw, ageatdc) aren't factors, so droplevels() is a no-op on
# them -- the is.factor() guard below just makes that explicit.

library(dplyr)

df <- read.csv(file.path("data-raw", "local_extract.csv"), stringsAsFactors = TRUE)

predictor_vars <- c(
  "coi", "maternal_education", "Race_maternal", "Ethnicity_maternal",
  "pat_place_of_birth", "Race_paternal", "Race_patient",
  "GA", "Length_of_stay", "mat_years_in_us", "pat_time_in_us",
  "delivery_bw", "ageatdc"
)

# Drop unused levels only for factor predictors; leave numeric columns untouched.
df <- df %>%
  mutate(across(all_of(predictor_vars), ~ if (is.factor(.x)) droplevels(.x) else .x))

# Sanity check: confirm no predictor is left with fewer than 2 observed levels/values.
check_levels <- sapply(df[predictor_vars], function(x) length(unique(na.omit(x))))
print(check_levels)

formula_string <- paste("Referred_vs_NotReferred ~", paste(predictor_vars, collapse = " + "))
fml <- as.formula(formula_string)

fit_bin <- glm(fml, data = df, family = binomial(link = "logit"))
summary(fit_bin)
