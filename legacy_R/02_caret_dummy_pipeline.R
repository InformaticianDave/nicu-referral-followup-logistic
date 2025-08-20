# 02_caret_dummy_pipeline.R
#
# Full pipeline: nominal/continuous/ordinal/binary predictor typing,
# mode/median imputation, caret dummy-encoding + center/scale, a
# class-weighted glm(), train/test AUC, and an OR/CI/p-value table.
#
# Odds-ratio / 95% CI / p-value table, sorted by significance, ranked
# top-20 for a clinician-facing summary.

library(readr)
library(dplyr)
library(caret)
library(pROC)
library(data.table)
library(e1071)   # required for the "dummy" preProcess method
library(stringr)

df <- read.csv(file.path("data-raw", "local_extract.csv"), stringsAsFactors = FALSE)

referral_cols <- paste0("referral_indications___", 1:19)
additional_cols <- paste0("additional___", 16:31)
iugr_cols <- c("delivery_iugrsga___1", "delivery_iugrsga___2", "delivery_iugrsga___3")
private_public_other_cols <- paste0("privatepublicother___", 1:3)

predictor_vars <- c(
  "coi", "Insurance_type", "patientinfo_interpneed",
  "mat_birthplace", "mat_years_in_us", "maternal_education",
  "mat_employed_during_pregnancy", "Race_maternal", "Ethnicity_maternal",
  "mat_wic_yn", "pat_place_of_birth", "pat_time_in_us",
  "pat_edu", "Race_paternal", "Ethnicity_paternal",
  "delivery_inborn_outborn", "GA", "delivery_singlevsmultiple",
  "dc_dispo", "Length_of_stay",
  "patientinfo_sex", private_public_other_cols, "Race_patient",
  "mat_edu_collapsed", "pat_edu_collapsed",
  "mat_edu_group", "pat_edu_group",
  "delivery_bw", "ageatdc", "dc_eiatdc", "dc_ptotatdc",
  "dc_slpreferralatdc", "pmd_in_system", "dc_wcconnect",
  "outpt_vv_vs_inperson_fup", "outpt_dd", "graduate",
  referral_cols, additional_cols, iugr_cols
)

target_col <- "Referred_vs_NotReferred"
all_cols <- c(target_col, predictor_vars)
existing_cols <- all_cols[all_cols %in% names(df)]
predictor_vars <- predictor_vars[predictor_vars %in% names(df)]

df_analysis <- df %>%
  select(all_of(existing_cols)) %>%
  filter(!is.na(!!sym(target_col)))

df_analysis[[target_col]] <- factor(
  df_analysis[[target_col]],
  levels = c("Not Referred", "Referred"),
  labels = c("NotReferred", "Referred")
)

nominal_cols <- c(
  "Insurance_type", "patientinfo_interpneed", "mat_birthplace", "maternal_education",
  "mat_employed_during_pregnancy", "Race_maternal", "Ethnicity_maternal", "mat_wic_yn",
  "pat_place_of_birth", "pat_edu", "Race_paternal", "Ethnicity_paternal",
  "delivery_inborn_outborn", "delivery_singlevsmultiple", "dc_dispo",
  "patientinfo_sex", "Race_patient", "mat_edu_collapsed", "pat_edu_collapsed",
  "mat_edu_group", "pat_edu_group", "pmd_in_system", "dc_wcconnect",
  "outpt_vv_vs_inperson_fup", "outpt_dd", "graduate"
)
continuous_cols <- c(
  "mat_years_in_us", "pat_time_in_us", "GA", "Length_of_stay", "delivery_bw",
  "ageatdc", "dc_eiatdc", "dc_ptotatdc", "dc_slpreferralatdc"
)
ordinal_col <- "coi"
binary_cols <- setdiff(predictor_vars, c(nominal_cols, continuous_cols, ordinal_col))

mode_impute <- function(x) {
  ux <- na.omit(x)
  if (length(ux) == 0) return(NA)
  tab <- table(ux)
  names(tab)[which.max(tab)]
}

for (col in c(nominal_cols, binary_cols, ordinal_col)) {
  if (col %in% names(df_analysis)) {
    mode_val <- mode_impute(df_analysis[[col]])
    df_analysis[[col]][is.na(df_analysis[[col]])] <- mode_val
    df_analysis[[col]] <- factor(df_analysis[[col]])
  }
}
for (col in continuous_cols) {
  if (col %in% names(df_analysis)) {
    df_analysis[[col]][is.na(df_analysis[[col]])] <- median(df_analysis[[col]], na.rm = TRUE)
  }
}

coi_levels <- c("Very low", "Low", "Moderate", "High", "Very high")
df_analysis$coi <- factor(df_analysis$coi, levels = coi_levels, ordered = TRUE)

class_counts <- table(df_analysis[[target_col]])
class_weights <- 1 / class_counts
class_weights <- class_weights / sum(class_weights) * length(class_weights)
weights_vector <- as.vector(class_weights[df_analysis[[target_col]]])

set.seed(42)
trainIndex <- createDataPartition(df_analysis[[target_col]], p = .7, list = FALSE, times = 1)
df_train <- df_analysis[trainIndex, ]
train_weights <- weights_vector[trainIndex]

all_factors <- c(nominal_cols, binary_cols)
all_numeric <- continuous_cols

preProcValues <- preProcess(
  df_train[, c(all_factors, all_numeric)],
  method = c("center", "scale", "dummy")
)
df_train_processed <- predict(preProcValues, df_train)
df_train_processed$coi <- df_train$coi

is_constant <- apply(df_train_processed %>% select(-all_of(target_col)), 2, function(x) length(unique(x)) <= 1)
constant_cols <- names(df_train_processed %>% select(-all_of(target_col)))[is_constant]
df_train_processed <- df_train_processed %>% select(-all_of(constant_cols))

dt_train <- as.data.table(df_train_processed)
formula_str <- paste(target_col, "~", paste(names(dt_train)[!names(dt_train) %in% target_col], collapse = " + "))
model_formula <- as.formula(formula_str)

logreg_model <- glm(
  model_formula,
  data = df_train_processed,
  family = binomial,
  weights = train_weights,
  maxit = 100
)

coefs <- summary(logreg_model)$coefficients
cis <- confint.default(logreg_model, level = 0.95)

results_df <- data.frame(
  Feature = rownames(coefs),
  Coefficient = coefs[, "Estimate"],
  PValue = coefs[, "Pr(>|z|)"]
)
results_df$OddsRatio <- exp(results_df$Coefficient)

# FIX (2025-08-20): `95% CI Lower` / `95% CI Upper` are non-syntactic
# names (leading space, percent sign, digit) -- they must be backticked
# to be used as argument names in data.frame().
cis_df <- data.frame(Feature = rownames(cis), `95% CI Lower` = cis[, 1], `95% CI Upper` = cis[, 2], check.names = FALSE)
results_df <- merge(results_df, cis_df, by = "Feature", all.x = TRUE)

results_df$Feature <- gsub("`", "", results_df$Feature)
results_df <- results_df %>% filter(Feature != "(Intercept)") %>% arrange(PValue)

df_test <- df_analysis[-trainIndex, ]
df_test_processed <- predict(preProcValues, df_test)
df_test_processed <- df_test_processed %>% select(all_of(names(df_train_processed)))
df_test_processed$coi <- df_test$coi

test_probs <- predict(logreg_model, newdata = df_test_processed %>% select(-all_of(target_col)), type = "response")
auc_score <- auc(roc(df_test[[target_col]], test_probs))

top_20_results <- results_df %>%
  select(Feature, OddsRatio, `95% CI Lower`, `95% CI Upper`, PValue) %>%
  head(20)

cat("\n--- Model Performance ---\n")
cat("ROC AUC Score: ", round(auc_score, 4), "\n")
cat("\n--- Top 20 Predictors Ranked by P-value (< 0.1 for significance) ---\n")
print(top_20_results)
