# ============================================================
# nhanes_analysis.R
# Reproduces the NHANES real-data analysis from the paper
# (Section 5: blood pressure classification)
#
# Uses pre-processed data bundled in inst/extdata/:
#   internal_cleaned.csv    -- primary data (9186 units, 14 covariates)
#   internal_pred_full.csv  -- primary data with 3-class qhat from XGBoost
#   external_cleaned.csv    -- external data (12425 units, 8 covariates)
#
# Fits MLE and FMLE on two subsets:
#   fasting_all  -- all 9186 internal rows
#   random_600   -- random 600 rows (seed=42)
#
# Output: coefficient comparison data frame
# ============================================================

library(MLfused)
library(nnet)

# ---- Load data ----
data_path <- system.file("extdata", "internal_pred_full.csv", package = "MLfused")
if (data_path == "") stop("NHANES data not found. Reinstall MLfused.")

data <- read.csv(data_path, stringsAsFactors = FALSE)
data$sex    <- as.factor(data$sex)
data$race   <- as.factor(data$race)
data$bp_cat <- factor(data$bp_cat, levels = c("Normal", "Prehypertension", "Hypertension"))

# ---- Design matrix ----
vars_shared <- c("age", "sex", "race", "income_pir",
                 "bmi", "waist_cm", "height_cm", "weight_kg",
                 "glucose_mg_dl_log1p", "insulin_uU_mL_log1p",
                 "triglycerides_mg_dl_log1p", "ldl_mg_dl_log1p",
                 "hdl_mg_dl_log1p", "total_chol_mg_dl_log1p")
var_conti <- c("age", "income_pir", "bmi", "waist_cm", "height_cm", "weight_kg")

X_raw    <- model.matrix(~ . - 1, data = data[, vars_shared, drop = FALSE])
X_scaled <- scale(X_raw)
y        <- as.integer(data$bp_cat)
x_names  <- colnames(X_scaled)

# qhat: 3-class (L=3), use prehyp and hyp columns (Normal is reference)
qhat    <- as.matrix(data[, c("qhat_prehyp", "qhat_hyp")])
groups  <- list(c(1), c(2), c(3))
Lm1     <- length(groups) - 1L

# ---- Define subsets ----
set.seed(42)
subsets <- list(
  fasting_all = seq_len(nrow(X_scaled)),
  random_600  = sort(sample(seq_len(nrow(X_scaled)), 600, replace = FALSE))
)

# ---- Fit each subset ----
results <- list()

for (subset_name in names(subsets)) {
  idx <- subsets[[subset_name]]
  n_sub <- length(idx)
  message("\n--- ", subset_name, " (n=", n_sub, ") ---")

  X_sub    <- X_scaled[idx, , drop = FALSE]
  y_sub    <- y[idx]
  qhat_sub <- qhat[idx, , drop = FALSE]
  X_conti_sub <- as.data.frame(X_raw[idx, var_conti, drop = FALSE])

  # -- MLE --
  fit_mle <- nnet::multinom(y_sub ~ X_sub, trace = FALSE)
  cf <- coef(fit_mle)
  if (is.vector(cf)) cf <- matrix(cf, nrow = 1)
  beta_mle  <- cf[, "(Intercept)"]
  Theta_mle <- t(cf[, colnames(cf) != "(Intercept)", drop = FALSE])

  se_cf <- summary(fit_mle)$standard.errors
  if (is.vector(se_cf)) se_cf <- matrix(se_cf, nrow = 1)
  se_beta_mle  <- se_cf[, "(Intercept)"]
  se_Theta_mle <- t(se_cf[, colnames(se_cf) != "(Intercept)", drop = FALSE])

  message("  MLE done")

  # -- FMLE --
  H_base  <- build_H(X_conti_sub, phi_idx = seq_along(var_conti), n.q = 0)
  Hmat    <- build_Hmat_interleaved(H_base, Lm1)

  alpha0 <- rep(0, ncol(Theta_mle))
  tmat0  <- matrix(0, nrow = Lm1, ncol = ncol(Hmat) / Lm1)
  par0   <- pack_hard(beta_mle, Theta_mle, alpha0, tmat0)

  fit_fmle <- tryCatch(
    ml_fused(
      par = par0, X = X_sub, y = y_sub,
      qhat = qhat_sub, Hmat = Hmat,
      groups = groups,
      maxit = 500, tol = 1e-6,
      learning.rate = 0.1, tau_tmat = 0.1
    ),
    error = function(e) { message("  FMLE ERROR: ", e$message); NULL }
  )

  if (!is.null(fit_fmle)) {
    message("  FMLE done (conv=", fit_fmle$conv,
            ", iter=", fit_fmle$best_iter, ")")
  }

  # -- Collect results --
  K_minus1 <- length(beta_mle)
  coef_names <- c("intercept", x_names)

  for (ki in seq_len(K_minus1)) {
    # MLE
    est_mle <- c(beta_mle[ki], Theta_mle[, ki])
    se_mle  <- c(se_beta_mle[ki], se_Theta_mle[, ki])

    results[[length(results) + 1]] <- data.frame(
      subset    = subset_name,
      method    = "MLE",
      class_idx = ki + 1L,
      coef_name = coef_names,
      estimate  = est_mle,
      std_error = se_mle,
      stringsAsFactors = FALSE
    )

    # FMLE
    if (!is.null(fit_fmle)) {
      est_fmle <- c(fit_fmle$par$beta[ki], fit_fmle$par$Theta[, ki])
      se_fmle  <- c(fit_fmle$se$beta[ki], fit_fmle$se$Theta[, ki])

      results[[length(results) + 1]] <- data.frame(
        subset    = subset_name,
        method    = "FMLE",
        class_idx = ki + 1L,
        coef_name = coef_names,
        estimate  = est_fmle,
        std_error = se_fmle,
        stringsAsFactors = FALSE
      )
    }
  }
}

results_df <- do.call(rbind, results)
results_df$ci_lo <- results_df$estimate - 1.96 * results_df$std_error
results_df$ci_hi <- results_df$estimate + 1.96 * results_df$std_error

message("\nDone. Results in 'results_df'.")
