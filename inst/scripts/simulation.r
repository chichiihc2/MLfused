# Simulation pipeline for MLfused package
# Reproduces the simulation study from the paper
# These functions require: MLfused, nnet, xgboost, dplyr, purrr, MASS

## --- 0. Libraries ---
library(MLfused)
library(MASS)      # mvrnorm
library(nnet)      # multinom
library(xgboost)   # external ML for qhat
library(dplyr)
library(purrr)


## --- 1. Utility: build_H() ---
build_H <- function(x_mat, phi_idx, n.q = 2) {
  # Natural spline version (cubic, natural boundary conditions)
  # n.q = number of INTERNAL knots per feature; if <= 0 -> x only (no splines)

  stopifnot(length(phi_idx) >= 1)
  if (!is.data.frame(x_mat)) x_mat <- as.data.frame(x_mat)

  phi <- x_mat[, phi_idx, drop = FALSE]
  n   <- nrow(phi)

  # Start with intercept
  H_list <- list(h0 = rep(1, n))
  knots_info <- vector("list", length(phi_idx))

  # q = 0 -> just intercept + raw x (if include_x = TRUE)
  if (n.q <= 0) {
    for (j in seq_along(phi_idx)) {
      H_list[[paste0("phi", j)]] <- phi[[j]]-mean(phi[[j]])
    }

    Hmat <- as.matrix(as.data.frame(H_list))
    attr(Hmat, "basis")  <- "natural_spline"
    attr(Hmat, "n.q")    <- n.q
    attr(Hmat, "knots")  <- NULL
    attr(Hmat, "degree") <- 3
    return(Hmat)
  }

  # q > 0 -> natural spline basis per feature
  if (!requireNamespace("splines", quietly = TRUE)) {
    stop("Package 'splines' is required. Please install.packages('splines').")
  }
  probs <- seq(1, n.q) / (n.q + 1)

  for (j in seq_along(phi_idx)) {
    xj <- phi[[j]]

    # Internal knots at quantiles; drop duplicates if data are discrete/tied
    kj  <- unique(stats::quantile(xj, probs = probs, na.rm = TRUE, names = FALSE))
    bnd <- range(xj, na.rm = TRUE)

    # Build natural spline basis (cubic, natural boundary)
    # If kj happens to be empty (e.g., ties), ns() still works.
    Bj <- splines::ns(xj, knots = if (length(kj)) kj else NULL,
                      Boundary.knots = bnd, intercept = F)
    Bj <- as.matrix(Bj)

    # Name columns
    if (ncol(Bj) > 0) {
      colnames(Bj) <- paste0("ns_phi", j, "_b", seq_len(ncol(Bj)))
      for (k in seq_len(ncol(Bj))) {
        H_list[[colnames(Bj)[k]]] <- Bj[, k]
      }
    }

    knots_info[[j]] <- list(
      feature        = phi_idx[j],
      knots          = kj,
      boundary_knots = bnd,
      degree         = 3,
      type           = "natural"
    )
  }

  Hmat <- as.matrix(as.data.frame(H_list))
  attr(Hmat, "basis")  <- "natural_spline"
  attr(Hmat, "n.q")    <- n.q
  attr(Hmat, "knots")  <- knots_info
  attr(Hmat, "degree") <- 3
  return(Hmat)
}


## --- 2. External Model Training (fixed coefficients) ---
train_external_source <- function(seed, phi_idx, n2 = 10000, bootstrap = TRUE, V.ext = 2,
                                   ml_method = "xgboost", mu_ext = rep(0, p),
                                   Theta_true, beta_true, alpha_true, mu1.shift,
                                   rho, p, K) {
  # Uses explicit Theta_true parameter (no globals)

  # Independent seed for external data
  set.seed(seed + 1)

  Sigma <- matrix(rho, nrow = p, ncol = p); diag(Sigma) <- 1
  X2 <- mvrnorm(n2, mu = mu_ext, Sigma = V.ext * Sigma)

  eta2  <- X2 %*% Theta_true + matrix(rep(alpha_true, each = n2), nrow = n2)
  prob2 <- softmax_first(eta2)
  G2    <- apply(prob2, 1, function(pp) sample.int(K, 1, prob = pp))
  W2    <- ifelse(G2 %in% c(1, 2), 1L, 2L)

  boot_models <- list()

  if (ml_method == "rf") {
    phi_cols <- paste0("phi", seq_along(phi_idx))
    df_train <- data.frame(
      W = factor(as.integer(W2)),
      setNames(as.data.frame(X2[, phi_idx, drop = FALSE]), phi_cols)
    )
    bst_main <- ranger::ranger(
      W ~ ., data = df_train,
      probability  = TRUE,
      keep.inbag   = TRUE,
      num.threads  = 1
    )
  } else {
    # XGBoost branch
    phiX2  <- X2[, phi_idx, drop = FALSE]
    dtrain <- xgb.DMatrix(data = phiX2, label = as.numeric(W2 == 1))

    # nthread = 1 to prevent parallel crashes
    params <- list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      max_depth = 4,
      eta = 0.1,
      subsample = 0.5,
      min_child_weight = 1,
      nthread = 1
    )

    bst_main <- xgb.train(params = params, data = dtrain, nrounds = 300, verbose = 0)

    if (bootstrap) {
      B.boot <- 200
      for (b in seq_len(B.boot)) {
        idx_b    <- sample.int(nrow(X2), size = nrow(X2), replace = TRUE)
        dtrain_b <- xgb.DMatrix(data = X2[idx_b, phi_idx, drop = FALSE],
                                label = as.numeric(W2[idx_b] == 1))
        boot_models[[b]] <- xgb.train(params = params, data = dtrain_b, nrounds = 300, verbose = 0)
      }
    }
  }

  rf_cols <- if (ml_method == "rf") phi_cols else NULL
  return(list(main = bst_main, boots = boot_models, ml_method = ml_method, rf_cols = rf_cols))
}

## --- 3. Prediction ---
predict.external_probs <- function(models, X1, phi_idx) {
  if (models$ml_method == "rf") {
    df_new <- setNames(as.data.frame(X1[, phi_idx, drop = FALSE]), models$rf_cols)
    pred   <- predict(models$main, data = df_new, type = "se", se.method = "infjack")
    # pred$predictions: n x 2 (columns named "1" and "2")
    # pred$se:          n x 2 SEs for each probability
    q1_main <- pred$predictions[, "1"]
    qhat    <- cbind(q1_main, 1 - q1_main)
    sd_q1   <- pred$se[, "1"]
    qhat_se <- cbind(sd_q1, sd_q1)
  } else {
    dnew    <- xgb.DMatrix(data = X1[, phi_idx, drop = FALSE])
    q1_main <- as.numeric(predict(models$main, newdata = dnew))
    qhat    <- cbind(q1_main, 1 - q1_main)

    qhat_se <- matrix(0, nrow = nrow(X1), ncol = 2)
    if (length(models$boots) > 0) {
      preds_boot <- sapply(models$boots, function(m) as.numeric(predict(m, newdata = dnew)))
      sd_q1  <- apply(preds_boot, 1, sd)
      qhat_se <- cbind(sd_q1, sd_q1)
    }
  }
  return(list(qhat = qhat, qhat.se = qhat_se))
}


## --- 4. Main Simulation Loop (fixed coefficients) ---
run_simulation <- function(seed,
                                          n1_grid,
                                          n.q_grid,
                                          shift.levels,
                                          phi_idx,
                                          bootstrap = FALSE,
                                          V.ext = 2,
                                          V.int= 1,
                                          rho=0.8,
                                          ml_method = "xgboost",
                                          maxit = 500,
                                          tol = 1e-6,
                                          learning.rate = 0.1,
                                          B_boot = 0,
                                          Theta_true,
                                          beta_true,
                                          alpha_true,
                                          mu1.shift,
                                          p,
                                          K) {

  n2 <- 10000

  # A. Train External Model (ONCE per seed) — uses explicit Theta_true
  ext_models <- train_external_source(seed, phi_idx, bootstrap = bootstrap, V.ext = V.ext,
                                       ml_method = ml_method,
                                       Theta_true = Theta_true, beta_true = beta_true,
                                       alpha_true = alpha_true, mu1.shift = mu1.shift,
                                       rho = rho, p = p, K = K)

  out_list    <- list()
  out_ci_list <- list()
  counter     <- 1

  # B. Loop over Sample Sizes (n1)
  for (n1 in n1_grid) {

    # C. Loop over Shift Levels
    for (lvl in shift.levels) {

      # --- 1. Generate Internal Data ---
      mu.shift <- lvl * mu1.shift
      Sigma    <- matrix(rho, nrow = p, ncol = p); diag(Sigma) <- 1

      set.seed(seed + n1*10000 + lvl*1000)
      X1       <- mvrnorm(n1, mu = mu.shift, Sigma = V.int*Sigma)

      eta1  <- X1 %*% Theta_true + matrix(rep(beta_true, each = n1), nrow = n1)
      prob1 <- softmax_first(eta1)
      G1    <- apply(prob1, 1, function(pp) sample.int(K, 1, prob = pp))

      # --- 2. Predict qhat (Reused for all n.q) ---
      q_info  <- predict.external_probs(ext_models, X1, phi_idx)
      qhat    <- q_info$qhat[, 2, drop = FALSE]  # L=2: use qhat_2
      qhat.se <- q_info$qhat.se[, 2, drop = FALSE]

      # --- 3. Baseline Fit (Reused for all n.q) ---
      df1   <- data.frame(y = factor(G1), X1)
      fit0  <- nnet::multinom(y ~ ., data = df1, trace = FALSE)
      cf    <- coef(fit0); if (is.vector(cf)) cf <- matrix(cf, nrow = 1)

      beta0  <- cf[, "(Intercept)"]
      Theta0 <- t(cf[, colnames(cf) != "(Intercept)", drop = FALSE])

      # Get SEs for baseline (used for CI construction)
      summ0  <- summary(fit0)
      se_cf  <- summ0$standard.errors
      if (is.vector(se_cf)) se_cf <- matrix(se_cf, nrow = 1)
      se_beta0  <- se_cf[, "(Intercept)"]
      se_Theta0 <- t(se_cf[, colnames(se_cf) != "(Intercept)", drop = FALSE])

      # Prepare Original Fit Object
      alpha0    <- rep(0, length(beta0))
      se_alpha0 <- rep(NA_real_, length(beta0))

      fit.original <- list(
        par = list(beta = beta0, Theta = Theta0, alpha = alpha0),
        se  = list(beta = se_beta0, Theta = se_Theta0, alpha = se_alpha0)
      )

      # Original Error
      err_orig_mat  = (Theta0 - Theta_true)**2
      err_orig_int  = (beta0 - beta_true)**2
      err_orig_tot  = sum(err_orig_mat) + sum(err_orig_int)

      # D. Loop over n.q (Basis Dimensions)
      for (nq in n.q_grid) {

        # Build H
        H_X1 <- build_H(X1, phi_idx = phi_idx, n.q = nq)
        t0   <- matrix(0, nrow = ncol(qhat) , ncol = ncol(H_X1))

        # --- Fit Fused model ---
        par0.hard <- pack.hard( beta_true,Theta_true , alpha0, t0)

        fit.hard <- ML_fused_fast(
          par = par0.hard, X = X1, y = G1, qhat, qhat.se, Hmat = H_X1,
          groups = list(c(1, 2), c(3)),
          maxit = maxit, tol = tol,
          learning.rate = learning.rate, batch_frac = 1,
          lambda = n2, lambda.diag = 0
        )

        boot_diag <- list(n_kept = NA_integer_, n_nonconv = NA_integer_, n_error = NA_integer_)
        if (B_boot > 0) {
          boot_res <- bootstrap_hard_se_fast(
            par_fit = fit.hard$par, X = X1, y = G1,
            qhat = qhat, qhat.se = qhat.se, Hmat = H_X1,
            groups = list(c(1, 2), c(3)),
            B = B_boot, maxit = 500, tol = 1e-6,
            learning.rate = learning.rate, lambda = n2
          )
          boot_diag <- list(n_kept = boot_res$n_kept, n_nonconv = boot_res$n_nonconv, n_error = boot_res$n_error)
          se_boot_raw <- boot_res$se
          fit.hard$se.boot <- unpack.hard(se_boot_raw)

          correction_diag <- diag(fit.hard$conf$conf_matrix.adj - fit.hard$conf$conf_matrix)
          correction_diag <- pmax(correction_diag, 0)
          se_boot_adj_vec <- sqrt(as.numeric(se_boot_raw)^2 + correction_diag)
          attributes(se_boot_adj_vec) <- attributes(se_boot_raw)
          fit.hard$se.boot.adj <- unpack.hard(se_boot_adj_vec)

          # Quantile-based (percentile) bootstrap CIs
          fit.hard$ci.boot.q <- list(
            lwr = unpack.hard(boot_res$q_lwr),
            upr = unpack.hard(boot_res$q_upr)
          )
        }

        err.hard.beta  <- sum((fit.hard$par$beta  - beta_true)^2)
        err.hard.Theta <- sum((fit.hard$par$Theta - Theta_true)^2)

        out.hard <- data.frame(
          method      = "Fused.hard",
          seed        = seed,
          n1          = n1,
          n.q         = nq,
          shift.level = lvl,
          V.ext       = V.ext,
          error       = err.hard.beta + err.hard.Theta,
          error.beta  = err.hard.beta,
          error.Theta = err.hard.Theta,
          error.alpha = sum((fit.hard$par$alpha - alpha_true)^2),
          error.tmat  = NA_real_,
          conv        = fit.hard$conv,
          qhat.se.mean = mean(qhat.se)
        )

        out.orig <- data.frame(
          method      = "Original",
          seed        = seed,
          n1          = n1,
          n.q         = nq,
          shift.level = lvl,
          V.ext       = V.ext,
          error       = err_orig_tot,
          error.beta  = NA_real_,
          error.Theta = NA_real_,
          error.alpha = NA_real_,
          error.tmat  = NA_real_,
          conv        = NA,
          qhat.se.mean=NA
        )

        out_curr <- rbind(out.hard, out.orig)

        # --- Confidence Intervals ---
        fits <- list(Fused.hard = fit.hard, Original = fit.original)

        # Create CI table for this specific iteration
        ci_tbl <- purrr::imap_dfr(fits, ~ make_ci_tbl(.x, .y,
                                                       Theta_true = Theta_true,
                                                       beta_true = beta_true,
                                                       alpha_true = alpha_true))

        # Add identifiers to CI table
        ci_tbl$seed        <- seed
        ci_tbl$n1          <- n1
        ci_tbl$n.q         <- nq
        ci_tbl$shift.level <- lvl
        ci_tbl$V.ext       <- V.ext

        # Calculate coverage summary for this iteration
        ci_summary <- ci_tbl %>%
          filter(variable != "alpha") %>%
          group_by(method, adj) %>%
          summarise(
            sse        = sum((est - truth)**2),
            coverage   = mean(covered),
            avg_length = mean(length),
            sd_length  = sd(length),
            .groups    = "drop"
          )

        # Join summary back to 'out'
        out_curr <- out_curr %>% left_join(ci_summary, by = "method")

        out_curr$ml_method <- ml_method
        ci_tbl$ml_method   <- ml_method

        # Store results
        out_list[[counter]]    <- out_curr
        out_ci_list[[counter]] <- ci_tbl

        counter <- counter + 1
      } # end n.q
    } # end shift
  } # end n1

  return(list(
    out    = do.call(rbind, out_list),
    out_ci = do.call(rbind, out_ci_list),
    boot_diag = boot_diag
  ))
}


## --- 5. CI / Coverage Table ---
make_ci_tbl <- function(fit, method, Theta_true, beta_true, alpha_true) {
  z  <- qnorm(0.975)
  Kb <- length(fit$par$beta)      # K-1
  p  <- nrow(fit$par$Theta)

  safe_between <- function(x, a, b) {
    ifelse(is.na(x) | is.na(a) | is.na(b), NA, x >= a & x <= b)
  }

  # helper to build one CI table given a SE object (fit$se or fit$se.adj)
  build_ci_tbl_one <- function(se_obj, adj_flag) {
    # beta
    est_beta <- as.numeric(fit$par$beta)
    se_beta  <- as.numeric(se_obj$beta)
    lwr_b    <- est_beta - z * se_beta
    upr_b    <- est_beta + z * se_beta
    tb_beta <- tibble(
      method   = method,
      adj      = adj_flag,        # <--- indicate adjusted or not
      variable = "beta",
      i = seq_len(Kb), j = NA_integer_,
      est = est_beta, se = se_beta,
      lwr = lwr_b, upr = upr_b, length = upr_b - lwr_b,
      truth = as.numeric(beta_true),
      covered = safe_between(truth, lwr, upr)
    )

    # theta (vectorize column-major to match est)
    est_theta <- as.vector(fit$par$Theta)
    se_theta  <- as.vector(se_obj$Theta)
    lwr_t     <- est_theta - z * se_theta
    upr_t     <- est_theta + z * se_theta
    idx <- expand.grid(i = seq_len(p), j = seq_len(Kb))
    tb_theta <- tibble(
      method   = method,
      adj      = adj_flag,
      variable = "theta",
      i = idx$i, j = idx$j,
      est = est_theta, se = se_theta,
      lwr = lwr_t, upr = upr_t, length = upr_t - lwr_t,
      truth = as.vector(Theta_true),
      covered = safe_between(truth, lwr, upr)
    )

    # alpha
    est_alpha <- as.numeric(fit$par$alpha)
    se_alpha  <- as.numeric(se_obj$alpha)
    lwr_a     <- est_alpha - z * se_alpha
    upr_a     <- est_alpha + z * se_alpha
    tb_alpha <- tibble(
      method   = method,
      adj      = adj_flag,
      variable = "alpha",
      i = seq_len(Kb), j = NA_integer_,
      est = est_alpha, se = se_alpha,
      lwr = lwr_a, upr = upr_a, length = upr_a - lwr_a,
      truth = as.numeric(alpha_true),
      covered = safe_between(truth, lwr, upr)
    )

    bind_rows(tb_beta, tb_theta, tb_alpha)
  }

  # always include unadjusted SEs
  out <- build_ci_tbl_one(fit$se, adj_flag = "unadj")

  # if adjusted SEs are available, append them
  if (!is.null(fit$se.adj)) {
    out_adj <- build_ci_tbl_one(fit$se.adj, adj_flag = "adj")
    out <- bind_rows(out, out_adj)
  }

  if (!is.null(fit$se.adj.noq)) {
    out_adj <- build_ci_tbl_one(fit$se.adj.noq, adj_flag = "adj.noq")
    out <- bind_rows(out, out_adj)
  }

  if (!is.null(fit$se.score)) {
    out_score <- build_ci_tbl_one(fit$se.score, adj_flag = "score")
    out <- bind_rows(out, out_score)
  }

  if (!is.null(fit$se.score.adj)) {
    out_score_adj <- build_ci_tbl_one(fit$se.score.adj, adj_flag = "score.adj")
    out <- bind_rows(out, out_score_adj)
  }

  if (!is.null(fit$se.thm3)) {
    out_thm3 <- build_ci_tbl_one(fit$se.thm3, adj_flag = "thm3")
    out <- bind_rows(out, out_thm3)
  }

  if (!is.null(fit$se.thm3.adj)) {
    out_thm3_adj <- build_ci_tbl_one(fit$se.thm3.adj, adj_flag = "thm3.adj")
    out <- bind_rows(out, out_thm3_adj)
  }

  if (!is.null(fit$se.thm3.score)) {
    out <- bind_rows(out, build_ci_tbl_one(fit$se.thm3.score,     adj_flag = "thm3.score"))
  }
  if (!is.null(fit$se.thm3.score.adj)) {
    out <- bind_rows(out, build_ci_tbl_one(fit$se.thm3.score.adj, adj_flag = "thm3.score.adj"))
  }

  if (!is.null(fit$se.boot)) {
    out <- bind_rows(out, build_ci_tbl_one(fit$se.boot,     adj_flag = "boot"))
  }
  if (!is.null(fit$se.boot.adj)) {
    out <- bind_rows(out, build_ci_tbl_one(fit$se.boot.adj, adj_flag = "boot.adj"))
  }

  # Quantile (percentile) bootstrap CI — uses bootstrap quantiles directly
  if (!is.null(fit$ci.boot.q)) {
    Kb <- length(fit$par$beta)
    p  <- nrow(fit$par$Theta)

    lwr_q <- fit$ci.boot.q$lwr
    upr_q <- fit$ci.boot.q$upr

    # beta
    est_beta <- as.numeric(fit$par$beta)
    lwr_b <- as.numeric(lwr_q$beta);  upr_b <- as.numeric(upr_q$beta)
    tb_beta <- tibble(
      method = method, adj = "boot.q", variable = "beta",
      i = seq_len(Kb), j = NA_integer_,
      est = est_beta, se = (upr_b - lwr_b) / (2 * qnorm(0.975)),
      lwr = lwr_b, upr = upr_b, length = upr_b - lwr_b,
      truth = as.numeric(beta_true),
      covered = safe_between(truth, lwr, upr)
    )

    # theta
    est_theta <- as.vector(fit$par$Theta)
    lwr_t <- as.vector(lwr_q$Theta);  upr_t <- as.vector(upr_q$Theta)
    idx <- expand.grid(i = seq_len(p), j = seq_len(Kb))
    tb_theta <- tibble(
      method = method, adj = "boot.q", variable = "theta",
      i = idx$i, j = idx$j,
      est = est_theta, se = (upr_t - lwr_t) / (2 * qnorm(0.975)),
      lwr = lwr_t, upr = upr_t, length = upr_t - lwr_t,
      truth = as.vector(Theta_true),
      covered = safe_between(truth, lwr, upr)
    )

    # alpha
    est_alpha <- as.numeric(fit$par$alpha)
    lwr_a <- as.numeric(lwr_q$alpha);  upr_a <- as.numeric(upr_q$alpha)
    tb_alpha <- tibble(
      method = method, adj = "boot.q", variable = "alpha",
      i = seq_len(Kb), j = NA_integer_,
      est = est_alpha, se = (upr_a - lwr_a) / (2 * qnorm(0.975)),
      lwr = lwr_a, upr = upr_a, length = upr_a - lwr_a,
      truth = as.numeric(alpha_true),
      covered = safe_between(truth, lwr, upr)
    )

    out <- bind_rows(out, tb_beta, tb_theta, tb_alpha)
  }

  out
}
