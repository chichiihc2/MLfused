#' Bootstrap Standard Errors
#'
#' Computes bootstrap SEs by resampling and refitting via \code{\link{ml_fused}}.
#' Returns SD-based and MAD-based robust SEs, plus percentile CIs.
#'
#' @param par_fit Fitted parameter list (from \code{ml_fused()$par}).
#' @param X n x p covariate matrix.
#' @param y Integer response vector.
#' @param qhat n x (L-1) external probability matrix.
#' @param Hmat Basis matrix.
#' @param groups Group assignment list.
#' @param B Number of bootstrap replicates (default 200).
#' @param maxit Max iterations per replicate (default 300).
#' @param tol Convergence tolerance (default 1e-4).
#' @param learning.rate Step size (default 0.1).
#' @param lambda.diag Ridge stabilization (default 0).
#' @param tau_tmat L2 penalty on tmat (default 1).
#'
#' @return List with \code{se}, \code{se_mad}, \code{q_lwr}, \code{q_upr},
#'   \code{n_kept}, \code{n_nonconv}, \code{n_error}.
#' @export
bootstrap_se <- function(par_fit, X, y, qhat, Hmat, groups,
                         B = 200, maxit = 300, tol = 1e-4,
                         learning.rate = 0.1,
                         lambda.diag = 0, tau_tmat = 1) {
  n <- nrow(X)
  A <- group_matrix(groups, length(unique(y)))

  tmat0 <- matrix(0, nrow = nrow(par_fit$tmat), ncol = ncol(par_fit$tmat))
  par0  <- pack_hard(par_fit$beta, par_fit$Theta, par_fit$alpha, tmat0)
  p <- ncol(X); K <- length(unique(y)); L <- ncol(A)
  Lm1 <- L - 1; H <- ncol(Hmat) / Lm1
  attributes(par0)$dims <- list(p = p, K = K, L = L, H = H)

  Dnoq <- length(par0)
  boot_pars <- matrix(NA_real_, nrow = B, ncol = Dnoq)
  n_error <- 0L; n_nonconv <- 0L

  for (b in seq_len(B)) {
    idx <- sample.int(n, size = n, replace = TRUE)
    Xb <- X[idx, , drop = FALSE]; yb <- y[idx]
    qb <- qhat[idx, , drop = FALSE]
    Hb <- Hmat[idx, , drop = FALSE]

    fit_b <- tryCatch(
      ml_fused(par = par0, X = Xb, y = yb, qhat = qb,
               Hmat = Hb, groups = groups,
               maxit = maxit, tol = tol,
               learning.rate = learning.rate,
               lambda.diag = lambda.diag,
               compute_se = FALSE, tau_tmat = tau_tmat),
      error = function(e) NULL
    )
    if (is.null(fit_b)) {
      n_error <- n_error + 1L
    } else if (fit_b$conv != 0L) {
      n_nonconv <- n_nonconv + 1L
    } else {
      pb <- pack_hard(fit_b$par$beta, fit_b$par$Theta,
                      fit_b$par$alpha, fit_b$par$tmat)
      boot_pars[b, ] <- as.numeric(pb)
    }
  }

  keep <- complete.cases(boot_pars)
  bp_kept <- boot_pars[keep, , drop = FALSE]

  boot_stats <- apply(bp_kept, 2, function(x) {
    c(sd = sd(x), quantile(x, c(0.025, 0.975)))
  })
  se_boot <- boot_stats["sd", ]
  q_lwr   <- boot_stats["2.5%", ]
  q_upr   <- boot_stats["97.5%", ]
  attributes(se_boot) <- attributes(par0)
  attributes(q_lwr)   <- attributes(par0)
  attributes(q_upr)   <- attributes(par0)

  se_boot_mad <- apply(bp_kept, 2, function(x) mad(x, constant = 1.4826))
  attributes(se_boot_mad) <- attributes(par0)

  list(se = se_boot, se_mad = se_boot_mad,
       q_lwr = q_lwr, q_upr = q_upr,
       n_kept = sum(keep), n_nonconv = n_nonconv, n_error = n_error)
}


#' Diagnostic Bootstrap with Per-Replicate Tracking
#'
#' Extended bootstrap that records convergence status, gradient norms,
#' iteration counts, objective values, and error messages per replicate.
#'
#' @inheritParams bootstrap_se
#' @param B Number of replicates (default 500).
#' @param store_indices Store bootstrap sample indices (default FALSE).
#'
#' @return List with summary stats and per-replicate diagnostics.
#' @export
bootstrap_se_diagnostic <- function(par_fit, X, y, qhat, Hmat, groups,
                                    B = 500, maxit = 300, tol = 1e-4,
                                    learning.rate = 0.1,
                                    lambda.diag = 0, store_indices = FALSE,
                                    tau_tmat = 1) {
  n <- nrow(X)
  A <- group_matrix(groups, length(unique(y)))

  tmat0 <- matrix(0, nrow = nrow(par_fit$tmat), ncol = ncol(par_fit$tmat))
  par0  <- pack_hard(par_fit$beta, par_fit$Theta, par_fit$alpha, tmat0)
  p <- ncol(X); K <- length(unique(y)); L <- ncol(A)
  Lm1 <- L - 1; H <- ncol(Hmat) / Lm1
  attributes(par0)$dims <- list(p = p, K = K, L = L, H = H)

  Dnoq <- length(par0)
  boot_pars      <- matrix(NA_real_, nrow = B, ncol = Dnoq)
  boot_conv      <- rep(NA_integer_, B)
  boot_grad_norm <- rep(NA_real_, B)
  boot_n_iter    <- rep(NA_integer_, B)
  boot_obj       <- rep(NA_real_, B)
  boot_error_msg <- rep(NA_character_, B)
  boot_indices   <- if (store_indices) vector("list", B) else NULL

  for (b in seq_len(B)) {
    idx <- sample.int(n, size = n, replace = TRUE)
    if (store_indices) boot_indices[[b]] <- idx

    Xb <- X[idx, , drop = FALSE]; yb <- y[idx]
    qb <- qhat[idx, , drop = FALSE]
    Hb <- Hmat[idx, , drop = FALSE]

    fit_b <- tryCatch(
      ml_fused(par = par0, X = Xb, y = yb, qhat = qb,
               Hmat = Hb, groups = groups,
               maxit = maxit, tol = tol,
               learning.rate = learning.rate,
               lambda.diag = lambda.diag,
               compute_se = FALSE, tau_tmat = tau_tmat),
      error = function(e) { attr(e, "msg") <- conditionMessage(e); e }
    )

    if (inherits(fit_b, "error")) {
      boot_error_msg[b] <- conditionMessage(fit_b)
      boot_conv[b] <- -1L
    } else {
      boot_conv[b]      <- fit_b$conv
      boot_grad_norm[b] <- fit_b$best_grad_norm
      boot_n_iter[b]    <- fit_b$best_iter
      boot_obj[b]       <- tail(fit_b$obj.record, 1)
      pb <- pack_hard(fit_b$par$beta, fit_b$par$Theta,
                      fit_b$par$alpha, fit_b$par$tmat)
      boot_pars[b, ] <- as.numeric(pb)
    }
  }

  converged <- (boot_conv == 0L) & !is.na(boot_conv)
  bp_conv   <- boot_pars[converged, , drop = FALSE]

  if (nrow(bp_conv) > 1) {
    boot_stats <- apply(bp_conv, 2, function(x) {
      c(sd = sd(x), quantile(x, c(0.025, 0.975)))
    })
    se_boot <- boot_stats["sd", ]
    q_lwr   <- boot_stats["2.5%", ]
    q_upr   <- boot_stats["97.5%", ]
    attributes(se_boot) <- attributes(par0)
    attributes(q_lwr)   <- attributes(par0)
    attributes(q_upr)   <- attributes(par0)
  } else {
    se_boot <- q_lwr <- q_upr <- rep(NA_real_, Dnoq)
    attributes(se_boot) <- attributes(par0)
    attributes(q_lwr)   <- attributes(par0)
    attributes(q_upr)   <- attributes(par0)
  }

  list(
    se = se_boot, q_lwr = q_lwr, q_upr = q_upr,
    n_kept = sum(converged),
    n_nonconv = sum(boot_conv == 1L, na.rm = TRUE),
    n_error = sum(boot_conv == -1L, na.rm = TRUE),
    boot_pars = boot_pars, boot_conv = boot_conv,
    boot_grad_norm = boot_grad_norm, boot_n_iter = boot_n_iter,
    boot_obj = boot_obj, boot_error_msg = boot_error_msg,
    boot_indices = boot_indices, par0_attrs = attributes(par0)
  )
}
