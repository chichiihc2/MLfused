#' Bootstrap Standard Errors for Hard-Mode Fused Multinomial Model
#'
#' Computes bootstrap standard errors for the hard-mode (fixed-q)
#' empirical-likelihood fused multinomial model by resampling observations
#' and refitting via \code{\link{ML_fused}}.
#'
#' @param par_fit A fitted parameter list as returned by \code{ML_fused()$par},
#'   containing elements \code{beta}, \code{Theta}, \code{alpha}, and \code{tmat}.
#' @param X Numeric matrix of covariates (n x p).
#' @param y Integer vector of class labels (length n).
#' @param qhat Numeric matrix of external predicted probabilities for non-reference
#'   groups (n x (L-1)).
#' @param qhat.se Numeric matrix of standard errors for \code{qhat} (n x (L-1)).
#' @param Hmat Numeric basis matrix for the EL constraint (n x ((L-1)*H)).
#' @param groups Integer vector of group assignments (length n).
#' @param B Integer number of bootstrap replicates (default 200).
#' @param maxit Integer maximum iterations per bootstrap fit (default 100).
#' @param tol Numeric convergence tolerance (default 1e-4).
#' @param learning.rate Numeric step-size scaling factor (default 0.1).
#' @param lambda Numeric regularization parameter for the q-penalty. If \code{NULL},
#'   defaults to \code{n}.
#' @param lambda.diag Numeric ridge stabilization added to the Hessian diagonal
#'   (default 0).
#'
#' @return A numeric vector of bootstrap standard errors with the same length and
#'   attributes as the packed hard parameter vector.
#'
#' @export
bootstrap_hard_se <- function(par_fit, X, y, qhat, qhat.se, Hmat, groups,
                               B = 200, maxit = 100, tol = 1e-4,
                               learning.rate = 0.1, lambda = NULL,
                               lambda.diag = 0) {
  n <- nrow(X)
  if (is.null(lambda)) lambda <- n
  A <- group_matrix(groups, length(unique(y)))

  # Warm-start: pack fitted params into hard format
  par0 <- pack.hard(par_fit$beta, par_fit$Theta, par_fit$alpha, par_fit$tmat)
  p   <- ncol(X); K <- length(unique(y)); L <- ncol(A)
  Lm1 <- L - 1; H <- ncol(Hmat) / Lm1
  attributes(par0)$dims <- list(p = p, K = K, L = L, H = H, n = n, Lm1 = Lm1)

  Dnoq <- length(par0)
  boot_pars <- matrix(NA_real_, nrow = B, ncol = Dnoq)

  for (b in seq_len(B)) {
    idx <- sample.int(n, size = n, replace = TRUE)
    Xb  <- X[idx, , drop = FALSE]
    yb  <- y[idx]
    qb  <- qhat[idx, , drop = FALSE]
    qsb <- qhat.se[idx, , drop = FALSE]
    Hb  <- Hmat[idx, , drop = FALSE]

    fit_b <- tryCatch(
      ML_fused(par = par0, X = Xb, y = yb, qhat = qb, qhat.se = qsb,
               Hmat = Hb, groups = groups,
               maxit = maxit, tol = tol,
               learning.rate = learning.rate, lambda = lambda,
               lambda.diag = lambda.diag, compute_se = FALSE),
      error = function(e) NULL
    )
    if (!is.null(fit_b)) {
      pb <- pack.hard(fit_b$par$beta, fit_b$par$Theta,
                      fit_b$par$alpha, fit_b$par$tmat)
      boot_pars[b, ] <- as.numeric(pb)
    }
  }

  se_boot <- apply(boot_pars, 2, function(x) sd(x, na.rm = TRUE))
  attributes(se_boot) <- attributes(par0)
  se_boot
}

#' Fast Bootstrap Standard Errors for Hard-Mode Fused Multinomial Model
#'
#' Computes bootstrap standard errors using the fast hard-mode solver
#' \code{\link{ML_fused_fast}}. Resets \code{tmat} to zero for warm-start
#' stability, tracks convergence and error counts, and returns both
#' SD-based and MAD-based robust SEs along with percentile confidence intervals.
#'
#' @param par_fit A fitted parameter list as returned by \code{ML_fused_fast()$par},
#'   containing elements \code{beta}, \code{Theta}, \code{alpha}, and \code{tmat}.
#' @param X Numeric matrix of covariates (n x p).
#' @param y Integer vector of class labels (length n).
#' @param qhat Numeric matrix of external predicted probabilities for non-reference
#'   groups (n x (L-1)).
#' @param qhat.se Numeric matrix of standard errors for \code{qhat} (n x (L-1)).
#' @param Hmat Numeric basis matrix for the EL constraint (n x ((L-1)*H)).
#' @param groups Integer vector of group assignments (length n).
#' @param B Integer number of bootstrap replicates (default 200).
#' @param maxit Integer maximum iterations per bootstrap fit (default 300).
#' @param tol Numeric convergence tolerance (default 1e-4).
#' @param learning.rate Numeric step-size scaling factor (default 0.1).
#' @param lambda Numeric regularization parameter for the q-penalty. If \code{NULL},
#'   defaults to \code{n}.
#' @param lambda.diag Numeric ridge stabilization added to the Hessian diagonal
#'   (default 0).
#'
#' @return A list with components:
#'   \describe{
#'     \item{se}{Numeric vector of SD-based bootstrap standard errors.}
#'     \item{se_mad}{Numeric vector of MAD-based robust bootstrap standard errors.}
#'     \item{q_lwr}{Numeric vector of 2.5 percent percentile bootstrap quantiles.}
#'     \item{q_upr}{Numeric vector of 97.5 percent percentile bootstrap quantiles.}
#'     \item{n_kept}{Integer count of converged replicates used for SE computation.}
#'     \item{n_nonconv}{Integer count of non-converged replicates.}
#'     \item{n_error}{Integer count of replicates that produced errors.}
#'   }
#'
#' @export
bootstrap_hard_se_fast <- function(par_fit, X, y, qhat, qhat.se, Hmat, groups,
                                   B = 200, maxit = 300, tol = 1e-4,
                                   learning.rate = 0.1, lambda = NULL,
                                   lambda.diag = 0) {
  n <- nrow(X)
  if (is.null(lambda)) lambda <- n
  A <- group_matrix(groups, length(unique(y)))

  # Warm-start: keep beta/Theta/alpha but reset tmat to zero
  # (original tmat warm-start causes H_tt ill-conditioning and divergence)
  tmat0 <- matrix(0, nrow = nrow(par_fit$tmat), ncol = ncol(par_fit$tmat))
  par0 <- pack.hard(par_fit$beta, par_fit$Theta, par_fit$alpha, tmat0)
  p   <- ncol(X); K <- length(unique(y)); L <- ncol(A)
  Lm1 <- L - 1; H <- ncol(Hmat) / Lm1
  attributes(par0)$dims <- list(p = p, K = K, L = L, H = H, n = n, Lm1 = Lm1)

  Dnoq <- length(par0)
  boot_pars <- matrix(NA_real_, nrow = B, ncol = Dnoq)
  n_error <- 0L
  n_nonconv <- 0L

  for (b in seq_len(B)) {
    idx <- sample.int(n, size = n, replace = TRUE)
    Xb  <- X[idx, , drop = FALSE]
    yb  <- y[idx]
    qb  <- qhat[idx, , drop = FALSE]
    qsb <- qhat.se[idx, , drop = FALSE]
    Hb  <- Hmat[idx, , drop = FALSE]

    fit_b <- tryCatch(
      ML_fused_fast(par = par0, X = Xb, y = yb, qhat = qb, qhat.se = qsb,
                    Hmat = Hb, groups = groups,
                    maxit = maxit, tol = tol,
                    learning.rate = learning.rate,
                    lambda = lambda, lambda.diag = lambda.diag,
                    compute_se = FALSE),
      error = function(e) NULL
    )
    if (is.null(fit_b)) {
      n_error <- n_error + 1L
    } else if (fit_b$conv != 0L) {
      n_nonconv <- n_nonconv + 1L
    } else {
      pb <- pack.hard(fit_b$par$beta, fit_b$par$Theta,
                      fit_b$par$alpha, fit_b$par$tmat)
      boot_pars[b, ] <- as.numeric(pb)
    }
  }

  keep <- apply(boot_pars, 1, function(row) !anyNA(row))

  bp_kept <- boot_pars[keep, , drop = FALSE]
  se_boot <- apply(bp_kept, 2, function(x) sd(x, na.rm = TRUE))
  attributes(se_boot) <- attributes(par0)

  # Percentile-based quantile CIs (2.5% and 97.5%)
  q_lwr <- apply(bp_kept, 2, function(x) quantile(x, 0.025, na.rm = TRUE))
  q_upr <- apply(bp_kept, 2, function(x) quantile(x, 0.975, na.rm = TRUE))
  attributes(q_lwr) <- attributes(par0)
  attributes(q_upr) <- attributes(par0)

  # MAD-based robust bootstrap SE
  se_boot_mad <- apply(bp_kept, 2, function(x) mad(x, constant = 1.4826))
  attributes(se_boot_mad) <- attributes(par0)

  list(se = se_boot, se_mad = se_boot_mad, q_lwr = q_lwr, q_upr = q_upr,
       n_kept = sum(keep), n_nonconv = n_nonconv, n_error = n_error)
}

#' Diagnostic Bootstrap Standard Errors with Per-Replicate Tracking
#'
#' Extended bootstrap procedure that records per-replicate diagnostics including
#' convergence status, gradient norm, iteration count, objective value, and
#' error messages. Uses the fast hard-mode solver \code{\link{ML_fused_fast}}.
#' Useful for diagnosing bootstrap convergence issues.
#'
#' @param par_fit A fitted parameter list as returned by \code{ML_fused_fast()$par},
#'   containing elements \code{beta}, \code{Theta}, \code{alpha}, and \code{tmat}.
#' @param X Numeric matrix of covariates (n x p).
#' @param y Integer vector of class labels (length n).
#' @param qhat Numeric matrix of external predicted probabilities for non-reference
#'   groups (n x (L-1)).
#' @param qhat.se Numeric matrix of standard errors for \code{qhat} (n x (L-1)).
#' @param Hmat Numeric basis matrix for the EL constraint (n x ((L-1)*H)).
#' @param groups Integer vector of group assignments (length n).
#' @param B Integer number of bootstrap replicates (default 500).
#' @param maxit Integer maximum iterations per bootstrap fit (default 300).
#' @param tol Numeric convergence tolerance (default 1e-4).
#' @param learning.rate Numeric step-size scaling factor (default 0.1).
#' @param lambda Numeric regularization parameter for the q-penalty. If \code{NULL},
#'   defaults to \code{n}.
#' @param lambda.diag Numeric ridge stabilization added to the Hessian diagonal
#'   (default 0).
#' @param store_indices Logical; if \code{TRUE}, store the bootstrap sample indices
#'   for each replicate (default \code{FALSE}).
#'
#' @return A list with components:
#'   \describe{
#'     \item{se}{Numeric vector of SD-based bootstrap standard errors (converged replicates only).}
#'     \item{q_lwr}{Numeric vector of 2.5 percent percentile bootstrap quantiles.}
#'     \item{q_upr}{Numeric vector of 97.5 percent percentile bootstrap quantiles.}
#'     \item{n_kept}{Integer count of converged replicates.}
#'     \item{n_nonconv}{Integer count of non-converged replicates.}
#'     \item{n_error}{Integer count of replicates that produced errors.}
#'     \item{boot_pars}{Matrix (B x D) of parameter estimates per replicate (NA rows for failures).}
#'     \item{boot_conv}{Integer vector of convergence codes per replicate (-1 = error, 0 = converged, 1 = not converged).}
#'     \item{boot_grad_norm}{Numeric vector of best gradient norms per replicate.}
#'     \item{boot_n_iter}{Integer vector of iteration counts per replicate.}
#'     \item{boot_obj}{Numeric vector of final objective values per replicate.}
#'     \item{boot_error_msg}{Character vector of error messages per replicate (NA if no error).}
#'     \item{boot_indices}{List of bootstrap sample index vectors (only if \code{store_indices = TRUE}).}
#'     \item{par0_attrs}{Attributes of the initial packed parameter vector (for unpacking).}
#'   }
#'
#' @export
bootstrap_hard_se_diagnostic <- function(par_fit, X, y, qhat, qhat.se, Hmat, groups,
                                          B = 500, maxit = 300, tol = 1e-4,
                                          learning.rate = 0.1, lambda = NULL,
                                          lambda.diag = 0, store_indices = FALSE) {
  n <- nrow(X)
  if (is.null(lambda)) lambda <- n
  A <- group_matrix(groups, length(unique(y)))

  tmat0 <- matrix(0, nrow = nrow(par_fit$tmat), ncol = ncol(par_fit$tmat))
  par0 <- pack.hard(par_fit$beta, par_fit$Theta, par_fit$alpha, tmat0)
  p <- ncol(X); K <- length(unique(y)); L <- ncol(A)
  Lm1 <- L - 1; H <- ncol(Hmat) / Lm1
  attributes(par0)$dims <- list(p = p, K = K, L = L, H = H, n = n, Lm1 = Lm1)

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

    Xb  <- X[idx, , drop = FALSE]
    yb  <- y[idx]
    qb  <- qhat[idx, , drop = FALSE]
    qsb <- qhat.se[idx, , drop = FALSE]
    Hb  <- Hmat[idx, , drop = FALSE]

    fit_b <- tryCatch(
      ML_fused_fast(par = par0, X = Xb, y = yb, qhat = qb, qhat.se = qsb,
                    Hmat = Hb, groups = groups,
                    maxit = maxit, tol = tol,
                    learning.rate = learning.rate,
                    lambda = lambda, lambda.diag = lambda.diag,
                    compute_se = FALSE),
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
      pb <- pack.hard(fit_b$par$beta, fit_b$par$Theta,
                      fit_b$par$alpha, fit_b$par$tmat)
      boot_pars[b, ] <- as.numeric(pb)
    }
  }

  # Summary stats from converged only (backward compat)
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
    # Summary (backward compatible)
    se = se_boot, q_lwr = q_lwr, q_upr = q_upr,
    n_kept = sum(converged), n_nonconv = sum(boot_conv == 1L, na.rm = TRUE),
    n_error = sum(boot_conv == -1L, na.rm = TRUE),
    # Per-replicate diagnostics
    boot_pars = boot_pars,
    boot_conv = boot_conv,
    boot_grad_norm = boot_grad_norm,
    boot_n_iter = boot_n_iter,
    boot_obj = boot_obj,
    boot_error_msg = boot_error_msg,
    boot_indices = boot_indices,
    # Dimensions for unpacking
    par0_attrs = attributes(par0)
  )
}
