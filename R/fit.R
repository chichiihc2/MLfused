# ============================================================
# fit.R — Optimizer functions for fused multinomial models
# ============================================================

# ----------------------------------------------------------
# .build_hard_result_orig
# ----------------------------------------------------------

#' Build Result List for Hard-Mode Optimizer (Original)
#'
#' @title Build hard-mode result list (original)
#'
#' @description Internal helper that constructs the return list for the
#'   hard-mode (fixed-q) branch of \code{\link{ML_fused}}.
#'   Computes sandwich and Theorem 3 standard errors when
#'   \code{compute_se = TRUE}; otherwise returns \code{NULL} SE slots.
#'
#' @param par_best Numeric vector of best parameters found during
#'   optimization, in hard-packed format (see \code{pack.hard}).
#' @param X Numeric matrix of primary covariates (\eqn{n \times p}).
#' @param y Integer vector of class labels of length \eqn{n}.
#' @param qhat Numeric matrix of external grouped-probability predictions
#'   (\eqn{n \times (L-1)}).
#' @param qhat.se Numeric matrix of standard errors for \code{qhat}
#'   (\eqn{n \times (L-1)}).
#' @param Hmat Numeric matrix of constraint basis functions
#'   (\eqn{n \times (L-1)H}).
#' @param A Group matrix (\eqn{K \times L}) from \code{group_matrix}.
#' @param lambda.diag Numeric ridge stabilization for the Hessian.
#' @param compute_se Logical; if \code{TRUE}, compute standard errors.
#' @param best_grad_norm Numeric; smallest gradient infinity-norm found.
#' @param best_iter Integer; iteration at which \code{best_grad_norm} was
#'   attained.
#' @param grad.error Numeric vector of gradient norms across iterations.
#' @param obj.record Numeric vector of objective values across iterations.
#' @param conv Integer convergence flag: 0 = converged, 1 = max iterations.
#' @param par_trace List of parameter vectors across iterations, or
#'   \code{NULL} if tracing was disabled.
#'
#' @return A list with components:
#'   \describe{
#'     \item{par}{Unpacked best parameters (from \code{unpack.hard}).}
#'     \item{best_grad_norm}{Smallest gradient norm attained.}
#'     \item{best_iter}{Iteration of best gradient norm.}
#'     \item{grad.error}{Per-iteration gradient norms.}
#'     \item{obj.record}{Per-iteration objective values.}
#'     \item{conv}{Convergence flag (0 or 1).}
#'     \item{conf}{Full output of \code{conf.est.hard}, or \code{NULL}.}
#'     \item{thm3}{Full output of \code{conf.est.thm3}, or \code{NULL}.}
#'     \item{se, se.adj}{Hessian-based SEs (unpacked), with/without qhat
#'       uncertainty adjustment.}
#'     \item{se.score, se.score.adj}{Score-based sandwich SEs (unpacked).}
#'     \item{se.thm3, se.thm3.adj}{Theorem 3 SEs (unpacked).}
#'     \item{se.thm3.score, se.thm3.score.adj}{Theorem 3 plugin SEs
#'       (unpacked).}
#'     \item{par_trace}{Parameter trace list, or \code{NULL}.}
#'   }
#'
#' @keywords internal
.build_hard_result_orig <- function(par_best, X, y, qhat, qhat.se, Hmat, A,
                                    lambda.diag, compute_se, best_grad_norm,
                                    best_iter, grad.error, obj.record, conv,
                                    par_trace) {
  up_best <- unpack.hard(par_best)
  if (!compute_se) {
    return(list(par=up_best, best_grad_norm=best_grad_norm,
                best_iter=best_iter, grad.error=grad.error,
                obj.record=obj.record, conv=conv,
                conf=NULL, thm3=NULL,
                se=NULL, se.adj=NULL, se.score=NULL,
                se.score.adj=NULL, se.thm3=NULL, se.thm3.adj=NULL,
                se.thm3.score=NULL, se.thm3.score.adj=NULL,
                par_trace=par_trace))
  }

  conf  <- conf.est.hard(par_best, X, y, qhat, qhat.se, Hmat, A=A, lambda.diag=lambda.diag)
  thm3  <- conf.est.thm3(par_best, X, y, qhat, qhat.se, Hmat, A=A)
  thm3p <- conf.est.thm3.plugin(par_best, X, y, qhat, qhat.se, Hmat, A=A)

  fill_na <- function(vec, fallback) {
    vec[is.na(vec)] <- fallback[is.na(vec)]
    attributes(vec) <- attributes(par_best)
    vec
  }

  se_thm3_vec      <- fill_na(thm3$se.thm3,            conf$se)
  se_thm3_adj_vec  <- fill_na(thm3$se.thm3.adj,        conf$se.adj)
  se_thm3p_vec     <- fill_na(thm3p$se.thm3.plugin,    conf$se)
  se_thm3p_adj_vec <- fill_na(thm3p$se.thm3.plugin.adj, conf$se.adj)

  list(par=up_best, best_grad_norm=best_grad_norm,
       best_iter=best_iter, grad.error=grad.error,
       obj.record=obj.record, conv=conv,
       conf=conf, thm3=thm3,
       se=unpack.hard(conf$se),
       se.adj=unpack.hard(conf$se.adj),
       se.score=unpack.hard(conf$se.score),
       se.score.adj=unpack.hard(conf$se.score.adj),
       se.thm3=unpack.hard(se_thm3_vec),
       se.thm3.adj=unpack.hard(se_thm3_adj_vec),
       se.thm3.score=unpack.hard(se_thm3p_vec),
       se.thm3.score.adj=unpack.hard(se_thm3p_adj_vec),
       par_trace=par_trace)
}


# ----------------------------------------------------------
# ML_fused
# ----------------------------------------------------------

#' Fused Multinomial Learning with Empirical-Likelihood Utilized Machine Learning Results
#'
#' @title Fused multinomial optimizer
#'
#' @description Fits a multinomial logistic regression model on the primary data
#'   while incorporating external machine learning results via an
#'   empirical-likelihood constraint encoded by \code{Hmat} and group mapping
#'   \code{groups}. The optimizer uses a Newton update based on the analytic
#'   gradient and Hessian, with optional feasibility masking and mini-batching.
#'
#'   \strong{Reference conventions.}
#'   \itemize{
#'     \item \strong{Class reference (baseline):} class \eqn{1} is the baseline
#'       in the multinomial softmax parameterization used by
#'       \code{softmax_first()}. Consequently, \eqn{\beta} and \eqn{\Theta}
#'       correspond to classes \eqn{2,\dots,K}.
#'     \item \strong{Group reference:} \code{groups[[1]]} is the reference
#'       group. The grouped-probability quantities \eqn{P_{\mathrm{ext},L}} and
#'       \code{qhat} are represented in \strong{non-reference form}, i.e., only
#'       groups \eqn{2,\dots,L} are parameterized/used in the constraint term.
#'   }
#'
#' @param par Numeric vector of initial parameters in hard-packed format
#'   (via \code{pack.hard()}). The packing order is
#'   \eqn{(\beta, \Theta, \alpha, t)}.
#' @param X Numeric matrix of primary covariates with dimension
#'   \eqn{n \times p}.
#' @param y Vector of primary class labels of length \eqn{n}. Labels should
#'   correspond to \eqn{\{1,\dots,K\}} with class 1 treated as the baseline
#'   under the \code{softmax_first()} convention.
#' @param qhat Numeric matrix of external grouped-probability predictions on the
#'   primary sample, with dimension \eqn{n \times (L-1)}. Column \eqn{m}
#'   corresponds to external \strong{non-reference} group \eqn{m+1}, i.e.,
#'   groups \eqn{2,\dots,L}. (Group \eqn{1} is the reference and is not
#'   explicitly represented.)
#' @param qhat.se Numeric matrix (or compatible) of uncertainty estimates for
#'   \code{qhat}, typically standard errors, with dimension
#'   \eqn{n \times (L-1)}. Used by \code{conf.est.hard()}
#'   for inference; not required for the Newton step.
#' @param Hmat Numeric matrix of constraint functions with dimension
#'   \eqn{n \times ((L-1)H)}.
#' @param groups List of length \eqn{L} defining the external grouping of the
#'   \eqn{K} original classes.
#' @param maxit Integer maximum number of Newton iterations (default 500).
#' @param tol Numeric convergence tolerance (default \code{1e-6}).
#' @param learning.rate Numeric step size multiplying the Newton direction
#'   (default 0.1).
#' @param batch_frac Numeric in \eqn{(0,1]} controlling the mini-batch fraction
#'   (default 1).
#' @param use_mask Logical; if \code{TRUE} (default), restricts to feasible
#'   indices before mini-batch sampling.
#' @param lambda Numeric tuning parameter (default 100).
#' @param lambda.diag Numeric diagonal stabilization added to the Hessian
#'   (default 0).
#' @param compute_se Logical; if \code{TRUE} (default), compute sandwich and
#'   Theorem 3 standard errors at convergence.
#' @param store_trace Logical; if \code{TRUE}, store parameter vectors at each
#'   iteration (default \code{FALSE}).
#'
#' @return A list with fitted parameters, diagnostics, and inference objects.
#'   See \code{\link{ML_fused_fast}} for details.
#'
#' @export
ML_fused <- function(par, X, y, qhat, qhat.se, Hmat, groups,
                    maxit = 500, tol = 1e-6,
                    learning.rate = 0.1,
                    batch_frac = 1,
                    use_mask = TRUE,
                    lambda = 100,
                    lambda.diag = 0,
                    compute_se = TRUE,
                    store_trace = FALSE) {
  p <- ncol(X); K <- length(unique(y)); L <- length(groups); n <- nrow(X)
  H <- dim(Hmat)[2] / (L - 1)
  attributes(par)$dims <- list(p = p, K = K, L = L, H = H)
  A <- group_matrix(groups, K)

  up0 <- unpack.hard(par)
  par_h <- pack.hard(up0$beta, up0$Theta, up0$alpha, up0$tmat)

  par_curr <- par_h; par_best <- par_h
  best_grad_norm <- Inf; best_iter <- 0
  grad.error <- numeric(0); obj.record <- numeric(0)
  par_trace <- list()

  for (it in seq_len(maxit)) {
    if (use_mask) {
      kept_ids <- which(keep.hard(par_curr, X, y, qhat, Hmat, A))
      if (length(kept_ids) == 0) {
        message(sprintf("Iter %d: no valid samples. Stopping.", it)); break
      }
    } else {
      kept_ids <- seq_len(n)
    }

    bsize     <- max(1L, floor(length(kept_ids) * batch_frac))
    batch_ids <- sort(sample(kept_ids, size = min(bsize, length(kept_ids)), replace = FALSE))
    Xi   <- X   [batch_ids, , drop = FALSE]; yi   <- y   [batch_ids]
    qh_i <- qhat[batch_ids, , drop = FALSE]; Hi   <- Hmat[batch_ids, , drop = FALSE]

    grad_curr <- obj.gradient.hard(par_curr, Xi, yi, qh_i, Hi, A, tau_tmat = 0)
    Hess      <- obj.hessian.hard(par_curr, Xi, yi, qh_i, Hi, A,
                                  lambda.diag = lambda.diag, tau_tmat = 0)

    obj.record <- c(obj.record,
                    objective.hard(par_curr, Xi, yi, qh_i, Hi, A)$obj)

    grad_norm <- max(abs(grad_curr))
    grad.error <- c(grad.error, grad_norm)
    if (is.finite(grad_norm) && grad_norm < best_grad_norm) {
      best_grad_norm <- grad_norm; par_best <- par_curr; best_iter <- it
    }

    delta <- tryCatch(solve(Hess, -grad_curr), error = function(e) {
      H2 <- Hess; diag(H2) <- diag(H2) + max(1e-6, lambda.diag)
      tryCatch(solve(H2, -grad_curr),
               error = function(e2) MASS::ginv(H2) %*% (-grad_curr))
    })

    if (grad_norm < tol || max(abs(delta)) < tol) {
      message(sprintf("Converged in %d iterations.", it))
      return(.build_hard_result_orig(par_best, X, y, qhat, qhat.se, Hmat, A,
                                     lambda.diag, compute_se, best_grad_norm,
                                     best_iter, grad.error, obj.record, 0L,
                                     if (store_trace) par_trace else NULL))
    }
    attributes(delta) <- attributes(grad_curr)
    par_curr <- par_curr + learning.rate * as.numeric(delta)
    if (store_trace) par_trace[[it]] <- par_curr
  }

  message("Reached maximum iterations without convergence.")
  .build_hard_result_orig(par_best, X, y, qhat, qhat.se, Hmat, A,
                          lambda.diag, compute_se, best_grad_norm,
                          best_iter, grad.error, obj.record, 1L,
                          if (store_trace) par_trace else NULL)
}


# ----------------------------------------------------------
# .build_fast_result
# ----------------------------------------------------------

#' Build Result List for Fast Hard-Mode Optimizer
#'
#' @title Build fast hard-mode result list
#'
#' @description Internal helper that constructs the return list for
#'   \code{\link{ML_fused_fast}}. Computes sandwich and Theorem 3 standard
#'   errors using the fast variants (\code{conf.est.hard.fast},
#'   \code{conf.est.thm3.fast}, \code{conf.est.thm3.plugin.fast}) when
#'   \code{compute_se = TRUE}; otherwise returns \code{NULL} SE slots.
#'   Precomputes the Hessian at the converged parameters and reuses it across
#'   SE estimators for efficiency.
#'
#' @param par_best Numeric vector of best parameters found during
#'   optimization, in hard-packed format (see \code{pack.hard}).
#' @param X Numeric matrix of primary covariates (\eqn{n \times p}).
#' @param y Integer vector of class labels of length \eqn{n}.
#' @param qhat Numeric matrix of external grouped-probability predictions
#'   (\eqn{n \times (L-1)}).
#' @param qhat.se Numeric matrix of standard errors for \code{qhat}
#'   (\eqn{n \times (L-1)}).
#' @param Hmat Numeric matrix of constraint basis functions
#'   (\eqn{n \times (L-1)H}).
#' @param A Group matrix (\eqn{K \times L}) from \code{group_matrix}.
#' @param lambda.diag Numeric ridge stabilization for the Hessian.
#' @param compute_se Logical; if \code{TRUE}, compute standard errors.
#' @param best_grad_norm Numeric; smallest gradient infinity-norm found.
#' @param best_iter Integer; iteration at which \code{best_grad_norm} was
#'   attained.
#' @param grad.error Numeric vector of gradient norms across iterations.
#' @param obj.record Numeric vector of objective values across iterations.
#' @param conv Integer convergence flag: 0 = converged, 1 = max iterations.
#' @param par_trace List of parameter vectors across iterations, or
#'   \code{NULL} if tracing was disabled.
#' @param diag_trace List of per-iteration diagnostic records, or
#'   \code{NULL} if diagnostic tracing was disabled.
#'
#' @return A list with components:
#'   \describe{
#'     \item{par}{Unpacked best parameters (from \code{unpack.hard}).}
#'     \item{best_grad_norm}{Smallest gradient norm attained.}
#'     \item{best_iter}{Iteration of best gradient norm.}
#'     \item{grad.error}{Per-iteration gradient norms.}
#'     \item{obj.record}{Per-iteration objective values.}
#'     \item{conv}{Convergence flag (0 or 1).}
#'     \item{conf}{Full output of \code{conf.est.hard.fast}, or \code{NULL}.}
#'     \item{thm3}{Full output of \code{conf.est.thm3.fast}, or \code{NULL}.}
#'     \item{se, se.adj}{Hessian-based SEs (unpacked), with/without qhat
#'       uncertainty adjustment.}
#'     \item{se.score, se.score.adj}{Score-based sandwich SEs (unpacked).}
#'     \item{se.thm3, se.thm3.adj}{Theorem 3 SEs (unpacked).}
#'     \item{se.thm3.score, se.thm3.score.adj}{Theorem 3 plugin SEs
#'       (unpacked).}
#'     \item{par_trace}{Parameter trace list, or \code{NULL}.}
#'     \item{diag_trace}{Diagnostic trace list, or \code{NULL}.}
#'   }
#'
#' @keywords internal
.build_fast_result <- function(par_best, X, y, qhat, qhat.se, Hmat, A,
                               lambda.diag, compute_se,
                               best_grad_norm, best_iter,
                               grad.error, obj.record, conv,
                               par_trace, diag_trace = NULL) {
  up_best <- unpack.hard(par_best)
  if (!compute_se) {
    return(list(par = up_best, best_grad_norm = best_grad_norm,
                best_iter = best_iter, grad.error = grad.error,
                obj.record = obj.record, conv = conv,
                conf = NULL, thm3 = NULL,
                se = NULL, se.adj = NULL, se.score = NULL,
                se.score.adj = NULL, se.thm3 = NULL, se.thm3.adj = NULL,
                se.thm3.score = NULL, se.thm3.score.adj = NULL,
                par_trace = par_trace,
                diag_trace = diag_trace))
  }

  # Precompute Hessians: one at converged par, one at t=0
  hess_conv <- obj.hessian.hard(par_best, X, y, qhat, Hmat, A,
                                       lambda.diag = lambda.diag)

  conf <- conf.est.hard.fast(par_best, X, y, qhat, qhat.se, Hmat, A,
                             lambda.diag = lambda.diag,
                             hess_precomputed = hess_conv)

  # thm3 needs Hessian at t=0 — compute separately
  thm3 <- conf.est.thm3.fast(par_best, X, y, qhat, qhat.se, Hmat, A)

  # thm3.plugin reuses converged Hessian (but at lambda.diag=0)
  hess_conv_noridge <- if (lambda.diag == 0) hess_conv else
    obj.hessian.hard(par_best, X, y, qhat, Hmat, A, lambda.diag = 0)
  thm3p <- conf.est.thm3.plugin.fast(par_best, X, y, qhat, qhat.se, Hmat, A,
                                      hess_precomputed = hess_conv_noridge)

  # Fill NA slots in thm3 with sandwich SEs
  se_thm3_vec <- thm3$se.thm3
  se_thm3_vec[is.na(se_thm3_vec)] <- conf$se[is.na(se_thm3_vec)]
  se_thm3_adj_vec <- thm3$se.thm3.adj
  se_thm3_adj_vec[is.na(se_thm3_adj_vec)] <- conf$se.adj[is.na(se_thm3_adj_vec)]
  attributes(se_thm3_vec) <- attributes(par_best)
  attributes(se_thm3_adj_vec) <- attributes(par_best)

  se_thm3p_vec <- thm3p$se.thm3.plugin
  se_thm3p_vec[is.na(se_thm3p_vec)] <- conf$se[is.na(se_thm3p_vec)]
  se_thm3p_adj_vec <- thm3p$se.thm3.plugin.adj
  se_thm3p_adj_vec[is.na(se_thm3p_adj_vec)] <- conf$se.adj[is.na(se_thm3p_adj_vec)]
  attributes(se_thm3p_vec) <- attributes(par_best)
  attributes(se_thm3p_adj_vec) <- attributes(par_best)

  list(par = up_best, best_grad_norm = best_grad_norm,
       best_iter = best_iter, grad.error = grad.error,
       obj.record = obj.record, conv = conv,
       conf = conf, thm3 = thm3,
       se = unpack.hard(conf$se),
       se.adj = unpack.hard(conf$se.adj),
       se.score = unpack.hard(conf$se.score),
       se.score.adj = unpack.hard(conf$se.score.adj),
       se.thm3 = unpack.hard(se_thm3_vec),
       se.thm3.adj = unpack.hard(se_thm3_adj_vec),
       se.thm3.score = unpack.hard(se_thm3p_vec),
       se.thm3.score.adj = unpack.hard(se_thm3p_adj_vec),
       se.adj.noq = NULL, par_trace = par_trace,
       diag_trace = diag_trace)
}


# ----------------------------------------------------------
# ML_fused_fast
# ----------------------------------------------------------

#' Fast Hard-Mode Fused Multinomial Optimizer
#'
#' @title Fast fused multinomial optimizer
#'
#' @description Optimized fused multinomial solver that vectorizes the
#'   per-observation Hessian loop by using \code{obj.grad_hess.hard()} for
#'   combined gradient and Hessian computation in a single softmax pass.
#'
#'   Same API and return structure as \code{ML_fused()}.
#'
#' @param par Numeric vector of initial parameters in hard-packed format
#'   (via \code{pack.hard()}).
#' @param X Numeric matrix of primary covariates with dimension
#'   \eqn{n \times p}.
#' @param y Vector of primary class labels of length \eqn{n}. Labels should
#'   correspond to \eqn{\{1,\dots,K\}} with class 1 treated as the baseline
#'   under the \code{softmax_first()} convention.
#' @param qhat Numeric matrix of external grouped-probability predictions on the
#'   primary sample, with dimension \eqn{n \times (L-1)}.
#' @param qhat.se Numeric matrix of uncertainty estimates for \code{qhat},
#'   with dimension \eqn{n \times (L-1)}.
#' @param Hmat Numeric matrix of constraint functions with dimension
#'   \eqn{n \times ((L-1)H)}.
#' @param groups List of length \eqn{L} defining the external grouping of the
#'   \eqn{K} original classes. \code{groups[[1]]} is the reference group.
#' @param maxit Integer maximum number of Newton iterations (default 500).
#' @param tol Numeric convergence tolerance (default \code{1e-6}).
#' @param learning.rate Numeric step size for the Newton update (default 0.1).
#' @param batch_frac Numeric in \eqn{(0,1]} controlling the mini-batch fraction
#'   (default 1).
#' @param use_mask Logical; currently unused in the fast solver (all
#'   observations are used each iteration). Retained for API compatibility
#'   (default \code{TRUE}).
#' @param lambda Numeric tuning parameter for the objective (default 100).
#' @param lambda.diag Numeric diagonal ridge stabilization for the Hessian
#'   (default 0).
#' @param compute_se Logical; if \code{TRUE} (default), compute sandwich and
#'   Theorem 3 standard errors at convergence using fast SE estimators.
#' @param store_trace Logical; if \code{TRUE}, store parameter vectors at each
#'   iteration (default \code{FALSE}).
#' @param diagnostic_trace Logical; if \code{TRUE}, collect per-iteration
#'   diagnostic information including gradient norms, Hessian condition numbers,
#'   feasibility counts, and t-parameter dynamics (default \code{FALSE}).
#' @param tau_tmat Numeric scaling factor for the t-parameter block in the
#'   gradient/Hessian computation (default 1).
#'
#' @return A list with fitted parameters, diagnostics, and inference objects.
#'   Same structure as \code{ML_fused()}, with the addition of:
#'   \describe{
#'     \item{par}{Unpacked best parameters (from \code{unpack.hard}).}
#'     \item{best_grad_norm}{Smallest gradient infinity-norm attained.}
#'     \item{best_iter}{Iteration of best gradient norm.}
#'     \item{grad.error}{Per-iteration gradient norms.}
#'     \item{obj.record}{Per-iteration objective values.}
#'     \item{conv}{Convergence flag: 0 = converged, 1 = max iterations reached.}
#'     \item{conf}{Full output of \code{conf.est.hard.fast}, or \code{NULL}.}
#'     \item{thm3}{Full output of \code{conf.est.thm3.fast}, or \code{NULL}.}
#'     \item{se, se.adj}{Hessian-based SEs (unpacked).}
#'     \item{se.score, se.score.adj}{Score-based sandwich SEs (unpacked).}
#'     \item{se.thm3, se.thm3.adj}{Theorem 3 SEs (unpacked).}
#'     \item{se.thm3.score, se.thm3.score.adj}{Theorem 3 plugin SEs
#'       (unpacked).}
#'     \item{par_trace}{Parameter trace list, or \code{NULL}.}
#'     \item{diag_trace}{List of per-iteration diagnostic records when
#'       \code{diagnostic_trace = TRUE}, or \code{NULL}.}
#'   }
#'
#' @export
ML_fused_fast <- function(par, X, y, qhat, qhat.se, Hmat, groups,
                          maxit = 500, tol = 1e-6,
                          learning.rate = 0.1,
                          batch_frac = 1,
                          use_mask = TRUE,
                          lambda = 100,
                          lambda.diag = 0,
                          compute_se = TRUE,
                          store_trace = FALSE,
                          diagnostic_trace = FALSE,
                          tau_tmat = 1) {
  p <- ncol(X); K <- length(unique(y)); L <- length(groups); n <- nrow(X)
  H <- dim(Hmat)[2] / (L - 1)
  attributes(par)$dims <- list(p = p, K = K, L = L, H = H)
  A <- group_matrix(groups, K)

  up0 <- unpack.hard(par)
  par_h <- pack.hard(up0$beta, up0$Theta, up0$alpha, up0$tmat)

  par_curr <- par_h; par_best <- par_h
  best_grad_norm <- Inf; best_iter <- 0
  grad.error <- numeric(maxit)
  obj.record <- numeric(maxit)
  actual_iters <- 0
  par_trace <- list()
  diag_trace <- if (diagnostic_trace) list() else NULL
  ix_diag <- if (diagnostic_trace) make_hard_indices(K - 1, p, L - 1, H) else NULL

  for (it in seq_len(maxit)) {
    kept_ids <- seq_len(n)

    bsize     <- max(1L, floor(length(kept_ids) * batch_frac))
    batch_ids <- sort(sample(kept_ids, size = min(bsize, length(kept_ids)), replace = FALSE))
    Xi   <- X   [batch_ids, , drop = FALSE]; yi   <- y   [batch_ids]
    qh_i <- qhat[batch_ids, , drop = FALSE]; Hi   <- Hmat[batch_ids, , drop = FALSE]

    # Combined gradient + Hessian (one softmax pass)
    gh <- obj.grad_hess.hard(par_curr, Xi, yi, qh_i, Hi, A,
                                    lambda.diag = lambda.diag,
                                    return_shared = diagnostic_trace,
                                    tau_tmat = tau_tmat)
    grad_curr <- gh$grad
    Hess      <- gh$hess

    # Objective tracking
    obj_val <- objective.hard(par_curr, Xi, yi, qh_i, Hi, A)$obj

    grad_norm <- max(abs(grad_curr))
    actual_iters <- it
    grad.error[it] <- grad_norm
    obj.record[it] <- obj_val

    if (is.finite(grad_norm) && grad_norm < best_grad_norm) {
      best_grad_norm <- grad_norm; par_best <- par_curr; best_iter <- it
    }

    solve_method <- "solve"
    delta <- tryCatch(solve(Hess, -grad_curr), error = function(e) {
      solve_method <<- "ridge"
      H2 <- Hess; diag(H2) <- diag(H2) + max(1e-6, lambda.diag)
      tryCatch(solve(H2, -grad_curr),
               error = function(e2) { solve_method <<- "ginv"; MASS::ginv(H2) %*% (-grad_curr) })
    })

    # Collect per-iteration diagnostic trace
    if (diagnostic_trace) {
      up_diag <- unpack.hard(par_curr)
      tmat_curr <- up_diag$tmat
      tmat_delta <- delta[ix_diag$idx_t]
      sh_diag <- gh$shared
      hess_rcond <- tryCatch(rcond(Hess), error = function(e) NA_real_)
      diag_trace[[it]] <- list(
        iter             = it,
        grad_norm        = grad_norm,
        obj_val          = obj_val,
        n_feasible       = length(kept_ids),
        infeasible_flag  = sh_diag$infeasible,
        hess_rcond       = hess_rcond,
        max_delta        = max(abs(delta)),
        solve_method     = solve_method,
        tmat_norm        = sqrt(sum(tmat_curr^2)),
        tmat_delta_norm  = sqrt(sum(tmat_delta^2)),
        min_S            = min(1 + sh_diag$Svec),
        max_w            = max(sh_diag$w),
        n_infeasible_obs = sum(1 + sh_diag$Svec <= 1e-8),
        grad_tmat_norm   = max(abs(grad_curr[ix_diag$idx_t]))
      )
    }

    if (grad_norm < tol || max(abs(delta)) < tol) {
      message(sprintf("Converged in %d iterations (ML_fused_fast).", it))
      return(.build_fast_result(par_best, X, y, qhat, qhat.se, Hmat, A,
                                lambda.diag, compute_se,
                                best_grad_norm, best_iter,
                                grad.error[1:it], obj.record[1:it], 0L,
                                if (store_trace) par_trace else NULL,
                                diag_trace = diag_trace))
    }
    attributes(delta) <- attributes(grad_curr)
    par_curr <- par_curr + learning.rate * as.numeric(delta)
    if (store_trace) par_trace[[it]] <- par_curr
  }

  message("Reached maximum iterations without convergence (ML_fused_fast).")
  .build_fast_result(par_best, X, y, qhat, qhat.se, Hmat, A,
                     lambda.diag, compute_se,
                     best_grad_norm, best_iter,
                     grad.error[1:actual_iters], obj.record[1:actual_iters], 1L,
                     if (store_trace) par_trace else NULL,
                     diag_trace = diag_trace)
}
