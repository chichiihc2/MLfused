#' Fused Multinomial Optimizer
#'
#' Fits a multinomial logistic regression model on primary data while
#' incorporating external ML predictions via an empirical-likelihood
#' constraint. Uses Newton updates with analytic gradient and Hessian.
#'
#' @section Parameterization:
#' The code uses class 1 as the softmax baseline (linear predictor fixed
#' at 0), so parameters correspond to classes 2, ..., K.
#' The paper (Dai and Shao, 2025) uses class K as the baseline.
#' The two are equivalent under relabeling: the code's class 1 corresponds
#' to the paper's class K, and vice versa.
#' \code{groups[[1]]} is the reference group.
#'
#' @param par Initial parameters in packed format (via \code{pack_hard()}).
#' @param X n x p covariate matrix.
#' @param y Integer class labels (1..K).
#' @param qhat n x (L-1) external predicted group probabilities.
#' @param Hmat n x ((L-1)*H) basis matrix.
#' @param groups List of length L defining class-to-group mapping.
#' @param maxit Maximum Newton iterations (default 500).
#' @param tol Convergence tolerance on gradient infinity-norm (default 1e-6).
#' @param learning.rate Step size for Newton update (default 0.1).
#' @param batch_frac Mini-batch fraction in (0,1] (default 1).
#' @param lambda.diag Ridge stabilization for Hessian diagonal (default 0).
#' @param compute_se Compute standard errors at convergence (default TRUE).
#' @param store_trace Store parameter trace (default FALSE).
#' @param diagnostic_trace Per-iteration diagnostics (default FALSE).
#' @param tau_tmat L2 penalty on tmat parameters (default 1).
#'
#' @return A list with:
#'   \describe{
#'     \item{par}{Unpacked best parameters.}
#'     \item{conv}{0 = converged, 1 = max iterations.}
#'     \item{best_grad_norm, best_iter}{Convergence diagnostics.}
#'     \item{grad.error, obj.record}{Per-iteration traces.}
#'     \item{conf}{Full sandwich SE output (from \code{sandwich_se}).}
#'     \item{se}{Sandwich SEs (unpacked list: beta, Theta, alpha, tmat).}
#'     \item{diag_trace}{Diagnostic trace (if requested).}
#'   }
#'
#' @export
ml_fused <- function(par, X, y, qhat, Hmat, groups,
                     maxit = 500, tol = 1e-6,
                     learning.rate = 0.1,
                     batch_frac = 1,
                     lambda.diag = 0,
                     compute_se = TRUE,
                     store_trace = FALSE,
                     diagnostic_trace = FALSE,
                     tau_tmat = 1) {

  p <- ncol(X); L <- length(groups); n <- nrow(X)
  K <- sum(lengths(groups))
  H <- dim(Hmat)[2] / (L - 1)
  attributes(par)$dims <- list(p = p, K = K, L = L, H = H)
  A <- group_matrix(groups, K)

  up0 <- unpack_hard(par)
  par_h <- pack_hard(up0$beta, up0$Theta, up0$alpha, up0$tmat)

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
    batch_ids <- sort(sample(kept_ids, size = min(bsize, length(kept_ids)),
                             replace = FALSE))
    Xi   <- X   [batch_ids, , drop = FALSE]; yi   <- y   [batch_ids]
    qh_i <- qhat[batch_ids, , drop = FALSE]; Hi   <- Hmat[batch_ids, , drop = FALSE]

    gh <- grad_hess_hard(par_curr, Xi, yi, qh_i, Hi, A,
                         lambda.diag = lambda.diag,
                         return_shared = diagnostic_trace,
                         tau_tmat = tau_tmat)
    grad_curr <- gh$grad
    Hess      <- gh$hess

    obj_val <- objective_hard(par_curr, Xi, yi, qh_i, Hi, A)$obj

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
               error = function(e2) {
                 solve_method <<- "ginv"
                 MASS::ginv(H2) %*% (-grad_curr)
               })
    })

    if (diagnostic_trace) {
      up_diag <- unpack_hard(par_curr)
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
      message(sprintf("Converged in %d iterations.", it))
      return(.build_result(par_best, X, y, qhat, Hmat, A,
                           lambda.diag, compute_se,
                           best_grad_norm, best_iter,
                           grad.error[1:it], obj.record[1:it], 0L,
                           if (store_trace) par_trace else NULL,
                           diag_trace))
    }
    attributes(delta) <- attributes(grad_curr)
    par_curr <- par_curr + learning.rate * as.numeric(delta)
    if (store_trace) par_trace[[it]] <- par_curr
  }

  message("Reached maximum iterations without convergence.")
  .build_result(par_best, X, y, qhat, Hmat, A,
                lambda.diag, compute_se,
                best_grad_norm, best_iter,
                grad.error[1:actual_iters], obj.record[1:actual_iters], 1L,
                if (store_trace) par_trace else NULL,
                diag_trace)
}


#' @keywords internal
.build_result <- function(par_best, X, y, qhat, Hmat, A,
                          lambda.diag, compute_se,
                          best_grad_norm, best_iter,
                          grad.error, obj.record, conv,
                          par_trace, diag_trace = NULL) {
  up_best <- unpack_hard(par_best)
  base <- list(
    par = up_best, best_grad_norm = best_grad_norm,
    best_iter = best_iter, grad.error = grad.error,
    obj.record = obj.record, conv = conv,
    conf = NULL, se = NULL,
    par_trace = par_trace, diag_trace = diag_trace
  )
  if (!compute_se) return(base)

  hess_conv <- hessian_hard(par_best, X, y, qhat, Hmat, A,
                            lambda.diag = lambda.diag, tau_tmat = 0)

  conf <- sandwich_se(par_best, X, y, qhat, Hmat, A,
                      lambda.diag = lambda.diag,
                      hess_precomputed = hess_conv)

  base$conf <- conf
  base$se   <- unpack_hard(conf$se)
  base
}
