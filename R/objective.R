# ============================================================
# objective.R — Objective, gradient, and Hessian for fused multinomial
# ============================================================

# --- Internal helpers ---

#' Parameter Index Ranges
#'
#' @param d Number of non-baseline classes (K - 1).
#' @param p Number of covariates.
#' @param Lm1 Number of non-reference groups (L - 1).
#' @param H Basis columns per group.
#' @return Named list of index vectors and block sizes.
#' @keywords internal
make_hard_indices <- function(d, p, Lm1, H) {
  d_beta  <- d
  d_Theta <- p * d
  d_alpha <- d
  d_t     <- Lm1 * H
  Dnoq    <- d_beta + d_Theta + d_alpha + d_t
  list(
    d_beta = d_beta, d_Theta = d_Theta, d_alpha = d_alpha, d_t = d_t,
    Dnoq = Dnoq,
    idx_beta  = 1:d_beta,
    idx_Theta = (d_beta + 1):(d_beta + d_Theta),
    idx_alpha = (d_beta + d_Theta + 1):(d_beta + d_Theta + d_alpha),
    idx_t     = (d_beta + d_Theta + d_alpha + 1):Dnoq,
    idx_theta = 1:(d_beta + d_Theta)
  )
}



#' Shared Quantities for Gradient and Hessian
#'
#' Precomputes softmax probabilities, EL constraint quantities, and
#' intermediate matrices shared between gradient and Hessian.
#'
#' @param par Packed hard-mode parameter vector.
#' @param X n x p covariate matrix.
#' @param y Integer response vector.
#' @param qhat n x (L-1) external probability matrix.
#' @param Hmat Basis matrix.
#' @param A K x L group matrix.
#' @param eps_feas Feasibility threshold.
#' @return Named list of intermediate quantities.
#' @keywords internal
compute_shared <- function(par, X, y, qhat, Hmat, A, eps_feas = 1e-8) {
  up    <- unpack_hard(par)
  beta  <- up$beta; Theta <- up$Theta; alpha <- up$alpha; tmat <- up$tmat

  n <- nrow(X); p <- ncol(X); K <- ncol(Theta) + 1; d <- K - 1
  A_LK <- t(A)
  L <- nrow(A_LK); Lm1 <- L - 1; H <- as.integer(ncol(Hmat) / Lm1)

  # Primary softmax
  eta  <- X %*% Theta + matrix(beta, n, d, byrow = TRUE)
  Ppri <- softmax_first(eta)
  Y    <- matrix(0, n, K); Y[cbind(seq_len(n), y)] <- 1
  R    <- Ppri[, 2:K, drop = FALSE] - Y[, 2:K, drop = FALSE]
  pPri <- Ppri[, -1, drop = FALSE]

  # External softmax
  etaE   <- X %*% Theta + matrix(alpha, n, d, byrow = TRUE)
  Pext_K <- softmax_first(etaE)
  Pext_L <- Pext_K %*% A
  pE     <- Pext_K[, -1, drop = FALSE]

  # EL constraint: z, v, S, w
  z_list <- v_list <- vector("list", Lm1)
  Svec   <- rep(0, n)
  for (m in seq_len(Lm1)) {
    Hm          <- Hmat[, m + (0:(H - 1)) * Lm1, drop = FALSE]
    v_list[[m]] <- Pext_L[, m + 1] - qhat[, m]
    z_list[[m]] <- as.numeric(Hm %*% tmat[m, ])
    Svec        <- Svec + v_list[[m]] * z_list[[m]]
  }

  w <- 1 / pmax(1 + Svec, 1e-10)
  infeasible <- FALSE

  z_matrix <- do.call(cbind, z_list)
  r_matrix <- Pext_L[, 2:L, drop = FALSE] - qhat

  B     <- sweep(A_LK[2:L, 2:K, drop = FALSE], 1, A_LK[2:L, 1], `-`)
  a_mat <- z_matrix %*% B

  c_ap <- rowSums(pE * a_mat)
  u    <- pE * a_mat - pE * c_ap
  pa   <- pE * a_mat

  A_sub <- A[-1, -1, drop = FALSE]
  zw    <- (z_matrix %*% t(A_sub)) * w
  s_val <- rowSums(zw * pE)
  G_ext <- zw * pE - pE * s_val

  list(n = n, p = p, K = K, d = d, L = L, Lm1 = Lm1, H = H,
       X = X, R = R, pPri = pPri, pE = pE, w = w,
       z_matrix = z_matrix, r_matrix = r_matrix,
       z_list = z_list, v_list = v_list, Svec = Svec,
       B = B, a_mat = a_mat, c_ap = c_ap, u = u, pa = pa,
       G_ext = G_ext, infeasible = infeasible,
       beta = up$beta, Theta = up$Theta, alpha = up$alpha, tmat = up$tmat)
}


# --- Exported functions ---

#' Objective Value
#'
#' Computes the negative log-likelihood of the fused multinomial model
#' with empirical-likelihood constraint.
#'
#' @param par Packed hard-mode parameter vector.
#' @param X n x p covariate matrix.
#' @param y Integer response vector (1..K).
#' @param qhat n x (L-1) external probability matrix.
#' @param Hmat Basis matrix.
#' @param A K x L group matrix.
#' @param lambda EL constraint weight. The EL term is scaled by
#'   \code{lambda / n}. Default \code{NULL} uses \code{lambda = n}
#'   (ratio = 1). Set to the external sample size to upweight the
#'   EL constraint.
#'
#' @return List with \code{obj} (scalar), \code{P} (n x K), \code{Pext_L} (n x L).
#' @export
objective_hard <- function(par, X, y, qhat, Hmat, A, lambda = NULL) {
  up    <- unpack_hard(par)
  beta  <- up$beta; Theta <- up$Theta; alpha <- up$alpha; tmat <- up$tmat

  n <- nrow(X); K <- ncol(Theta) + 1
  L <- ncol(A); Lm1 <- L - 1; H <- as.integer(ncol(Hmat) / Lm1)
  lam_ratio <- if (is.null(lambda)) 1 else lambda / n

  eta <- X %*% Theta + matrix(beta, n, K - 1, byrow = TRUE)
  P   <- softmax_first(eta)
  Y   <- matrix(0, n, K); Y[cbind(seq_len(n), y)] <- 1
  ell <- sum(Y * log(pmax(P, 1e-12)))

  eta_ext <- X %*% Theta + matrix(alpha, n, K - 1, byrow = TRUE)
  Pext_K  <- softmax_first(eta_ext)
  Pext_L  <- Pext_K %*% A

  Svec <- rep(0, n)
  for (m in seq_len(Lm1)) {
    Hm   <- Hmat[, m + (0:(H - 1)) * Lm1, drop = FALSE]
    v_m  <- Pext_L[, m + 1] - qhat[, m]
    z_m  <- as.numeric(Hm %*% tmat[m, ])
    Svec <- Svec + v_m * z_m
  }
  el_term <- sum(log(pmax(1 + Svec, 1e-10)))

  list(obj = -(ell - lam_ratio * el_term), P = P, Pext_L = Pext_L)
}


#' Gradient
#'
#' Analytic gradient of the fused negative log-likelihood.
#'
#' @param par Packed hard-mode parameter vector.
#' @param X n x p covariate matrix.
#' @param y Integer response vector.
#' @param qhat n x (L-1) external probability matrix.
#' @param Hmat Basis matrix.
#' @param A K x L group matrix.
#' @param tau_tmat L2 penalty on tmat (default 1).
#' @param lambda EL constraint weight (default NULL = n, giving ratio 1).
#'
#' @return Numeric gradient vector of length Dnoq.
#' @export
gradient_hard <- function(par, X, y, qhat, Hmat, A, tau_tmat = 1, lambda = NULL) {
  sh <- compute_shared(par, X, y, qhat, Hmat, A)
  lam_ratio <- if (is.null(lambda)) 1 else lambda / sh$n

  grad_beta  <- colSums(sh$R)
  grad_Theta <- crossprod(sh$X, sh$R) + lam_ratio * crossprod(sh$X, sh$G_ext)
  grad_alpha <- lam_ratio * colSums(sh$G_ext)

  grad_tmat <- matrix(0, sh$Lm1, sh$H)
  for (m in seq_len(sh$Lm1)) {
    Hm <- Hmat[, m + (0:(sh$H - 1)) * sh$Lm1, drop = FALSE]
    grad_tmat[m, ] <- lam_ratio * crossprod(Hm, sh$w * sh$v_list[[m]])
  }

  grad <- pack_hard(grad_beta, grad_Theta, grad_alpha, grad_tmat)

  ix <- make_hard_indices(sh$d, sh$p, sh$Lm1, sh$H)
  grad[ix$idx_t] <- grad[ix$idx_t] + tau_tmat * par[ix$idx_t]

  grad
}


#' Hessian
#'
#' Analytic Hessian of the fused negative log-likelihood (Dnoq x Dnoq).
#'
#' @param par Packed hard-mode parameter vector.
#' @param X n x p covariate matrix.
#' @param y Integer response vector.
#' @param qhat n x (L-1) external probability matrix.
#' @param Hmat Basis matrix.
#' @param A K x L group matrix.
#' @param lambda.diag Ridge regularisation on the diagonal (default 0).
#' @param eps_feas Feasibility threshold (default 1e-8).
#' @param tau_tmat L2 penalty on tmat (default 1).
#' @param lambda EL constraint weight (default NULL = n, giving ratio 1).
#'
#' @return Symmetric Dnoq x Dnoq Hessian matrix.
#' @export
hessian_hard <- function(par, X, y, qhat, Hmat, A,
                         lambda.diag = 0, eps_feas = 1e-8,
                         tau_tmat = 1, lambda = NULL) {
  sh <- compute_shared(par, X, y, qhat, Hmat, A, eps_feas)
  lam_ratio <- if (is.null(lambda)) 1 else lambda / sh$n
  .build_hessian_from_shared(sh, Hmat, lambda.diag, tau_tmat, lam_ratio)
}


#' Combined Gradient and Hessian
#'
#' Single-pass computation sharing softmax evaluation.
#'
#' @inheritParams hessian_hard
#' @param return_shared If TRUE, include intermediate quantities in output.
#'
#' @return List with \code{grad}, \code{hess}, and optionally \code{shared}.
#' @export
grad_hess_hard <- function(par, X, y, qhat, Hmat, A,
                           lambda.diag = 0, eps_feas = 1e-8,
                           return_shared = FALSE, tau_tmat = 1,
                           lambda = NULL) {
  sh <- compute_shared(par, X, y, qhat, Hmat, A, eps_feas)
  lam_ratio <- if (is.null(lambda)) 1 else lambda / sh$n

  # Gradient
  grad_beta  <- colSums(sh$R)
  grad_Theta <- crossprod(sh$X, sh$R) + lam_ratio * crossprod(sh$X, sh$G_ext)
  grad_alpha <- lam_ratio * colSums(sh$G_ext)

  grad_tmat <- matrix(0, sh$Lm1, sh$H)
  for (m in seq_len(sh$Lm1)) {
    Hm <- Hmat[, m + (0:(sh$H - 1)) * sh$Lm1, drop = FALSE]
    grad_tmat[m, ] <- lam_ratio * crossprod(Hm, sh$w * sh$v_list[[m]])
  }
  grad <- pack_hard(grad_beta, grad_Theta, grad_alpha, grad_tmat)

  ix <- make_hard_indices(sh$d, sh$p, sh$Lm1, sh$H)
  grad[ix$idx_t] <- grad[ix$idx_t] + tau_tmat * par[ix$idx_t]

  # Hessian
  hess <- .build_hessian_from_shared(sh, Hmat, lambda.diag, tau_tmat, lam_ratio)

  result <- list(grad = grad, hess = hess)
  if (return_shared) result$shared <- sh
  result
}


# --- Internal Hessian builder ---

#' @keywords internal
.build_hessian_from_shared <- function(sh, Hmat, lambda.diag = 0, tau_tmat = 1, lam_ratio = 1) {
  d <- sh$d; p <- sh$p; n <- sh$n; Lm1 <- sh$Lm1; H <- sh$H
  Xm <- sh$X; pPri <- sh$pPri; pE <- sh$pE; w <- sh$w

  ix <- make_hard_indices(d, p, Lm1, H)
  Dnoq <- ix$Dnoq

  pa <- sh$pa; c_ap <- sh$c_ap; u <- sh$u
  r_matrix <- sh$r_matrix; B <- sh$B

  Hfull <- matrix(0, Dnoq, Dnoq)
  place <- function(M, rr, cc) Hfull[rr, cc] <<- Hfull[rr, cc] + M

  # Primary block
  H_bb      <- matrix(0, d, d)
  H_bTheta  <- matrix(0, d, ix$d_Theta)
  H_ThetaTh <- matrix(0, ix$d_Theta, ix$d_Theta)

  for (j in seq_len(d)) {
    for (k in j:d) {
      wjk <- -pPri[, j] * pPri[, k]
      if (j == k) wjk <- wjk + pPri[, j]
      H_bb[j, k] <- sum(wjk)
      if (j != k) H_bb[k, j] <- H_bb[j, k]
      cols_k <- ((k - 1) * p + 1):(k * p)
      H_bTheta[j, cols_k] <- colSums(Xm * wjk)
      if (j != k) {
        cols_j <- ((j - 1) * p + 1):(j * p)
        H_bTheta[k, cols_j] <- colSums(Xm * wjk)
      }
      rows_j <- ((j - 1) * p + 1):(j * p)
      XtWX <- crossprod(Xm, Xm * wjk)
      H_ThetaTh[rows_j, cols_k] <- XtWX
      if (j != k) H_ThetaTh[cols_k, rows_j] <- XtWX
    }
  }

  # EL blocks
  H_aa        <- matrix(0, d, d)
  H_aTheta    <- matrix(0, d, ix$d_Theta)
  H_ThetaThEL <- matrix(0, ix$d_Theta, ix$d_Theta)

  for (j in seq_len(d)) {
    for (k in j:d) {
      H_S_jk <- -pa[, j] * pE[, k] - pE[, j] * pa[, k] +
                 2 * c_ap * pE[, j] * pE[, k]
      if (j == k) H_S_jk <- H_S_jk + pa[, j] - c_ap * pE[, j]
      H_eta_jk <- w * H_S_jk - w^2 * u[, j] * u[, k]
      H_aa[j, k] <- sum(H_eta_jk)
      if (j != k) H_aa[k, j] <- H_aa[j, k]
      cols_k <- ((k - 1) * p + 1):(k * p)
      H_aTheta[j, cols_k] <- colSums(Xm * H_eta_jk)
      if (j != k) {
        cols_j <- ((j - 1) * p + 1):(j * p)
        H_aTheta[k, cols_j] <- colSums(Xm * H_eta_jk)
      }
      rows_j <- ((j - 1) * p + 1):(j * p)
      XtWX <- crossprod(Xm, Xm * H_eta_jk)
      H_ThetaThEL[rows_j, cols_k] <- XtWX
      if (j != k) H_ThetaThEL[cols_k, rows_j] <- XtWX
    }
  }

  # t blocks
  H_at     <- matrix(0, d, ix$d_t)
  H_Thetat <- matrix(0, ix$d_Theta, ix$d_t)
  H_tt     <- matrix(0, ix$d_t, ix$d_t)

  for (m in seq_len(Lm1)) {
    Hm   <- Hmat[, m + (0:(H - 1)) * Lm1, drop = FALSE]
    bm   <- B[m, ]
    rm   <- r_matrix[, m]
    tidx <- m + (0:(H - 1)) * Lm1
    Jbm  <- sweep(pE, 2, bm, `*`) - pE * as.numeric(pE %*% bm)
    for (j in seq_len(d)) {
      v_j <- w * Jbm[, j] - w^2 * rm * u[, j]
      H_at[j, tidx] <- colSums(Hm * v_j)
      rows_j <- ((j - 1) * p + 1):(j * p)
      H_Thetat[rows_j, tidx] <- crossprod(Xm, Hm * v_j)
    }
    for (m2 in seq_len(Lm1)) {
      Hm2   <- Hmat[, m2 + (0:(H - 1)) * Lm1, drop = FALSE]
      r2    <- r_matrix[, m2]
      tidx2 <- m2 + (0:(H - 1)) * Lm1
      H_tt[tidx, tidx2] <- H_tt[tidx, tidx2] +
        crossprod(Hm, Hm2 * (-w^2 * rm * r2))
    }
  }

  # Assemble
  place(H_bb,        ix$idx_beta,  ix$idx_beta)
  place(H_bTheta,    ix$idx_beta,  ix$idx_Theta)
  place(t(H_bTheta), ix$idx_Theta, ix$idx_beta)
  place(H_ThetaTh,   ix$idx_Theta, ix$idx_Theta)

  place(lam_ratio * H_aa,            ix$idx_alpha, ix$idx_alpha)
  place(lam_ratio * H_aTheta,        ix$idx_alpha, ix$idx_Theta)
  place(lam_ratio * t(H_aTheta),     ix$idx_Theta, ix$idx_alpha)
  place(lam_ratio * H_ThetaThEL,     ix$idx_Theta, ix$idx_Theta)

  place(lam_ratio * H_at,        ix$idx_alpha, ix$idx_t)
  place(lam_ratio * t(H_at),     ix$idx_t,     ix$idx_alpha)
  place(lam_ratio * H_Thetat,    ix$idx_Theta, ix$idx_t)
  place(lam_ratio * t(H_Thetat), ix$idx_t,     ix$idx_Theta)
  place(lam_ratio * H_tt,        ix$idx_t,     ix$idx_t)

  diag(Hfull[ix$idx_t, ix$idx_t]) <- diag(Hfull[ix$idx_t, ix$idx_t]) + tau_tmat
  diag(Hfull) <- diag(Hfull) + lambda.diag
  0.5 * (Hfull + t(Hfull))
}
