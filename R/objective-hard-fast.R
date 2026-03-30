# ============================================================
# Hard-mode objective helpers (objective, gradient, Hessian, combined)
# ============================================================

#' Hard-Mode Parameter Index Ranges
#'
#' Computes named index ranges for each block of the hard-mode
#' (fixed-q) parameter vector: beta, Theta, alpha, and tmat.
#'
#' @param d Integer. Number of non-baseline classes (K - 1).
#' @param p Integer. Number of covariates.
#' @param Lm1 Integer. Number of non-reference groups (L - 1).
#' @param H Integer. Number of basis columns per non-reference group in
#'   the H matrix.
#'
#' @return A named list with elements:
#'   \describe{
#'     \item{d_beta, d_Theta, d_alpha, d_t}{Lengths of each block.}
#'     \item{Dnoq}{Total parameter dimension (excluding q).}
#'     \item{idx_beta, idx_Theta, idx_alpha, idx_t}{Integer vectors of
#'       indices into the packed parameter vector.}
#'     \item{idx_theta}{Combined indices for beta and Theta.}
#'   }
#'
#' @keywords internal
make_hard_indices <- function(d, p, Lm1, H) {
  d_beta  <- d
  d_Theta <- p * d
  d_alpha <- d
  d_t     <- Lm1 * H
  Dnoq    <- d_beta + d_Theta + d_alpha + d_t
  list(
    d_beta = d_beta, d_Theta = d_Theta, d_alpha = d_alpha, d_t = d_t, Dnoq = Dnoq,
    idx_beta  = 1:d_beta,
    idx_Theta = (d_beta + 1):(d_beta + d_Theta),
    idx_alpha = (d_beta + d_Theta + 1):(d_beta + d_Theta + d_alpha),
    idx_t     = (d_beta + d_Theta + d_alpha + 1):Dnoq,
    idx_theta = c(1:(d_beta + d_Theta))
  )
}

#' Adjustment Term for qhat Uncertainty
#'
#' Computes the block-diagonal adjustment matrix that accounts for
#' estimation uncertainty in the external predicted probabilities
#' (\code{qhat}). Used to inflate the variance of the t-block in
#' sandwich standard errors.
#'
#' @param Hmat An n x (Lm1 * H) basis matrix (e.g. from \code{build_H}).
#' @param var.qhat An n x Lm1 matrix of pointwise variances of qhat
#'   (i.e. \code{qhat.se^2}).
#' @param Lm1 Integer. Number of non-reference groups (L - 1).
#' @param H Integer. Number of basis columns per non-reference group.
#'
#' @return A (Lm1 * H) x (Lm1 * H) symmetric matrix representing the
#'   qhat-uncertainty adjustment.
#'
#' @keywords internal
compute_adj_term <- function(Hmat, var.qhat, Lm1, H) {
  d_t <- Lm1 * H
  adj <- matrix(0, d_t, d_t)
  for (m in seq_len(Lm1)) {
    Hm   <- Hmat[, m + (0:(H-1)) * Lm1, drop = FALSE]  # n x H
    tidx <- m + (0:(H-1)) * Lm1
    adj[tidx, tidx] <- crossprod(Hm, Hm * var.qhat[, m])
  }
  adj
}

#' Shared Quantities for Hard-Mode Gradient and Hessian
#'
#' Precomputes primary and external softmax probabilities, EL constraint
#' quantities (z, v, S, w), and various intermediate matrices that are
#' shared between the gradient and Hessian computations. Called once per
#' Newton step by \code{\link{obj.grad_hess.hard}}.
#'
#' @param par Packed hard-mode parameter vector (see \code{\link{pack.hard}}).
#' @param X An n x p covariate matrix.
#' @param y An integer vector of length n with class labels in 1, ..., K.
#' @param qhat An n x (L-1) matrix of external predicted group
#'   probabilities for non-reference groups.
#' @param Hmat An n x ((L-1)*H) basis matrix for the EL constraint.
#' @param A A K x L group membership matrix (see \code{\link{group_matrix}}).
#' @param eps_feas Numeric scalar. Minimum feasibility threshold for
#'   \code{1 + S} (default 1e-8).
#'
#' @return A named list of intermediate quantities including dimensions
#'   (n, p, K, d, L, Lm1, H), matrices (R, pPri, pE, w, z_matrix,
#'   r_matrix, B, a_mat, c_ap, u, pa, G_ext), and the unpacked
#'   parameters (beta, Theta, alpha, tmat).
#'
#' @keywords internal
compute_shared <- function(par, X, y, qhat, Hmat, A, eps_feas = 1e-8) {
  up    <- unpack.hard(par)
  beta  <- up$beta; Theta <- up$Theta; alpha <- up$alpha; tmat <- up$tmat

  n <- nrow(X); p <- ncol(X); K <- ncol(Theta) + 1; d <- K - 1
  A_LK <- t(A)  # L x K  (A is K x L from group_matrix)
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
  Pext_L <- Pext_K %*% A   # n x L
  pE     <- Pext_K[, -1, drop = FALSE]

  # EL constraint: z, v (= r), S, w
  z_list <- v_list <- vector("list", Lm1)
  Svec   <- rep(0, n)
  for (m in seq_len(Lm1)) {
    Hm         <- Hmat[, m + (0:(H-1)) * Lm1, drop = FALSE]
    v_list[[m]] <- Pext_L[, m + 1] - qhat[, m]
    z_list[[m]] <- as.numeric(Hm %*% tmat[m, ])
    Svec        <- Svec + v_list[[m]] * z_list[[m]]
  }

  w <- 1 / pmax(1 + Svec, 1e-10)
  infeasible <- FALSE

  z_matrix <- do.call(cbind, z_list)   # n x Lm1
  r_matrix <- Pext_L[, 2:L, drop = FALSE] - qhat

  # B = A_LK[2:L, 2:K] - A_LK[2:L, 1]  (Lm1 x d)
  B     <- sweep(A_LK[2:L, 2:K, drop = FALSE], 1, A_LK[2:L, 1], `-`)
  a_mat <- z_matrix %*% B   # n x d

  # u = Jacobian of a_mat through softmax
  c_ap <- rowSums(pE * a_mat)
  u    <- pE * a_mat - pE * c_ap  # n x d
  pa   <- pE * a_mat              # n x d

  # G_ext (for gradient)
  A_sub    <- A[-1, -1, drop = FALSE]   # (K-1) x (L-1)
  zw       <- (z_matrix %*% t(A_sub)) * w
  s_val    <- rowSums(zw * pE)
  G_ext    <- zw * pE - pE * s_val

  list(n = n, p = p, K = K, d = d, L = L, Lm1 = Lm1, H = H,
       X = X, R = R, pPri = pPri, pE = pE, w = w,
       z_matrix = z_matrix, r_matrix = r_matrix,
       z_list = z_list, v_list = v_list, Svec = Svec,
       B = B, a_mat = a_mat, c_ap = c_ap, u = u, pa = pa,
       G_ext = G_ext, infeasible = infeasible,
       beta = up$beta, Theta = up$Theta, alpha = up$alpha, tmat = up$tmat)
}

#' Hard-Mode Objective Value
#'
#' Computes the negative log-likelihood of the fused multinomial model
#' with empirical-likelihood constraint, evaluated at hard-mode
#' parameters (q fixed at qhat).
#'
#' @param par Packed hard-mode parameter vector (see \code{\link{pack.hard}}).
#' @param X An n x p covariate matrix.
#' @param y An integer vector of length n with class labels in 1, ..., K.
#' @param qhat An n x (L-1) matrix of external predicted group
#'   probabilities for non-reference groups.
#' @param Hmat An n x ((L-1)*H) basis matrix for the EL constraint.
#' @param A A K x L group membership matrix (see \code{\link{group_matrix}}).
#'
#' @return A list with elements:
#'   \describe{
#'     \item{obj}{Scalar. The negative log-likelihood value.}
#'     \item{P}{An n x K matrix of primary softmax probabilities.}
#'     \item{Pext_L}{An n x L matrix of external group probabilities.}
#'   }
#'
#' @export
objective.hard <- function(par, X, y, qhat, Hmat, A) {
  up    <- unpack.hard(par)
  beta  <- up$beta; Theta <- up$Theta; alpha <- up$alpha; tmat <- up$tmat

  n <- nrow(X); K <- ncol(Theta) + 1
  L <- ncol(A); Lm1 <- L - 1; H <- as.integer(ncol(Hmat) / Lm1)

  # Primary multinomial log-likelihood
  eta <- X %*% Theta + matrix(beta, n, K - 1, byrow = TRUE)
  P   <- softmax_first(eta)
  Y   <- matrix(0, n, K); Y[cbind(seq_len(n), y)] <- 1
  ell <- sum(Y * log(pmax(P, 1e-12)))

  # External group probabilities
  eta_ext <- X %*% Theta + matrix(alpha, n, K - 1, byrow = TRUE)
  Pext_K  <- softmax_first(eta_ext)
  Pext_L  <- Pext_K %*% A

  # EL constraint
  Svec <- rep(0, n)
  for (m in seq_len(Lm1)) {
    Hm   <- Hmat[, m + (0:(H - 1)) * Lm1, drop = FALSE]
    v_m  <- Pext_L[, m + 1] - qhat[, m]
    z_m  <- as.numeric(Hm %*% tmat[m, ])
    Svec <- Svec + v_m * z_m
  }
  el_term <- sum(log(pmax(1 + Svec, 1e-10)))

  list(obj = -(ell - el_term), P = P, Pext_L = Pext_L)
}


#' Hard-Mode Feasibility Mask
#'
#' Returns a logical vector indicating which observations satisfy the
#' feasibility constraint \code{1 + S_i > eps} under hard-mode parameters.
#'
#' @param par Packed hard-mode parameter vector (see \code{\link{pack.hard}}).
#' @param X An n x p covariate matrix.
#' @param y An integer vector of length n with class labels in 1, ..., K.
#' @param qhat An n x (L-1) matrix of external predicted group
#'   probabilities for non-reference groups.
#' @param Hmat An n x ((L-1)*H) basis matrix for the EL constraint.
#' @param A A K x L group membership matrix (see \code{\link{group_matrix}}).
#' @param eps Numeric scalar. Minimum feasibility threshold (default 1e-8).
#'
#' @return A logical vector of length n.
#'
#' @export
keep.hard <- function(par, X, y, qhat, Hmat, A, eps = 1e-8) {
  up    <- unpack.hard(par)
  Theta <- up$Theta; alpha <- up$alpha; tmat <- up$tmat

  n <- nrow(X); K <- ncol(Theta) + 1
  L <- ncol(A); Lm1 <- L - 1; H <- as.integer(ncol(Hmat) / Lm1)

  eta_ext <- X %*% Theta + matrix(alpha, n, K - 1, byrow = TRUE)
  Pext_L  <- softmax_first(eta_ext) %*% A

  Svec <- rep(0, n)
  for (m in seq_len(Lm1)) {
    Hm   <- Hmat[, m + (0:(H - 1)) * Lm1, drop = FALSE]
    v_m  <- Pext_L[, m + 1] - qhat[, m]
    z_m  <- as.numeric(Hm %*% tmat[m, ])
    Svec <- Svec + v_m * z_m
  }
  (1 + Svec) > eps
}


#' Hard-Mode Gradient
#'
#' Computes the gradient of the fused negative log-likelihood (primary
#' model plus empirical-likelihood constraint) with respect to the
#' hard-mode parameter vector (q fixed at qhat).
#'
#' @param par Packed hard-mode parameter vector (see \code{\link{pack.hard}}).
#' @param X An n x p covariate matrix.
#' @param y An integer vector of length n with class labels in 1, ..., K.
#' @param qhat An n x (L-1) matrix of external predicted group
#'   probabilities for non-reference groups.
#' @param Hmat An n x ((L-1)*H) basis matrix for the EL constraint.
#' @param A A K x L group membership matrix (see \code{\link{group_matrix}}).
#' @param tau_tmat Numeric scalar. L2 regularisation penalty on the
#'   tmat parameters (default 1).
#'
#' @return A numeric vector of length Dnoq (same length as \code{par})
#'   containing the gradient.
#'
#' @export
obj.gradient.hard <- function(par, X, y, qhat, Hmat, A, tau_tmat = 1) {
  sh <- compute_shared(par, X, y, qhat, Hmat, A)

  grad_beta  <- colSums(sh$R)
  grad_Theta <- crossprod(sh$X, sh$R) + crossprod(sh$X, sh$G_ext)
  grad_alpha <- colSums(sh$G_ext)

  grad_tmat <- matrix(0, sh$Lm1, sh$H)
  for (m in seq_len(sh$Lm1)) {
    Hm <- Hmat[, m + (0:(sh$H - 1)) * sh$Lm1, drop = FALSE]
    grad_tmat[m, ] <- crossprod(Hm, sh$w * sh$v_list[[m]])
  }

  grad <- pack.hard(grad_beta, grad_Theta, grad_alpha, grad_tmat)

  # L2 penalty on tmat
  ix <- make_hard_indices(sh$d, sh$p, sh$Lm1, sh$H)
  grad[ix$idx_t] <- grad[ix$idx_t] + tau_tmat * par[ix$idx_t]

  grad
}

#' Hard-Mode Hessian (Vectorised, Dnoq x Dnoq)
#'
#' Computes the Hessian of the fused negative log-likelihood with
#' respect to the hard-mode parameter vector. The per-observation
#' Hessian contributions are vectorised for speed. Returns a
#' Dnoq x Dnoq symmetric matrix (no q-block).
#'
#' @param par Packed hard-mode parameter vector (see \code{\link{pack.hard}}).
#' @param X An n x p covariate matrix.
#' @param y An integer vector of length n with class labels in 1, ..., K.
#' @param qhat An n x (L-1) matrix of external predicted group
#'   probabilities for non-reference groups.
#' @param Hmat An n x ((L-1)*H) basis matrix for the EL constraint.
#' @param A A K x L group membership matrix (see \code{\link{group_matrix}}).
#' @param lambda.diag Numeric scalar. Ridge regularisation added to the
#'   diagonal of the Hessian (default 0).
#' @param eps_feas Numeric scalar. Minimum feasibility threshold for
#'   \code{1 + S} (default 1e-8).
#' @param tau_tmat Numeric scalar. L2 regularisation penalty on the
#'   tmat parameters (default 1).
#'
#' @return A Dnoq x Dnoq symmetric numeric matrix (the Hessian).
#'
#' @export
obj.hessian.hard <- function(par, X, y, qhat, Hmat, A,
                                    lambda.diag = 0, eps_feas = 1e-8,
                                    tau_tmat = 1) {
  sh <- compute_shared(par, X, y, qhat, Hmat, A, eps_feas)
  d <- sh$d; p <- sh$p; n <- sh$n; Lm1 <- sh$Lm1; H <- sh$H
  X <- sh$X; pPri <- sh$pPri; pE <- sh$pE; w <- sh$w

  d_beta <- d; d_Theta <- p * d; d_alpha <- d; d_t <- Lm1 * H
  Dnoq <- d_beta + d_Theta + d_alpha + d_t

  idx_beta  <- 1:d_beta
  idx_Theta <- (d_beta + 1):(d_beta + d_Theta)
  idx_alpha <- (d_beta + d_Theta + 1):(d_beta + d_Theta + d_alpha)
  idx_t     <- (d_beta + d_Theta + d_alpha + 1):Dnoq

  Hfull <- matrix(0, Dnoq, Dnoq)
  place <- function(M, rr, cc) Hfull[rr, cc] <<- Hfull[rr, cc] + M

  pa   <- sh$pa; c_ap <- sh$c_ap; u <- sh$u
  r_matrix <- sh$r_matrix; B <- sh$B

  # ---- Vectorized primary block ----
  H_bb      <- matrix(0, d, d)
  H_bTheta  <- matrix(0, d, d_Theta)
  H_ThetaTh <- matrix(0, d_Theta, d_Theta)

  for (j in seq_len(d)) {
    for (k in j:d) {
      wjk <- -pPri[, j] * pPri[, k]
      if (j == k) wjk <- wjk + pPri[, j]

      H_bb[j, k] <- sum(wjk)
      if (j != k) H_bb[k, j] <- H_bb[j, k]

      cols_k <- ((k-1)*p + 1):(k*p)
      H_bTheta[j, cols_k] <- colSums(X * wjk)
      if (j != k) {
        cols_j <- ((j-1)*p + 1):(j*p)
        H_bTheta[k, cols_j] <- colSums(X * wjk)
      }

      rows_j <- ((j-1)*p + 1):(j*p)
      XtWX <- crossprod(X, X * wjk)
      H_ThetaTh[rows_j, cols_k] <- XtWX
      if (j != k) H_ThetaTh[cols_k, rows_j] <- XtWX
    }
  }

  # ---- Vectorized EL blocks (alpha, Theta_EL) ----
  H_aa        <- matrix(0, d, d)
  H_aTheta    <- matrix(0, d, d_Theta)
  H_ThetaThEL <- matrix(0, d_Theta, d_Theta)

  for (j in seq_len(d)) {
    for (k in j:d) {
      # H_S_jk vectorized over all observations
      H_S_jk <- -pa[, j] * pE[, k] - pE[, j] * pa[, k] +
                 2 * c_ap * pE[, j] * pE[, k]
      if (j == k) H_S_jk <- H_S_jk + pa[, j] - c_ap * pE[, j]

      H_eta_jk <- w * H_S_jk - w^2 * u[, j] * u[, k]

      H_aa[j, k] <- sum(H_eta_jk)
      if (j != k) H_aa[k, j] <- H_aa[j, k]

      cols_k <- ((k-1)*p + 1):(k*p)
      H_aTheta[j, cols_k] <- colSums(X * H_eta_jk)
      if (j != k) {
        cols_j <- ((j-1)*p + 1):(j*p)
        H_aTheta[k, cols_j] <- colSums(X * H_eta_jk)
      }

      rows_j <- ((j-1)*p + 1):(j*p)
      XtWX <- crossprod(X, X * H_eta_jk)
      H_ThetaThEL[rows_j, cols_k] <- XtWX
      if (j != k) H_ThetaThEL[cols_k, rows_j] <- XtWX
    }
  }

  # ---- eta-t and t-t blocks ----
  H_at     <- matrix(0, d, d_t)
  H_Thetat <- matrix(0, d_Theta, d_t)
  H_tt     <- matrix(0, d_t, d_t)

  for (m in seq_len(Lm1)) {
    Hm   <- Hmat[, m + (0:(H-1)) * Lm1, drop = FALSE]
    bm   <- B[m, ]
    rm   <- r_matrix[, m]
    tidx <- m + (0:(H-1)) * Lm1

    # Jbm vectorized: pE * bm - pE * (pE %*% bm)
    Jbm <- sweep(pE, 2, bm, `*`) - pE * as.numeric(pE %*% bm)  # n x d

    for (j in seq_len(d)) {
      v_j <- w * Jbm[, j] - w^2 * rm * u[, j]
      H_at[j, tidx] <- colSums(Hm * v_j)
      rows_j <- ((j-1)*p + 1):(j*p)
      H_Thetat[rows_j, tidx] <- crossprod(X, Hm * v_j)
    }

    for (m2 in seq_len(Lm1)) {
      Hm2   <- Hmat[, m2 + (0:(H-1)) * Lm1, drop = FALSE]
      r2    <- r_matrix[, m2]
      tidx2 <- m2 + (0:(H-1)) * Lm1
      wt    <- -w^2 * rm * r2
      H_tt[tidx, tidx2] <- H_tt[tidx, tidx2] + crossprod(Hm, Hm2 * wt)
    }
  }

  # ---- Assemble ----
  place(H_bb,        idx_beta,  idx_beta)
  place(H_bTheta,    idx_beta,  idx_Theta)
  place(t(H_bTheta), idx_Theta, idx_beta)
  place(H_ThetaTh,   idx_Theta, idx_Theta)

  place(H_aa,            idx_alpha, idx_alpha)
  place(H_aTheta,        idx_alpha, idx_Theta)
  place(t(H_aTheta),     idx_Theta, idx_alpha)
  place(H_ThetaThEL,     idx_Theta, idx_Theta)

  place(H_at,            idx_alpha, idx_t)
  place(t(H_at),         idx_t,     idx_alpha)
  place(H_Thetat,        idx_Theta, idx_t)
  place(t(H_Thetat),     idx_t,     idx_Theta)
  place(H_tt,            idx_t,     idx_t)

  # L2 penalty on tmat: add tau_tmat to H_tt diagonal
  diag(Hfull[idx_t, idx_t]) <- diag(Hfull[idx_t, idx_t]) + tau_tmat

  diag(Hfull) <- diag(Hfull) + lambda.diag
  0.5 * (Hfull + t(Hfull))
}

#' Combined Hard-Mode Gradient and Hessian
#'
#' Computes both the gradient and Hessian of the fused negative
#' log-likelihood in a single pass, sharing the softmax and
#' intermediate quantity computations. This is the workhorse called
#' at each Newton step by the fast hard-mode solver.
#'
#' @param par Packed hard-mode parameter vector (see \code{\link{pack.hard}}).
#' @param X An n x p covariate matrix.
#' @param y An integer vector of length n with class labels in 1, ..., K.
#' @param qhat An n x (L-1) matrix of external predicted group
#'   probabilities for non-reference groups.
#' @param Hmat An n x ((L-1)*H) basis matrix for the EL constraint.
#' @param A A K x L group membership matrix (see \code{\link{group_matrix}}).
#' @param lambda.diag Numeric scalar. Ridge regularisation added to the
#'   diagonal of the Hessian (default 0).
#' @param eps_feas Numeric scalar. Minimum feasibility threshold for
#'   \code{1 + S} (default 1e-8).
#' @param return_shared Logical. If \code{TRUE}, the list of shared
#'   intermediate quantities (from \code{\link{compute_shared}}) is
#'   included in the return value under \code{$shared} (default
#'   \code{FALSE}).
#' @param tau_tmat Numeric scalar. L2 regularisation penalty on the
#'   tmat parameters (default 1).
#'
#' @return A list with elements:
#'   \describe{
#'     \item{grad}{Numeric vector of length Dnoq -- the gradient.}
#'     \item{hess}{Dnoq x Dnoq symmetric numeric matrix -- the Hessian.}
#'     \item{shared}{(only if \code{return_shared = TRUE}) The list
#'       returned by \code{\link{compute_shared}}.}
#'   }
#'
#' @export
obj.grad_hess.hard <- function(par, X, y, qhat, Hmat, A,
                                      lambda.diag = 0, eps_feas = 1e-8,
                                      return_shared = FALSE,
                                      tau_tmat = 1) {
  sh <- compute_shared(par, X, y, qhat, Hmat, A, eps_feas)
  d <- sh$d; p <- sh$p; n <- sh$n; Lm1 <- sh$Lm1; H <- sh$H
  Xm <- sh$X; pPri <- sh$pPri; pE <- sh$pE; w <- sh$w

  # ---- GRADIENT ----
  grad_beta  <- colSums(sh$R)
  grad_Theta <- crossprod(Xm, sh$R) + crossprod(Xm, sh$G_ext)
  grad_alpha <- colSums(sh$G_ext)

  grad_tmat <- matrix(0, Lm1, H)
  for (m in seq_len(Lm1)) {
    Hm <- Hmat[, m + (0:(H-1)) * Lm1, drop = FALSE]
    grad_tmat[m, ] <- crossprod(Hm, w * sh$v_list[[m]])
  }
  grad <- pack.hard(grad_beta, grad_Theta, grad_alpha, grad_tmat)

  # L2 penalty on tmat
  ix_pen <- make_hard_indices(d, p, Lm1, H)
  grad[ix_pen$idx_t] <- grad[ix_pen$idx_t] + tau_tmat * par[ix_pen$idx_t]

  # ---- HESSIAN ----
  d_beta <- d; d_Theta <- p * d; d_alpha <- d; d_t <- Lm1 * H
  Dnoq <- d_beta + d_Theta + d_alpha + d_t

  idx_beta  <- 1:d_beta
  idx_Theta <- (d_beta + 1):(d_beta + d_Theta)
  idx_alpha <- (d_beta + d_Theta + 1):(d_beta + d_Theta + d_alpha)
  idx_t     <- (d_beta + d_Theta + d_alpha + 1):Dnoq

  Hfull <- matrix(0, Dnoq, Dnoq)
  place <- function(M, rr, cc) Hfull[rr, cc] <<- Hfull[rr, cc] + M

  pa <- sh$pa; c_ap <- sh$c_ap; u <- sh$u
  r_matrix <- sh$r_matrix; B <- sh$B

  # Primary block
  H_bb      <- matrix(0, d, d)
  H_bTheta  <- matrix(0, d, d_Theta)
  H_ThetaTh <- matrix(0, d_Theta, d_Theta)

  for (j in seq_len(d)) {
    for (k in j:d) {
      wjk <- -pPri[, j] * pPri[, k]
      if (j == k) wjk <- wjk + pPri[, j]
      H_bb[j, k] <- sum(wjk)
      if (j != k) H_bb[k, j] <- H_bb[j, k]
      cols_k <- ((k-1)*p + 1):(k*p)
      H_bTheta[j, cols_k] <- colSums(Xm * wjk)
      if (j != k) {
        cols_j <- ((j-1)*p + 1):(j*p)
        H_bTheta[k, cols_j] <- colSums(Xm * wjk)
      }
      rows_j <- ((j-1)*p + 1):(j*p)
      XtWX <- crossprod(Xm, Xm * wjk)
      H_ThetaTh[rows_j, cols_k] <- XtWX
      if (j != k) H_ThetaTh[cols_k, rows_j] <- XtWX
    }
  }

  # EL alpha/Theta blocks
  H_aa        <- matrix(0, d, d)
  H_aTheta    <- matrix(0, d, d_Theta)
  H_ThetaThEL <- matrix(0, d_Theta, d_Theta)

  for (j in seq_len(d)) {
    for (k in j:d) {
      H_S_jk <- -pa[, j] * pE[, k] - pE[, j] * pa[, k] +
                 2 * c_ap * pE[, j] * pE[, k]
      if (j == k) H_S_jk <- H_S_jk + pa[, j] - c_ap * pE[, j]
      H_eta_jk <- w * H_S_jk - w^2 * u[, j] * u[, k]
      H_aa[j, k] <- sum(H_eta_jk)
      if (j != k) H_aa[k, j] <- H_aa[j, k]
      cols_k <- ((k-1)*p + 1):(k*p)
      H_aTheta[j, cols_k] <- colSums(Xm * H_eta_jk)
      if (j != k) {
        cols_j <- ((j-1)*p + 1):(j*p)
        H_aTheta[k, cols_j] <- colSums(Xm * H_eta_jk)
      }
      rows_j <- ((j-1)*p + 1):(j*p)
      XtWX <- crossprod(Xm, Xm * H_eta_jk)
      H_ThetaThEL[rows_j, cols_k] <- XtWX
      if (j != k) H_ThetaThEL[cols_k, rows_j] <- XtWX
    }
  }

  # t blocks
  H_at     <- matrix(0, d, d_t)
  H_Thetat <- matrix(0, d_Theta, d_t)
  H_tt     <- matrix(0, d_t, d_t)

  for (m in seq_len(Lm1)) {
    Hm   <- Hmat[, m + (0:(H-1)) * Lm1, drop = FALSE]
    bm   <- B[m, ]
    rm   <- r_matrix[, m]
    tidx <- m + (0:(H-1)) * Lm1
    Jbm  <- sweep(pE, 2, bm, `*`) - pE * as.numeric(pE %*% bm)
    for (j in seq_len(d)) {
      v_j <- w * Jbm[, j] - w^2 * rm * u[, j]
      H_at[j, tidx] <- colSums(Hm * v_j)
      rows_j <- ((j-1)*p + 1):(j*p)
      H_Thetat[rows_j, tidx] <- crossprod(Xm, Hm * v_j)
    }
    for (m2 in seq_len(Lm1)) {
      Hm2   <- Hmat[, m2 + (0:(H-1)) * Lm1, drop = FALSE]
      r2    <- r_matrix[, m2]
      tidx2 <- m2 + (0:(H-1)) * Lm1
      H_tt[tidx, tidx2] <- H_tt[tidx, tidx2] + crossprod(Hm, Hm2 * (-w^2 * rm * r2))
    }
  }

  # Assemble
  place(H_bb, idx_beta, idx_beta)
  place(H_bTheta, idx_beta, idx_Theta)
  place(t(H_bTheta), idx_Theta, idx_beta)
  place(H_ThetaTh, idx_Theta, idx_Theta)

  place(H_aa, idx_alpha, idx_alpha)
  place(H_aTheta, idx_alpha, idx_Theta)
  place(t(H_aTheta), idx_Theta, idx_alpha)
  place(H_ThetaThEL, idx_Theta, idx_Theta)

  place(H_at, idx_alpha, idx_t)
  place(t(H_at), idx_t, idx_alpha)
  place(H_Thetat, idx_Theta, idx_t)
  place(t(H_Thetat), idx_t, idx_Theta)
  place(H_tt, idx_t, idx_t)

  # L2 penalty on tmat: add tau_tmat to H_tt diagonal
  diag(Hfull[idx_t, idx_t]) <- diag(Hfull[idx_t, idx_t]) + tau_tmat

  diag(Hfull) <- diag(Hfull) + lambda.diag
  Hfull <- 0.5 * (Hfull + t(Hfull))

  result <- list(grad = grad, hess = Hfull)
  if (return_shared) result$shared <- sh
  result
}
