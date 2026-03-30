#' Per-sample score matrix for hard-mode sandwich variance
#'
#' Computes the per-sample score (gradient) matrix used in score-based
#' sandwich variance estimation for the hard-mode (fixed-q) fused
#' multinomial model.
#'
#' @param par Hard-format parameter vector (beta, Theta, alpha, tmat).
#' @param X Design matrix (n x p).
#' @param y Integer response vector of length n (class labels 1..K).
#' @param qhat External probability matrix (n x (L-1)).
#' @param Hmat Basis matrix for EL constraints.
#' @param A Group matrix (K x L) from \code{group_matrix()}.
#'
#' @return An n x Dnoq matrix of per-sample score vectors, with columns
#'   ordered as (beta, Theta, alpha, t).
#'
#' @export
compute_per_sample_scores_hard <- function(par, X, y, qhat, Hmat, A) {
  up    <- unpack.hard(par)
  beta  <- up$beta; Theta <- up$Theta; alpha <- up$alpha; tmat <- up$tmat
  n     <- nrow(X); p <- ncol(X); K <- ncol(Theta) + 1; d <- K - 1
  L     <- ncol(A); Lm1 <- nrow(tmat); H <- ncol(Hmat) / Lm1

  # --- Primary model (beta, Theta) ---
  eta   <- X %*% Theta + matrix(rep(beta, each = n), nrow = n)
  Pmat  <- softmax_first(eta)                         # n x K
  Y     <- matrix(0, n, K); Y[cbind(seq_len(n), y)] <- 1
  R     <- Pmat[, 2:K, drop = FALSE] - Y[, 2:K, drop = FALSE]  # n x (K-1)

  # --- External model (alpha, Theta) ---
  eta_ext  <- X %*% Theta + matrix(rep(alpha, each = n), nrow = n)
  Pext_K   <- softmax_first(eta_ext)                   # n x K
  Pext_L   <- Pext_K %*% A                             # n x L

  # --- EL weights ---
  Svec <- rep(0, n)
  z_list <- v_list <- vector("list", Lm1)
  for (m in seq_len(Lm1)) {
    v_list[[m]] <- Pext_L[, m + 1] - qhat[, m]
    z_list[[m]] <- as.numeric(Hmat[, m + (0:(H-1)) * Lm1] %*% tmat[m, ])
    Svec <- Svec + v_list[[m]] * z_list[[m]]
  }
  w <- 1 / (1 + Svec)                                  # n-vector

  # --- EL gradient contribution to alpha/Theta (G_ext) ---
  z_matrix <- do.call(cbind, z_list)                   # n x Lm1
  A_sub    <- A[-1, -1, drop = FALSE]                  # Lm1 x (K-1)
  P_ext_nob <- Pext_K[, -1, drop = FALSE]              # n x (K-1)
  zw       <- (z_matrix %*% t(A_sub)) * w              # n x (K-1)
  s_val    <- rowSums(zw * P_ext_nob)                  # n
  G_ext    <- zw * P_ext_nob - P_ext_nob * s_val       # n x (K-1)

  # --- Per-sample scores ---
  # beta block: primary likelihood residuals
  sc_beta  <- -R                                        # n x (K-1)

  # alpha block: EL gradient w.r.t. external intercepts
  sc_alpha <- -G_ext                                    # n x (K-1)

  # Theta block: X x (R + G_ext)   [p*(K-1) columns]
  sc_Theta <- matrix(0, n, p * d)
  for (k in seq_len(d)) {
    sc_Theta[, ((k-1)*p + 1):(k*p)] <- X * (-(R[, k] + G_ext[, k]))
  }

  # t block: -w_i * v_m[i] * H[i,:]   [Lm1*H columns]
  sc_t <- matrix(0, n, Lm1 * H)
  for (m in seq_len(Lm1)) {
    cols_m <- (m - 1) * H + seq_len(H)
    Hm     <- Hmat[, m + (0:(H-1)) * Lm1, drop = FALSE]  # n x H
    sc_t[, cols_m] <- (-w) * v_list[[m]] * Hm
  }

  cbind(sc_beta, sc_Theta, sc_alpha, sc_t)  # n x Dnoq
}


#' Hard-mode sandwich inference with score-based and adjusted SEs
#'
#' Computes sandwich variance estimates for the hard-mode (fixed-q)
#' fused multinomial model, including Hessian-based, score-based, and
#' qhat-uncertainty-adjusted variants.
#'
#' @param par Hard-format parameter vector (beta, Theta, alpha, tmat).
#' @param X Design matrix (n x p).
#' @param y Integer response vector of length n (class labels 1..K).
#' @param qhat External probability matrix (n x (L-1)).
#' @param qhat.se Standard errors of external probabilities (n x (L-1)).
#' @param Hmat Basis matrix for EL constraints.
#' @param A Group matrix (K x L) from \code{group_matrix()}.
#' @param lambda.diag Ridge stabilization added to the Hessian diagonal. Default 0.
#'
#' @return A list with components:
#'   \describe{
#'     \item{Ihat}{Hessian-based meat matrix.}
#'     \item{Jhat}{Bread matrix.}
#'     \item{conf_matrix}{Hessian-based sandwich covariance.}
#'     \item{se}{Hessian-based standard errors.}
#'     \item{conf_matrix.adj}{Adjusted Hessian-based sandwich covariance.}
#'     \item{se.adj}{Adjusted Hessian-based standard errors.}
#'     \item{conf_matrix.score}{Score-based sandwich covariance.}
#'     \item{se.score}{Score-based standard errors.}
#'     \item{conf_matrix.score.adj}{Adjusted score-based sandwich covariance.}
#'     \item{se.score.adj}{Adjusted score-based standard errors.}
#'   }
#'
#' @export
conf.est.hard <- function(par, X, y, qhat, qhat.se, Hmat, A, lambda.diag = 0) {
  # par: hard-format (beta, Theta, alpha, tmat)
  up    <- unpack.hard(par)
  beta  <- up$beta; Theta <- up$Theta; alpha <- up$alpha; tmat <- up$tmat

  n   <- nrow(X); p <- ncol(X); K <- ncol(Theta) + 1; d <- K - 1
  L   <- ncol(A); Lm1 <- nrow(tmat); H <- ncol(Hmat) / Lm1

  hess0 <- obj.hessian.hard(par, X, y, qhat, Hmat, A, lambda.diag = lambda.diag, tau_tmat = 0)

  d_beta  <- d;  d_Theta <- p * d;  d_alpha <- d;  d_t <- Lm1 * H
  Dnoq    <- d_beta + d_Theta + d_alpha + d_t
  idx_beta  <- 1:d_beta
  idx_Theta <- (d_beta + 1):(d_beta + d_Theta)
  idx_alpha <- (d_beta + d_Theta + 1):(d_beta + d_Theta + d_alpha)
  idx_t     <- (d_beta + d_Theta + d_alpha + 1):Dnoq

  Jhat <- hess0 / n

  Ihat <- matrix(0, Dnoq, Dnoq)
  Ihat[idx_beta,  idx_beta]  <-  Jhat[idx_beta,  idx_beta]
  Ihat[idx_Theta, idx_Theta] <-  Jhat[idx_Theta, idx_Theta]
  # J_{phi_free} = 0 per Theorem 2: the alpha block stays zero
  Ihat[idx_t,     idx_t]     <- -Jhat[idx_t,     idx_t]

  conf_matrix <- ginv(Jhat) %*% Ihat %*% ginv(Jhat) / n
  se <- sqrt(diag(conf_matrix))
  attributes(se) <- attributes(par)

  # qhat-uncertainty adjustment on t-block
  var.qhat <- qhat.se^2
  adj.term <- matrix(0, nrow = d_t, ncol = d_t)
  for (i in seq_len(n)) {
    vv <- diag(c(var.qhat[i, , drop = TRUE]), nrow = Lm1, ncol = Lm1)
    hi <- as.numeric(Hmat[i, 1 + (0:(H-1)) * Lm1])
    adj.term <- adj.term + kronecker(tcrossprod(hi), vv)
  }

  Ihat.adj <- Ihat
  Ihat.adj[idx_t, idx_t] <- Ihat[idx_t, idx_t] + adj.term / n

  conf_matrix.adj <- ginv(Jhat) %*% Ihat.adj %*% ginv(Jhat) / n
  se.adj <- sqrt(diag(conf_matrix.adj))
  attributes(se.adj) <- attributes(par)

  # --- Score-based sandwich (full off-diagonal meat) ---
  score_mat  <- compute_per_sample_scores_hard(par, X, y, qhat, Hmat, A)
  Ihat_score <- crossprod(score_mat) / n               # Dnoq x Dnoq
  Jinv       <- ginv(Jhat)
  conf_matrix.score <- Jinv %*% Ihat_score %*% Jinv / n
  se.score <- sqrt(diag(conf_matrix.score))
  attributes(se.score) <- attributes(par)

  Ihat_score.adj <- Ihat_score
  Ihat_score.adj[idx_t, idx_t] <- Ihat_score[idx_t, idx_t] + adj.term / n
  conf_matrix.score.adj <- Jinv %*% Ihat_score.adj %*% Jinv / n
  se.score.adj <- sqrt(diag(conf_matrix.score.adj))
  attributes(se.score.adj) <- attributes(par)

  list(Ihat = Ihat, Jhat = Jhat,
       conf_matrix = conf_matrix, se = se,
       conf_matrix.adj = conf_matrix.adj, se.adj = se.adj,
       conf_matrix.score = conf_matrix.score, se.score = se.score,
       conf_matrix.score.adj = conf_matrix.score.adj, se.score.adj = se.score.adj)
}


#' Shared Theorem 3 computation core
#'
#' Internal helper that performs the shared core computation for
#' Theorem 3 standard errors, used by both \code{conf.est.thm3} and
#' \code{conf.est.thm3.plugin}.
#'
#' @param par Hard-format parameter vector (beta, Theta, alpha, tmat).
#' @param X Design matrix (n x p).
#' @param y Integer response vector of length n (class labels 1..K).
#' @param qhat External probability matrix (n x (L-1)).
#' @param qhat.se Standard errors of external probabilities (n x (L-1)).
#' @param Hmat Basis matrix for EL constraints.
#' @param A Group matrix (K x L) from \code{group_matrix()}.
#' @param hess0 Precomputed Hessian matrix.
#'
#' @return A list with components:
#'   \describe{
#'     \item{se}{Unadjusted SE vector (Dnoq), with NA for alpha and t blocks.}
#'     \item{se.adj}{Adjusted SE vector (Dnoq), with NA for alpha and t blocks.}
#'     \item{Sigma_theta}{Unadjusted theta-block covariance (scaled by 1/n).}
#'     \item{Sigma_theta_adj}{Adjusted theta-block covariance (scaled by 1/n).}
#'     \item{I_theta_inv}{Inverse Fisher information for theta (scaled by 1/n).}
#'   }
#'
#' @keywords internal
.thm3_core_orig <- function(par, X, y, qhat, qhat.se, Hmat, A, hess0) {
  up    <- unpack.hard(par)
  n   <- nrow(X); p <- ncol(X); K <- ncol(up$Theta) + 1; d <- K - 1
  L   <- ncol(A); Lm1 <- nrow(up$tmat); H <- ncol(Hmat) / Lm1

  d_beta  <- d;  d_Theta <- p * d;  d_alpha <- d;  d_t <- Lm1 * H
  Dnoq    <- d_beta + d_Theta + d_alpha + d_t
  idx_beta  <- 1:d_beta
  idx_Theta <- (d_beta + 1):(d_beta + d_Theta)
  idx_alpha <- (d_beta + d_Theta + 1):(d_beta + d_Theta + d_alpha)
  idx_t     <- (d_beta + d_Theta + d_alpha + 1):Dnoq
  idx_theta <- c(idx_beta, idx_Theta)

  G <- hess0 / n

  G_tt      <- G[idx_t, idx_t, drop = FALSE]
  G_t_theta <- G[idx_t, idx_theta, drop = FALSE]
  G_theta_t <- G[idx_theta, idx_t, drop = FALSE]
  G_t_phi   <- G[idx_t, idx_alpha, drop = FALSE]
  G_phi_t   <- G[idx_alpha, idx_t, drop = FALSE]
  I_theta   <- G[idx_theta, idx_theta, drop = FALSE]

  I_theta_inv <- ginv(I_theta)

  compute_sigma <- function(J_t_val) {
    D <- -J_t_val - G_t_theta %*% I_theta_inv %*% G_theta_t
    D_inv <- ginv(D)
    inner     <- G_phi_t %*% D_inv %*% G_t_phi
    inner_inv <- ginv(inner)
    L_t_theta <- (D_inv - D_inv %*% G_t_phi %*% inner_inv %*% G_phi_t %*% D_inv) %*%
                  G_t_theta %*% I_theta_inv
    I_theta_inv + t(L_t_theta) %*% D %*% L_t_theta
  }

  build_se_vec <- function(se_theta) {
    se_vec <- numeric(Dnoq)
    se_vec[idx_theta] <- se_theta
    se_vec[idx_alpha] <- NA_real_
    se_vec[idx_t]     <- NA_real_
    attributes(se_vec) <- attributes(par)
    se_vec
  }

  # Unadjusted
  J_t <- -G_tt
  Sigma_theta <- compute_sigma(J_t)
  se_vec <- build_se_vec(sqrt(pmax(diag(Sigma_theta) / n, 0)))

  # Adjusted
  var.qhat <- qhat.se^2
  adj.term <- matrix(0, nrow = d_t, ncol = d_t)
  for (i in seq_len(n)) {
    vv <- diag(c(var.qhat[i, , drop = TRUE]), nrow = Lm1, ncol = Lm1)
    hi <- as.numeric(Hmat[i, 1 + (0:(H-1)) * Lm1])
    adj.term <- adj.term + kronecker(tcrossprod(hi), vv)
  }

  J_t_adj <- J_t + adj.term / n
  Sigma_theta_adj <- compute_sigma(J_t_adj)
  se_adj_vec <- build_se_vec(sqrt(pmax(diag(Sigma_theta_adj) / n, 0)))

  list(se = se_vec, se.adj = se_adj_vec,
       Sigma_theta = Sigma_theta / n, Sigma_theta_adj = Sigma_theta_adj / n,
       I_theta_inv = I_theta_inv / n)
}


#' Theorem 3 standard errors
#'
#' Computes standard errors using Theorem 3 of the fused multinomial
#' model, evaluating the Hessian at t=0 (unfused model).
#'
#' @param par Hard-format parameter vector (beta, Theta, alpha, tmat).
#' @param X Design matrix (n x p).
#' @param y Integer response vector of length n (class labels 1..K).
#' @param qhat External probability matrix (n x (L-1)).
#' @param qhat.se Standard errors of external probabilities (n x (L-1)).
#' @param Hmat Basis matrix for EL constraints.
#' @param A Group matrix (K x L) from \code{group_matrix()}.
#'
#' @return A list with components:
#'   \describe{
#'     \item{se.thm3}{Unadjusted Theorem 3 SE vector.}
#'     \item{se.thm3.adj}{Adjusted Theorem 3 SE vector.}
#'     \item{Sigma_theta}{Unadjusted theta-block covariance.}
#'     \item{Sigma_theta_adj}{Adjusted theta-block covariance.}
#'     \item{I_theta_inv}{Inverse Fisher information for theta.}
#'   }
#'
#' @export
conf.est.thm3 <- function(par, X, y, qhat, qhat.se, Hmat, A) {
  up    <- unpack.hard(par)
  Lm1 <- nrow(up$tmat); H <- ncol(Hmat) / Lm1
  tmat0 <- matrix(0, nrow = Lm1, ncol = H)
  par0  <- pack.hard(up$beta, up$Theta, up$alpha, tmat0)
  hess0 <- obj.hessian.hard(par0, X, y, qhat, Hmat, A, lambda.diag = 0, tau_tmat = 0)

  res <- .thm3_core_orig(par, X, y, qhat, qhat.se, Hmat, A, hess0)
  list(se.thm3 = res$se, se.thm3.adj = res$se.adj,
       Sigma_theta = res$Sigma_theta, Sigma_theta_adj = res$Sigma_theta_adj,
       I_theta_inv = res$I_theta_inv)
}


#' Theorem 3 plugin standard errors
#'
#' Computes standard errors using Theorem 3 of the fused multinomial
#' model with a plugin Hessian (evaluated at the converged parameter
#' values rather than at t=0).
#'
#' @param par Hard-format parameter vector (beta, Theta, alpha, tmat).
#' @param X Design matrix (n x p).
#' @param y Integer response vector of length n (class labels 1..K).
#' @param qhat External probability matrix (n x (L-1)).
#' @param qhat.se Standard errors of external probabilities (n x (L-1)).
#' @param Hmat Basis matrix for EL constraints.
#' @param A Group matrix (K x L) from \code{group_matrix()}.
#'
#' @return A list with components:
#'   \describe{
#'     \item{se.thm3.plugin}{Unadjusted plugin Theorem 3 SE vector.}
#'     \item{se.thm3.plugin.adj}{Adjusted plugin Theorem 3 SE vector.}
#'     \item{Sigma_theta}{Unadjusted theta-block covariance.}
#'     \item{Sigma_theta_adj}{Adjusted theta-block covariance.}
#'     \item{I_theta_inv}{Inverse Fisher information for theta.}
#'   }
#'
#' @export
conf.est.thm3.plugin <- function(par, X, y, qhat, qhat.se, Hmat, A) {
  hess0 <- obj.hessian.hard(par, X, y, qhat, Hmat, A, lambda.diag = 0, tau_tmat = 0)

  res <- .thm3_core_orig(par, X, y, qhat, qhat.se, Hmat, A, hess0)
  list(se.thm3.plugin = res$se, se.thm3.plugin.adj = res$se.adj,
       Sigma_theta = res$Sigma_theta, Sigma_theta_adj = res$Sigma_theta_adj,
       I_theta_inv = res$I_theta_inv)
}


#' Fast hard-mode sandwich inference
#'
#' Optimized version of \code{\link{conf.est.hard}} that uses the fast
#' direct Hessian computation (\code{obj.hessian.hard}) and
#' vectorized qhat-uncertainty adjustment (\code{compute_adj_term}).
#' Supports optional precomputed Hessian to avoid redundant computation.
#'
#' @param par Hard-format parameter vector (beta, Theta, alpha, tmat).
#' @param X Design matrix (n x p).
#' @param y Integer response vector of length n (class labels 1..K).
#' @param qhat External probability matrix (n x (L-1)).
#' @param qhat.se Standard errors of external probabilities (n x (L-1)).
#' @param Hmat Basis matrix for EL constraints.
#' @param A Group matrix (K x L) from \code{group_matrix()}.
#' @param lambda.diag Ridge stabilization added to the Hessian diagonal. Default 0.
#' @param hess_precomputed Optional precomputed Hessian matrix. If NULL (default),
#'   the Hessian is computed internally via \code{obj.hessian.hard}.
#'
#' @return A list with the same structure as \code{\link{conf.est.hard}}.
#'
#' @export
conf.est.hard.fast <- function(par, X, y, qhat, qhat.se, Hmat, A,
                               lambda.diag = 0, hess_precomputed = NULL) {
  up   <- unpack.hard(par)
  n    <- nrow(X); p <- ncol(X); K <- ncol(up$Theta) + 1; d <- K - 1
  L    <- ncol(A); Lm1 <- nrow(up$tmat); H <- ncol(Hmat) / Lm1

  hess0 <- if (!is.null(hess_precomputed)) hess_precomputed else
           obj.hessian.hard(par, X, y, qhat, Hmat, A, lambda.diag = lambda.diag)

  d_beta <- d; d_Theta <- p * d; d_alpha <- d; d_t <- Lm1 * H
  Dnoq <- d_beta + d_Theta + d_alpha + d_t
  idx_beta  <- 1:d_beta
  idx_Theta <- (d_beta + 1):(d_beta + d_Theta)
  idx_alpha <- (d_beta + d_Theta + 1):(d_beta + d_Theta + d_alpha)
  idx_t     <- (d_beta + d_Theta + d_alpha + 1):Dnoq

  Jhat <- hess0 / n

  Ihat <- matrix(0, Dnoq, Dnoq)
  Ihat[idx_beta,  idx_beta]  <-  Jhat[idx_beta,  idx_beta]
  Ihat[idx_Theta, idx_Theta] <-  Jhat[idx_Theta, idx_Theta]
  Ihat[idx_t,     idx_t]     <- -Jhat[idx_t,     idx_t]

  Jinv <- ginv(Jhat)
  conf_matrix <- Jinv %*% Ihat %*% Jinv / n
  se <- sqrt(diag(conf_matrix))
  attributes(se) <- attributes(par)

  # Vectorized adj.term
  var.qhat <- qhat.se^2
  adj.term <- compute_adj_term(Hmat, var.qhat, Lm1, H)

  Ihat.adj <- Ihat
  Ihat.adj[idx_t, idx_t] <- Ihat[idx_t, idx_t] + adj.term / n

  conf_matrix.adj <- Jinv %*% Ihat.adj %*% Jinv / n
  se.adj <- sqrt(diag(conf_matrix.adj))
  attributes(se.adj) <- attributes(par)

  # Score-based sandwich
  score_mat <- compute_per_sample_scores_hard(par, X, y, qhat, Hmat, A)
  Ihat_score <- crossprod(score_mat) / n
  conf_matrix.score <- Jinv %*% Ihat_score %*% Jinv / n
  se.score <- sqrt(diag(conf_matrix.score))
  attributes(se.score) <- attributes(par)

  Ihat_score.adj <- Ihat_score
  Ihat_score.adj[idx_t, idx_t] <- Ihat_score[idx_t, idx_t] + adj.term / n
  conf_matrix.score.adj <- Jinv %*% Ihat_score.adj %*% Jinv / n
  se.score.adj <- sqrt(diag(conf_matrix.score.adj))
  attributes(se.score.adj) <- attributes(par)

  list(Ihat = Ihat, Jhat = Jhat,
       conf_matrix = conf_matrix, se = se,
       conf_matrix.adj = conf_matrix.adj, se.adj = se.adj,
       conf_matrix.score = conf_matrix.score, se.score = se.score,
       conf_matrix.score.adj = conf_matrix.score.adj, se.score.adj = se.score.adj)
}


#' Fast Theorem 3 standard errors
#'
#' Optimized version of \code{\link{conf.est.thm3}} that uses the fast
#' direct Hessian computation and vectorized qhat-uncertainty adjustment.
#' Evaluates the Hessian at t=0 (unfused model).
#'
#' @param par Hard-format parameter vector (beta, Theta, alpha, tmat).
#' @param X Design matrix (n x p).
#' @param y Integer response vector of length n (class labels 1..K).
#' @param qhat External probability matrix (n x (L-1)).
#' @param qhat.se Standard errors of external probabilities (n x (L-1)).
#' @param Hmat Basis matrix for EL constraints.
#' @param A Group matrix (K x L) from \code{group_matrix()}.
#' @param hess_precomputed Optional precomputed Hessian matrix (at t=0). If NULL
#'   (default), the Hessian is computed internally.
#'
#' @return A list with components:
#'   \describe{
#'     \item{se.thm3}{Unadjusted Theorem 3 SE vector.}
#'     \item{se.thm3.adj}{Adjusted Theorem 3 SE vector.}
#'     \item{Sigma_theta}{Unadjusted theta-block covariance.}
#'     \item{Sigma_theta_adj}{Adjusted theta-block covariance.}
#'     \item{I_theta_inv}{Inverse Fisher information for theta.}
#'   }
#'
#' @export
conf.est.thm3.fast <- function(par, X, y, qhat, qhat.se, Hmat, A,
                               hess_precomputed = NULL) {
  up   <- unpack.hard(par)
  beta <- up$beta; Theta <- up$Theta; alpha <- up$alpha; tmat <- up$tmat
  n    <- nrow(X); p <- ncol(X); K <- ncol(Theta) + 1; d <- K - 1
  L    <- ncol(A); Lm1 <- nrow(tmat); H <- ncol(Hmat) / Lm1

  # Theorem 3 requires Hessian at t=0
  tmat0 <- matrix(0, nrow = Lm1, ncol = H)
  par0  <- pack.hard(beta, Theta, alpha, tmat0)
  hess0 <- if (!is.null(hess_precomputed)) hess_precomputed else
           obj.hessian.hard(par0, X, y, qhat, Hmat, A, lambda.diag = 0)

  d_beta <- d; d_Theta <- p * d; d_alpha <- d; d_t <- Lm1 * H
  Dnoq <- d_beta + d_Theta + d_alpha + d_t
  idx_beta  <- 1:d_beta
  idx_Theta <- (d_beta + 1):(d_beta + d_Theta)
  idx_alpha <- (d_beta + d_Theta + 1):(d_beta + d_Theta + d_alpha)
  idx_t     <- (d_beta + d_Theta + d_alpha + 1):Dnoq
  idx_theta <- c(idx_beta, idx_Theta)

  G <- hess0 / n

  G_tt      <- G[idx_t, idx_t, drop = FALSE]
  G_t_theta <- G[idx_t, idx_theta, drop = FALSE]
  G_theta_t <- G[idx_theta, idx_t, drop = FALSE]
  G_t_phi   <- G[idx_t, idx_alpha, drop = FALSE]
  G_phi_t   <- G[idx_alpha, idx_t, drop = FALSE]
  I_theta   <- G[idx_theta, idx_theta, drop = FALSE]

  I_theta_inv <- ginv(I_theta)

  compute_sigma <- function(J_t_val) {
    D <- -J_t_val - G_t_theta %*% I_theta_inv %*% G_theta_t
    D_inv <- ginv(D)
    inner     <- G_phi_t %*% D_inv %*% G_t_phi
    inner_inv <- ginv(inner)
    L_t_theta <- (D_inv - D_inv %*% G_t_phi %*% inner_inv %*% G_phi_t %*% D_inv) %*%
                  G_t_theta %*% I_theta_inv
    I_theta_inv + t(L_t_theta) %*% D %*% L_t_theta
  }

  # Unadjusted
  J_t <- -G_tt
  Sigma_theta <- compute_sigma(J_t)
  se_theta <- sqrt(pmax(diag(Sigma_theta) / n, 0))

  se_vec <- numeric(Dnoq)
  se_vec[idx_theta] <- se_theta
  se_vec[idx_alpha] <- NA_real_
  se_vec[idx_t]     <- NA_real_
  attributes(se_vec) <- attributes(par)

  # Adjusted
  var.qhat <- qhat.se^2
  adj.term <- compute_adj_term(Hmat, var.qhat, Lm1, H)

  J_t_adj <- J_t + adj.term / n
  Sigma_theta_adj <- compute_sigma(J_t_adj)
  se_theta_adj <- sqrt(pmax(diag(Sigma_theta_adj) / n, 0))

  se_adj_vec <- numeric(Dnoq)
  se_adj_vec[idx_theta] <- se_theta_adj
  se_adj_vec[idx_alpha] <- NA_real_
  se_adj_vec[idx_t]     <- NA_real_
  attributes(se_adj_vec) <- attributes(par)

  list(se.thm3 = se_vec, se.thm3.adj = se_adj_vec,
       Sigma_theta = Sigma_theta / n, Sigma_theta_adj = Sigma_theta_adj / n,
       I_theta_inv = I_theta_inv / n)
}


#' Fast Theorem 3 plugin standard errors
#'
#' Optimized version of \code{\link{conf.est.thm3.plugin}} that uses
#' the fast direct Hessian computation and vectorized qhat-uncertainty
#' adjustment. Uses the Hessian evaluated at the converged parameter
#' values (plugin approach).
#'
#' @param par Hard-format parameter vector (beta, Theta, alpha, tmat).
#' @param X Design matrix (n x p).
#' @param y Integer response vector of length n (class labels 1..K).
#' @param qhat External probability matrix (n x (L-1)).
#' @param qhat.se Standard errors of external probabilities (n x (L-1)).
#' @param Hmat Basis matrix for EL constraints.
#' @param A Group matrix (K x L) from \code{group_matrix()}.
#' @param hess_precomputed Optional precomputed Hessian matrix (at converged params).
#'   If NULL (default), the Hessian is computed internally.
#'
#' @return A list with components:
#'   \describe{
#'     \item{se.thm3.plugin}{Unadjusted plugin Theorem 3 SE vector.}
#'     \item{se.thm3.plugin.adj}{Adjusted plugin Theorem 3 SE vector.}
#'     \item{Sigma_theta}{Unadjusted theta-block covariance.}
#'     \item{Sigma_theta_adj}{Adjusted theta-block covariance.}
#'     \item{I_theta_inv}{Inverse Fisher information for theta.}
#'   }
#'
#' @export
conf.est.thm3.plugin.fast <- function(par, X, y, qhat, qhat.se, Hmat, A,
                                      hess_precomputed = NULL) {
  up   <- unpack.hard(par)
  beta <- up$beta; Theta <- up$Theta; alpha <- up$alpha; tmat <- up$tmat
  n    <- nrow(X); p <- ncol(X); K <- ncol(Theta) + 1; d <- K - 1
  L    <- ncol(A); Lm1 <- nrow(tmat); H <- ncol(Hmat) / Lm1

  # Plugin uses Hessian at converged tmat (not t=0)
  hess0 <- if (!is.null(hess_precomputed)) hess_precomputed else
           obj.hessian.hard(par, X, y, qhat, Hmat, A, lambda.diag = 0)

  d_beta <- d; d_Theta <- p * d; d_alpha <- d; d_t <- Lm1 * H
  Dnoq <- d_beta + d_Theta + d_alpha + d_t
  idx_beta  <- 1:d_beta
  idx_Theta <- (d_beta + 1):(d_beta + d_Theta)
  idx_alpha <- (d_beta + d_Theta + 1):(d_beta + d_Theta + d_alpha)
  idx_t     <- (d_beta + d_Theta + d_alpha + 1):Dnoq
  idx_theta <- c(idx_beta, idx_Theta)

  G <- hess0 / n

  G_tt      <- G[idx_t, idx_t, drop = FALSE]
  G_t_theta <- G[idx_t, idx_theta, drop = FALSE]
  G_theta_t <- G[idx_theta, idx_t, drop = FALSE]
  G_t_phi   <- G[idx_t, idx_alpha, drop = FALSE]
  G_phi_t   <- G[idx_alpha, idx_t, drop = FALSE]
  I_theta   <- G[idx_theta, idx_theta, drop = FALSE]

  I_theta_inv <- ginv(I_theta)

  compute_sigma <- function(J_t_val) {
    D <- -J_t_val - G_t_theta %*% I_theta_inv %*% G_theta_t
    D_inv <- ginv(D)
    inner     <- G_phi_t %*% D_inv %*% G_t_phi
    inner_inv <- ginv(inner)
    L_t_theta <- (D_inv - D_inv %*% G_t_phi %*% inner_inv %*% G_phi_t %*% D_inv) %*%
                  G_t_theta %*% I_theta_inv
    I_theta_inv + t(L_t_theta) %*% D %*% L_t_theta
  }

  J_t <- -G_tt
  Sigma_theta <- compute_sigma(J_t)
  se_theta <- sqrt(pmax(diag(Sigma_theta) / n, 0))

  se_vec <- numeric(Dnoq)
  se_vec[idx_theta] <- se_theta
  se_vec[idx_alpha] <- NA_real_
  se_vec[idx_t]     <- NA_real_
  attributes(se_vec) <- attributes(par)

  var.qhat <- qhat.se^2
  adj.term <- compute_adj_term(Hmat, var.qhat, Lm1, H)

  J_t_adj <- J_t + adj.term / n
  Sigma_theta_adj <- compute_sigma(J_t_adj)
  se_theta_adj <- sqrt(pmax(diag(Sigma_theta_adj) / n, 0))

  se_adj_vec <- numeric(Dnoq)
  se_adj_vec[idx_theta] <- se_theta_adj
  se_adj_vec[idx_alpha] <- NA_real_
  se_adj_vec[idx_t]     <- NA_real_
  attributes(se_adj_vec) <- attributes(par)

  list(se.thm3.plugin = se_vec, se.thm3.plugin.adj = se_adj_vec,
       Sigma_theta = Sigma_theta / n, Sigma_theta_adj = Sigma_theta_adj / n,
       I_theta_inv = I_theta_inv / n)
}
