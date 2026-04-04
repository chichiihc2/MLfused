#' Sandwich Standard Errors
#'
#' Computes Hessian-based sandwich variance estimates for the fused
#' multinomial model.
#'
#' @param par Packed hard-mode parameter vector.
#' @param X n x p covariate matrix.
#' @param y Integer response vector.
#' @param qhat n x (L-1) external probability matrix.
#' @param Hmat Basis matrix.
#' @param A K x L group matrix.
#' @param lambda.diag Ridge stabilization (default 0).
#' @param hess_precomputed Optional precomputed Hessian (at tau_tmat=0).
#'
#' @return List with:
#'   \describe{
#'     \item{se}{Sandwich standard errors (numeric vector with dims attribute).}
#'     \item{conf_matrix}{Sandwich covariance matrix.}
#'     \item{Ihat}{Meat matrix.}
#'     \item{Jhat}{Bread matrix.}
#'   }
#' @export
sandwich_se <- function(par, X, y, qhat, Hmat, A,
                        lambda.diag = 0, hess_precomputed = NULL) {
  up  <- unpack_hard(par)
  n   <- nrow(X); p <- ncol(X); K <- ncol(up$Theta) + 1; d <- K - 1
  Lm1 <- nrow(up$tmat); H <- ncol(Hmat) / Lm1

  hess0 <- if (!is.null(hess_precomputed)) hess_precomputed else
    hessian_hard(par, X, y, qhat, Hmat, A,
                 lambda.diag = lambda.diag, tau_tmat = 0)

  ix <- make_hard_indices(d, p, Lm1, H)

  Jhat <- hess0 / n
  Jinv <- MASS::ginv(Jhat)

  Ihat <- matrix(0, ix$Dnoq, ix$Dnoq)
  Ihat[ix$idx_beta,  ix$idx_beta]  <-  Jhat[ix$idx_beta,  ix$idx_beta]
  Ihat[ix$idx_Theta, ix$idx_Theta] <-  Jhat[ix$idx_Theta, ix$idx_Theta]
  Ihat[ix$idx_t,     ix$idx_t]     <- -Jhat[ix$idx_t,     ix$idx_t]

  conf_matrix <- Jinv %*% Ihat %*% Jinv / n
  dvar <- diag(conf_matrix)
  if (any(dvar < 0))
    warning("Negative variance estimates detected; sandwich covariance may be unreliable.")
  se <- sqrt(pmax(dvar, 0))
  attributes(se) <- attributes(par)

  list(Ihat = Ihat, Jhat = Jhat, conf_matrix = conf_matrix, se = se)
}
