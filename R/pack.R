#' Pack Hard-Mode Parameters into a Single Vector
#'
#' Concatenates model parameters for the empirical-likelihood
#' fused multinomial model into a single numeric vector.
#' The packing order is \code{(beta, Theta, alpha, tmat)}.
#'
#' @param beta Numeric vector of length K-1. Intercepts for the primary
#'   multinomial model.
#' @param Theta Numeric matrix of dimension p x (K-1). Coefficient matrix
#'   for the primary model.
#' @param alpha Numeric vector of length K-1. Intercepts for the external
#'   model.
#' @param tmat Numeric matrix of dimension (L-1) x H. Constraint parameters
#'   for the empirical-likelihood term.
#'
#' @return A numeric vector containing all parameters, with a \code{"dims"}
#'   attribute storing a list of dimension information
#'   (\code{p}, \code{K}, \code{L}, \code{H}).
#'
#' @export
pack.hard <- function(beta, Theta, alpha, tmat) {
  dims <- list(p = nrow(Theta), K = ncol(Theta)+1, L = nrow(tmat)+1, H = ncol(tmat))
  par  <- c(beta, as.vector(Theta), alpha, as.vector(tmat))
  attr(par, "dims") <- dims
  par
}

#' Unpack Hard-Mode Parameters from a Packed Vector
#'
#' Reverses \code{\link{pack.hard}}, extracting individual parameter
#' components from a single numeric vector using the dimension metadata
#' stored in its \code{"dims"} attribute.
#'
#' @param par A numeric vector produced by \code{\link{pack.hard}}, with
#'   a \code{"dims"} attribute.
#'
#' @return A named list with elements:
#'   \describe{
#'     \item{beta}{Numeric vector of length K-1. Primary model intercepts.}
#'     \item{Theta}{Numeric matrix of dimension p x (K-1). Primary model
#'       coefficients.}
#'     \item{alpha}{Numeric vector of length K-1. External model intercepts.}
#'     \item{tmat}{Numeric matrix of dimension (L-1) x H. Constraint
#'       parameters.}
#'   }
#'
#' @export
unpack.hard <- function(par) {
  d   <- attr(par, "dims")
  p   <- d$p; K <- d$K; L <- d$L; H <- d$H; Lm1 <- L - 1
  idx <- 0
  beta  <- par[(idx+1):(idx+K-1)];                                idx <- idx + (K-1)
  Theta <- matrix(par[(idx+1):(idx+p*(K-1))], nrow=p, ncol=K-1); idx <- idx + p*(K-1)
  alpha <- par[(idx+1):(idx+K-1)];                                idx <- idx + (K-1)
  tmat  <- matrix(par[(idx+1):(idx+Lm1*H)], nrow=Lm1, byrow=FALSE)
  list(beta=beta, Theta=Theta, alpha=alpha, tmat=tmat)
}
