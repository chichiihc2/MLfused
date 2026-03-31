#' Pack Hard-Mode Parameters into a Single Vector
#'
#' Concatenates the four parameter blocks into a single numeric vector
#' with dimension metadata. Packing order: (beta, Theta, alpha, tmat).
#'
#' @param beta Numeric vector of length K-1 (primary model intercepts).
#' @param Theta Numeric matrix p x (K-1) (primary model coefficients).
#' @param alpha Numeric vector of length K-1 (external model intercepts).
#' @param tmat Numeric matrix (L-1) x H (EL constraint parameters).
#'
#' @return A numeric vector with a \code{dims} attribute storing
#'   \code{p}, \code{K}, \code{L}, \code{H}.
#'
#' @export
pack_hard <- function(beta, Theta, alpha, tmat) {
  dims <- list(p = nrow(Theta), K = ncol(Theta) + 1,
               L = nrow(tmat) + 1, H = ncol(tmat))
  par <- c(beta, as.vector(Theta), alpha, as.vector(tmat))
  attr(par, "dims") <- dims
  par
}


#' Unpack Hard-Mode Parameters
#'
#' Reverses \code{\link{pack_hard}}, recovering the four parameter blocks.
#'
#' @param par A packed parameter vector with a \code{dims} attribute
#'   (from \code{\link{pack_hard}}).
#'
#' @return A named list: \code{beta}, \code{Theta}, \code{alpha}, \code{tmat}.
#'
#' @export
unpack_hard <- function(par) {
  d <- attr(par, "dims")
  p <- d$p; K <- d$K; L <- d$L; H <- d$H; Lm1 <- L - 1
  idx <- 0
  beta  <- par[(idx + 1):(idx + K - 1)];                                      idx <- idx + (K - 1)
  Theta <- matrix(par[(idx + 1):(idx + p * (K - 1))], nrow = p, ncol = K - 1); idx <- idx + p * (K - 1)
  alpha <- par[(idx + 1):(idx + K - 1)];                                      idx <- idx + (K - 1)
  tmat  <- matrix(par[(idx + 1):(idx + Lm1 * H)], nrow = Lm1, byrow = FALSE)
  list(beta = beta, Theta = Theta, alpha = alpha, tmat = tmat)
}
