#' Build Constraint Basis Matrix
#'
#' Constructs a basis matrix H for the empirical-likelihood constraint,
#' using either centered raw covariates (\code{n.q <= 0}) or natural cubic
#' spline bases (\code{n.q > 0}).
#'
#' @param x_mat A data frame or matrix of covariates.
#' @param phi_idx Integer vector of column indices to use.
#' @param n.q Number of internal knots per feature. If \code{<= 0}, uses
#'   centered raw covariates. Default 2.
#'
#' @return An n x H matrix. Column 1 is an intercept. Attributes include
#'   \code{basis}, \code{n.q}, \code{knots}, \code{degree}.
#'
#' @export
build_H <- function(x_mat, phi_idx, n.q = 2) {
  stopifnot(length(phi_idx) >= 1)
  if (!is.data.frame(x_mat)) x_mat <- as.data.frame(x_mat)

  phi <- x_mat[, phi_idx, drop = FALSE]
  n   <- nrow(phi)

  H_list <- list(h0 = rep(1, n))
  knots_info <- vector("list", length(phi_idx))

  if (n.q <= 0) {
    for (j in seq_along(phi_idx)) {
      H_list[[paste0("phi", j)]] <- phi[[j]] - mean(phi[[j]])
    }
    Hmat <- as.matrix(as.data.frame(H_list))
    attr(Hmat, "basis")  <- "centered_raw"
    attr(Hmat, "n.q")    <- n.q
    attr(Hmat, "knots")  <- NULL
    attr(Hmat, "degree") <- NA_integer_
    return(Hmat)
  }

  probs <- seq(1, n.q) / (n.q + 1)

  for (j in seq_along(phi_idx)) {
    xj  <- phi[[j]]
    kj  <- unique(stats::quantile(xj, probs = probs, na.rm = TRUE, names = FALSE))
    bnd <- range(xj, na.rm = TRUE)

    Bj <- splines::ns(xj, knots = if (length(kj)) kj else NULL,
                      Boundary.knots = bnd, intercept = FALSE)
    Bj <- as.matrix(Bj)

    if (ncol(Bj) > 0) {
      colnames(Bj) <- paste0("ns_phi", j, "_b", seq_len(ncol(Bj)))
      for (k in seq_len(ncol(Bj))) {
        H_list[[colnames(Bj)[k]]] <- Bj[, k]
      }
    }

    knots_info[[j]] <- list(
      feature        = phi_idx[j],
      knots          = kj,
      boundary_knots = bnd,
      degree         = 3,
      type           = "natural"
    )
  }

  Hmat <- as.matrix(as.data.frame(H_list))
  attr(Hmat, "basis")  <- "natural_spline"
  attr(Hmat, "n.q")    <- n.q
  attr(Hmat, "knots")  <- knots_info
  attr(Hmat, "degree") <- 3
  Hmat
}


#' Interleave Basis Matrix for Multiple Groups
#'
#' Replicates a single-group basis matrix into an interleaved layout
#' for Lm1 non-reference groups. Column ordering: for basis k and
#' group m, output column = m + (k-1)*Lm1.
#'
#' @param H_base An n x H matrix (single-group basis).
#' @param Lm1 Integer. Number of non-reference groups (L - 1).
#'
#' @return An n x (H * Lm1) interleaved matrix.
#'
#' @export
build_Hmat_interleaved <- function(H_base, Lm1) {
  n <- nrow(H_base)
  H <- ncol(H_base)
  out <- matrix(0, n, H * Lm1)
  for (k in seq_len(H)) {
    for (m in seq_len(Lm1)) {
      out[, m + (k - 1) * Lm1] <- H_base[, k]
    }
  }
  out
}
