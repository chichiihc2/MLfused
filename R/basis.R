#' Build H constraint basis matrix
#'
#' Constructs a constraint basis matrix using an intercept column plus either
#' centered raw features (when \code{n.q <= 0}) or natural cubic spline bases
#' (when \code{n.q > 0}) for selected columns of the input matrix.
#'
#' @param x_mat A data frame or matrix of covariates (n rows).
#' @param phi_idx Integer vector of column indices in \code{x_mat} to include
#'   in the basis.
#' @param n.q Number of internal knots per feature for the natural spline basis.
#'   If \code{n.q <= 0}, centered raw features are used instead of splines.
#'   Defaults to \code{2}.
#'
#' @return A numeric matrix with \code{n} rows. The first column is an
#'   intercept (all ones). Remaining columns are either centered raw features
#'   (\code{n.q <= 0}) or natural cubic spline basis columns (\code{n.q > 0}).
#'   The matrix carries the following attributes:
#'   \describe{
#'     \item{\code{basis}}{Character string \code{"natural_spline"}.}
#'     \item{\code{n.q}}{The number of internal knots requested.}
#'     \item{\code{knots}}{A list of knot information per feature (or
#'       \code{NULL} when \code{n.q <= 0}).}
#'     \item{\code{degree}}{Integer \code{3} (cubic).}
#'   }
#'
#' @export
build_H <- function(x_mat, phi_idx, n.q = 2) {
  # Natural spline version (cubic, natural boundary conditions)
  # n.q = number of INTERNAL knots per feature; if <= 0 -> x only (no splines)

  stopifnot(length(phi_idx) >= 1)
  if (!is.data.frame(x_mat)) x_mat <- as.data.frame(x_mat)

  phi <- x_mat[, phi_idx, drop = FALSE]
  n   <- nrow(phi)

  # Start with intercept
  H_list <- list(h0 = rep(1, n))
  knots_info <- vector("list", length(phi_idx))

  # q = 0 -> just intercept + raw x (if include_x = TRUE)
  if (n.q <= 0) {
    for (j in seq_along(phi_idx)) {
      H_list[[paste0("phi", j)]] <- phi[[j]]-mean(phi[[j]])
    }

    Hmat <- as.matrix(as.data.frame(H_list))
    attr(Hmat, "basis")  <- "natural_spline"
    attr(Hmat, "n.q")    <- n.q
    attr(Hmat, "knots")  <- NULL
    attr(Hmat, "degree") <- 3
    return(Hmat)
  }

  # q > 0 -> natural spline basis per feature
  if (!requireNamespace("splines", quietly = TRUE)) {
    stop("Package 'splines' is required. Please install.packages('splines').")
  }
  probs <- seq(1, n.q) / (n.q + 1)

  for (j in seq_along(phi_idx)) {
    xj <- phi[[j]]

    # Internal knots at quantiles; drop duplicates if data are discrete/tied
    kj  <- unique(stats::quantile(xj, probs = probs, na.rm = TRUE, names = FALSE))
    bnd <- range(xj, na.rm = TRUE)

    # Build natural spline basis (cubic, natural boundary)
    # If kj happens to be empty (e.g., ties), ns() still works.
    Bj <- splines::ns(xj, knots = if (length(kj)) kj else NULL,
                      Boundary.knots = bnd, intercept = F)
    Bj <- as.matrix(Bj)

    # Name columns
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
  return(Hmat)
}

#' Interleave H basis columns for multinomial grouping
#'
#' Takes an \code{n x H_per} base constraint matrix and interleaves its columns
#' for \code{Lm1} non-reference groups. The output column ordering follows the
#' convention used by the EL constraint: for basis index \code{k} and
#' non-reference group index \code{m}, the output column is
#' \code{m + (k - 1) * Lm1} (1-based).
#'
#' @param H_base Numeric matrix of dimension \code{n x H_per} (e.g., intercept
#'   plus centered covariates or spline bases).
#' @param Lm1 Integer; the number of non-reference groups (\code{L - 1}).
#'
#' @return A numeric matrix of dimension \code{n x (H_per * Lm1)} with
#'   interleaved columns.
#'
#' @export
build_Hmat_interleaved <- function(H_base, Lm1) {
  # H_base: n x H  (intercept + centered covariates)
  # Lm1: number of non-reference groups
  # Returns: n x (H * Lm1)  with interleaved column order
  n <- nrow(H_base)
  H <- ncol(H_base)
  result <- matrix(NA_real_, n, H * Lm1)
  for (k in seq_len(H)) {
    for (m in seq_len(Lm1)) {
      col_idx <- m + (k - 1) * Lm1
      result[, col_idx] <- H_base[, k]
    }
  }
  result
}
