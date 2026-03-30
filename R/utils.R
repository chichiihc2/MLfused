#' Create a Diagonal Matrix from a Vector
#'
#' Constructs a diagonal matrix with the given values on the diagonal.
#' Unlike \code{\link[base]{diag}}, this function always returns a matrix
#' (even for length-1 input) by explicitly building an n x n zero matrix
#' and setting its diagonal.
#'
#' @param variables A numeric vector of diagonal entries.
#'
#' @return A square numeric matrix of dimension \code{length(variables)} with
#'   \code{variables} on the diagonal and zeros elsewhere.
#'
#' @keywords internal
Diag <- function(variables) {
  v <- as.numeric(variables)
  n <- length(v)
  out <- matrix(0, n, n)
  diag(out) <- v
  out
}

#' Create a Group Membership Matrix
#'
#' Builds a K x L binary matrix \code{A} indicating which classes (rows)
#' belong to which groups (columns). Each class must belong to exactly one
#' group.
#'
#' @param groups A list of length L, where each element is an integer vector
#'   of class indices (1-based) belonging to that group.
#' @param K Integer. The total number of classes.
#'
#' @return A K x L binary matrix where \code{A[k, l] = 1} if class \code{k}
#'   belongs to group \code{l}, and 0 otherwise.
#'
#' @export
group_matrix <- function(groups, K){
  L <- length(groups)
  A <- matrix(0, nrow = K, ncol = L)
  for (l in seq_len(L)) A[groups[[l]], l] <- 1
  stopifnot(all(rowSums(A) == 1))
  A
}

#' Baseline-Constrained Softmax
#'
#' Computes row-wise softmax probabilities with the first class as the
#' baseline (its linear predictor is fixed to zero). This is the standard
#' parameterization for multinomial logistic regression where class 1 is
#' the reference category.
#'
#' @param eta_mat An n x (K-1) matrix of linear predictors for classes
#'   2 through K.
#'
#' @return An n x K matrix of probabilities. Column 1 corresponds to the
#'   baseline class.
#'
#' @export
softmax_first <- function(eta_mat){
  eta_full <- cbind(0, eta_mat)                              # baseline column = 0
  e <- exp(eta_full - apply(eta_full, 1, max))               # stabilize
  e / rowSums(e)
}
