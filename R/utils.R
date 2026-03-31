#' Group Membership Matrix
#'
#' Creates a K x L binary partition matrix from a list of group assignments.
#'
#' @param groups A list of length L, where each element is an integer vector
#'   of class indices belonging to that group.
#' @param K Integer. Total number of classes.
#'
#' @return A K x L binary matrix where \code{A[k, l] = 1} if class k belongs
#'   to group l, and 0 otherwise. Each class belongs to exactly one group.
#'
#' @export
group_matrix <- function(groups, K) {
  L <- length(groups)
  A <- matrix(0, nrow = K, ncol = L)
  for (l in seq_len(L)) A[groups[[l]], l] <- 1
  stopifnot(all(rowSums(A) == 1))
  A
}


#' Baseline-Constrained Softmax
#'
#' Computes softmax probabilities with class 1 as the reference category
#' (linear predictor fixed at 0).
#'
#' @param eta_mat An n x (K-1) matrix of linear predictors for classes
#'   2, ..., K.
#'
#' @return An n x K probability matrix. Column 1 is the baseline class.
#'   Rows sum to 1.
#'
#' @export
softmax_first <- function(eta_mat) {
  eta_full <- cbind(0, eta_mat)
  e <- exp(eta_full - apply(eta_full, 1, max))
  e / rowSums(e)
}
