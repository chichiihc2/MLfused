test_that("softmax_first rows sum to 1", {
  eta <- matrix(rnorm(20), nrow = 5, ncol = 4)
  P <- softmax_first(eta)
  expect_equal(rowSums(P), rep(1, 5), tolerance = 1e-12)
  expect_equal(ncol(P), 5)
  expect_true(all(P > 0))
})

test_that("softmax_first baseline column matches exp(0)", {
  eta <- matrix(c(1, 2), nrow = 1)
  P <- softmax_first(eta)
  expected <- exp(c(0, 1, 2)) / sum(exp(c(0, 1, 2)))
  expect_equal(as.numeric(P), expected, tolerance = 1e-12)
})

test_that("group_matrix creates valid partition", {
  groups <- list(c(1, 2), c(3))
  A <- group_matrix(groups, K = 3)
  expect_equal(dim(A), c(3, 2))
  expect_equal(rowSums(A), rep(1, 3))
  expect_equal(colSums(A), c(2, 1))
})

test_that("group_matrix rejects invalid partition", {
  groups_bad <- list(c(1, 2), c(2, 3))
  expect_error(group_matrix(groups_bad, K = 3))
})
