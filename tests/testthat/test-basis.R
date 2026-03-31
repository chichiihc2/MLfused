test_that("build_H with n.q=0 returns intercept + raw features", {
  x <- matrix(rnorm(100), ncol = 2)
  H <- build_H(x, phi_idx = c(1, 2), n.q = 0)
  expect_equal(ncol(H), 3)
  expect_equal(nrow(H), 50)
  expect_true(all(H[, 1] == 1))
})

test_that("build_H with n.q=1 returns correct dimensions", {
  x <- matrix(rnorm(100), ncol = 2)
  H <- build_H(x, phi_idx = c(1, 2), n.q = 1)
  expect_equal(ncol(H), 5)
  expect_equal(nrow(H), 50)
})

test_that("build_H attaches correct attributes", {
  x <- matrix(rnorm(100), ncol = 2)
  H <- build_H(x, phi_idx = c(1, 2), n.q = 2)
  expect_equal(attr(H, "n.q"), 2)
  expect_true(!is.null(attr(H, "basis")))
})

test_that("build_Hmat_interleaved produces correct column ordering", {
  H_base <- matrix(1:12, nrow = 4, ncol = 3)
  Hmat <- build_Hmat_interleaved(H_base, Lm1 = 2)
  expect_equal(ncol(Hmat), 6)
  expect_equal(nrow(Hmat), 4)
  expect_equal(Hmat[, 1], H_base[, 1])
  expect_equal(Hmat[, 2], H_base[, 1])
})
