test_that("build_H with n.q=0 returns intercept + raw features", {
  x <- matrix(rnorm(100), ncol = 2)
  H <- build_H(x, phi_idx = c(1, 2), n.q = 0)
  expect_equal(ncol(H), 3)  # intercept + 2 features
  expect_equal(nrow(H), 50)
  expect_true(all(H[, 1] == 1))  # intercept column
})

test_that("build_H with n.q=1 returns correct dimensions", {
  x <- matrix(rnorm(100), ncol = 2)
  H <- build_H(x, phi_idx = c(1, 2), n.q = 1)
  # intercept + n.q+1 basis per feature * 2 features = 1 + 2*2 = 5
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
  H_base <- matrix(1:12, nrow = 4, ncol = 3)  # 4 obs, 3 basis functions
  Hmat <- build_Hmat_interleaved(H_base, Lm1 = 2)
  expect_equal(ncol(Hmat), 6)  # 3 basis * 2 groups
  expect_equal(nrow(Hmat), 4)
  # Column 1 should equal column 1 of H_base (group 1, basis 1)
  expect_equal(Hmat[, 1], H_base[, 1])
  # Column 2 should also equal column 1 of H_base (group 2, basis 1)
  expect_equal(Hmat[, 2], H_base[, 1])
})
