test_that("pack.hard / unpack.hard round-trip", {
  beta <- c(0.2, -0.1)
  Theta <- matrix(1:4, 2, 2)
  alpha <- c(0.3, -0.2)
  tmat <- matrix(c(0.1, 0.2, 0.3), nrow = 1)

  par <- pack.hard(beta, Theta, alpha, tmat)
  up <- unpack.hard(par)

  expect_equal(up$beta, beta)
  expect_equal(up$Theta, Theta)
  expect_equal(up$alpha, alpha)
  expect_equal(up$tmat, tmat)
})

test_that("pack preserves dims attribute", {
  par <- pack.hard(c(0.1, 0.2), matrix(1:4, 2, 2), c(0.3, 0.4), matrix(0, 1, 3))
  d <- attr(par, "dims")
  expect_equal(d$p, 2)
  expect_equal(d$K, 3)
  expect_equal(d$L, 2)
  expect_equal(d$H, 3)
})
