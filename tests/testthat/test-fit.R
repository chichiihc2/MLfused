test_that("ml_fused converges on test data", {
  skip_on_cran()

  fit <- ml_fused(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 500, tol = 1e-6,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = TRUE
  )

  expect_equal(fit$conv, 0)
  expect_true(all(is.finite(fit$par$beta)))
  expect_true(all(is.finite(fit$par$Theta)))
})

test_that("ml_fused produces finite sandwich SEs", {
  skip_on_cran()

  fit <- ml_fused(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 500, tol = 1e-6,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = TRUE
  )

  expect_true(all(is.finite(fit$se$beta)))
  expect_true(all(is.finite(fit$se$Theta)))
})

test_that("ml_fused with compute_se=FALSE skips inference", {
  skip_on_cran()

  fit <- ml_fused(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 100, tol = 1e-4,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = FALSE
  )

  expect_null(fit$conf)
  expect_null(fit$se)
})

test_that("tau_tmat affects optimization", {
  skip_on_cran()

  fit1 <- ml_fused(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 200, tol = 1e-6,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = FALSE, tau_tmat = 0.01
  )

  fit2 <- ml_fused(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 200, tol = 1e-6,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = FALSE, tau_tmat = 10
  )

  expect_false(isTRUE(all.equal(fit1$par$tmat, fit2$par$tmat, tolerance = 1e-3)))
})
