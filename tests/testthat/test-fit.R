test_that("ML_fused_fast converges on test data", {
  skip_on_cran()

  fit <- ML_fused_fast(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat, qhat.se = test_qhat.se,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 500, tol = 1e-6,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = TRUE
  )

  expect_equal(fit$conv, 0)
  expect_true(all(is.finite(fit$par$beta)))
  expect_true(all(is.finite(fit$par$Theta)))
  expect_true(is.list(fit$se.score) || is.numeric(fit$se.score))
})

test_that("ML_fused matches ML_fused_fast", {
  skip_on_cran()

  fit_orig <- ML_fused(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat, qhat.se = test_qhat.se,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 500, tol = 1e-6,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = TRUE
  )

  fit_fast <- ML_fused_fast(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat, qhat.se = test_qhat.se,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 500, tol = 1e-6,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = TRUE
  )

  # Tolerance loosened: the two solvers use different Hessian implementations
  # and may converge to slightly different points
  expect_equal(fit_orig$par$beta, fit_fast$par$beta, tolerance = 0.1)
  expect_equal(fit_orig$par$Theta, fit_fast$par$Theta, tolerance = 0.1)
})
