test_that("bootstrap_se runs and returns correct structure", {
  skip_on_cran()

  fit <- ml_fused(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 200, tol = 1e-4,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = FALSE, tau_tmat = 1
  )

  res <- bootstrap_se(
    par_fit = fit$par, X = test_X, y = test_y,
    qhat = test_qhat,
    Hmat = test_Hmat, groups = test_groups,
    B = 10, maxit = 100, tol = 1e-3,
    learning.rate = 0.1, lambda = test_lambda,
    tau_tmat = 1
  )

  expect_true(is.numeric(res$se))
  expect_true(is.numeric(res$se_mad))
  expect_true(res$n_kept >= 0)
  expect_true(res$n_kept + res$n_nonconv + res$n_error == 10)
})
