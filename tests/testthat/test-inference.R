test_that("sandwich_se returns correct structure", {
  skip_on_cran()

  fit <- ml_fused(
    par = test_par_hard, X = test_X, y = test_y,
    qhat = test_qhat,
    Hmat = test_Hmat, groups = test_groups,
    maxit = 500, tol = 1e-6,
    learning.rate = 0.1, lambda = test_lambda,
    compute_se = FALSE
  )

  par_best <- pack_hard(fit$par$beta, fit$par$Theta,
                        fit$par$alpha, fit$par$tmat)
  res <- sandwich_se(par_best, test_X, test_y, test_qhat,
                     test_Hmat, test_A)

  expect_true(all(is.finite(res$se)))
  expect_true(all(as.numeric(res$se) >= 0))
  expect_equal(nrow(res$conf_matrix), length(par_best))
  expect_equal(ncol(res$conf_matrix), length(par_best))
})
