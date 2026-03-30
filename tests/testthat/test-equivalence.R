test_that("Hessian matches finite differences", {
  skip_on_cran()

  H_mat <- obj.hessian.hard(test_par_hard, test_X, test_y, test_qhat,
                            test_Hmat, test_A, lambda.diag = 0, tau_tmat = 0)

  # Verify symmetry
  expect_equal(H_mat, t(H_mat), tolerance = 1e-10)

  # Verify against finite differences of gradient
  eps <- 1e-5
  Dnoq <- length(test_par_hard)
  fd_hess <- matrix(0, Dnoq, Dnoq)
  for (j in seq_len(min(Dnoq, 10))) {
    par_plus <- test_par_hard
    par_plus[j] <- par_plus[j] + eps
    attr(par_plus, "dims") <- attr(test_par_hard, "dims")
    par_minus <- test_par_hard
    par_minus[j] <- par_minus[j] - eps
    attr(par_minus, "dims") <- attr(test_par_hard, "dims")

    g_plus <- obj.gradient.hard(par_plus, test_X, test_y, test_qhat,
                                test_Hmat, test_A, tau_tmat = 0)
    g_minus <- obj.gradient.hard(par_minus, test_X, test_y, test_qhat,
                                 test_Hmat, test_A, tau_tmat = 0)
    fd_hess[, j] <- (as.numeric(g_plus) - as.numeric(g_minus)) / (2 * eps)
  }
  expect_equal(H_mat[, 1:min(Dnoq, 10)], fd_hess[, 1:min(Dnoq, 10)], tolerance = 1e-3)
})

test_that("combined grad_hess matches separate calls", {
  skip_on_cran()

  gh <- obj.grad_hess.hard(test_par_hard, test_X, test_y, test_qhat,
                           test_Hmat, test_A, lambda.diag = 0, tau_tmat = 0)
  g_sep <- obj.gradient.hard(test_par_hard, test_X, test_y, test_qhat,
                             test_Hmat, test_A, tau_tmat = 0)
  H_sep <- obj.hessian.hard(test_par_hard, test_X, test_y, test_qhat,
                            test_Hmat, test_A, lambda.diag = 0, tau_tmat = 0)

  expect_equal(as.numeric(gh$grad), as.numeric(g_sep), tolerance = 1e-10)
  expect_equal(gh$hess, H_sep, tolerance = 1e-10)
})
