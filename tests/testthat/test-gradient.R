test_that("obj.gradient.hard matches finite differences", {
  skip_on_cran()

  grad <- obj.gradient.hard(test_par_hard, test_X, test_y, test_qhat,
                            test_Hmat, test_A, tau_tmat = 0)

  eps <- 1e-5
  Dnoq <- length(test_par_hard)
  fd_grad <- numeric(Dnoq)
  for (j in seq_len(Dnoq)) {
    par_plus <- test_par_hard
    par_plus[j] <- par_plus[j] + eps
    attr(par_plus, "dims") <- attr(test_par_hard, "dims")
    par_minus <- test_par_hard
    par_minus[j] <- par_minus[j] - eps
    attr(par_minus, "dims") <- attr(test_par_hard, "dims")

    f_plus <- objective.hard(par_plus, test_X, test_y, test_qhat, test_Hmat, test_A)$obj
    f_minus <- objective.hard(par_minus, test_X, test_y, test_qhat, test_Hmat, test_A)$obj
    fd_grad[j] <- (f_plus - f_minus) / (2 * eps)
  }

  expect_equal(as.numeric(grad), fd_grad, tolerance = 1e-4)
})
