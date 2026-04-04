# Shared test fixtures for MLfused tests
# Small deterministic dataset: n=50, p=2, K=3, L=2

set.seed(42)
test_n <- 50
test_p <- 2
test_K <- 3

test_Theta <- matrix(c(1, -0.5, -0.3, 0.8), nrow = 2, ncol = 2)
test_beta  <- c(0.2, -0.1)
test_alpha <- c(0.35, -0.25)

test_X <- MASS::mvrnorm(test_n, mu = rep(0, test_p), Sigma = diag(test_p))

test_eta <- test_X %*% test_Theta + matrix(test_beta, test_n, test_K - 1, byrow = TRUE)
test_P <- softmax_first(test_eta)
test_y <- apply(test_P, 1, function(p) sample.int(test_K, 1, prob = p))

test_groups <- list(c(1, 2), c(3))
test_A <- group_matrix(test_groups, test_K)

# Synthetic qhat for group 2 (class 3)
test_qhat <- matrix(runif(test_n, 0.2, 0.5), ncol = 1)

# Build H basis
test_Hmat <- build_H(test_X, phi_idx = c(1, 2), n.q = 0)

# Initial parameters
test_t0 <- matrix(0, nrow = 1, ncol = ncol(test_Hmat))
test_par_hard <- pack_hard(test_beta, test_Theta, test_alpha, test_t0)
