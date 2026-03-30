# MLfused

Multinomial Logistic Regression with Empirical-Likelihood Data Fusion

MLfused fits multinomial logistic regression models that incorporate external
machine learning predictions via empirical-likelihood constraints,
improving estimation efficiency when auxiliary data is available.

## Installation

```r
# Install from GitHub
devtools::install_github("username/MLfused")
```

## Quick example

```r
library(MLfused)
set.seed(1)

# Synthetic data: n=500, p=3 covariates, K=3 classes
n <- 500; p <- 3; K <- 3
X <- matrix(rnorm(n * p), n, p)
eta <- cbind(0, X %*% matrix(c(1,-1,0.5,-0.5,0.3,0.3), p, K-1))
P <- exp(eta) / rowSums(exp(eta))
y <- apply(P, 1, function(pr) sample(K, 1, prob = pr))

# External predictions: groups = {class 1} vs {classes 2,3}
groups <- list(c(1), c(2, 3))
qhat   <- matrix(pmin(pmax(P[,2] + P[,3] + rnorm(n, 0, 0.05), 0.01), 0.99))
qhat_se <- matrix(0.05, n, 1)

# Build constraint basis and initial parameters
Hmat <- build_Hmat_interleaved(build_H(X, phi_idx = 1:2, n.q = 0), Lm1 = 1)
par0 <- pack.hard(rep(0, K-1), matrix(0, p, K-1), rep(0, K-1),
                  matrix(0, 1, ncol(Hmat)))

# Fit fused model
fit <- ML_fused_fast(par0, X, y, qhat, qhat_se, Hmat, groups,
                     lambda = 5000, compute_se = TRUE, tau_tmat = 0.01)
fit$par$beta      # intercept estimates
fit$par$Theta     # slope matrix (p x (K-1))
fit$se.score.adj  # score-based adjusted standard errors
```

## Features

- **Newton optimizer** with analytic gradient and Hessian for the
  empirical-likelihood fused multinomial objective.
- **Fast solver** (`ML_fused_fast`) with vectorized per-observation Hessian
  computation and combined gradient-Hessian passes.
- **Multiple SE estimators**: Hessian-based sandwich, score-based sandwich,
  Theorem 3, and bootstrap -- each with optional qhat-uncertainty adjustment.
- **Flexible constraint basis**: intercept + raw covariates or natural cubic
  splines via `build_H()`.
- Supports **coarsened external labels** (binary grouping of K classes) and
  **full external labels** (one-to-one class mapping).

## Vignettes

- `vignette("getting-started", package = "MLfused")` -- Synthetic data
  walkthrough: generate data, fit the model, compare with `nnet::multinom()`.
- `vignette("nhanes-application", package = "MLfused")` -- Real-data
  application: NHANES blood pressure classification with coarsened and full
  external predictions.

## Reproducing the paper

The `inst/scripts/` directory contains the analysis pipeline scripts. The
processed NHANES datasets used in the paper are bundled in `inst/extdata/`:

- `internal_pred_coarsened.csv` -- primary data with coarsened (binary)
  external predictions
- `internal_pred_full.csv` -- primary data with full (3-class) external
  predictions
- `internal_cleaned.csv` -- primary data without external predictions

## Citation

If you use this package, please cite:

> [Author names]. "Data Fusion for Multinomial Logistic Regression via
> Empirical Likelihood." *[Journal]*, [Year].

## License

MIT
