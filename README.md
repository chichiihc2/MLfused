# MLfused

Fused Multinomial Logistic Regression Utilizing Summary-Level External Machine-Learning Information

## Overview

MLfused implements the fused maximum likelihood estimator (FMLE) proposed by Dai and Shao (2025). The method incorporates external nonparametric machine-learning predictions (e.g., XGBoost, random forests, neural networks) into multinomial logistic regression via empirical-likelihood moment constraints, improving estimation efficiency without requiring individual-level external data.

The framework handles common data-quality issues in external sources:

- **Coarsened outcome labels** -- the external source may record only grouped categories (e.g., binary) while the primary study has fine-grained K-class labels.
- **Partially observed covariates** -- the external source may observe only a subset of the primary covariates.
- **Covariate shift** -- the covariate distributions may differ across sources; robustness is achieved through nonparametric ML predictions without density-ratio estimation.
- **Concept shift** -- outcome-generating mechanisms may differ across sources, accommodated by separating shared and source-specific (free) parameters.

## Installation

```r
# Without vignettes (faster)
devtools::install_github("chichiihc2/MLfused")

# With vignettes (needed for vignette() to work)
devtools::install_github("chichiihc2/MLfused", build_vignettes = TRUE)
```

## Quick Example

```r
library(MLfused)
library(xgboost)

set.seed(43)

# True parameters (Example 2: proportion heterogeneity)
p <- 4; K <- 3; n1 <- 500; n2 <- 10000
Theta_true <- matrix(c(1,-1,-1,1, 1,1,-1,1), nrow = p, ncol = K-1, byrow = TRUE)
beta_true  <- c(0.2, -0.1)    # primary intercepts
alpha_true <- c(0.35, -0.25)  # external intercepts
groups <- list(c(1,2), c(3))  # coarsened: {class 1,2} vs {class 3}

Sigma <- matrix(0.8, p, p); diag(Sigma) <- 1

# External data and XGBoost
X2 <- MASS::mvrnorm(n2, rep(0, p), Sigma)
eta2 <- X2 %*% Theta_true + matrix(alpha_true, n2, K-1, byrow = TRUE)
G2 <- apply(softmax_first(eta2), 1, function(pp) sample.int(K, 1, prob = pp))
W2 <- ifelse(G2 %in% c(1,2), 1L, 2L)

xgb_fit <- xgb.train(
  params = list(objective = "binary:logistic", max_depth = 4,
                eta = 0.1, subsample = 0.5, nthread = 1),
  data = xgb.DMatrix(X2, label = as.numeric(W2 == 1)),
  nrounds = 300, verbose = 0
)

# Primary data
X1 <- MASS::mvrnorm(n1, rep(0, p), Sigma)
eta1 <- X1 %*% Theta_true + matrix(beta_true, n1, K-1, byrow = TRUE)
y <- apply(softmax_first(eta1), 1, function(pp) sample.int(K, 1, prob = pp))

# qhat = P(Group 2 | X)
qhat <- matrix(1 - predict(xgb_fit, xgb.DMatrix(X1)), ncol = 1)

# MLE (primary data only)
fit_mle <- nnet::multinom(y ~ ., data = data.frame(y = factor(y), X1), trace = FALSE)

# FMLE (fused with external ML)
cf <- coef(fit_mle); if (is.vector(cf)) cf <- matrix(cf, nrow = 1)
beta0 <- cf[, 1]; Theta0 <- t(cf[, -1])
Hmat <- build_H(X1, phi_idx = 1:p, n.q = 1)
par0 <- pack_hard(beta0, Theta0, rep(0, K-1),
                  matrix(0, ncol(qhat), ncol(Hmat)))

fit <- ml_fused(par0, X1, y, qhat, Hmat, groups, tau_tmat = 0.1)

fit$par$beta   # intercept estimates
fit$par$Theta  # slope matrix (p x (K-1))
fit$se$Theta   # sandwich standard errors
```

## Main Functions

| Function | Description |
|----------|-------------|
| `ml_fused()` | Fit the fused multinomial model via Newton optimization |
| `pack_hard()` / `unpack_hard()` | Pack/unpack parameter blocks (beta, Theta, alpha, tmat) |
| `build_H()` | Construct constraint basis (raw covariates or natural splines) |
| `build_Hmat_interleaved()` | Interleave basis for multiple non-reference groups |
| `sandwich_se()` | Hessian-based sandwich standard errors (Theorem 2) |
| `bootstrap_se()` | Bootstrap standard errors with SD and MAD variants |
| `objective_hard()` | Profile log-pseudo-likelihood value |
| `gradient_hard()` / `hessian_hard()` | Analytic gradient and Hessian |

## Method

Given primary data `(Y_i, X_i)` from a multinomial logistic model and a black-box external prediction `q_hat(Z)` trained on a large auxiliary sample, the FMLE maximizes the profile log-pseudo-likelihood (Dai and Shao, 2025, eq. 7):

```
l_n(gamma | q_hat) = l_n(theta) - (1/n) sum_i log(1 + sum_{l,h} g_{l,h}(X_i) * lambda_{l,h})
```

where `l_n(theta)` is the primary multinomial log-likelihood, and the second term encodes empirical-likelihood constraints linking the primary model to external predictions through a basis function set H.

An L2 penalty `tau * ||lambda||^2` regularizes the Lagrange multipliers for numerical stability (Section 3.5 of the paper, default `tau = 0.1`). The Newton step size is damped (`learning.rate = 0.1` by default) for robustness of the EL log-term.

## Parameterization

The paper uses class K as the softmax baseline; the code uses class 1. The two are equivalent under relabeling. In the code:

- `beta` = intercepts for classes 2, ..., K (the paper's `theta_{k,1}`)
- `Theta` = slope matrix, p x (K-1) (the paper's `theta_{k,2:p}`)
- `alpha` = external intercepts (the paper's `phi_{k,1}`)
- `tmat` = Lagrange multipliers (the paper's `lambda_{l,h}`)

Under Example 2 (proportion heterogeneity), the shared parameter `psi` consists of the slope coefficients in `Theta`, while the free parameters are `beta` (primary) and `alpha` (external).

## Vignette

```r
vignette("getting-started", package = "MLfused")
```

Reproduces the paper's simulation design: K=3 classes, L=2 groups (coarsened external labels), XGBoost external predictor, comparing MLE and FMLE with a coefficient plot.

## Real Data (NHANES)

The package bundles pre-processed NHANES 2013-2018 data used in Section 5 of the paper (blood pressure classification: Normal / Prehypertension / Hypertension):

| File | Description |
|------|-------------|
| `inst/extdata/internal_cleaned.csv` | Primary data (9,186 units, 14 covariates) |
| `inst/extdata/external_cleaned.csv` | External data (12,425 units, 8 covariates) |
| `inst/extdata/internal_pred_full.csv` | Primary data with 3-class XGBoost predictions |

To reproduce the paper's real-data analysis:

```r
source(system.file("scripts", "nhanes_analysis.R", package = "MLfused"))
```

This fits MLE and FMLE on the full primary sample (n=9,186) and a random subsample (n=600), matching Figure 2 in the paper.

## Citation

If you use this package, please cite:

> Dai, C.-S. and Shao, J. (2025). "Fused Multinomial Logistic Regression Utilizing Summary-Level External Machine-Learning Information." Submitted to *Journal of Machine Learning Research*.

## License

MIT
