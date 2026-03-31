# Migration Guide: MLfused 0.1.0 → 0.2.0

## Why the rewrite

v0.1.0 shipped two parallel code paths ("original" and "fast") for every major
function. The "original" path contained statistical bugs:

1. **`ML_fused()` hardcoded `tau_tmat=0`** — the L2 penalty on tmat was never
   applied during optimization, producing different (worse) estimates than
   `ML_fused_fast()`.
2. **Bootstrap functions missing `tau_tmat`** — refits used default penalty
   regardless of what the original fit used.

v0.2.0 removes all duplicate/buggy code and simplifies the SE interface to
**sandwich** and **bootstrap** only (score-based, adjusted, and Theorem 3
variants removed).

## Function name mapping

| v0.1.0 (old)                    | v0.2.0 (new)              | Notes                     |
|---------------------------------|---------------------------|---------------------------|
| `ML_fused_fast()`               | `ml_fused()`              | Renamed                   |
| `ML_fused()`                    | *removed*                 | Buggy; use `ml_fused()`   |
| `pack.hard()`                   | `pack_hard()`             | Renamed (snake_case)      |
| `unpack.hard()`                 | `unpack_hard()`           | Renamed (snake_case)      |
| `objective.hard()`              | `objective_hard()`        | Renamed                   |
| `obj.gradient.hard()`           | `gradient_hard()`         | Renamed                   |
| `obj.hessian.hard()`            | `hessian_hard()`          | Renamed                   |
| `obj.grad_hess.hard()`          | `grad_hess_hard()`        | Renamed                   |
| `conf.est.hard.fast()`          | `sandwich_se()`           | Simplified (sandwich only)|
| `conf.est.hard()`               | *removed*                 | Buggy                     |
| `conf.est.thm3*`               | *removed*                 | Dropped                   |
| `bootstrap_hard_se_fast()`      | `bootstrap_se()`          | Renamed; added `tau_tmat` |
| `bootstrap_hard_se()`           | *removed*                 | Used buggy `ML_fused()`   |
| `bootstrap_hard_se_diagnostic()`| `bootstrap_se_diagnostic()`| Renamed; added `tau_tmat`|
| `compute_per_sample_scores_hard()` | *removed*              | Dropped with score SEs    |
| `keep.hard()`                   | *removed*                 | Only used by `ML_fused()` |
| `Diag()`                        | *removed*                 | Unused                    |

## SE methods

v0.2.0 provides two SE methods:

- **`sandwich_se()`** — Hessian-based sandwich variance (computed automatically by `ml_fused()`)
- **`bootstrap_se()`** — Nonparametric bootstrap (call separately after fitting)

Removed: score-based sandwich, qhat-adjusted variants, Theorem 3, Theorem 3 plugin.

## Return structure of `ml_fused()`

`ml_fused()` now returns:
- `$se` — unpacked sandwich SEs (list with `beta`, `Theta`, `alpha`, `tmat`)
- `$conf` — full sandwich output (covariance matrix, Ihat, Jhat)

Removed fields: `se.adj`, `se.score`, `se.score.adj`, `se.thm3`, `se.thm3.adj`,
`se.thm3.score`, `se.thm3.score.adj`, `thm3`.

## Quick migration

```r
# Old → New
ML_fused_fast  → ml_fused
pack.hard      → pack_hard
unpack.hard    → unpack_hard
objective.hard → objective_hard
obj.gradient.hard → gradient_hard
obj.hessian.hard  → hessian_hard
obj.grad_hess.hard → grad_hess_hard
conf.est.hard.fast → sandwich_se
bootstrap_hard_se_fast → bootstrap_se
bootstrap_hard_se_diagnostic → bootstrap_se_diagnostic

# Old SE access → New
fit$se.score$Theta → fit$se$Theta
fit$se.thm3$Theta  → fit$se$Theta  (sandwich replaces all)
```

If you were using `ML_fused()` (without `_fast`), switch to `ml_fused()` and
set `tau_tmat` to the value you intend. The old `ML_fused()` always used
`tau_tmat=0` regardless of the argument, so if you relied on that behavior,
pass `tau_tmat=0` explicitly.
