# Example 5: MPH* with unknown CTMC path and non-invertible reward mapping
---

## Scripts

### Algorithm script
- **`Example_5_algorithm.py`**

This script contains all simulation, E-step, CM-step rate updates, observed
likelihood evaluation, and full ECM estimation routines for example 5

Key implementation features:
- Adaptive truncation of the infinite sum over \(V=1,2,\dots\) using a
  negligible-tail stopping rule
- Caching of expensive quadrature integrals inside the E-step over z_3 to not calculate them over and over again
- Log-likelihood tracking obtained directly from E-step weights
- Reward parameters \((r_{31}, r_{32})\) updated by monotone ascent using a
  log-parameterization and backtracking line search

### Run script 
- **`Run_example_5.py`**

This script runs 7 Monte Carlo experiments using the algorithm . It simulates synthetic data, fits the model
from chosen random initializations, tracks convergence via likelihood diagnostics, and
exports results to CSV files.

---
## Overview of the algorithmic components

### 1. Stable numerical utilities

- `logsumexp(logw)`
- `logaddexp(a, b)`
- `log_trapz_from_logf(logf, x)`
- `safe_exp(x)`

These provide stable computations for:
- summing log-weights,
- combining reward components in log space,
- trapezoidal quadrature computed in log space to solve non-closed form intrgrals,
- exponentials to avoid overflow/underflow.

---

### 2. Simulation functions (optional)

- `simulate_noninv_unknown_one(...)`  
  Simulates a single CTMC trajectory and returns:
  - `Y = (y1,y2)`,
  - latent times `Z = (z1,z2,z3)`,
  - `V` (number of visits to state 2),
  - `H` (indicator of exiting via state 3).

- `simulate_noninv_unknown_dataset(..., N, ...)`  
  Simulates `N` i.i.d. observations and returns arrays `Y`, `Z`, `V`, `H`.

---

### 3. Conditional densities and helper functions

- `log_gamma_pdf(x, shape, rate)`  
  Log-density of \(\mathrm{Gamma}(v,\text{rate})\) with integer shape \(v\).

- `M_of_y(y1, y2, r31, r32)`  
  Feasible upper bound for \(Z_3\):
\[
M(y) = \min\!\left(\frac{y_1}{r_{31}},\frac{y_2}{r_{32}}\right).
\]

- `log_fY_given_v_h0(y1, y2, v, lam, q2)`  
  Conditional log-density for \(H=0\):
\[
\log f(y\mid v, H=0) = \log \Gamma(v,\lambda)(y_1) + \log \Gamma(v,q_2)(y_2).
\]

- `log_integrals_for_h1(...)`  
  Quadrature-based computation for \(H=1\):
\[
I_0 = \int_0^{M(y)} g(z)\,dz,\qquad
I_1 = \int_0^{M(y)} z\,g(z)\,dz,
\]
where
\[
g(z) = \Gamma(v,\lambda)(y_1-r_{31}z)\;\Gamma(v,q_2)(y_2-r_{32}z)\;\lambda_3 e^{-\lambda_3 z}.
\]
Returns `(logI0, logI1)` computed by trapezoidal integration on a uniform grid.

- `log_fY_given_v_h1(...)`  
  Returns \(\log f(y\mid v,H=1)=\log I_0\).

---

### 4. Adaptive E-step with posterior weights \(w_{v,h}(y)\)

#### Single-observation E-step (fast, cached)
- `E_for_one_obs_adaptive_cached(...)`

For each observation \(y=(y_1,y_2)\), the posterior over \((V,H)\) is proportional to:
\[
w_{v,0}(y) \propto P(V=v,H=0)\,f(y\mid v,H=0),
\qquad
w_{v,1}(y) \propto P(V=v,H=1)\,f(y\mid v,H=1).
\]

The prior terms are:
- \(P(V=1,H=0)=p_{2a}\), \(P(V=1,H=1)=p_{23}\)
- For \(v\ge 2\): \(P(V=v,H=h)=p_{21}^{v-1}p_{2h}\) with \(p_{20}=p_{2a}\), \(p_{21}=p_{23}\)

**Adaptive truncation rule:**  
Sum over \(v=1,2,\dots\) until the incremental mass contribution becomes smaller
than `tail_rel_tol` for `tail_patience` consecutive terms, after at least `v_min`
terms, with an absolute cap `v_cap`.

This returns:
- `EV = E[V|y]`
- `EH = E[H|y]`
- `EHZ3 = E[H Z3|y]` using cached \((\log I_0,\log I_1)\)
- `EZ1, EZ2` via the reward relations:
  - `EZ1 = max(y1 - r31*EHZ3, 0)`
  - `EZ2 = max(y2 - r32*EHZ3, 0)`
- `ll_y = log f_Y(y)` computed directly from the same log-weights

**Integral caching:**  
For each observation, the E-step caches \((\log I_0,\log I_1)\) for each visited
\(v\), avoiding recomputation when forming \(E[Z_3\mid y,v,H=1]\).

#### Full-sample E-step (sufficient statistics)
- `E_step_adaptive(...)`

Returns aggregated expected sufficient statistics:
- \(S_1 = \sum_n E[Z_1^{(n)}\mid Y]\)
- \(S_2 = \sum_n E[Z_2^{(n)}\mid Y]\)
- \(V_\Sigma = \sum_n E[V^{(n)}\mid Y]\)
- \(H_\Sigma = \sum_n E[H^{(n)}\mid Y]\)
- \((HZ_3)_\Sigma = \sum_n E[H^{(n)}Z_3^{(n)}\mid Y]\)
- mean truncation level `v_used_mean` (diagnostic)

#### Full-sample E-step + observed log-likelihood (preferred for speed)
- `E_step_adaptive_with_ll(...)`

Same as `E_step_adaptive`, but also returns:
- `ll_total = sum_n log f_Y(y^(n))`

This avoids redundant calls to the observed likelihood during convergence tracking.

---

### 5. CM-step: closed-form updates for CTMC rates

- `M_step_rates(S1_hat, S2_hat, V_hat, H_hat, HZ3_hat, N, eps=1e-12)`

Closed-form updates for \((\lambda,\rho,\mu,\kappa,\lambda_3)\) holding rewards fixed:
\[
\hat{\lambda}   = \frac{\hat V}{\hat S_1},\qquad
\hat{\rho}      = \frac{\hat V - N}{\hat S_2},\qquad
\hat{\mu}       = \frac{N-\hat H}{\hat S_2},\qquad
\hat{\kappa}    = \frac{\hat H}{\hat S_2},\qquad
\hat{\lambda}_3 = \frac{\hat H}{\widehat{HZ_3}}.
\]

---

### 6. Observed log-likelihood (adaptive truncation)

- `loglik_adaptive(...)`

Computes:
\[
\ell(\theta) = \sum_{n=1}^N \log f_Y(y^{(n)})
\]
by summing the mixture over \(v\) and \(h\) using the same adaptive tail rule.

In `run_ecm`, this is used primarily for the **R-update line search**, while
per-iteration tracking is obtained from `E_step_adaptive_with_ll`.

---

### 7. ECM update for reward parameters \(r_{31}, r_{32}\)

Rewards are updated via monotone ascent on the observed log-likelihood under a
log-parameterization:

- `r_from_v(v)` with \(r_{31}=e^{v_1}\), \(r_{32}=e^{v_2}\)
- `fd_grad_v(...)` finite-difference gradient of `loglik_adaptive` w.r.t. \(v\)
- `update_v_linesearch(...)` gradient-ascent step with backtracking line search,
  accepting only strict likelihood improvements

---

## Overall ECM procedure

The full estimation routine `run_ecm(...)` proceeds as follows:

1. **Initialize** \(\lambda,\rho,\mu,\kappa,\lambda_3,r_{31},r_{32}\)
2. Compute an initial observed log-likelihood (one-time call to `loglik_adaptive`)
3. Repeat until convergence (or `max_iter`):
   - **E-step:** compute sufficient statistics and the current log-likelihood
     using `E_step_adaptive_with_ll`
   - **CM-step (rates):** update \(\lambda,\rho,\mu,\kappa,\lambda_3\) in closed form
   - **ECM-step (rewards):** update \(r_{31},r_{32}\) via monotone ascent with
     backtracking line search (up to `max_R_inner_steps`)
   - **Stopping rule:** Kulkarni-style likelihood criterion using
     \(|\Delta \ell|/N < \texttt{eps_ll}\) after at least `min_iters_ll` iterations
   - Optionally apply a secondary relative parameter-change criterion

The function returns final parameter estimates and (optionally) the log-likelihood
history if `track_ll=True`.

---
