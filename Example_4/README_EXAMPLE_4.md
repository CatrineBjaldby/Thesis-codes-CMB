# Example 2: Bivariate Clara model with non-invertible reward mapping

This repository contains an ECM (Expectation–Conditional Maximization)
implementation for the **Clara bivariate reward model** with a
**non-invertible reward mapping**, corresponding to the Clara example in the
thesis.

---

## Scripts

### Algorithm script
- **`Example_4_algorithm.py`**

This script contains all simulation, E-step, M-step, likelihood evaluation, and
ECM estimation routines for the Clara non-invertible reward example. It
simulates synthetic data from a three-state CTMC with an optional visit to
state 3, evaluates the observed-data likelihood, and fits parameters via an
ECM procedure with closed-form rate updates and numerical updates for the
reward parameters.

### Run script (recommended)
- **`Run_Example_4.py`**

This script runs Monte Carlo experiments using the algorithm implemented in
`clara_ecm_algorithm.py`. It simulates synthetic data, fits the model from
chosen initializations, computes diagnostics (e.g. log-likelihood traces),
and exports results to CSV files.

---

## Model overview

- Latent process: CTMC with transient states \(1 \to 2\), optional visit to
  state 3, then absorption
- Indicator \(H \sim \mathrm{Bernoulli}(q)\) determines whether state 3 is visited
- Sojourn rates: \(\lambda_1>0, \lambda_2>0, \lambda_3>0\)
- Reward parameters: \(r_{31}>0, r_{32}>0\)

### Latent sojourn times

- \(Z_1 \sim \mathrm{Exp}(\lambda_1)\)
- \(Z_2 \sim \mathrm{Exp}(\lambda_2)\)
- \(H \sim \mathrm{Bernoulli}(q)\)
- \(Z_3 \sim \mathrm{Exp}(\lambda_3)\) if \(H=1\), and \(Z_3=0\) if \(H=0\)

### Observation model

Observed data are given by a non-invertible reward mapping:
\[
Y_1 = Z_1 + r_{31} Z_3,\qquad
Y_2 = Z_2 + r_{32} Z_3,
\]
so the latent vector \((Z_1,Z_2,Z_3)\in\mathbb{R}^3\) is mapped to the observed
vector \((Y_1,Y_2)\in\mathbb{R}^2\).

---

## Overview of the algorithmic components

### 1. Simulation

- `simulate_clara_dataset(lam1, lam2, lam3, q, r31, r32, N, rng)`

Simulates `N` independent observations and returns:
- `Y`: observed data (Nx2),
- `Z`: latent sojourn times (Nx3),
- `H`: indicator of visiting state 3 (N,).

---

### 2. Integrals for the E-step

- `integrals_for_x3(M, lam1, lam2, lam3, r31, r32, ...)`

For each observation, define:
\[
M = \min\!\left(\frac{y_1}{r_{31}},\frac{y_2}{r_{32}}\right),\qquad
a = \lambda_1 r_{31} + \lambda_2 r_{32} - \lambda_3.
\]

The routine computes:
\[
I_0(M) = \int_0^M e^{a u}\,du,\qquad
I_1(M) = \int_0^M u e^{a u}\,du,
\]
using closed-form expressions when possible, with special handling for
\(a\approx 0\), and a numerical fallback.

---

### 3. E-step (single observation)

- `E_for_one_obs(y1, y2, lam1, lam2, lam3, q, r31, r32)`

Computes conditional expectations:
- \(E[Z_1 \mid Y]\),
- \(E[Z_2 \mid Y]\),
- \(E[H Z_3 \mid Y]\),
- \(E[H \mid Y]\).

If the feasible range for \(Z_3\) is negligible, the observation is treated as
effectively \(H=0\).

---

### 4. E-step (full sample)

- `E_step_clara(Y, lam1, lam2, lam3, q, r31, r32)`

Aggregates expected sufficient statistics:
- \(S_1 = \sum_n E[Z_1^{(n)} \mid Y]\),
- \(S_2 = \sum_n E[Z_2^{(n)} \mid Y]\),
- \(S_{3H} = \sum_n E[H^{(n)} Z_3^{(n)} \mid Y]\),
- \(H_\Sigma = \sum_n E[H^{(n)} \mid Y]\).

---

### 5. M-step (rates and mixture probability)

- `M_step_clara(S1_hat, S2_hat, S3H_hat, H_sum_hat, N)`

Closed-form updates:
\[
\hat{\lambda}_1 = \frac{N}{S_1},\qquad
\hat{\lambda}_2 = \frac{N}{S_2},\qquad
\hat{q} = \frac{H_\Sigma}{N}.
\]

The update
\[
\hat{\lambda}_3 = \frac{H_\Sigma}{S_{3H}}
\]
is skipped when posterior mass on \(H=1\) is too small, to avoid instability.

---

### 6. Observed log-likelihood

- `loglik_clara(Y, lam1, lam2, lam3, q, r31, r32)`

Evaluates the observed-data log-likelihood:
\[
f_Y(y_1,y_2) = (1-q) A(y_1,y_2) + q B(y_1,y_2),
\]
with
\[
A(y_1,y_2) = \lambda_1 \lambda_2 e^{-\lambda_1 y_1 - \lambda_2 y_2},
\]
\[
B(y_1,y_2) = \lambda_1 \lambda_2 \lambda_3
e^{-\lambda_1 y_1 - \lambda_2 y_2} I_0(M).
\]

---

### 7. CM-step for reward parameters

Reward parameters are updated via conditional maximization using an
unconstrained parameterization:

- `r_from_v(v)` with \(r_{31} = e^{v_1}, r_{32} = e^{v_2}\)
- `fd_grad_v(...)`: finite-difference gradients of the log-likelihood
- `update_v_with_linesearch(...)`: gradient ascent with backtracking line search

---

## Overall ECM procedure

The full estimation routine `run_ecm_clara` proceeds as follows:

1. Initialize \(\lambda_1, \lambda_2, \lambda_3, q, r_{31}, r_{32}\)
2. Repeat until convergence:
   - **E-step:** compute expected sufficient statistics
   - **M-step:** update \(\lambda_1, \lambda_2, \lambda_3, q\) in closed form
   - **CM-step:** update \(r_{31}, r_{32}\) via likelihood ascent
3. Stop when the relative change in the full parameter vector
   \((\lambda_1, \lambda_2, \lambda_3, q, r_{31}, r_{32})\)
   falls below the specified tolerance
4. Report final parameter estimates and the final observed log-likelihood

---
