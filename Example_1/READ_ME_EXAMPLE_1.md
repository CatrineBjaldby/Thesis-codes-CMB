# Example 1: Bivariate with known CTMC and inversible R
## Scripts

### Algorithm script
- **`Example_1_algorithm.py`**

This script contains all simulation, likelihood evaluation, and estimation
routines for example 1. Note that notation is not the same as in the derivation in the thesis. 

### Run script
- **`_Run_Example_1.py`**

This script runs 10 Monte Carlo experiments using the algorithm implemented in
`Example_1_algorithm.py`. It simulates synthetic data, fits the model, computes
diagnostics, and exports results to CSV files.

## Model overview
- Initial state: phase 1
- Transition \(1 \to 2\) with rate \(\lambda > 0\)
- Absorption from phase 2 with rate \(\mu > 0\)

Latent phase sojourn times are:
- \(Z_1 \sim \mathrm{Exp}(\lambda)\)
- \(Z_2 \sim \mathrm{Exp}(\mu)\)

Observed data are given by a linear transformation:
\[
Y = Z R,
\]
where \(Z = (Z_1, Z_2)\) and \(R \in \mathbb{R}^{2\times2}\) is an unknown
row-stochastic mixing matrix

---

## Overview of the algorithmic components

### 1. Simulation functions

- `simulate_kulkarni_path(lam, mu, rng)`  
  Simulates a single absorption path, returning latent times and transition counts.

- `simulate_kulkarni_dataset(lam, mu, R, N, rng)`  
  Simulates `N` independent observations and returns latent times `Z` and
  observed data `Y`.

---

### 2. Parameterization of the mixing matrix R

The reward matrix `R` is parameterized via a two-dimensional unconstrained vector `u = (u₁, u₂)`:

- `R_from_u(u)` maps `u` to a valid row-stochastic matrix `R`
- `u_from_R(R)` provides the inverse mapping

A bounded logistic transform ensures:
- all entries of `R` lie in \((\varepsilon, 1-\varepsilon)\)
- `R` remains well-conditioned and non-degenerate

---

### 3. Latent time reconstruction and feasibility checks

- `Z_from_Y(Y, R)`  
  Reconstructs latent times via the linear system \(Z R = Y\), solved using stable linear algebra routines.

- `feasibility_info(Y, R)`  
  Checks whether reconstructed latent times are physically meaningful
  (non-negative up to tolerance) and rejects ill-conditioned matrices `R`.

Only feasible observations are used in rate updates.

---

### 4. Likelihood evaluation

- `loglik_Y(Y, lam, mu, R)`  

The log-likelihood is evaluated using a change-of-variables formulation:
\[
\ell(Y;\lambda,\mu,R)
= \sum_i \left(
\log\lambda + \log\mu
- \lambda Z_{1,i} - \mu Z_{2,i}
\right)
- N \log|\det(R)|
- \text{penalty}(Z),
\]
where \(Z = YR^{-1}\).

A smooth penalty is applied when reconstructed latent times become negative.
This likelihood is used only for numerical optimization of `R`.

---

### 5. Symmetry handling

- `symmetries_of(...)`
- `choose_best_by_loglik(...)`

The model admits multiple equivalent parameterizations due to label-switching
and sign symmetries. All symmetric configurations are evaluated, and the one
with highest likelihood is selected to ensure numerical stability and
identifiability across iterations.

---

### 6. E-step: deterministic latent reconstruction

- `E_step(Y, R)`

This step reconstructs latent times deterministically and computes sufficient
statistics using only feasible observations:
- total latent time in phase 1
- total latent time in phase 2
- number of transitions and absorptions

This is not a classical expectation step, since no conditional expectations are
computed.

---

### 7. M-step: closed-form rate updates

- `M_step_rates(Estats)`

Given reconstructed latent times, the rate parameters are updated in closed
form:
\[
\hat{\lambda} = \frac{N_{\text{feas}}}{\sum Z_1},
\qquad
\hat{\mu} = \frac{N_{\text{feas}}}{\sum Z_2}.
\]

---

### 8. Conditional maximization for R

- `fd_grad_u(...)`
- `update_u_with_linesearch(...)`
- `feasibility_hill_climb(...)`

The mixing matrix `R` is updated by numerical gradient ascent in the
unconstrained parameter `u`, using:
- finite-difference gradients
- backtracking line search
- determinant and condition-number checks
- feasibility repair via randomized hill-climbing if needed

---

## Overall EM / ECM procedure

The full estimation algorithm (`run_em`) proceeds as follows:

1. Initialize \(\lambda\), \(\mu\), and `u`
2. Normalize parameters using symmetry selection
3. Repeat until convergence:
   - Reconstruct latent times \(Z = YR^{-1}\)
   - **E-step:** compute sufficient statistics from feasible observations
   - **M-step:** update \(\lambda\) and \(\mu\) in closed form
   - **CM-step:** update `R` numerically via likelihood ascent
   - Apply symmetry normalization
4. Stop when the per-observation improvement in log-likelihood falls below a
   specified tolerance
---

