# Example 2: p-phase model (multivariate) with known CTMC and inversible reward matrix R
---

## Scripts

### Algorithm script
- **`Example_2_algorithm.py`**

Implements simulation, feasibility checks, likelihood evaluation, and an ECM-style estimation
routine for the p-phase model.

### Run script
- **`_Run_Example_2.py`**

Runs 10 Monte Carlo experiments using the implementation in `Example_2_algorithm.py`. It simulates
synthetic data, fits the model, computes diagnostics, and exports results to CSV files.

---

## Algorithmic components (what each function does)

### 1) Simulation of synthetic data
- `simulate_kulkarni_p_dataset(lambdas, R, N, rng)`
  Simulates latent sojourn times \(Z\) with independent exponentials and returns observed
  data \(Y=ZR\).

### 2) Parameterization of R
- `R_from_u_p(u, p)`
  Maps an unconstrained vector \(u \in \mathbb{R}^{p^2}\) (reshaped into \(U\in\mathbb{R}^{p\times p}\))
  to a valid row-stochastic matrix \(R\) using a row-wise softmax. This ensures:
  - \(R_{ij} > 0\) for all \(i,j\)
  - \(\sum_j R_{ij}=1\) for each row \(i\)

### 3) Latent reconstruction and feasibility
- `Z_from_Y(Y, R)`
  Reconstructs latent times by solving \(ZR=Y\) via stable linear solves (no explicit inverse).

- `feasibility_info(Y, R, ...)`
  Computes reconstructed \(Z=YR^{-1}\) and marks an observation feasible if all components satisfy
  \(Z_k \ge -\texttt{tol}\). Matrices \(R\) are rejected if they are near-singular or ill-conditioned.

- `feasibility_hill_climb(Y, u0, p, ...)`
  A repair heuristic that perturbs \(u\) to increase the number of feasible reconstructions when the
  current iterate yields too many infeasible rows (or a non-finite objective).

### 4) Closed-form updates for the rates (on feasible subset)
- `E_step_p(Y, R, ...)`
  Reconstructs \(Z\), keeps only feasible rows, and forms the sufficient-statistic-like sums needed for
  rate updates.

- `M_step_lambdas_p(Estats)`
  Updates \(\lambda_k\) in closed form using feasible reconstructions (componentwise):
  \[
  \hat\lambda_k = \frac{\text{exits}_k}{\sum_{n\in\mathcal{F}} Z^{(n)}_k},
  \]
  where \(\mathcal{F}\) denotes the feasible set. In this implementation, \(\text{exits}_k\) corresponds to
  the assumed “forward/Coxian-style” single visitation structure used in `E_step_p`.

### 5) Penalized likelihood used for R updates
Two likelihood functions are provided:

- `loglik_Y_p_soft(Y, lambdas, R, ...)`
  The objective optimized for updating \(R\). It is a change-of-variables / complete-data-style criterion
  based on deterministic reconstruction \(Z=YR^{-1}\), including:
  - the base exponential log-likelihood terms in \(Z\)
  - a Jacobian term \(-N\log|\det(R)|\)
  - a smooth penalty for negative components of reconstructed \(Z\)

- `loglik_Y_p_hard_subset(...)` (diagnostic only)
  A hard-feasibility diagnostic that evaluates the criterion only on feasible rows and returns \(-\infty\)
  if no rows are feasible. This is not used for optimization; it is used for debugging/diagnostics.

### 6) Numerical update of R (in u-space)
- `fd_grad_u_p(...)`
  Finite-difference approximation of the gradient of the soft objective w.r.t. \(u\).

- `update_u_with_linesearch_p(...)`
  Gradient ascent with backtracking line search to enforce monotone improvement in the soft objective.

### 7) Identifiability / phase canonicalization
- `canonicalize_phases(lambdas, u, p)`
  Enforces a canonical representation to stabilize label switching by sorting phases according to the
  current rate parameters \(\lambda_k\), and permuting the corresponding rows of \(U\) (equivalently, rows
  of \(R\)).

---

## Overall ECM-style estimation procedure

The `run_em_p` alternates between:

1. Construct \(R\) from \(u\) via row-wise softmax.
2. Reconstruct latent times \(Z = YR^{-1}\) and identify feasible observations.
3. Update \(\boldsymbol{\lambda}\) in closed form using feasible reconstructed times only.
4. Update \(u\) (hence \(R\)) by maximizing a penalized log-likelihood with respect to \(u\).
5. Apply phase canonicalization to avoid instability from label switching.
6. Stop when the per-observation improvement in the soft objective is below a tolerance.

---
