# Example 3 — Extended Kulkarni ECM/EM (Soft Support + Adaptive Truncation)

## Scripts

### Algorithm script
- **`Example_3_algorithm.py`**

This script contains all simulation, likelihood evaluation, and estimation
routines for example 3, including the ECM-type estimation algorithm.

### Run script
- **`_Run_Example_3.py`**

This script runs 10 Monte Carlo experiments using the algorithm implemented in
`Example_3_algorithm.py`. It simulates synthetic data, fits the model, computes
diagnostics, and exports results to CSV files.

---

### Latent sufficient statistics
For each trajectory, the latent data include:

- `Z1`, `Z2`: total sojourn times in states 1 and 2  
- `N12`, `N21`: number of transitions 1→2 and 2→1  
- `N2a`: absorption count from state 2 (always 1 per trajectory)

### Observed data (bivariate reward)
We observe a bivariate random vector `Y ∈ R^2` related to sojourn times by:

\[
Y = Z R,
\]

where \(Z = (Z_1, Z_2)\) and \(R \in \mathbb{R}^{2\times 2}\) is a **row-stochastic reward matrix** (each row sums to 1).

---

## What this implementation does

The main estimator is an **ECM/GEM** loop:

1. **E-step**  
   Reconstruct feasible `Z` from `Y` and `R`, then compute expected latent path counts (especially the expected loop count in the Kulkarni series).

2. **CM-step (rates)**  
   Closed-form updates for `(λ, ρ, μ)` using expected sufficient statistics.

3. **GEM step (rewards)**  
   Update reward parameters via monotone ascent on a **penalized observed log-likelihood** using finite-difference gradients and backtracking line search.

---

## Key Features

### 1) Soft support (penalized feasibility)
Not all `Y` are compatible with a given `R` because reconstructing

\[
Z = Y R^{-1}
\]

may produce negative values or fail when `R` is ill-conditioned. This implementation uses:

- feasibility checks via `feasibility_info(...)`
- a **penalized log-likelihood**:
  - subtract `infeas_penalty` for infeasible observations (cannot yield valid `Z`)
  - subtract `negz_penalty` for numerically invalid/near-zero reconstructed `z1,z2`

This avoids hard failure and enables stable optimization of `R`.

---

### 2) Adaptive truncation of the infinite m-sum (**m_cap / “Vcap”**)

#### Why truncation is needed
The extended Kulkarni likelihood involves an **infinite series** over an integer index `m ≥ 1` (interpretable as a latent loop/visit count). Both the E-step and the observed likelihood require:

- \(\log \sum_{m\ge 1} w_m(z_1,z_2)\)
- \(\mathbb{E}[m \mid z_1,z_2]\)

#### What `m_cap` is (“Vcap”)
`m_cap` is the **hard maximum number of m-terms** allowed in the series for **each observation**. In other words:

- **`m_cap` (aka “Vcap” in discussion) = maximum latent loop/visit count included in the approximation.**

It guarantees an upper bound on runtime even in difficult cases.

#### Adaptive stopping (tail rule)
Instead of always summing to `m_cap`, the code stops early when the incremental contribution becomes negligible:

- `tail_rel_tol`: threshold for “small relative contribution”
- `tail_patience`: number of consecutive “small” terms required before stopping
- `m_min`: minimum number of terms before the stopping rule is allowed
- `m_cap`: absolute ceiling if the tail never becomes negligible

This is implemented in:

- `_adaptive_m_posterior_stats(...)`

A diagnostic `m_used_mean` is reported (average last `m` used across feasible observations). If `m_used_mean` is close to `m_cap`, increase `m_cap` or loosen tolerances.

---

## Reward matrix parameterization (sigmoid)
The reward matrix is parameterized via an unconstrained vector `u ∈ R^2` mapped through a bounded sigmoid:

- `r1 = sigmoid(u1)`, `r2 = sigmoid(u2)` with bounds controlled by `EPS_R`
- \[
R =
\begin{pmatrix}
r_1 & 1-r_1\\
r_2 & 1-r_2
\end{pmatrix}
\]

This ensures:
- row-stochasticity,
- entries bounded away from {0,1} for stability,
- a smooth unconstrained optimization variable `u`.

---

## Main functions

### Simulation
- `simulate_kulkarni_path_extended(lam, rho, mu, rng)`  
  Simulates one absorption path; returns `Z1,Z2`, transition counts, etc.

- `simulate_kulkarni_dataset_extended(lam, rho, mu, R, N, rng)`  
  Generates an i.i.d. dataset and outputs `Y` (and latent summaries).

### Estimation
- `run_ecm_extended_monotone_penalized_adaptive(Y, ...)`  
  Runs the ECM loop and returns `(lam_hat, rho_hat, mu_hat, R_hat [, ll_hist])`.

---

## Typical workflow

1. **Choose true parameters** `(λ, ρ, μ)` and a reward matrix `R`.
2. **Simulate data** using `simulate_kulkarni_dataset_extended(...)`.
3. **Estimate parameters** using `run_ecm_extended_monotone_penalized_adaptive(Y, ...)`.
4. Inspect:
   - log-likelihood trace (`ll_hist`)
   - feasibility rate (`feasible / N`)
   - adaptive truncation depth (`m_used_mean`)

---

## Important tuning parameters

### Feasibility / numerical stability
- `tol` (default `1e-12`): feasibility and near-zero guards
- `cond_max` (default `1e12`): reject ill-conditioned `R`
- `infeas_penalty`, `negz_penalty`: soft support penalties

### Adaptive truncation controls (m-sum)
- `m_cap` (default `500`): hard cap (“Vcap”)
- `tail_rel_tol` (default `1e-12`): relative tail threshold
- `tail_patience` (default `6`): consecutive small terms required
- `m_min` (default `10`): minimum terms before stopping allowed

### GEM step for rewards
- `lr0` (default `1e-2`): initial step size for line search
- `max_R_steps` (default `10`): max accepted reward updates per ECM iteration

### Convergence
- `eps_ll` (default `1e-8`): stop if `|Δll| / N < eps_ll`

---

## Notes and limitations
- The observed likelihood is penalized to handle infeasible reconstructions (“soft support”).
- Adaptive truncation is an approximation of an infinite series; accuracy depends on the tail controls.
- If you frequently hit `m_cap`, increase it or adjust tail tolerances.

---

## Suggested citation / context
This code corresponds to the **extended Kulkarni example** with back-jumps and bivariate MPH\* rewards, estimated via ECM with reward optimization and adaptive series truncation.
