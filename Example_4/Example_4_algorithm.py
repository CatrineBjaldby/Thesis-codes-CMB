import numpy as np
from math import exp, isfinite


# DATA GENERATION 
def simulate_clara_dataset(lam1, lam2, lam3, q, r31, r32, N, rng=None):
    """
    Simulate N i.i.d. observations (Y1, Y2) from the Clara non-invertible R example.

    Model:
      - CTMC with 3 transient states (1 -> 2 -> {a or 3}, 3 -> a)
      - Sojourn times:
          Z1 ~ Exp(lam1)
          Z2 ~ Exp(lam2)
          Z3 ~ Exp(lam3) if H=1, Z3=0 if H=0
      - H ~ Bernoulli(q): indicator of visiting state 3
      - Rewards:
          Y1 = Z1 + r31 * Z3
          Y2 = Z2 + r32 * Z3
    """

    if rng is None:
        rng = np.random.default_rng()

    Y = np.zeros((N, 2))
    Z = np.zeros((N, 3))
    H = np.zeros(N)

    for n in range(N):
        # sojourn times in states 1 and 2
        z1 = rng.exponential(scale=1.0 / lam1)
        z2 = rng.exponential(scale=1.0 / lam2)

        # indicator of visiting state 3
        h = rng.binomial(1, q)

        if h == 1:
            z3 = rng.exponential(scale=1.0 / lam3)
        else:
            z3 = 0.0

        y1 = z1 + r31 * z3
        y2 = z2 + r32 * z3

        Y[n, 0] = y1
        Y[n, 1] = y2
        Z[n, :] = [z1, z2, z3]
        H[n] = h

    return Y, Z, H


# INTEGRALS NEEDED FOR E-STEP to make the intractable z_3 with quadrature
def integrals_for_x3(M, lam1, lam2, lam3, r31, r32,
                     use_closed_form=True, n_grid=200):
    """
    Compute
      I0 = ∫_0^M exp(a u) du
      I1 = ∫_0^M u exp(a u) du
    where a = lam1*r31 + lam2*r32 - lam3.

    If use_closed_form=True, use analytic formulas, with a~0 handled separately.
    """
    a = lam1 * r31 + lam2 * r32 - lam3

    if M <= 0:
        return 0.0, 0.0

    if use_closed_form and abs(a) > 1e-12:
        I0 = (exp(a * M) - 1.0) / a
        I1 = (exp(a * M) * (a * M - 1.0) + 1.0) / (a**2)
        return I0, I1

    if use_closed_form and abs(a) <= 1e-12:
        # a ~ 0 => exp(a u) ~ 1
        I0 = M
        I1 = 0.5 * M**2
        return I0, I1

    # Numerical fallback
    u = np.linspace(0.0, M, n_grid)
    f0 = np.exp(a * u)
    f1 = u * f0
    I0 = np.trapz(f0, u)
    I1 = np.trapz(f1, u)
    return I0, I1


# E-STEP FOR A SINGLE OBSERVATION
def E_for_one_obs(y1, y2, lam1, lam2, lam3, q, r31, r32, tol=1e-12):
    """
    E-step contributions for a single observed pair (y1,y2) for the clara model
    Returns:
      Ez1, Ez2, EHz3, Eh
    corresponding to:
      E[Z1 | Y], E[Z2 | Y], E[H Z3 | Y], E[H | Y].
    """

    if r31 <= 0 or r32 <= 0:
        raise ValueError("This implementation assumes r31, r32 > 0.")

    # feasible range for Z3 contribution
    M = min(y1 / r31, y2 / r32)
    if M <= tol:
        # H must effectively be 0
        Eh = 0.0
        Ez3 = 0.0
        EHz3 = 0.0
        Ez1 = y1
        Ez2 = y2
        return Ez1, Ez2, EHz3, Eh

    # integrals for the X3 density when H=1
    I0, I1 = integrals_for_x3(M, lam1, lam2, lam3, r31, r32)
    C_val = lam3 * I0  # C(y1,y2) from the derivation

    denom = (1.0 - q) + q * C_val
    if denom <= tol:
        # Degenerate: treat as H=0
        Eh = 0.0
        Ez3 = 0.0
        EHz3 = 0.0
        Ez1 = y1
        Ez2 = y2
        return Ez1, Ez2, EHz3, Eh

    # Posterior probabilities
    p_H1 = q * C_val / denom
    p_H0 = 1.0 - p_H1

    # E[Z3 | Y, H=1]
    if I0 <= tol:
        Ez3 = 0.0
    else:
        Ez3 = I1 / I0

    # E[H Z3 | Y] = P(H=1|Y) * E[Z3 | Y,H=1]
    EHz3 = p_H1 * Ez3

    # E[Z1 | Y]
    Ez1 = p_H0 * y1 + p_H1 * (y1 - r31 * Ez3)

    # E[Z2 | Y]
    Ez2 = p_H0 * y2 + p_H1 * (y2 - r32 * Ez3)

    Eh = p_H1

    return Ez1, Ez2, EHz3, Eh


#E-STEP FOR THE WHOLE SAMPLE
def E_step_clara(Y, lam1, lam2, lam3, q, r31, r32, tol=1e-12):
    """
    E-step for the Clara example.

    Y: Nx2 array of observations (Y1,Y2).

    Returns expected sufficient statistics:
      S1_hat   ≈ E[ sum_n Z1^{(n)} | Y ],
      S2_hat   ≈ E[ sum_n Z2^{(n)} | Y ],
      S3H_hat  ≈ E[ sum_n H^{(n)} Z3^{(n)} | Y ],
      H_sum_hat≈ E[ sum_n H^{(n)} | Y ].
    """
    S1_hat = 0.0
    S2_hat = 0.0
    S3H_hat = 0.0
    H_sum_hat = 0.0

    for n in range(Y.shape[0]):
        y1, y2 = Y[n, :]
        Ez1, Ez2, EHz3, Eh = E_for_one_obs(
            y1, y2, lam1, lam2, lam3, q, r31, r32, tol=tol
        )
        S1_hat += Ez1
        S2_hat += Ez2
        S3H_hat += EHz3
        H_sum_hat += Eh

    return S1_hat, S2_hat, S3H_hat, H_sum_hat


# M-STEP FOR (lambda1, lambda2, lambda3, q)

def M_step_clara(S1_hat, S2_hat, S3H_hat, H_sum_hat, N, eps=1e-12):
    """
    Closed-form M-step for (lambda1, lambda2, lambda3, q)
    using expected sufficient statistics from the E-step.

    N: number of observations.
    """

    lam1_new = N / max(S1_hat, eps)
    lam2_new = N / max(S2_hat, eps)

    if H_sum_hat <= eps or S3H_hat <= eps:
        lam3_new = None  # do not update if not enough mass on H=1
    else:
        lam3_new = H_sum_hat / max(S3H_hat, eps)

    q_new = H_sum_hat / N

    return lam1_new, lam2_new, lam3_new, q_new


# OBSERVED LOG-LIKELIHOOD 
def loglik_clara(Y, lam1, lam2, lam3, q, r31, r32, tol=1e-300):
    """
    Observed log-likelihood of Y under the Clara example
    with parameters (lam1, lam2, lam3, q, r31, r32).

    Uses the analytical joint density:
      f_Y(y1,y2) = (1-q)*A(y1,y2) + q*B(y1,y2)
    with A and B defined as in the thesis derivation.
    """

    N = Y.shape[0]
    total = 0.0

    for n in range(N):
        y1, y2 = Y[n, :]

        # A term (H=0)
        A_val = lam1 * lam2 * np.exp(-lam1 * y1 - lam2 * y2)

        # B term (H=1)
        if r31 > 0 and r32 > 0:
            M = min(y1 / r31, y2 / r32)
            if M > 0:
                I0, _ = integrals_for_x3(M, lam1, lam2, lam3, r31, r32)
                B_val = lam1 * lam2 * lam3 * np.exp(-lam1 * y1 - lam2 * y2) * I0
            else:
                B_val = 0.0
        else:
            B_val = 0.0

        dens = (1.0 - q) * A_val + q * B_val

        if dens <= tol or not isfinite(dens):
            return -np.inf

        total += np.log(dens)

    return float(total)


# ECM STEP FOR R: PARAMETERISATION r31 = exp(v1), r32 = exp(v2)
def r_from_v(v):
    """
    Map unconstrained v in R^2 to positive reward parameters:
      r31 = exp(v[0]),
      r32 = exp(v[1]).
    """
    r31 = float(np.exp(v[0]))
    r32 = float(np.exp(v[1]))
    return r31, r32


def fd_grad_v(Y, lam1, lam2, lam3, q, v, step=1e-5, tol=1e-300):
    """
    Finite-difference gradient of loglik_clara with respect to v = (v1,v2),
    where r31 = exp(v1), r32 = exp(v2).

    Uses central differences where possible, falls back to one-sided
    if needed, and sets gradient component to 0 if both sides are -inf.
    """
    r31, r32 = r_from_v(v)
    base_ll = loglik_clara(Y, lam1, lam2, lam3, q, r31, r32, tol=tol)
    g = np.zeros_like(v)

    if not np.isfinite(base_ll):
        return base_ll, g

    for j in range(len(v)):
        v_p = v.copy()
        v_m = v.copy()
        v_p[j] += step
        v_m[j] -= step

        r31_p, r32_p = r_from_v(v_p)
        r31_m, r32_m = r_from_v(v_m)

        ll_p = loglik_clara(Y, lam1, lam2, lam3, q, r31_p, r32_p, tol=tol)
        ll_m = loglik_clara(Y, lam1, lam2, lam3, q, r31_m, r32_m, tol=tol)

        if np.isfinite(ll_p) and np.isfinite(ll_m):
            g[j] = (ll_p - ll_m) / (2.0 * step)
        elif np.isfinite(ll_p):
            g[j] = (ll_p - base_ll) / step
        elif np.isfinite(ll_m):
            g[j] = (base_ll - ll_m) / step
        else:
            g[j] = 0.0

    return base_ll, g


# CM step with linesearch where rates fro CTMC kept fixed
def update_v_with_linesearch(
    Y,
    lam1, lam2, lam3, q,
    v0,
    lr=1e-2,
    max_backtracks=12,
    tol=1e-300,
    verbose=False
):
    """
    One gradient-ascent + backtracking line-search step for v (hence r31,r32).

    Returns:
      v_new, ll_new
    """

    base_ll, g = fd_grad_v(Y, lam1, lam2, lam3, q, v0, step=1e-5, tol=tol)

    if not np.isfinite(base_ll):
        return v0, -np.inf

    v = v0.copy()

    for _ in range(max_backtracks):
        v_try = v + lr * g
        r31_try, r32_try = r_from_v(v_try)
        ll_try = loglik_clara(Y, lam1, lam2, lam3, q, r31_try, r32_try, tol=tol)

        if np.isfinite(ll_try) and ll_try > base_ll + 1e-8:
            if verbose:
                print(f"  [ECM-R] improved ll from {base_ll:.6f} to {ll_try:.6f}")
            return v_try, ll_try

        lr *= 0.5  # backtrack

    # No improvement
    if verbose:
        print("  [ECM-R] no improvement, staying at current v")
    return v0, base_ll


# FULL ECM ALGORITHM
def run_ecm_clara(
    Y,
    lam1_0,
    lam2_0,
    lam3_0,
    q_0,
    r31_0,
    r32_0,
    max_iter=100,
    max_R_inner_steps=10,
    rel_tol=1e-6,
    tol=1e-300,
    verbose=True,
    true_params=None,  
    track_ll=False,     
):
    """
    ECM algorithm for the Clara non-invertible R example.

    If track_ll=True, the function returns:
      lam1, lam2, lam3, q, r31, r32, ll_history
    where ll_history[k] is the log-likelihood after iteration k+1.
    """

    N = Y.shape[0]

    lam1 = float(lam1_0)
    lam2 = float(lam2_0)
    lam3 = float(lam3_0)
    q    = float(q_0)

    # log-parameterisation for r31, r32
    v = np.array([np.log(r31_0), np.log(r32_0)], dtype=float)

    # initial parameter vector for convergence check
    r31, r32 = r_from_v(v)
    prev_theta = np.array([lam1, lam2, lam3, q, r31, r32])

    # history of log-likelihood values 
    ll_history = [] if track_ll else None

    for it in range(1, max_iter + 1):
        # ---------- E-step ----------
        S1_hat, S2_hat, S3H_hat, H_sum_hat = E_step_clara(
            Y, lam1, lam2, lam3, q, r31, r32, tol=tol
        )

        if verbose:
            print(f"\n=== ECM Iteration {it} ===")
            print(f"  S1_hat   = {S1_hat:.6f}")
            print(f"  S2_hat   = {S2_hat:.6f}")
            print(f"  S3H_hat  = {S3H_hat:.6f}")
            print(f"  H_sum    = {H_sum_hat:.6f}")

        # ---------- M-step for (lambda1, lambda2, lambda3, q) ----------
        lam1_new, lam2_new, lam3_new, q_new = M_step_clara(
            S1_hat, S2_hat, S3H_hat, H_sum_hat, N
        )

        lam1 = lam1_new
        lam2 = lam2_new
        if lam3_new is not None:
            lam3 = lam3_new
        # keep q in (0,1)
        q = np.clip(q_new, 1e-6, 1.0 - 1e-6)

        if verbose:
            print("  M-step for (lambda1, lambda2, lambda3, q):")
            print(f"    lambda1 = {lam1:.6f}")
            print(f"    lambda2 = {lam2:.6f}")
            print(f"    lambda3 = {lam3:.6f}")
            print(f"    q       = {q:.6f}")

        # ---------- ECM-step for R (r31, r32) ----------
        v_candidate = v.copy()
        r31_cand, r32_cand = r_from_v(v_candidate)
        base_ll = loglik_clara(Y, lam1, lam2, lam3, q, r31_cand, r32_cand, tol=tol)

        if verbose:
            print(f"  Initial log-likelihood (before R-update) = {base_ll:.6f}")

        for k in range(max_R_inner_steps):
            v_try, ll_try = update_v_with_linesearch(
                Y, lam1, lam2, lam3, q, v_candidate,
                lr=1e-2, tol=tol, verbose=False
            )

            if not np.isfinite(ll_try) or ll_try <= base_ll + 1e-8:
                # no improvement
                break

            v_candidate = v_try
            base_ll = ll_try

        v = v_candidate
        r31, r32 = r_from_v(v)

        if verbose:
            print("  ECM-step for R (r31, r32):")
            print(f"    r31 = {r31:.6f}")
            print(f"    r32 = {r32:.6f}")
            print(f"    log-likelihood after R-update = {base_ll:.6f}")

        # ---------- Convergence check ----------
        theta = np.array([lam1, lam2, lam3, q, r31, r32])
        rel_change = np.linalg.norm(theta - prev_theta) / (1e-8 + np.linalg.norm(prev_theta))

        if track_ll:
            ll_cur = loglik_clara(Y, lam1, lam2, lam3, q, r31, r32, tol=tol)
            ll_history.append(ll_cur)

        if verbose:
            print(f"  relative parameter change = {rel_change:.3e}")
            if track_ll:
                print(f"  log-likelihood at this iteration = {ll_cur:.6f}")

        if rel_change < rel_tol:
            if verbose:
                print("Converged (parameter change criterion).")
            break

        prev_theta = theta

    # ---------- Final report ----------
    ll_est = loglik_clara(Y, lam1, lam2, lam3, q, r31, r32, tol=tol)

    if verbose:
        print("\n=== Final ECM estimates (Clara) ===")
        print(f"lambda1 = {lam1:.6f}")
        print(f"lambda2 = {lam2:.6f}")
        print(f"lambda3 = {lam3:.6f}")
        print(f"q       = {q:.6f}")
        print(f"r31     = {r31:.6f}")
        print(f"r32     = {r32:.6f}")
        print(f"log-likelihood at final estimates = {ll_est:.6f}")

        if true_params is not None:
            lam1_t, lam2_t, lam3_t, q_t, r31_t, r32_t = true_params
            ll_true = loglik_clara(Y, lam1_t, lam2_t, lam3_t, q_t, r31_t, r32_t, tol=tol)
            print(f"log-likelihood under true parameters = {ll_true:.6f}")

    if track_ll:
        return lam1, lam2, lam3, q, r31, r32, np.asarray(ll_history)

    return lam1, lam2, lam3, q, r31, r32

