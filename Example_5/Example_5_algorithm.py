
from __future__ import annotations

import math
from typing import Tuple, Optional, List, Dict

import numpy as np


# =========================================================
# 0) Stable utilities
# =========================================================

def logsumexp(logw: np.ndarray) -> float:
    """Compute log(sum(exp(logw))) stably for 1D array."""
    m = float(np.max(logw))
    if not np.isfinite(m):
        return -np.inf
    s = float(np.sum(np.exp(logw - m)))
    if s <= 0.0 or not np.isfinite(s):
        return -np.inf
    return m + math.log(s)


def logaddexp(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b))."""
    if not np.isfinite(a):
        return b
    if not np.isfinite(b):
        return a
    m = a if a >= b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def log_trapz_from_logf(logf: np.ndarray, x: np.ndarray) -> float:
    """
    Compute log(∫ exp(logf(x)) dx) via trapezoidal rule, stably.
    Assumes x is strictly increasing, length>=2.
    """
    if len(x) < 2:
        return -np.inf
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        raise ValueError("x must be strictly increasing for trapezoidal integration.")
    a = logf[:-1]
    b = logf[1:]
    m = np.maximum(a, b)
    pair = m + np.log(np.exp(a - m) + np.exp(b - m))  # log(e^a + e^b)
    log_terms = np.log(dx) + math.log(0.5) + pair     # log(0.5*(e^a+e^b)*dx)
    return logsumexp(log_terms)


def safe_exp(x: float) -> float:
    """Exponentiate with basic clipping to avoid overflow/underflow."""
    if x < -745:
        return 0.0
    if x > 709:
        return float("inf")
    return math.exp(x)


# =========================================================
# 1) Simulation (optional)
# =========================================================

def simulate_noninv_unknown_one(
    lam: float,
    rho: float,
    mu: float,
    kappa: float,
    lam3: float,
    r31: float,
    r32: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Simulate one trajectory and return:
      Y=(y1,y2), Z=(z1,z2,z3), V (visits to state 2), H (exit via state 3 indicator).
    """
    if lam <= 0 or rho < 0 or mu <= 0 or kappa <= 0 or lam3 <= 0:
        raise ValueError("Rates must satisfy lam>0, mu>0, kappa>0, lam3>0, rho>=0.")
    if r31 <= 0 or r32 <= 0:
        raise ValueError("Reward parameters r31,r32 must be >0.")
    q2 = rho + mu + kappa
    if q2 <= 0:
        raise ValueError("q2 must be >0.")

    z1 = 0.0
    z2 = 0.0
    z3 = 0.0
    V = 0
    H = 0

    state = 1  # start in 1
    while True:
        if state == 1:
            z1 += rng.exponential(scale=1.0 / lam)
            state = 2
            continue

        if state == 2:
            V += 1
            z2 += rng.exponential(scale=1.0 / q2)
            u = rng.uniform()
            if u < rho / q2:
                state = 1
            elif u < (rho + mu) / q2:
                H = 0
                z3 = 0.0
                break
            else:
                H = 1
                state = 3
            continue

        if state == 3:
            z3 += rng.exponential(scale=1.0 / lam3)
            break

        raise RuntimeError("Invalid state encountered.")

    y1 = z1 + r31 * z3
    y2 = z2 + r32 * z3
    return np.array([y1, y2], float), np.array([z1, z2, z3], float), V, H


def simulate_noninv_unknown_dataset(
    lam: float,
    rho: float,
    mu: float,
    kappa: float,
    lam3: float,
    r31: float,
    r32: float,
    N: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate N i.i.d. observations; return arrays Y, Z, V, H."""
    if rng is None:
        rng = np.random.default_rng()

    Y = np.zeros((N, 2), float)
    Z = np.zeros((N, 3), float)
    V = np.zeros(N, int)
    H = np.zeros(N, int)

    for n in range(N):
        y, z, v, h = simulate_noninv_unknown_one(lam, rho, mu, kappa, lam3, r31, r32, rng)
        Y[n, :] = y
        Z[n, :] = z
        V[n] = v
        H[n] = h

    return Y, Z, V, H


# =========================================================
# 2) Densities and helper functions
# =========================================================

def log_gamma_pdf(x: float, shape: int, rate: float) -> float:
    """
    Gamma(shape=k, rate=beta): f(x)=beta^k / Gamma(k) * x^{k-1} exp(-beta x), x>0.
    Here shape is integer v>=1.
    """
    if x <= 0.0 or rate <= 0.0 or shape <= 0:
        return -np.inf
    k = int(shape)
    return k * math.log(rate) - math.lgamma(k) + (k - 1) * math.log(x) - rate * x


def M_of_y(y1: float, y2: float, r31: float, r32: float) -> float:
    """M(y)=min(y1/r31, y2/r32)."""
    if r31 <= 0 or r32 <= 0:
        return 0.0
    return min(y1 / r31, y2 / r32)


def log_fY_given_v_h0(y1: float, y2: float, v: int, lam: float, q2: float) -> float:
    """log f(Y|V=v,H=0) = log Gamma(v,lam)(y1) + log Gamma(v,q2)(y2)."""
    return log_gamma_pdf(y1, v, lam) + log_gamma_pdf(y2, v, q2)


def log_integrals_for_h1(
    y1: float,
    y2: float,
    v: int,
    lam: float,
    q2: float,
    lam3: float,
    r31: float,
    r32: float,
    n_grid: int = 200,
) -> Tuple[float, float]:
    """
    Compute (logI0, logI1) where
      I0 = ∫_0^M g(z) dz  = f(Y|v,H=1)
      I1 = ∫_0^M z g(z) dz
    with
      g(z)=Gamma(v,lam)(y1-r31 z)*Gamma(v,q2)(y2-r32 z)*Exp(lam3)(z).
    """
    M = M_of_y(y1, y2, r31, r32)
    if v <= 0 or M <= 0.0:
        return -np.inf, -np.inf
    if lam <= 0 or q2 <= 0 or lam3 <= 0:
        return -np.inf, -np.inf

    z = np.linspace(0.0, M, int(n_grid), dtype=float)
    x1 = y1 - r31 * z
    x2 = y2 - r32 * z

    if np.any(x1 < -1e-10) or np.any(x2 < -1e-10):
        return -np.inf, -np.inf

    tiny = 1e-300
    x1c = np.clip(x1, tiny, None)
    x2c = np.clip(x2, tiny, None)

    lgam_v = math.lgamma(v)
    logf1 = v * math.log(lam) - lgam_v + (v - 1) * np.log(x1c) - lam * x1c
    logf2 = v * math.log(q2)  - lgam_v + (v - 1) * np.log(x2c) - q2 * x2c
    logf3 = math.log(lam3) - lam3 * z

    logg = logf1 + logf2 + logf3
    logI0 = log_trapz_from_logf(logg, z)

    logz = np.full_like(z, -np.inf)
    mask = z > 0.0
    logz[mask] = np.log(z[mask])
    logI1 = log_trapz_from_logf(logg + logz, z)

    return float(logI0), float(logI1)


def log_fY_given_v_h1(
    y1: float,
    y2: float,
    v: int,
    lam: float,
    q2: float,
    lam3: float,
    r31: float,
    r32: float,
    n_grid: int = 200,
) -> float:
    """log f(Y|V=v,H=1) = logI0."""
    logI0, _ = log_integrals_for_h1(y1, y2, v, lam, q2, lam3, r31, r32, n_grid=n_grid)
    return logI0


# =========================================================
# 3) ADAPTIVE E-step (posterior weights w_{v,h}) 
# =========================================================

def E_for_one_obs_adaptive_cached(
    y1: float,
    y2: float,
    lam: float,
    rho: float,
    mu: float,
    kappa: float,
    lam3: float,
    r31: float,
    r32: float,
    # adaptive truncation controls
    v_cap: int = 500,
    tail_rel_tol: float = 1e-12,
    tail_patience: int = 6,
    v_min: int = 10,
    # quadrature
    n_grid: int = 200,
    # misc
    z3_tol: float = 1e-12,
) -> Tuple[float, float, float, float, float, int, float]:
    """
    Adaptive E-step for one observation with integral caching.

    Returns:
      EV, EH, EHZ3, EZ1, EZ2, v_used, ll_y

    where ll_y = log fY(y) computed from the same log-weights (no extra work).
    """
    if y1 <= 0.0 or y2 <= 0.0:
        # Degenerate safeguard
        return 1.0, 0.0, 0.0, max(y1, 0.0), max(y2, 0.0), 1, -np.inf

    if lam <= 0 or mu <= 0 or kappa <= 0 or lam3 <= 0 or rho < 0:
        raise ValueError("Invalid rate parameters.")
    if r31 <= 0 or r32 <= 0:
        raise ValueError("r31,r32 must be >0.")

    q2 = rho + mu + kappa
    if q2 <= 0.0:
        raise ValueError("q2 must be >0.")

    p21 = rho / q2
    p2a = mu / q2
    p23 = kappa / q2

    log_p21 = -np.inf if p21 <= 0 else math.log(p21)
    log_p2a = -np.inf if p2a <= 0 else math.log(p2a)
    log_p23 = -np.inf if p23 <= 0 else math.log(p23)

    M = M_of_y(y1, y2, r31, r32)

    logw0_list: List[float] = []
    logw1_list: List[float] = []

    # Cache integrals for H=1: v -> (logI0, logI1)
    # Only stored when H=1 term is finite/used.
    I_cache: Dict[int, Tuple[float, float]] = {}

    log_mass = -np.inf
    small_count = 0
    v_used = 0

    for v in range(1, v_cap + 1):
        v_used = v

        # log P(V=v,H=h)
        if v == 1:
            lp_v0 = log_p2a
            lp_v1 = log_p23
        else:
            if not np.isfinite(log_p21):
                lp_v0 = -np.inf
                lp_v1 = -np.inf
            else:
                lp_v0 = (v - 1) * log_p21 + log_p2a
                lp_v1 = (v - 1) * log_p21 + log_p23

        # h=0
        lw0 = lp_v0 + log_fY_given_v_h0(y1, y2, v, lam, q2)

        # h=1 (compute integral ONCE and cache)
        if M > z3_tol and np.isfinite(lp_v1):
            logI0, logI1 = log_integrals_for_h1(
                y1, y2, v, lam, q2, lam3, r31, r32, n_grid=n_grid
            )
            if np.isfinite(logI0):
                I_cache[v] = (logI0, logI1)
                lw1 = lp_v1 + logI0
            else:
                lw1 = -np.inf
        else:
            lw1 = -np.inf

        logw0_list.append(lw0)
        logw1_list.append(lw1)

        # incremental combined weight for this v
        log_incr = logaddexp(lw0, lw1)

        # update total mass
        new_log_mass = logaddexp(log_mass, log_incr)

        # stopping rule (relative contribution of this v)
        if np.isfinite(log_mass):
            rel = safe_exp(log_incr - new_log_mass)
        else:
            rel = 1.0

        log_mass = new_log_mass

        if v >= v_min and rel < tail_rel_tol:
            small_count += 1
        else:
            small_count = 0

        if v >= v_min and small_count >= tail_patience:
            break

    logw0 = np.asarray(logw0_list, dtype=float)
    logw1 = np.asarray(logw1_list, dtype=float)

    # ll_y = log fY(y)
    ll_y = logsumexp(np.concatenate([logw0, logw1]))
    if not np.isfinite(ll_y):
        return 1.0, 0.0, 0.0, y1, y2, v_used, -np.inf

    post0 = np.exp(logw0 - ll_y)
    post1 = np.exp(logw1 - ll_y)

    v_idx = np.arange(1, v_used + 1, dtype=float)
    EH = float(np.sum(post1))
    EV = float(np.sum(v_idx * (post0 + post1)))

    # EHZ3 = sum_v P(v,1|y) * E[Z3|y,v,1]
    # Use cached integrals: Ez3 = exp(logI1 - logI0)
    EHZ3 = 0.0
    if EH > z3_tol and M > z3_tol:
        for v in range(1, v_used + 1):
            pv1 = float(post1[v - 1])
            if pv1 <= 0.0:
                continue
            if v not in I_cache:
                continue
            logI0, logI1 = I_cache[v]
            if not (np.isfinite(logI0) and np.isfinite(logI1)):
                continue
            Ez3 = safe_exp(logI1 - logI0)
            EHZ3 += pv1 * Ez3

    EZ1 = max(float(y1 - r31 * EHZ3), 0.0)
    EZ2 = max(float(y2 - r32 * EHZ3), 0.0)

    return EV, EH, EHZ3, EZ1, EZ2, v_used, float(ll_y)


def E_step_adaptive(
    Y: np.ndarray,
    lam: float,
    rho: float,
    mu: float,
    kappa: float,
    lam3: float,
    r31: float,
    r32: float,
    # adaptive controls
    v_cap: int = 500,
    tail_rel_tol: float = 1e-12,
    tail_patience: int = 6,
    v_min: int = 10,
    # quadrature
    n_grid: int = 200,
) -> Tuple[float, float, float, float, float, float]:
    """
    Backward-compatible aggregated E-step sufficient statistics:
      S1_hat, S2_hat, V_hat, H_hat, HZ3_hat, v_used_mean
    """
    S1, S2, Vsum, Hsum, HZ3sum, v_used_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for n in range(Y.shape[0]):
        y1, y2 = float(Y[n, 0]), float(Y[n, 1])
        EV, EH, EHZ3, EZ1, EZ2, v_used, _ll_y = E_for_one_obs_adaptive_cached(
            y1, y2, lam, rho, mu, kappa, lam3, r31, r32,
            v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
            n_grid=n_grid
        )
        S1 += EZ1
        S2 += EZ2
        Vsum += EV
        Hsum += EH
        HZ3sum += EHZ3
        v_used_acc += float(v_used)

    v_used_mean = v_used_acc / max(float(Y.shape[0]), 1.0)
    return float(S1), float(S2), float(Vsum), float(Hsum), float(HZ3sum), float(v_used_mean)


def E_step_adaptive_with_ll(
    Y: np.ndarray,
    lam: float,
    rho: float,
    mu: float,
    kappa: float,
    lam3: float,
    r31: float,
    r32: float,
    # adaptive controls
    v_cap: int = 500,
    tail_rel_tol: float = 1e-12,
    tail_patience: int = 6,
    v_min: int = 10,
    # quadrature
    n_grid: int = 200,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Aggregated E-step sufficient statistics PLUS observed log-likelihood:

      S1_hat, S2_hat, V_hat, H_hat, HZ3_hat, v_used_mean, ll_total

    where ll_total = sum_n log fY(y^(n)) computed from E-step weights (no extra loops).
    """
    S1, S2, Vsum, Hsum, HZ3sum, v_used_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ll_total = 0.0

    for n in range(Y.shape[0]):
        y1, y2 = float(Y[n, 0]), float(Y[n, 1])
        EV, EH, EHZ3, EZ1, EZ2, v_used, ll_y = E_for_one_obs_adaptive_cached(
            y1, y2, lam, rho, mu, kappa, lam3, r31, r32,
            v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
            n_grid=n_grid
        )
        S1 += EZ1
        S2 += EZ2
        Vsum += EV
        Hsum += EH
        HZ3sum += EHZ3
        v_used_acc += float(v_used)
        ll_total += float(ll_y)

    v_used_mean = v_used_acc / max(float(Y.shape[0]), 1.0)
    return float(S1), float(S2), float(Vsum), float(Hsum), float(HZ3sum), float(v_used_mean), float(ll_total)


# =========================================================
# 4) CM-step: closed-form rate updates (THEORY)
# =========================================================

def M_step_rates(
    S1_hat: float,
    S2_hat: float,
    V_hat: float,
    H_hat: float,
    HZ3_hat: float,
    N: int,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float, float]:
    """Closed-form CM-step for (lam, rho, mu, kappa, lam3) holding (r31,r32) fixed."""
    S1 = max(S1_hat, eps)
    S2 = max(S2_hat, eps)
    HZ3 = max(HZ3_hat, eps)

    lam_new = max(V_hat / S1, eps)
    rho_new = max((V_hat - N) / S2, eps)
    mu_new = max((N - H_hat) / S2, eps)
    kappa_new = max(H_hat / S2, eps)
    lam3_new = max(H_hat / HZ3, eps)

    return float(lam_new), float(rho_new), float(mu_new), float(kappa_new), float(lam3_new)


# =========================================================
# 5) Observed log-likelihood (kept for R-update / diagnostics)
# =========================================================

def loglik_adaptive(
    Y: np.ndarray,
    lam: float,
    rho: float,
    mu: float,
    kappa: float,
    lam3: float,
    r31: float,
    r32: float,
    # adaptive controls
    v_cap: int = 500,
    tail_rel_tol: float = 1e-12,
    tail_patience: int = 6,
    v_min: int = 10,
    # quadrature
    n_grid: int = 200,
    tol: float = 1e-300,
) -> float:
    """
    Observed log-likelihood:
      ℓ(θ) = sum_n log fY(y^{(n)}),
    where fY(y) is computed by summing v=1,2,... adaptively until tail negligible.

    NOTE: In run_ecm, we avoid calling this for the main per-iteration convergence tracking,
    by using E_step_adaptive_with_ll (faster). We still keep this for the R-update line-search.
    """
    if lam <= 0 or mu <= 0 or kappa <= 0 or lam3 <= 0 or rho < 0:
        return -np.inf
    if r31 <= 0 or r32 <= 0:
        return -np.inf

    q2 = rho + mu + kappa
    if q2 <= 0.0:
        return -np.inf

    p21 = rho / q2
    p2a = mu / q2
    p23 = kappa / q2

    log_p21 = -np.inf if p21 <= 0 else math.log(p21)
    log_p2a = -np.inf if p2a <= 0 else math.log(p2a)
    log_p23 = -np.inf if p23 <= 0 else math.log(p23)

    total = 0.0

    for n in range(Y.shape[0]):
        y1, y2 = float(Y[n, 0]), float(Y[n, 1])
        if y1 <= 0.0 or y2 <= 0.0:
            return -np.inf

        M = M_of_y(y1, y2, r31, r32)

        log_mass = -np.inf
        small_count = 0

        for v in range(1, v_cap + 1):
            if v == 1:
                lp_v0 = log_p2a
                lp_v1 = log_p23
            else:
                if not np.isfinite(log_p21):
                    lp_v0 = -np.inf
                    lp_v1 = -np.inf
                else:
                    lp_v0 = (v - 1) * log_p21 + log_p2a
                    lp_v1 = (v - 1) * log_p21 + log_p23

            lw0 = lp_v0 + log_fY_given_v_h0(y1, y2, v, lam, q2)

            if M > 0.0 and np.isfinite(lp_v1):
                lw1 = lp_v1 + log_fY_given_v_h1(
                    y1, y2, v, lam, q2, lam3, r31, r32, n_grid=n_grid
                )
            else:
                lw1 = -np.inf

            log_incr = logaddexp(lw0, lw1)
            new_log_mass = logaddexp(log_mass, log_incr)

            if np.isfinite(log_mass):
                rel = safe_exp(log_incr - new_log_mass)
            else:
                rel = 1.0

            log_mass = new_log_mass

            if v >= v_min and rel < tail_rel_tol:
                small_count += 1
            else:
                small_count = 0

            if v >= v_min and small_count >= tail_patience:
                break

        if not np.isfinite(log_mass) or safe_exp(log_mass) <= tol:
            return -np.inf

        total += log_mass

    return float(total)


# =========================================================
# 6) ECM update for (r31,r32) by monotone ascent on observed log-likelihood
# =========================================================

def r_from_v(v: np.ndarray) -> Tuple[float, float]:
    """Enforce positivity: r31=exp(v0), r32=exp(v1)."""
    return float(np.exp(v[0])), float(np.exp(v[1]))


def fd_grad_v(
    Y: np.ndarray,
    lam: float,
    rho: float,
    mu: float,
    kappa: float,
    lam3: float,
    v: np.ndarray,
    # adaptive settings for likelihood
    v_cap: int,
    tail_rel_tol: float,
    tail_patience: int,
    v_min: int,
    n_grid: int,
    step: float = 1e-4,
) -> Tuple[float, np.ndarray]:
    """Finite-difference gradient of loglik_adaptive w.r.t v=(log r31, log r32)."""
    r31, r32 = r_from_v(v)
    base = loglik_adaptive(
        Y, lam, rho, mu, kappa, lam3, r31, r32,
        v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
        n_grid=n_grid
    )
    g = np.zeros_like(v, dtype=float)

    if not np.isfinite(base):
        return base, g

    for j in range(len(v)):
        vp = v.copy()
        vm = v.copy()
        vp[j] += step
        vm[j] -= step

        r31p, r32p = r_from_v(vp)
        r31m, r32m = r_from_v(vm)

        llp = loglik_adaptive(
            Y, lam, rho, mu, kappa, lam3, r31p, r32p,
            v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
            n_grid=n_grid
        )
        llm = loglik_adaptive(
            Y, lam, rho, mu, kappa, lam3, r31m, r32m,
            v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
            n_grid=n_grid
        )

        if np.isfinite(llp) and np.isfinite(llm):
            g[j] = (llp - llm) / (2.0 * step)
        elif np.isfinite(llp):
            g[j] = (llp - base) / step
        elif np.isfinite(llm):
            g[j] = (base - llm) / step
        else:
            g[j] = 0.0

    return float(base), g


def update_v_linesearch(
    Y: np.ndarray,
    lam: float,
    rho: float,
    mu: float,
    kappa: float,
    lam3: float,
    v0: np.ndarray,
    # adaptive likelihood settings
    v_cap: int,
    tail_rel_tol: float,
    tail_patience: int,
    v_min: int,
    n_grid: int,
    lr0: float = 1e-2,
    max_backtracks: int = 15,
    improve_tol: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Monotone gradient-ascent step with backtracking for v=(log r31,log r32)."""
    base, g = fd_grad_v(
        Y, lam, rho, mu, kappa, lam3, v0,
        v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
        n_grid=n_grid
    )
    if not np.isfinite(base):
        return v0, -np.inf

    lr = lr0
    for _ in range(max_backtracks):
        v_try = v0 + lr * g
        r31_try, r32_try = r_from_v(v_try)
        ll_try = loglik_adaptive(
            Y, lam, rho, mu, kappa, lam3, r31_try, r32_try,
            v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
            n_grid=n_grid
        )
        if np.isfinite(ll_try) and ll_try > base + improve_tol:
            return v_try, float(ll_try)
        lr *= 0.5

    return v0, float(base)


# =========================================================
# 7) Full ECM loop (E-step + CM rates + ECM for R)  [FASTER LL TRACKING]
# =========================================================

def run_ecm(
    Y: np.ndarray,
    lam0: float,
    rho0: float,
    mu0: float,
    kappa0: float,
    lam3_0: float,
    r31_0: float,
    r32_0: float,
    max_iter: int = 200,
    # --- Kulkarni-style stopping (likelihood per obs) ---
    eps_ll: float = 1e-8,
    min_iters_ll: int = 2,
    # (optional) keep parameter criterion as secondary safeguard
    rel_tol_param: Optional[float] = None,
    # adaptive controls for v-sum
    v_cap: int = 500,
    tail_rel_tol: float = 1e-12,
    tail_patience: int = 6,
    v_min: int = 10,
    # quadrature
    n_grid: int = 200,
    # R-update
    max_R_inner_steps: int = 2,
    lr0: float = 1e-2,
    # misc
    verbose: bool = True,
    track_ll: bool = True,
) -> Tuple[float, float, float, float, float, float, float, Optional[np.ndarray]]:
    """
    Same ECM as before, but faster due to:
      - cached integrals inside E-step
      - log-likelihood tracking obtained from E-step weights (no extra v-sums)
    """
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("Y must be an array of shape (N,2).")
    if np.any(Y <= 0.0):
        raise ValueError("This implementation assumes y1>0 and y2>0 for all observations.")

    N = int(Y.shape[0])

    lam = float(lam0)
    rho = float(rho0)
    mu = float(mu0)
    kappa = float(kappa0)
    lam3 = float(lam3_0)

    v = np.array([math.log(r31_0), math.log(r32_0)], dtype=float)
    r31, r32 = r_from_v(v)

    prev_params = np.array([lam, rho, mu, kappa, lam3, r31, r32], dtype=float)

    ll_hist: List[float] = []
    ll_prev: Optional[float] = None

    # initial ll via loglik_adaptive (kept as-is; one-time cost)
    ll0 = loglik_adaptive(
        Y, lam, rho, mu, kappa, lam3, r31, r32,
        v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
        n_grid=n_grid
    )
    ll_prev = float(ll0)
    if track_ll:
        ll_hist.append(float(ll0))

    for it in range(1, max_iter + 1):
        # ---------- E-step (FAST) + get ll under CURRENT params ----------
        S1_hat, S2_hat, V_hat, H_hat, HZ3_hat, v_used_mean, ll_cur = E_step_adaptive_with_ll(
            Y, lam, rho, mu, kappa, lam3, r31, r32,
            v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
            n_grid=n_grid
        )

        if verbose:
            print(f"\n=== ECM Iteration {it} ===")
            print(f"  S1_hat      = {S1_hat:.6f}")
            print(f"  S2_hat      = {S2_hat:.6f}")
            print(f"  V_hat       = {V_hat:.6f}")
            print(f"  H_hat       = {H_hat:.6f}")
            print(f"  HZ3_hat     = {HZ3_hat:.6f}")
            print(f"  mean v_used = {v_used_mean:.2f}")
            print(f"  ll (from E-step weights) = {ll_cur:.6f}")

        # ---------- CM-step (rates) ----------
        lam, rho, mu, kappa, lam3 = M_step_rates(S1_hat, S2_hat, V_hat, H_hat, HZ3_hat, N)

        if verbose:
            print("  CM-step (rates):")
            print(f"    lam   = {lam:.6f}")
            print(f"    rho   = {rho:.6f}")
            print(f"    mu    = {mu:.6f}")
            print(f"    kappa = {kappa:.6f}")
            print(f"    lam3  = {lam3:.6f}")

        # ---------- ECM-step (R) ----------
        # Need true observed ll for updated rates and current R:
        base_ll = loglik_adaptive(
            Y, lam, rho, mu, kappa, lam3, r31, r32,
            v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
            n_grid=n_grid
        )
        if verbose:
            print(f"  log-likelihood before R-update = {base_ll:.6f}")

        v_cand = v.copy()
        ll_cand = float(base_ll)

        for _ in range(max_R_inner_steps):
            v_try, ll_try = update_v_linesearch(
                Y, lam, rho, mu, kappa, lam3, v_cand,
                v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience, v_min=v_min,
                n_grid=n_grid, lr0=lr0
            )
            if (not np.isfinite(ll_try)) or (ll_try <= ll_cand + 1e-8):
                break
            v_cand, ll_cand = v_try, float(ll_try)

        v = v_cand
        r31, r32 = r_from_v(v)

        if verbose:
            print("  ECM-step (R):")
            print(f"    r31 = {r31:.6f}")
            print(f"    r32 = {r32:.6f}")
            print(f"    log-likelihood after R-update = {ll_cand:.6f}")

        # ---------- Convergence (Kulkarni-style) ----------
        if track_ll:
            ll_hist.append(float(ll_cand))

        dll_per_obs = abs(float(ll_cand) - float(ll_prev)) / max(N, 1)

        if verbose:
            print(f"  |Δll|/N = {dll_per_obs:.3e} (eps_ll={eps_ll:.3e})")

        if it >= min_iters_ll and np.isfinite(dll_per_obs) and dll_per_obs < eps_ll:
            if verbose:
                print("Converged (likelihood per-observation criterion).")
            break

        if rel_tol_param is not None:
            cur_params = np.array([lam, rho, mu, kappa, lam3, r31, r32], dtype=float)
            rel_change = float(np.linalg.norm(cur_params - prev_params) / (1e-8 + np.linalg.norm(prev_params)))
            if verbose:
                print(f"  relative parameter change = {rel_change:.3e} (rel_tol_param={rel_tol_param:.3e})")
            if rel_change < rel_tol_param:
                if verbose:
                    print("Converged (parameter change secondary criterion).")
                break
            prev_params = cur_params

        ll_prev = float(ll_cand)

    return lam, rho, mu, kappa, lam3, r31, r32, (np.asarray(ll_hist, float) if track_ll else None)
