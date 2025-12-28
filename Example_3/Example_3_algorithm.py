

import numpy as np
import math

# ------------------------------------------------------------
# 0) Simulation: extended Kulkarni with back-jumps
# ------------------------------------------------------------

def simulate_kulkarni_path_extended(lam, rho, mu, rng):
    """
    Simulate ONE absorption path for the extended Kulkarni CTMC:

        state 1 --(λ)--> state 2
        state 2 --(ρ)--> state 1
        state 2 --(μ)--> absorbing

    with initial distribution α = (1, 0).
    """
    B = np.array([1.0, 0.0])

    Z1 = 0.0
    Z2 = 0.0
    N_transition = np.zeros((2, 2))
    N_abs = np.zeros(2)

    state = 0  # 0 = state 1, 1 = state 2

    while True:
        if state == 0:
            rate = lam
            holding = rng.exponential(scale=1.0 / rate)
            Z1 += holding
            N_transition[0, 1] += 1.0
            state = 1

        elif state == 1:
            rate = rho + mu
            holding = rng.exponential(scale=1.0 / rate)
            Z2 += holding

            if rng.uniform() < rho / (rho + mu):
                N_transition[1, 0] += 1.0
                state = 0
            else:
                N_abs[1] += 1.0
                break
        else:
            raise RuntimeError("Invalid state index encountered in simulation.")

    Z = np.array([Z1, Z2])
    t_total = Z1 + Z2
    return t_total, B, Z, N_transition, N_abs


def simulate_kulkarni_dataset_extended(lam, rho, mu, R, N, rng=None):
    """
    Simulate N i.i.d. bivariate MPH* observations from the extended Kulkarni example.
    Reward relationship: Y = Z R, where Z = (Z1, Z2) are total sojourn times.
    """
    if rng is None:
        rng = np.random.default_rng()

    R = np.asarray(R, float)

    absorp_times = np.zeros(N)
    Z_all        = np.zeros((N, 2))
    Y_all        = np.zeros((N, 2))
    B_sum        = np.zeros(2)
    Z_sum        = np.zeros(2)
    N_sum        = np.zeros((2, 2))
    N_abs_sum    = np.zeros(2)

    for n in range(N):
        Z_total, B, Z, N_mat, N_abs = simulate_kulkarni_path_extended(lam, rho, mu, rng)
        Y = Z @ R

        absorp_times[n] = Z_total
        Z_all[n, :]     = Z
        Y_all[n, :]     = Y

        B_sum     += B
        Z_sum     += Z
        N_sum     += N_mat
        N_abs_sum += N_abs

    return absorp_times, Z_all, Y_all, B_sum, Z_sum, N_sum, N_abs_sum


# ------------------------------------------------------------
# 1) R-parameterization (same as before)
# ------------------------------------------------------------

EPS_R = 0.05

def sigmoid(x, eps=EPS_R):
    x = np.clip(x, -50.0, 50.0)
    s = 1.0 / (1.0 + np.exp(-x))
    return eps + (1.0 - 2.0 * eps) * s


def u_from_R(R, eps=EPS_R):
    R = np.asarray(R, float)
    r1 = R[0, 0]
    r2 = R[1, 0]

    denom = 1.0 - 2.0 * eps
    r1_bar = (r1 - eps) / denom
    r2_bar = (r2 - eps) / denom

    tiny = 1e-8
    r1_bar = np.clip(r1_bar, tiny, 1.0 - tiny)
    r2_bar = np.clip(r2_bar, tiny, 1.0 - tiny)

    u1 = np.log(r1_bar / (1.0 - r1_bar))
    u2 = np.log(r2_bar / (1.0 - r2_bar))
    return np.array([u1, u2], dtype=float)


def R_from_u(u):
    r1 = sigmoid(u[0])
    r2 = sigmoid(u[1])
    return np.array([[r1, 1.0 - r1],
                     [r2, 1.0 - r2]], dtype=float)


# ------------------------------------------------------------
# 2) Feasibility utilities
# ------------------------------------------------------------

def feasibility_info(Y, R, tol=1e-10, cond_max=1e12):
    R = np.asarray(R, float)

    c = np.linalg.cond(R)
    if (not np.isfinite(c)) or (c > cond_max):
        N = Y.shape[0]
        Z = np.full((N, 2), np.nan)
        mask = np.zeros(N, dtype=bool)
        return Z, mask, 0

    Z = np.linalg.solve(R.T, Y.T).T
    mask = (np.all(Z >= -tol, axis=1) & np.all(np.isfinite(Z), axis=1))
    return Z, mask, int(mask.sum())


# ------------------------------------------------------------
# 3) Adaptive truncation for m-sum (Kulkarni loop count)
# ------------------------------------------------------------

def _logaddexp(a: float, b: float) -> float:
    if not np.isfinite(a):
        return b
    if not np.isfinite(b):
        return a
    m = a if a >= b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _logsumexp(logw: np.ndarray) -> float:
    m = float(np.max(logw))
    if not np.isfinite(m):
        return -np.inf
    s = float(np.sum(np.exp(logw - m)))
    if s <= 0.0 or (not np.isfinite(s)):
        return -np.inf
    return m + math.log(s)


def _adaptive_m_posterior_stats(
    z1: float,
    z2: float,
    lam: float,
    rho: float,
    mu: float,
    # adaptive controls
    m_cap: int = 500,
    tail_rel_tol: float = 1e-12,
    tail_patience: int = 6,
    m_min: int = 10,
):
    """
    For a single feasible (z1,z2), compute:
      log_sum_w = log( sum_{m>=1} w_m(z1,z2) )   approximated adaptively
      N_hat     = E[m | z1,z2]                   approximated adaptively
      m_used    = last m included (diagnostic)

    Here w_m corresponds to your existing term:
      w_m ∝ p (1-p)^{m-1} * (lam*(rho+mu))^m / (Gamma(m)^2) * (z1 z2)^{m-1}
    (in log-form exactly as in your code).
    """
    if z1 <= 0.0 or z2 <= 0.0:
        return -np.inf, 1.0, 1

    p = mu / (rho + mu)
    if p <= 0.0 or p >= 1.0:
        return -np.inf, 1.0, 1

    log_p    = math.log(p)
    log_1mp  = math.log(1.0 - p)
    log_lam  = math.log(lam)
    log_rp   = math.log(rho + mu)

    log_zsum = math.log(z1) + math.log(z2)

    log_mass = -np.inf  # log(sum w_m)
    log_mmass = -np.inf # log(sum m*w_m)
    small_count = 0
    m_used = 0

    for m in range(1, m_cap + 1):
        m_used = m
        lgamma_m = math.lgamma(m)

        lw = (
            log_p
            + (m - 1) * log_1mp
            + m * (log_lam + log_rp)
            - 2.0 * lgamma_m
            + (m - 1) * log_zsum
        )

        # update mass and m-mass
        new_log_mass  = _logaddexp(log_mass, lw)
        new_log_mmass = _logaddexp(log_mmass, lw + math.log(m))

        # relative contribution of this term (approx)
        if np.isfinite(log_mass):
            rel = math.exp(lw - new_log_mass) if (lw - new_log_mass) > -745 else 0.0
        else:
            rel = 1.0

        log_mass, log_mmass = new_log_mass, new_log_mmass

        if m >= m_min and rel < tail_rel_tol:
            small_count += 1
        else:
            small_count = 0

        if m >= m_min and small_count >= tail_patience:
            break

    if not np.isfinite(log_mass) or not np.isfinite(log_mmass):
        return -np.inf, 1.0, m_used

    N_hat = math.exp(log_mmass - log_mass)
    return float(log_mass), float(N_hat), int(m_used)


# ------------------------------------------------------------
# 4) E-step (adaptive m truncation, feasible subset only)
# ------------------------------------------------------------

def E_step_extended_adaptive(
    Y, R, lam, rho, mu,
    tol=1e-12,
    cond_max=1e12,
    # adaptive controls
    m_cap=500,
    tail_rel_tol=1e-12,
    tail_patience=6,
    m_min=10,
):
    Z, mask, n_feas = feasibility_info(Y, R, tol=tol, cond_max=cond_max)
    if n_feas == 0:
        return {
            "mask": mask,
            "Zf": np.empty((0, 2)),
            "S1": 0.0,
            "S2": 0.0,
            "N12_hat": 0.0,
            "N21_hat": 0.0,
            "N2a_hat": 0.0,
            "N_hat_list": np.array([]),
            "m_used_mean": np.nan,
        }

    Zf = Z[mask]
    S1 = float(np.sum(Zf[:, 0]))
    S2 = float(np.sum(Zf[:, 1]))

    N_hat_list = []
    N12_hat = 0.0
    N21_hat = 0.0
    N2a_hat = float(n_feas)

    m_used_acc = 0.0

    for z1, z2 in Zf:
        # numeric guard
        if (not np.isfinite(z1)) or (not np.isfinite(z2)) or (z1 <= tol) or (z2 <= tol):
            N_hat = 1.0
            m_used = 1
        else:
            _, N_hat, m_used = _adaptive_m_posterior_stats(
                float(z1), float(z2), float(lam), float(rho), float(mu),
                m_cap=int(m_cap),
                tail_rel_tol=float(tail_rel_tol),
                tail_patience=int(tail_patience),
                m_min=int(m_min),
            )

        N_hat_list.append(N_hat)
        N12_hat += N_hat
        N21_hat += (N_hat - 1.0)
        m_used_acc += float(m_used)

    m_used_mean = m_used_acc / max(float(n_feas), 1.0)

    return {
        "mask": mask,
        "Zf": Zf,
        "S1": S1,
        "S2": S2,
        "N12_hat": float(N12_hat),
        "N21_hat": float(N21_hat),
        "N2a_hat": float(N2a_hat),
        "N_hat_list": np.asarray(N_hat_list, float),
        "m_used_mean": float(m_used_mean),
    }


# ------------------------------------------------------------
# 5) M-step for rates (unchanged)
# ------------------------------------------------------------

def M_step_rates_extended(Estats, eps=1e-15):
    S1      = max(Estats["S1"], eps)
    S2      = max(Estats["S2"], eps)
    N12_hat = Estats["N12_hat"]
    N21_hat = Estats["N21_hat"]
    N2a_hat = Estats["N2a_hat"]

    lam_new = N12_hat / S1
    rho_new = N21_hat / S2
    mu_new  = N2a_hat / S2

    return float(lam_new), float(rho_new), float(mu_new)


# ------------------------------------------------------------
# 6) Penalized observed log-likelihood (adaptive m truncation)
# ------------------------------------------------------------

def loglik_Y_extended_penalized_adaptive(
    Y, lam, rho, mu, R,
    tol=1e-10,
    cond_max=1e12,
    infeas_penalty=1e6,
    negz_penalty=1e6,
    # adaptive controls
    m_cap=500,
    tail_rel_tol=1e-12,
    tail_patience=6,
    m_min=10,
):
    R = np.asarray(R, float)

    c = np.linalg.cond(R)
    if (not np.isfinite(c)) or (c > cond_max):
        return -np.inf

    detR = np.linalg.det(R)
    if (not np.isfinite(detR)) or (detR == 0.0):
        return -np.inf

    if lam <= 0.0 or rho < 0.0 or mu <= 0.0 or (rho + mu) <= 0.0:
        return -np.inf

    p = mu / (rho + mu)
    if p <= 0.0 or p >= 1.0:
        return -np.inf

    Z, mask, n_feas = feasibility_info(Y, R, tol=tol, cond_max=cond_max)
    N = Y.shape[0]
    n_infeas = N - n_feas

    log_abs_detR = math.log(abs(detR))

    total_ll = 0.0
    if n_infeas > 0:
        total_ll -= infeas_penalty * float(n_infeas)

    Zf = Z[mask]
    for z1, z2 in Zf:
        if (not np.isfinite(z1)) or (not np.isfinite(z2)) or (z1 <= tol) or (z2 <= tol):
            total_ll -= negz_penalty
            continue

        log_sum_w, _, _ = _adaptive_m_posterior_stats(
            float(z1), float(z2), float(lam), float(rho), float(mu),
            m_cap=int(m_cap),
            tail_rel_tol=float(tail_rel_tol),
            tail_patience=int(tail_patience),
            m_min=int(m_min),
        )
        if not np.isfinite(log_sum_w):
            total_ll -= negz_penalty
            continue

        log_fZ = -lam * float(z1) - (rho + mu) * float(z2) + log_sum_w
        log_fY = log_fZ - log_abs_detR
        total_ll += log_fY

    return float(total_ll)


# ------------------------------------------------------------
# 7) Numerical gradient w.r.t u + line search (use penalized ll)
# ------------------------------------------------------------

def fd_grad_u_extended_penalized_adaptive(
    Y, lam, rho, mu, u,
    step=1e-4,
    tol=1e-12,
    infeas_penalty=1e6,
    negz_penalty=1e6,
    # adaptive controls
    m_cap=500,
    tail_rel_tol=1e-12,
    tail_patience=6,
    m_min=10,
):
    base_ll = loglik_Y_extended_penalized_adaptive(
        Y, lam, rho, mu, R_from_u(u),
        tol=tol,
        infeas_penalty=infeas_penalty,
        negz_penalty=negz_penalty,
        m_cap=m_cap,
        tail_rel_tol=tail_rel_tol,
        tail_patience=tail_patience,
        m_min=m_min,
    )
    g = np.zeros_like(u)

    if not np.isfinite(base_ll):
        return base_ll, g

    for j in range(len(u)):
        u_p = u.copy(); u_p[j] += step
        u_m = u.copy(); u_m[j] -= step

        ll_p = loglik_Y_extended_penalized_adaptive(
            Y, lam, rho, mu, R_from_u(u_p),
            tol=tol,
            infeas_penalty=infeas_penalty,
            negz_penalty=negz_penalty,
            m_cap=m_cap,
            tail_rel_tol=tail_rel_tol,
            tail_patience=tail_patience,
            m_min=m_min,
        )
        ll_m = loglik_Y_extended_penalized_adaptive(
            Y, lam, rho, mu, R_from_u(u_m),
            tol=tol,
            infeas_penalty=infeas_penalty,
            negz_penalty=negz_penalty,
            m_cap=m_cap,
            tail_rel_tol=tail_rel_tol,
            tail_patience=tail_patience,
            m_min=m_min,
        )

        if np.isfinite(ll_p) and np.isfinite(ll_m):
            g[j] = (ll_p - ll_m) / (2.0 * step)
        elif np.isfinite(ll_p):
            g[j] = (ll_p - base_ll) / step
        elif np.isfinite(ll_m):
            g[j] = (base_ll - ll_m) / step
        else:
            g[j] = 0.0

    return float(base_ll), g


def canonicalize_u_by_ll_penalized_adaptive(
    Y, lam, rho, mu, u,
    tol=1e-12,
    infeas_penalty=1e6,
    negz_penalty=1e6,
    # adaptive controls
    m_cap=500,
    tail_rel_tol=1e-12,
    tail_patience=6,
    m_min=10,
):
    ll_u = loglik_Y_extended_penalized_adaptive(
        Y, lam, rho, mu, R_from_u(u),
        tol=tol,
        infeas_penalty=infeas_penalty,
        negz_penalty=negz_penalty,
        m_cap=m_cap,
        tail_rel_tol=tail_rel_tol,
        tail_patience=tail_patience,
        m_min=m_min,
    )
    ll_mu = loglik_Y_extended_penalized_adaptive(
        Y, lam, rho, mu, R_from_u(-u),
        tol=tol,
        infeas_penalty=infeas_penalty,
        negz_penalty=negz_penalty,
        m_cap=m_cap,
        tail_rel_tol=tail_rel_tol,
        tail_patience=tail_patience,
        m_min=m_min,
    )
    if np.isfinite(ll_mu) and (not np.isfinite(ll_u) or ll_mu > ll_u):
        return -u, ll_mu
    return u, ll_u


def update_u_linesearch_monotone_penalized_adaptive(
    Y, lam, rho, mu, u0,
    tol=1e-12,
    step_fd=1e-4,
    lr0=1e-2,
    max_backtracks=15,
    det_tol=1e-10,
    infeas_penalty=1e6,
    negz_penalty=1e6,
    # adaptive controls
    m_cap=500,
    tail_rel_tol=1e-12,
    tail_patience=6,
    m_min=10,
):
    ll0 = loglik_Y_extended_penalized_adaptive(
        Y, lam, rho, mu, R_from_u(u0),
        tol=tol,
        infeas_penalty=infeas_penalty,
        negz_penalty=negz_penalty,
        m_cap=m_cap,
        tail_rel_tol=tail_rel_tol,
        tail_patience=tail_patience,
        m_min=m_min,
    )
    if not np.isfinite(ll0):
        return u0, ll0

    base_ll, g = fd_grad_u_extended_penalized_adaptive(
        Y, lam, rho, mu, u0,
        step=step_fd,
        tol=tol,
        infeas_penalty=infeas_penalty,
        negz_penalty=negz_penalty,
        m_cap=m_cap,
        tail_rel_tol=tail_rel_tol,
        tail_patience=tail_patience,
        m_min=m_min,
    )
    if not np.isfinite(base_ll):
        return u0, ll0

    lr = float(lr0)
    for _ in range(max_backtracks):
        u_try = u0 + lr * g
        R_try = R_from_u(u_try)
        if abs(np.linalg.det(R_try)) < det_tol:
            lr *= 0.5
            continue

        ll_try = loglik_Y_extended_penalized_adaptive(
            Y, lam, rho, mu, R_try,
            tol=tol,
            infeas_penalty=infeas_penalty,
            negz_penalty=negz_penalty,
            m_cap=m_cap,
            tail_rel_tol=tail_rel_tol,
            tail_patience=tail_patience,
            m_min=m_min,
        )
        if np.isfinite(ll_try) and ll_try > ll0:
            return u_try, float(ll_try)

        lr *= 0.5

    return u0, float(ll0)


# ------------------------------------------------------------
# 8) ECM loop (adaptive truncation + soft support)
# ------------------------------------------------------------

def run_ecm_extended_monotone_penalized_adaptive(
    Y,
    lam0=1.0, rho0=0.5, mu0=1.0, u0=np.array([0.0, 0.0]),
    max_iter=200,
    tol=1e-12,
    eps_ll=1e-8,
    max_R_steps=10,
    lr0=1e-2,
    verbose=False,
    track_ll=True,
    infeas_penalty=1e6,
    negz_penalty=1e6,
    # adaptive controls
    m_cap=500,
    tail_rel_tol=1e-12,
    tail_patience=6,
    m_min=10,
):
    lam = float(lam0)
    rho = float(rho0)
    mu  = float(mu0)
    u   = np.array(u0, dtype=float)

    u, ll_cur = canonicalize_u_by_ll_penalized_adaptive(
        Y, lam, rho, mu, u,
        tol=tol,
        infeas_penalty=infeas_penalty,
        negz_penalty=negz_penalty,
        m_cap=m_cap,
        tail_rel_tol=tail_rel_tol,
        tail_patience=tail_patience,
        m_min=m_min,
    )

    ll_hist = [ll_cur] if track_ll else None
    N = Y.shape[0]

    for k in range(max_iter):
        R = R_from_u(u)

        E = E_step_extended_adaptive(
            Y, R, lam, rho, mu,
            tol=tol,
            m_cap=m_cap,
            tail_rel_tol=tail_rel_tol,
            tail_patience=tail_patience,
            m_min=m_min,
        )

        if E["Zf"].shape[0] == 0:
            if verbose:
                print(f"[iter {k}] no feasible observations; stopping.")
            break

        lam_c, rho_c, mu_c = M_step_rates_extended(E)

        u_c = u.copy()
        ll_c = loglik_Y_extended_penalized_adaptive(
            Y, lam_c, rho_c, mu_c, R_from_u(u_c),
            tol=tol,
            infeas_penalty=infeas_penalty,
            negz_penalty=negz_penalty,
            m_cap=m_cap,
            tail_rel_tol=tail_rel_tol,
            tail_patience=tail_patience,
            m_min=m_min,
        )

        for _ in range(max_R_steps):
            u_try, ll_try = update_u_linesearch_monotone_penalized_adaptive(
                Y, lam_c, rho_c, mu_c, u_c,
                tol=tol,
                lr0=lr0,
                infeas_penalty=infeas_penalty,
                negz_penalty=negz_penalty,
                m_cap=m_cap,
                tail_rel_tol=tail_rel_tol,
                tail_patience=tail_patience,
                m_min=m_min,
            )
            if (not np.isfinite(ll_try)) or (ll_try <= ll_c):
                break
            u_c, ll_c = u_try, ll_try

        u_c, ll_c = canonicalize_u_by_ll_penalized_adaptive(
            Y, lam_c, rho_c, mu_c, u_c,
            tol=tol,
            infeas_penalty=infeas_penalty,
            negz_penalty=negz_penalty,
            m_cap=m_cap,
            tail_rel_tol=tail_rel_tol,
            tail_patience=tail_patience,
            m_min=m_min,
        )

        if np.isfinite(ll_c) and (not np.isfinite(ll_cur) or ll_c >= ll_cur - 1e-12):
            lam, rho, mu, u = lam_c, rho_c, mu_c, u_c
            ll_next = ll_c
        else:
            lr0 *= 0.5
            ll_next = ll_cur
            if verbose:
                print(f"[iter {k}] reject update; lr0 -> {lr0:g}")

        if track_ll:
            ll_hist.append(ll_next)

        if verbose:
            _, _, feas = feasibility_info(Y, R_from_u(u), tol=tol)
            print(f"[iter {k}] ll={ll_next:.6f}, feas={feas}/{N}, lam={lam:.4f}, rho={rho:.4f}, mu={mu:.4f}, m_used_mean={E.get('m_used_mean', np.nan):.2f}")

        if k >= 1 and np.isfinite(ll_next) and np.isfinite(ll_cur):
            if abs(ll_next - ll_cur) / max(N, 1) < eps_ll:
                break

        ll_cur = ll_next

    R_hat = R_from_u(u)
    if track_ll:
        return lam, rho, mu, R_hat, np.asarray(ll_hist, float)
    return lam, rho, mu, R_hat
