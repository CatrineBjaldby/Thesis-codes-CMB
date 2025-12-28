# ============================================================
# Kulkarni Example Known CTMC and Irriversible R matrix 
#  ============================================================

# Importing packages 
import numpy as np
import math

# ------------------------------------------------------
# Numerical default values
# ------------------------------------------------------
# Keeping R away from being singular
EPS_R    = 0.05
# Setting the tolerence for fesible observations of Z (cumputational needed -> in theory this is 0)
TOL_Z    = 1e-8
# Maximum condition number for R to be accepted
COND_MAX = 1e12 

# Soft-feasibility penalty parameters: makes finite-difference gradients meaningful and stabilizes R-updates.
PENALTY_Z   = 1e6     
PENALTY_POW = 2.0    


# ============================================================
# 1) Simulation: Coxian-2 path and dataset
# ============================================================

def simulate_kulkarni_path(lam, mu, rng):
    """
    Simulate 1 absorption path: 1 --lam--> 2 --mu--> absorbing, alpha=(1,0).
    Returns: t_total, B, Z=(Z1,Z2), N_transition, N_abs
    """
    if lam <= 0 or mu <= 0:
        raise ValueError("lam and mu must be positive.")

    B = np.array([1.0, 0.0])

    Z1 = rng.exponential(scale=1.0 / lam)
    Z2 = rng.exponential(scale=1.0 / mu)
    Z_total = Z1 + Z2
    Z = np.array([Z1, Z2])

    N_transition = np.zeros((2, 2))
    N_transition[0, 1] = 1.0

    N_abs = np.array([0.0, 1.0])

    return Z_total, B, Z, N_transition, N_abs


def simulate_kulkarni_dataset(lam, mu, R, N, rng=None):
    """
    Simulate N i.i.d. observations: latent Z=(Z1,Z2), observed Y=Z R.
    Returns: absorp_times, Z_all, Y_all, B_sum, Z_sum, N_sum, N_abs_sum
    """
    if rng is None:
        rng = np.random.default_rng()

    R = np.asarray(R, float)
    if R.shape != (2, 2):
        raise ValueError("R must be 2x2.")

    absorp_times = np.zeros(N)
    Z_all = np.zeros((N, 2))
    Y_all = np.zeros((N, 2))
    B_sum = np.zeros(2)
    Z_sum = np.zeros(2)
    N_sum = np.zeros((2, 2))
    N_abs_sum = np.zeros(2)

    for n in range(N):
        Z_total, B, Z, N_mat, N_abs = simulate_kulkarni_path(lam, mu, rng)
        Y = Z @ R

        absorp_times[n] = Z_total
        Z_all[n, :] = Z
        Y_all[n, :] = Y

        B_sum += B
        Z_sum += Z
        N_sum += N_mat
        N_abs_sum += N_abs

    return absorp_times, Z_all, Y_all, B_sum, Z_sum, N_sum, N_abs_sum


# ============================================================
# 2) Robust numerical parameterization of R (EPS-bounded sigmoid)
# ============================================================

def sigmoid_eps(x, eps=EPS_R):
    #Preventing creating 0 an infinity 
    x = np.clip(x, -50.0, 50.0)
    s = 1.0 / (1.0 + np.exp(-x))
    return eps + (1.0 - 2.0 * eps) * s


def R_from_u(u, eps=EPS_R):
    u = np.asarray(u, float).reshape(2,)
    r1 = sigmoid_eps(u[0], eps=eps)
    r2 = sigmoid_eps(u[1], eps=eps)
    return np.array([[r1, 1.0 - r1],
                     [r2, 1.0 - r2]], dtype=float)


def u_from_R(R, eps=EPS_R):
    """
    Mapping R values back from u values
    Assumes R ≈ [[r1, 1-r1],[r2, 1-r2]]
    """
    R = np.asarray(R, float)
    r1 = float(R[0, 0])
    r2 = float(R[1, 0])

    denom = 1.0 - 2.0 * eps
    r1_bar = (r1 - eps) / denom
    r2_bar = (r2 - eps) / denom

    tiny = 1e-10
    r1_bar = np.clip(r1_bar, tiny, 1.0 - tiny)
    r2_bar = np.clip(r2_bar, tiny, 1.0 - tiny)

    u1 = np.log(r1_bar / (1.0 - r1_bar))
    u2 = np.log(r2_bar / (1.0 - r2_bar))
    return np.array([u1, u2], dtype=float)


# ============================================================
# 3) Linear algebra: solve for Z (avoid inv)
# ============================================================

def Z_from_Y(Y, R):
    """
    Solve Z R = Y stably:
    R^T Z^T = Y^T  =>  Z^T = solve(R^T, Y^T)
    """
    Y = np.asarray(Y, float)
    R = np.asarray(R, float)
    return np.linalg.solve(R.T, Y.T).T


def feasibility_info(Y, R, tol=TOL_Z, cond_max=COND_MAX):
    """
    Which observations produce a physical meaning of Z (Positive values)
    Returns (Z, mask, n_feas). Reject bad R using cond(R): preventing from using when Z dominated by noice
    mask True iff Z[i] >= -tol componentwise.
    """
    R = np.asarray(R, float)

    c = np.linalg.cond(R)
    if (not np.isfinite(c)) or (c > cond_max):
        n = Y.shape[0]
        return np.full((n, 2), np.nan), np.zeros(n, dtype=bool), 0

    try:
        Z = Z_from_Y(Y, R)
    except np.linalg.LinAlgError:
        n = Y.shape[0]
        return np.full((n, 2), np.nan), np.zeros(n, dtype=bool), 0

    mask = (np.all(Z >= -tol, axis=1) & np.all(np.isfinite(Z), axis=1))
    return Z, mask, int(mask.sum())


# ============================================================
# 4) Symmetry handling
# ============================================================

def symmetries_of(u, lam, mu):
    """
    For stability, consider all 4 symmetric parameter configurations, so it doesnt jump back and fourth do to the equivalent likelihood solutions
    """
    u = np.asarray(u, float).reshape(2,)
    return [
        (u.copy(), lam, mu),
        (-u.copy(), lam, mu),
        (u[::-1].copy(), mu, lam),
        (((-u)[::-1]).copy(), mu, lam),
    ]


# ============================================================
# 4b) feasibility log-likelihood (penalized if Z is wrong from bad R)
# ============================================================

def _negativity_penalty(Z, tol=TOL_Z, penalty=PENALTY_Z, power=PENALTY_POW):
    """
    Penalize negative components of Z smoothly.
    If Z is feasible, penalty is ~0.
    """
    # allow small negatives up to -tol without penalty; penalize only beyond that
    neg = np.maximum(-(Z + tol), 0.0)  # if Z < -tol => positive
    if neg.size == 0:
        return 0.0
    return float(penalty * np.sum(neg ** power))


def loglik_Y(Y, lam, mu, R, tol=TOL_Z, cond_max=COND_MAX,
            penalty=PENALTY_Z, power=PENALTY_POW):
    """
    Complete-data style likelihood for Y via Z=Y R^{-1}:
    sum_i [log lam + log mu - lam Z1_i - mu Z2_i] - N log|detR|
    with a penalty if Z has negative components.
    """
    if lam <= 0.0 or mu <= 0.0 or (not np.isfinite(lam)) or (not np.isfinite(mu)):
        return -np.inf

    R = np.asarray(R, float)
    c = np.linalg.cond(R)
    if (not np.isfinite(c)) or (c > cond_max):
        return -np.inf

    detR = np.linalg.det(R)
    if (not np.isfinite(detR)) or (abs(detR) < 1e-15):
        return -np.inf

    try:
        Z = Z_from_Y(Y, R)
    except np.linalg.LinAlgError:
        return -np.inf

    if not np.all(np.isfinite(Z)):
        return -np.inf

    jac_term = -math.log(abs(detR))
    vals = math.log(lam) + math.log(mu) - lam * Z[:, 0] - mu * Z[:, 1] + jac_term
    if not np.all(np.isfinite(vals)):
        return -np.inf

    ll = float(np.sum(vals))
    ll -= _negativity_penalty(Z, tol=tol, penalty=penalty, power=power)
    return ll


def choose_best_by_loglik(Y, lam, mu, u, tol=TOL_Z):
    """
    Among the 4 symmetric parameter configurations, choose the one with highest loglik_Y
    prefer configurations with lam <= mu, then u[0] >= u[1], then col_sums[0] >= col_sums[1]
    """
    best = None
    best_ll = -np.inf
    best_score = None

    def tie_break_score(lam_c, mu_c, u_c):
        R_c = R_from_u(u_c)
        col_sums = np.sum(R_c, axis=0)
        return (
            float(lam_c <= mu_c),
            float(u_c[0] >= u_c[1]),
            float(col_sums[0] >= col_sums[1]),
        )

    for u_c, lam_c, mu_c in symmetries_of(u, lam, mu):
        R_c = R_from_u(u_c)
        ll = loglik_Y(Y, lam_c, mu_c, R_c, tol=tol)

        if np.isfinite(ll):
            score = tie_break_score(lam_c, mu_c, u_c)
            if (best is None) or (ll > best_ll + 1e-12) or (abs(ll - best_ll) <= 1e-12 and score > best_score):
                best = (lam_c, mu_c, u_c)
                best_ll = ll
                best_score = score

    # Fallback (should happen rarely): if everything is -inf, just keep original
    if best is None:
        return (lam, mu, np.asarray(u, float).reshape(2,))
    return best


# ============================================================
# 5) E-step and M-step (rates)
# ============================================================

def E_step(Y, R, tol=TOL_Z):
    """
    Deterministic E-step (not really an e-step): use feasible subset only for the closed-form rate updates
    """
    Z, mask, n_feas = feasibility_info(Y, R, tol=tol)
    Zf = Z[mask]

    EZ1 = float(np.sum(Zf[:, 0])) if n_feas > 0 else 0.0
    EZ2 = float(np.sum(Zf[:, 1])) if n_feas > 0 else 0.0
    EN12 = float(n_feas)
    EN2 = float(n_feas)

    return {"mask": mask, "Zf": Zf, "EZ1": EZ1, "EZ2": EZ2, "EN12": EN12, "EN2": EN2}


def M_step_rates(Estats, eps=1e-15):
    """ Updating the rates from the deterministic E-step"""
    lam = Estats["EN12"] / max(Estats["EZ1"], eps)
    mu  = Estats["EN2"]  / max(Estats["EZ2"],  eps)
    return float(lam), float(mu)


# ============================================================
# 6) Feasibility hill-climb (random perturbations)
# ============================================================

def feasibility_hill_climb(Y, u_init, max_tries=200, lr=5e-2, tol=TOL_Z, det_tol=1e-12, verbose=False):
    """
    Randomly perturb u to improve feasibility (number of feasible observations)
    0 < feasibility <= N
    """
    
    rng = np.random.default_rng()
    u_best = np.array(u_init, float).reshape(2,)
    R_best = R_from_u(u_best)
    _, _, feas_best = feasibility_info(Y, R_best, tol=tol)

    for k in range(max_tries):
        u_try = u_best + lr * rng.normal(size=u_best.shape)
        R_try = R_from_u(u_try)

        detR = np.linalg.det(R_try)
        if (not np.isfinite(detR)) or (abs(detR) < det_tol):
            continue

        c = np.linalg.cond(R_try)
        if (not np.isfinite(c)) or (c > COND_MAX):
            continue

        _, _, feas_try = feasibility_info(Y, R_try, tol=tol)
        if feas_try > feas_best:
            u_best, feas_best = u_try, feas_try
            if verbose:
                print(f"  [hill-climb] improved feasibility to {feas_best}/{Y.shape[0]} at try {k+1}")
            if feas_best == Y.shape[0]:
                break

    return u_best


# ============================================================
# 7) Numerical gradient and line-search update for u
# ============================================================

def fd_grad_u(Y, lam, mu, u, step=1e-4, tol=TOL_Z):

    """
    Finite-difference gradient of loglik_Y w.r.t. u
    𝛁_u loglik_Y(Y; lam, mu, R(u))
    """
    base_ll = loglik_Y(Y, lam, mu, R_from_u(u), tol=tol)
    g = np.zeros_like(u, dtype=float)

    if not np.isfinite(base_ll):
        return base_ll, g

    for j in range(len(u)):
        u_p = u.copy(); u_m = u.copy()
        u_p[j] += step
        u_m[j] -= step

        ll_p = loglik_Y(Y, lam, mu, R_from_u(u_p), tol=tol)
        ll_m = loglik_Y(Y, lam, mu, R_from_u(u_m), tol=tol)

        if np.isfinite(ll_p) and np.isfinite(ll_m):
            g[j] = (ll_p - ll_m) / (2.0 * step)
        elif np.isfinite(ll_p):
            g[j] = (ll_p - base_ll) / step
        elif np.isfinite(ll_m):
            g[j] = (base_ll - ll_m) / step
        else:
            g[j] = 0.0

    return base_ll, g


def update_u_with_linesearch(Y, lam, mu, u0, lr=1e-2, max_backtracks=12, tol=TOL_Z, det_tol=1e-12):
    """
    Update u via gradient ascent with backtracking line search on loglik_Y, ensuring mononotone improvement.
    If you cant evaluate at the current points, then it moves to a feasible reagion with the feasibility hill climb
    """
    ll0 = loglik_Y(Y, lam, mu, R_from_u(u0), tol=tol)
    if not np.isfinite(ll0):
        u0 = feasibility_hill_climb(Y, u0, max_tries=200, lr=5e-2, tol=tol, det_tol=det_tol, verbose=False)
        ll0 = loglik_Y(Y, lam, mu, R_from_u(u0), tol=tol)
        if not np.isfinite(ll0):
            return u0, -np.inf

    base_ll, g = fd_grad_u(Y, lam, mu, u0, step=1e-4, tol=tol)
    if not np.isfinite(base_ll):
        return u0, -np.inf

    gnorm = np.linalg.norm(g)
    if gnorm > 0 and np.isfinite(gnorm):
        g = g / gnorm

    u = u0.copy()
    for _ in range(max_backtracks):
        u_try = u + lr * g
        R_try = R_from_u(u_try)

        detR = np.linalg.det(R_try)
        if (not np.isfinite(detR)) or (abs(detR) < det_tol):
            lr *= 0.5
            continue

        c = np.linalg.cond(R_try)
        if (not np.isfinite(c)) or (c > COND_MAX):
            lr *= 0.5
            continue

        ll1 = loglik_Y(Y, lam, mu, R_try, tol=tol)
        if np.isfinite(ll1) and ll1 > base_ll + 1e-10:
            return u_try, ll1

        lr *= 0.5

    return u0, base_ll


# ============================================================
# 8) Main EM/ECM driver
# ============================================================

def run_em(
    Y,
    lam0=1.0,
    mu0=1.0,
    u0=np.array([0.0, 0.0]),
    max_iter=200,              
    max_R_inner_steps=50,     
    ll_tol_per_obs=1e-7,  
    min_iter=5,
    verbose=False,
    track_ll=False,
    tol=TOL_Z,
):
    """
    EM/ECM:
      - deterministic E-step Z=Y R^{-1} with feasibility mask
      - closed-form M-step for (lam, mu) on feasible subset
      - ECM step for R via ascent in u (using SOFT loglik)
      - symmetry cleanup per iteration
      - saving the history of the ll
    """
    Y = np.asarray(Y, float)
    N = Y.shape[0]

    lam = float(lam0)
    mu = float(mu0)
    u = np.asarray(u0, float).reshape(2,)

    ll_history = [] if track_ll else None

    lam, mu, u = choose_best_by_loglik(Y, lam, mu, u, tol=tol)

    if track_ll:
        ll0 = loglik_Y(Y, lam, mu, R_from_u(u), tol=tol)
        if not np.isfinite(ll0):
            u = feasibility_hill_climb(Y, u, max_tries=200, lr=5e-2, tol=tol, verbose=False)
            lam, mu, u = choose_best_by_loglik(Y, lam, mu, u, tol=tol)
            ll0 = loglik_Y(Y, lam, mu, R_from_u(u), tol=tol)
        ll_history.append(ll0)

    prev_ll = ll_history[-1] if track_ll else loglik_Y(Y, lam, mu, R_from_u(u), tol=tol)

    for t in range(1, max_iter + 1):
        R = R_from_u(u)

        E = E_step(Y, R, tol=tol)
        num_feas = int(np.sum(E["mask"]))
        if num_feas == 0:
            u = feasibility_hill_climb(Y, u, max_tries=200, lr=5e-2, tol=tol, verbose=verbose)
            lam, mu, u = choose_best_by_loglik(Y, lam, mu, u, tol=tol)
            R = R_from_u(u)
            E = E_step(Y, R, tol=tol)
            num_feas = int(np.sum(E["mask"]))
            if num_feas == 0:
                raise RuntimeError("No feasible observations under current R after repair; try different init u.")

        lam_new, mu_new = M_step_rates(E)

        # inner loop on u for R-update
        u_candidate = u.copy()
        ll_prev = loglik_Y(Y, lam_new, mu_new, R_from_u(u_candidate), tol=tol)

        if not np.isfinite(ll_prev):
            u_candidate = feasibility_hill_climb(Y, u_candidate, max_tries=200, lr=5e-2, tol=tol, verbose=False)
            ll_prev = loglik_Y(Y, lam_new, mu_new, R_from_u(u_candidate), tol=tol)

        for _ in range(max_R_inner_steps):
            u_try, ll_try = update_u_with_linesearch(Y, lam_new, mu_new, u_candidate, lr=1e-2, tol=tol)
            if (not np.isfinite(ll_try)) or (ll_try <= ll_prev + 1e-10):
                break
            u_candidate, ll_prev = u_try, ll_try

        u_new = u_candidate

        lam_new, mu_new, u_new = choose_best_by_loglik(Y, lam_new, mu_new, u_new, tol=tol)
        lam, mu, u = lam_new, mu_new, u_new

        ll_cur = loglik_Y(Y, lam, mu, R_from_u(u), tol=tol)

        if track_ll:
            ll_history.append(ll_cur)
            ll_diff_per_obs = (ll_history[-1] - ll_history[-2]) / N
            if verbose:
                print(f"[Iter {t}] feas={num_feas}/{N}, ll={ll_cur:.6g}, ll_diff_per_obs={ll_diff_per_obs:.3e}")
            if t >= min_iter and np.isfinite(ll_diff_per_obs) and abs(ll_diff_per_obs) < ll_tol_per_obs:
                break
        else:
            if np.isfinite(ll_cur) and np.isfinite(prev_ll):
                ll_diff_per_obs = (ll_cur - prev_ll) / N
                if t >= min_iter and abs(ll_diff_per_obs) < ll_tol_per_obs:
                    break
            prev_ll = ll_cur

    R_final = R_from_u(u)
    if track_ll:
        return lam, mu, R_final, np.asarray(ll_history, dtype=float)
    return lam, mu, R_final
