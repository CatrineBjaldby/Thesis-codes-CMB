import numpy as np
import time
import math

# ============================================================
# p-phase Kulkarni ECM (like the bivariate implementation logic)
# ============================================================


# Global numerical settings 
TOL_Z    = 1e-8
DET_TOL  = 1e-12
COND_MAX = 1e12

# Soft-feasibility penalty
PENALTY_Z   = 1e6
PENALTY_POW = 2.0


# ============================================================
# 1) Simulation of p-phase Kulkarni data
# ============================================================

def simulate_kulkarni_p_dataset(lambdas, R, N, rng=None):
    """
    Simulate N i.i.d. p-phase Kulkarni-like observations:

        Z_k ~ Exp(lambda_k), independent
        Y = Z R

    Returns:
      Z_all : (N,p) sojourn times
      Y_all : (N,p) observed rewards
    """
    lambdas = np.asarray(lambdas, dtype=float)
    p = lambdas.size
    if rng is None:
        rng = np.random.default_rng()

    if np.any(lambdas <= 0) or not np.all(np.isfinite(lambdas)):
        raise ValueError("All lambdas must be positive and finite.")

    Z_all = rng.exponential(scale=1.0 / lambdas, size=(N, p))
    Y_all = Z_all @ np.asarray(R, dtype=float)
    return Z_all, Y_all


# ============================================================
# 2) Basic linear algebra helpers
# ============================================================

def det_ok(R, threshold=DET_TOL):
    """
    Check that R is numerically invertible:
    determinant finite and not too close to zero.
    """
    sign, logdet = np.linalg.slogdet(R)
    if sign == 0 or (not np.isfinite(logdet)):
        return False
    return logdet > np.log(threshold)


def cond_ok(R, cond_max=COND_MAX):
    c = np.linalg.cond(R)
    return np.isfinite(c) and (c <= cond_max)


def R_from_u_p(u, p):
    """
    Parameterize p×p reward matrix R via u ∈ R^{p^2} using row-wise softmax.
    R has strictly positive entries and each row sums to 1.
    """
    u = np.asarray(u, dtype=float)
    if u.size != p * p:
        raise ValueError("u must have length p^2")
    U = u.reshape(p, p)

    # numerical stability: subtract row-wise max
    U_shifted = U - np.max(U, axis=1, keepdims=True)
    exp_U = np.exp(U_shifted)
    R = exp_U / np.sum(exp_U, axis=1, keepdims=True)
    return R


def Z_from_Y(Y, R):
    """
    Compute Z from Y via Y = Z R, without forming R^{-1}.
    Solve: Z R = Y  =>  (R^T)(Z^T) = Y^T  =>  Z^T = solve(R^T, Y^T)
    """
    Y = np.asarray(Y, dtype=float)
    R = np.asarray(R, dtype=float)
    return np.linalg.solve(R.T, Y.T).T


# ============================================================
# 3) Feasibility: Z must be nonnegative (within tolerance)
# ============================================================

def feasibility_info(Y, R, tol=TOL_Z, det_tol=DET_TOL, cond_max=COND_MAX):
    """
    Compute feasibility of Z = Y R^{-1} (via solve):
      mask[i] = True if all components of Z[i] >= -tol.
    Reject near-singular or ill-conditioned R.
    """
    Y = np.asarray(Y, dtype=float)
    R = np.asarray(R, dtype=float)
    n, p = Y.shape

    if (not det_ok(R, threshold=det_tol)) or (not cond_ok(R, cond_max=cond_max)):
        Z = np.full((n, p), np.nan)
        mask = np.zeros(n, dtype=bool)
        return Z, mask, 0

    try:
        Z = Z_from_Y(Y, R)
    except np.linalg.LinAlgError:
        Z = np.full((n, p), np.nan)
        mask = np.zeros(n, dtype=bool)
        return Z, mask, 0

    mask = np.all(Z >= -tol, axis=1) & np.all(np.isfinite(Z), axis=1)
    return Z, mask, int(mask.sum())


def feasibility_hill_climb(Y, u0, p,
                           max_tries=120,
                           lr=5e-2,
                           tol=TOL_Z,
                           det_tol=DET_TOL,
                           cond_max=COND_MAX,
                           verbose=False):
    """
    improve feasibility in u-space. Mirrors the "repair" idea in the bivariate code.
    """
    rng = np.random.default_rng()
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]

    u_best = np.asarray(u0, dtype=float).copy()
    R_best = R_from_u_p(u_best, p)
    _, _, feas_best = feasibility_info(Y, R_best, tol=tol, det_tol=det_tol, cond_max=cond_max)

    if verbose:
        print(f"[feasibility] start feas = {feas_best}/{n}")

    for k in range(max_tries):
        u_try = u_best + lr * rng.normal(size=u_best.shape)
        R_try = R_from_u_p(u_try, p)

        if (not det_ok(R_try, threshold=det_tol)) or (not cond_ok(R_try, cond_max=cond_max)):
            continue

        _, _, feas_try = feasibility_info(Y, R_try, tol=tol, det_tol=det_tol, cond_max=cond_max)
        if feas_try > feas_best:
            u_best, feas_best = u_try, feas_try
            if verbose:
                print(f"[feasibility] improved to {feas_best}/{n} at try {k+1}")
            if feas_best == n:
                break

    return u_best


# ============================================================
# 4) E-step and M-step for lambdas in p-phase Kulkarni model (just determinstik decising)
# ============================================================

def E_step_p(Y, R, tol=TOL_Z, det_tol=DET_TOL, cond_max=COND_MAX):
    """
    E-step:
      - reconstruct Z = Y R^{-1} via solve
      - keep feasible rows Z >= -tol
      - Coxian visitation: each feasible obs visits phases 0..p-1 once
    """
    Y = np.asarray(Y, dtype=float)
    N, p = Y.shape

    Z, mask, n_feas = feasibility_info(Y, R, tol=tol, det_tol=det_tol, cond_max=cond_max)
    Zf = Z[mask, :]
    num_feas = int(n_feas)

    B_sum = np.zeros(p)
    B_sum[0] = num_feas

    Z_sum = Zf.sum(axis=0) if num_feas > 0 else np.zeros(p)

    N_transient_sum = np.zeros((p, p))
    N_abs_sum = np.zeros(p)

    for k in range(p - 1):
        N_transient_sum[k, k + 1] = num_feas
    N_abs_sum[p - 1] = num_feas

    return {
        "mask": mask,
        "Zf": Zf,
        "B_sum": B_sum,
        "Z_sum": Z_sum,
        "N_transient_sum": N_transient_sum,
        "N_abs_sum": N_abs_sum,
    }


def M_step_lambdas_p(Estats, eps=1e-15):
    """
    M-step for lambdas:
      lambda_k = exits_k / Z_sum_k
    """
    Z_sum = Estats["Z_sum"]
    N_transient_sum = Estats["N_transient_sum"]
    N_abs_sum = Estats["N_abs_sum"]

    p = Z_sum.size
    lambdas = np.zeros(p)

    for k in range(p):
        exits_k = N_abs_sum[k] + np.sum(N_transient_sum[k, :]) - N_transient_sum[k, k]
        denom = max(float(Z_sum[k]), eps)
        lambdas[k] = exits_k / denom

    return lambdas


# ============================================================
# 5) Likelihoods: hard diagnostic + soft feasibility penalty (bivariate-style)
# ============================================================

def loglik_Y_p_hard_subset(Y, lambdas, R, tol=TOL_Z, det_tol=DET_TOL, cond_max=COND_MAX):
    """
    HARD diagnostic likelihood on feasible subset only (like your old 'hard'):
      - compute Z = Y R^{-1}
      - keep only rows with Z >= -tol
      - return -inf if zero feasible
    NOTE: diagnostic only; feasible set depends on R.
    """
    Y = np.asarray(Y, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    N, p = Y.shape

    if lambdas.size != p:
        raise ValueError("lambdas must have length p")
    if np.any(lambdas <= 0) or not np.all(np.isfinite(lambdas)):
        return -np.inf

    if (not det_ok(R, threshold=det_tol)) or (not cond_ok(R, cond_max=cond_max)):
        return -np.inf

    try:
        Z = Z_from_Y(Y, R)
    except np.linalg.LinAlgError:
        return -np.inf

    mask = np.all(Z >= -tol, axis=1) & np.all(np.isfinite(Z), axis=1)
    Zf = Z[mask, :]
    Nf = Zf.shape[0]
    if Nf == 0:
        return -np.inf

    sign, logdet = np.linalg.slogdet(R)
    if sign == 0 or (not np.isfinite(logdet)) or logdet <= np.log(det_tol):
        return -np.inf

    ll = 0.0
    ll += Nf * np.sum(np.log(lambdas))
    ll -= np.sum(lambdas * Zf.sum(axis=0))
    ll -= Nf * logdet
    return float(ll)


def _negativity_penalty(Z, tol=TOL_Z, penalty=PENALTY_Z, power=PENALTY_POW):
    """
    Same structure as bivariate:
    penalize only components below -tol, quadratically by default.
    """
    neg = np.maximum(-(Z + tol), 0.0)
    if neg.size == 0:
        return 0.0
    return float(penalty * np.sum(neg ** power))


def loglik_Y_p_soft(Y, lambdas, R,
                    tol=TOL_Z,
                    det_tol=DET_TOL,
                    cond_max=COND_MAX,
                    penalty=PENALTY_Z,
                    power=PENALTY_POW):
    """
    SOFT (penalized) log-likelihood aligned with bivariate case:

      ll = sum_i [ sum_k log(lambda_k) - sum_k lambda_k Z_{ik} ] - N log|det(R)|
      with Z = Y R^{-1}

    PLUS:
      - soft penalty for negative components of Z (quadratic by default).
      - rejects near-singular/ill-conditioned R.
    """
    Y = np.asarray(Y, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    N, p = Y.shape

    if lambdas.size != p:
        raise ValueError("lambdas must have length p")
    if np.any(lambdas <= 0) or not np.all(np.isfinite(lambdas)):
        return -np.inf

    R = np.asarray(R, dtype=float)
    if (not det_ok(R, threshold=det_tol)) or (not cond_ok(R, cond_max=cond_max)):
        return -np.inf

    sign, logdet = np.linalg.slogdet(R)
    if sign == 0 or (not np.isfinite(logdet)) or logdet <= np.log(det_tol):
        return -np.inf

    try:
        Z = Z_from_Y(Y, R)
    except np.linalg.LinAlgError:
        return -np.inf

    if not np.all(np.isfinite(Z)):
        return -np.inf

    # Base complete-data-style likelihood (all observations)
    ll = 0.0
    ll += N * np.sum(np.log(lambdas))
    ll -= float(np.sum(lambdas * np.sum(Z, axis=0)))
    ll -= N * float(logdet)

    # Soft feasibility penalty
    ll -= _negativity_penalty(Z, tol=tol, penalty=penalty, power=power)

    return float(ll)


# ============================================================
# 6) Gradient + line search for u (optimize SOFT objective)
# ============================================================

def fd_grad_u_p(Y, lambdas, u, p,
                step=1e-4,
                tol=TOL_Z,
                det_tol=DET_TOL,
                cond_max=COND_MAX,
                penalty=PENALTY_Z,
                power=PENALTY_POW):
    """
    Numerical gradient of SOFT objective w.r.t u using finite differences.
    """
    R0 = R_from_u_p(u, p)
    base_ll = loglik_Y_p_soft(Y, lambdas, R0, tol=tol, det_tol=det_tol, cond_max=cond_max,
                             penalty=penalty, power=power)
    g = np.zeros_like(u, dtype=float)

    if not np.isfinite(base_ll):
        return base_ll, g

    for j in range(u.size):
        u_p = u.copy(); u_m = u.copy()
        u_p[j] += step
        u_m[j] -= step

        ll_p = loglik_Y_p_soft(Y, lambdas, R_from_u_p(u_p, p),
                               tol=tol, det_tol=det_tol, cond_max=cond_max,
                               penalty=penalty, power=power)
        ll_m = loglik_Y_p_soft(Y, lambdas, R_from_u_p(u_m, p),
                               tol=tol, det_tol=det_tol, cond_max=cond_max,
                               penalty=penalty, power=power)

        if np.isfinite(ll_p) and np.isfinite(ll_m):
            g[j] = (ll_p - ll_m) / (2.0 * step)
        elif np.isfinite(ll_p):
            g[j] = (ll_p - base_ll) / step
        elif np.isfinite(ll_m):
            g[j] = (base_ll - ll_m) / step
        else:
            g[j] = 0.0

    return base_ll, g


def update_u_with_linesearch_p(Y, lambdas, u0, p,
                               lr=1e-2,
                               max_backtracks=12,
                               tol=TOL_Z,
                               det_tol=DET_TOL,
                               cond_max=COND_MAX,
                               penalty=PENALTY_Z,
                               power=PENALTY_POW):
    """
    Gradient ascent + backtracking on SOFT objective.
    Enforces monotone increase in the SOFT objective.
    """
    base_ll, g = fd_grad_u_p(Y, lambdas, u0, p,
                            step=1e-4,
                            tol=tol, det_tol=det_tol, cond_max=cond_max,
                            penalty=penalty, power=power)
    if not np.isfinite(base_ll):
        return u0, -np.inf

    gnorm = np.linalg.norm(g)
    if gnorm == 0 or (not np.isfinite(gnorm)):
        return u0, base_ll

    g = g / gnorm

    u = u0.copy()
    cur_lr = lr

    for _ in range(max_backtracks):
        u_try = u + cur_lr * g
        R_try = R_from_u_p(u_try, p)

        if (not det_ok(R_try, threshold=det_tol)) or (not cond_ok(R_try, cond_max=cond_max)):
            cur_lr *= 0.5
            continue

        ll1 = loglik_Y_p_soft(Y, lambdas, R_try,
                              tol=tol, det_tol=det_tol, cond_max=cond_max,
                              penalty=penalty, power=power)
        if np.isfinite(ll1) and ll1 > base_ll + 1e-10:
            return u_try, ll1

        cur_lr *= 0.5

    return u0, base_ll


# ============================================================
# 7) Identifiability convention 
# ============================================================

def canonicalize_phases(lambdas, u, p):
    """
    Sort phases by lambda_k and permute rows of U accordingly.
    This mirrors the bivariate "canonical representative" idea.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    U = np.asarray(u, dtype=float).reshape(p, p)

    perm = np.argsort(lambdas)
    lambdas_new = lambdas[perm]
    U_new = U[perm, :]

    return lambdas_new, U_new.ravel()


# ============================================================
# 8) Main EM/ECM driver for p-phase model
# ============================================================

def run_em_p(
    Y,
    lambdas0,
    u0,
    max_iter=80,
    max_R_inner_steps=10,
    tol=TOL_Z,
    det_tol=DET_TOL,
    cond_max=COND_MAX,
    penalty=PENALTY_Z,
    power=PENALTY_POW,
    ll_tol_per_obs=1e-7,
    min_iter=5,
    verbose=False,
    track_ll=False,
    return_hard_diagnostic=False,
):
    """
    ECM for p-phase Kulkarni/Coxian forward model.

    Alignment with bivariate:
      - deterministic reconstruction Z = Y R^{-1} via solve
      - feasible-mask used ONLY for lambda-updates (closed form)
      - R-step increases a soft objective with negativity penalty
      - convergence monitored on SAME soft objective (per observation)
      - canonicalization to stabilize label switching
      - reject near-singular / ill-conditioned R
    """
    Y = np.asarray(Y, dtype=float)
    N, p = Y.shape

    lambdas = np.asarray(lambdas0, dtype=float).copy()
    if lambdas.size != p:
        raise ValueError("lambdas0 must have length p")

    u = np.asarray(u0, dtype=float).copy()
    if u.size != p * p:
        raise ValueError("u0 must have length p^2")

    ll_history_soft = [] if track_ll else None
    ll_history_hard = [] if (track_ll and return_hard_diagnostic) else None

    prev_ll_soft = None

    # initial canonicalization
    lambdas, u = canonicalize_phases(lambdas, u, p)

    for t in range(1, max_iter + 1):
        R = R_from_u_p(u, p)

        # If objective is non-finite, attempt feasibility repair once
        ll_soft_cur = loglik_Y_p_soft(Y, lambdas, R,
                                      tol=tol, det_tol=det_tol, cond_max=cond_max,
                                      penalty=penalty, power=power)
        if not np.isfinite(ll_soft_cur):
            u = feasibility_hill_climb(Y, u, p, max_tries=120, lr=5e-2,
                                       tol=tol, det_tol=det_tol, cond_max=cond_max, verbose=False)
            R = R_from_u_p(u, p)
            ll_soft_cur = loglik_Y_p_soft(Y, lambdas, R,
                                          tol=tol, det_tol=det_tol, cond_max=cond_max,
                                          penalty=penalty, power=power)
            if not np.isfinite(ll_soft_cur):
                raise RuntimeError("Non-finite soft objective after feasibility repair; try different init.")

        # ---- E-step (mask for lambdas only) ----
        Estats = E_step_p(Y, R, tol=tol, det_tol=det_tol, cond_max=cond_max)
        num_feas = int(np.sum(Estats["mask"]))
        if num_feas == 0:
            # repair and retry
            u = feasibility_hill_climb(Y, u, p, max_tries=120, lr=5e-2,
                                       tol=tol, det_tol=det_tol, cond_max=cond_max, verbose=False)
            R = R_from_u_p(u, p)
            Estats = E_step_p(Y, R, tol=tol, det_tol=det_tol, cond_max=cond_max)
            num_feas = int(np.sum(Estats["mask"]))
            if num_feas == 0:
                raise RuntimeError("No feasible observations under current R after repair; try different init.")

        # ---- M-step for lambdas (closed form on feasible subset) ----
        lambdas_new = M_step_lambdas_p(Estats)
        if np.any(lambdas_new <= 0) or (not np.all(np.isfinite(lambdas_new))):
            raise RuntimeError("Invalid lambdas update (non-positive or non-finite).")

        # ---- ECM-step for R via u (ascent on SOFT objective) ----
        u_candidate = u.copy()
        ll_prev = loglik_Y_p_soft(Y, lambdas_new, R_from_u_p(u_candidate, p),
                                  tol=tol, det_tol=det_tol, cond_max=cond_max,
                                  penalty=penalty, power=power)

        if not np.isfinite(ll_prev):
            u_candidate = feasibility_hill_climb(Y, u_candidate, p, max_tries=120, lr=5e-2,
                                                 tol=tol, det_tol=det_tol, cond_max=cond_max, verbose=False)
            ll_prev = loglik_Y_p_soft(Y, lambdas_new, R_from_u_p(u_candidate, p),
                                      tol=tol, det_tol=det_tol, cond_max=cond_max,
                                      penalty=penalty, power=power)

        for _ in range(max_R_inner_steps):
            u_try, ll_try = update_u_with_linesearch_p(
                Y, lambdas_new, u_candidate, p,
                lr=1e-2,
                max_backtracks=12,
                tol=tol, det_tol=det_tol, cond_max=cond_max,
                penalty=penalty, power=power
            )
            if (not np.isfinite(ll_try)) or (ll_try <= ll_prev + 1e-10):
                break
            u_candidate, ll_prev = u_try, ll_try

        # ---- Canonicalize ----
        lambdas, u = canonicalize_phases(lambdas_new, u_candidate, p)
        R_new = R_from_u_p(u, p)

        # ---- Track + stop on SOFT objective (aligned) ----
        ll_soft = loglik_Y_p_soft(Y, lambdas, R_new,
                                  tol=tol, det_tol=det_tol, cond_max=cond_max,
                                  penalty=penalty, power=power)

        if track_ll:
            ll_history_soft.append(ll_soft)
            if return_hard_diagnostic:
                ll_history_hard.append(loglik_Y_p_hard_subset(Y, lambdas, R_new,
                                                             tol=tol, det_tol=det_tol, cond_max=cond_max))

        if verbose:
            _, _, nf = feasibility_info(Y, R_new, tol=tol, det_tol=det_tol, cond_max=cond_max)
            ll_h_dbg = loglik_Y_p_hard_subset(Y, lambdas, R_new, tol=tol, det_tol=det_tol, cond_max=cond_max)
            print(f"[Iter {t}] feas={nf}/{N}, ll_soft={ll_soft:.6g}, ll_hard(dbg)={ll_h_dbg:.6g}")

        if prev_ll_soft is not None and np.isfinite(ll_soft) and np.isfinite(prev_ll_soft):
            ll_diff_per_obs = (ll_soft - prev_ll_soft) / N
            if t >= min_iter and abs(ll_diff_per_obs) < ll_tol_per_obs:
                if verbose:
                    print("Converged (soft objective criterion).")
                break
        prev_ll_soft = ll_soft

    R_final = R_from_u_p(u, p)
    if track_ll:
        if return_hard_diagnostic:
            return lambdas, R_final, np.asarray(ll_history_soft, dtype=float), np.asarray(ll_history_hard, dtype=float)
        return lambdas, R_final, np.asarray(ll_history_soft, dtype=float)
    return lambdas, R_final
