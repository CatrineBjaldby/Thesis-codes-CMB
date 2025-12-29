# ============================================================
# Parallel (N,m) Adaptive-truncation Extended Kulkarni ECM (penalized)
# + diagnostics + incremental CSV export + robust summary build
#
# Key features:
# - Parallelizes independent Monte Carlo runs over (N, m)
# - Writes run-level rows incrementally to CSV (safe against crashes)
# - Optionally writes per-iteration ll trajectories incrementally (long format)
# - Builds summary CSV at the end by aggregating runs CSV (robust)
# ============================================================

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import ECM_back_jumping_bivariate as ext


# -----------------------------
# USER SETTINGS
# -----------------------------
OUT_DIR = r"C:\Users\hhn54\OneDrive\Dokumenter\Speciale\PARARUNS_EXAMPLE_3"
os.makedirs(OUT_DIR, exist_ok=True)

Ns = [100, 500, 1000, 10000]
M  = 10  # replications per N

# If you're under time pressure, you can do:
# M_bigN = 5 and use that only for N=10000.
M_bigN = None  # e.g. set to 5 to do only 5 runs for N=10000

# Parallelism:
MAX_WORKERS = None  # None => uses os.cpu_count(); or set e.g. 4

# Output names
RUNS_CSV   = os.path.join(OUT_DIR, "kulkarni_ext_penalized_adaptive_runs_parallel_example_3.csv")
TRAJ_CSV   = os.path.join(OUT_DIR, "kulkarni_ext_penalized_adaptive_ll_traj_parallel_example_3.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "kulkarni_ext_penalized_adaptive_summary_parallel_example_3.csv")

# Trajectory logging (can be huge for N=10000). Consider False if storage/time matters.
WRITE_TRAJ = True

lam_true = 2.0
rho_true = 0.7
mu_true  = 1.5
R_true   = np.array([[0.70, 0.30],
                     [0.20, 0.80]], float)

TOL_Z  = 1e-10
EPS_LL = 1e-8

# penalties
INFEAS_PEN = 1e6
NEGZ_PEN   = 1e6

# adaptive truncation controls
M_CAP         = 500
TAIL_REL_TOL  = 1e-12
TAIL_PATIENCE = 6
M_MIN         = 10

# ECM controls
MAX_ITER = 300
MAX_R_STEPS = 5
LR0 = 1e-2

VERBOSE = False  # IMPORTANT: keep False in parallel runs to avoid log chaos
TRACK_LL = True  # keep True if you want ll_hist

# -----------------------------
# Utility / diagnostics
# -----------------------------
def is_monotone_nondecreasing(x: np.ndarray, tol: float = 1e-10) -> bool:
    x = np.asarray(x, float)
    if x.size < 2:
        return True
    return bool(np.all(np.diff(x) >= -tol))


def seed_for(N: int, m: int) -> int:
    return 10_000 * m + N + 1234


def append_df_to_csv(path: str, df: pd.DataFrame) -> None:
    """Append dataframe to CSV with header if file doesn't exist."""
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


# -----------------------------
# Work unit (runs in worker process)
# -----------------------------
def run_one_job(N: int, m: int) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Returns:
      run_row: dict of run-level outputs (always returned, even on failures)
      traj_df: DataFrame with trajectory rows (or None)
    """
    seed = seed_for(N, m)
    rng = np.random.default_rng(seed)

    # simulate
    try:
        _, _, Y, *_ = ext.simulate_kulkarni_dataset_extended(
            lam_true, rho_true, mu_true, R_true, N=N, rng=rng
        )
    except Exception as e:
        run_row = {
            "N": N, "m": m, "seed": seed,
            "fit_success": False,
            "fit_error": f"simulate_failed: {e}",
            "runtime_sec": np.nan,
            "lam0": np.nan, "rho0": np.nan, "mu0": np.nan,
            "u0_1": np.nan, "u0_2": np.nan,
            "lam_hat": np.nan, "rho_hat": np.nan, "mu_hat": np.nan,
            "R11": np.nan, "R12": np.nan, "R21": np.nan, "R22": np.nan,
            "ll_true": np.nan, "ll_est": np.nan,
            "ll_true_per_obs": np.nan, "ll_est_per_obs": np.nan,
            "delta_ll_per_obs": np.nan,
            "iters": np.nan,
            "ll_monotone": np.nan,
            "converged_flag": np.nan,
            "final_abs_dll_per_obs": np.nan,
            "ll_hist_len": np.nan,
            "m_cap": M_CAP,
            "tail_rel_tol": TAIL_REL_TOL,
            "tail_patience": TAIL_PATIENCE,
            "m_min": M_MIN,
            "infeas_penalty": INFEAS_PEN,
            "negz_penalty": NEGZ_PEN,
            "max_iter": MAX_ITER,
            "max_R_steps": MAX_R_STEPS,
            "eps_ll": EPS_LL,
        }
        return run_row, None

    # init near truth
    lam0 = max(1e-6, lam_true + rng.normal(scale=0.5))
    rho0 = max(1e-8, rho_true + rng.normal(scale=0.2))
    mu0  = max(1e-6, mu_true  + rng.normal(scale=0.5))
    u0   = ext.u_from_R(R_true) + rng.normal(scale=0.5, size=2)

    # fit
    t0 = time.perf_counter()
    try:
        lam_hat, rho_hat, mu_hat, R_hat, ll_hist = ext.run_ecm_extended_monotone_penalized_adaptive(
            Y,
            lam0=lam0, rho0=rho0, mu0=mu0, u0=u0,
            max_iter=MAX_ITER, tol=TOL_Z,
            eps_ll=EPS_LL, max_R_steps=MAX_R_STEPS, lr0=LR0,
            verbose=VERBOSE, track_ll=TRACK_LL,
            infeas_penalty=INFEAS_PEN, negz_penalty=NEGZ_PEN,
            m_cap=M_CAP,
            tail_rel_tol=TAIL_REL_TOL,
            tail_patience=TAIL_PATIENCE,
            m_min=M_MIN
        )
        t1 = time.perf_counter()
        fit_success = True
        fit_error = ""
    except Exception as e:
        t1 = time.perf_counter()
        fit_success = False
        fit_error = f"{e}"
        # If you want deeper traceback text (bigger CSV), uncomment:
        # fit_error = traceback.format_exc()

        run_row = {
            "N": N, "m": m, "seed": seed,
            "fit_success": False,
            "fit_error": fit_error,
            "runtime_sec": float(t1 - t0),
            "lam0": float(lam0), "rho0": float(rho0), "mu0": float(mu0),
            "u0_1": float(u0[0]), "u0_2": float(u0[1]),
            "lam_hat": np.nan, "rho_hat": np.nan, "mu_hat": np.nan,
            "R11": np.nan, "R12": np.nan, "R21": np.nan, "R22": np.nan,
            "ll_true": np.nan, "ll_est": np.nan,
            "ll_true_per_obs": np.nan, "ll_est_per_obs": np.nan,
            "delta_ll_per_obs": np.nan,
            "iters": np.nan,
            "ll_monotone": np.nan,
            "converged_flag": np.nan,
            "final_abs_dll_per_obs": np.nan,
            "ll_hist_len": np.nan,
            "m_cap": M_CAP,
            "tail_rel_tol": TAIL_REL_TOL,
            "tail_patience": TAIL_PATIENCE,
            "m_min": M_MIN,
            "infeas_penalty": INFEAS_PEN,
            "negz_penalty": NEGZ_PEN,
            "max_iter": MAX_ITER,
            "max_R_steps": MAX_R_STEPS,
            "eps_ll": EPS_LL,
        }
        return run_row, None

    runtime = float(t1 - t0)

    # likelihoods (penalized adaptive)
    ll_true = ext.loglik_Y_extended_penalized_adaptive(
        Y, lam_true, rho_true, mu_true, R_true,
        tol=TOL_Z,
        infeas_penalty=INFEAS_PEN, negz_penalty=NEGZ_PEN,
        m_cap=M_CAP,
        tail_rel_tol=TAIL_REL_TOL,
        tail_patience=TAIL_PATIENCE,
        m_min=M_MIN
    )
    ll_est = ext.loglik_Y_extended_penalized_adaptive(
        Y, lam_hat, rho_hat, mu_hat, R_hat,
        tol=TOL_Z,
        infeas_penalty=INFEAS_PEN, negz_penalty=NEGZ_PEN,
        m_cap=M_CAP,
        tail_rel_tol=TAIL_REL_TOL,
        tail_patience=TAIL_PATIENCE,
        m_min=M_MIN
    )

    # trajectory + diagnostics
    traj_df = None
    iters = np.nan
    mono = np.nan
    converged_flag = np.nan
    final_abs_dll_po = np.nan
    ll_hist_len = np.nan

    if ll_hist is not None:
        ll_hist = np.asarray(ll_hist, float)
        ll_hist = ll_hist[np.isfinite(ll_hist)]
        iters = int(ll_hist.size)
        ll_hist_len = iters

        if iters > 0:
            ll_hist_po = ll_hist / float(N)
            mono = is_monotone_nondecreasing(ll_hist, tol=1e-10)

            if iters >= 2:
                final_abs_dll_po = float(abs(ll_hist_po[-1] - ll_hist_po[-2]))

            converged_flag = bool(
                iters >= 5 and np.isfinite(final_abs_dll_po) and final_abs_dll_po <= EPS_LL
            )

            if WRITE_TRAJ:
                traj_df = pd.DataFrame({
                    "N": N,
                    "m": m,
                    "seed": seed,
                    "iter": np.arange(1, iters + 1, dtype=int),
                    "ll_per_obs": ll_hist_po.astype(float),
                })

    # per-obs + delta
    ll_true_po = float(ll_true) / float(N) if np.isfinite(ll_true) else np.nan
    ll_est_po  = float(ll_est)  / float(N) if np.isfinite(ll_est)  else np.nan
    delta_po   = (ll_est_po - ll_true_po) if (np.isfinite(ll_true_po) and np.isfinite(ll_est_po)) else np.nan

    run_row = {
        "N": N, "m": m, "seed": seed,
        "fit_success": True,
        "fit_error": "",
        "runtime_sec": runtime,
        "lam0": float(lam0), "rho0": float(rho0), "mu0": float(mu0),
        "u0_1": float(u0[0]), "u0_2": float(u0[1]),
        "lam_hat": float(lam_hat), "rho_hat": float(rho_hat), "mu_hat": float(mu_hat),
        "R11": float(R_hat[0, 0]), "R12": float(R_hat[0, 1]),
        "R21": float(R_hat[1, 0]), "R22": float(R_hat[1, 1]),
        "ll_true": float(ll_true),
        "ll_est": float(ll_est),
        "ll_true_per_obs": ll_true_po,
        "ll_est_per_obs": ll_est_po,
        "delta_ll_per_obs": float(delta_po) if np.isfinite(delta_po) else np.nan,
        "iters": iters,
        "ll_monotone": mono,
        "converged_flag": converged_flag,
        "final_abs_dll_per_obs": final_abs_dll_po,
        "ll_hist_len": ll_hist_len,
        "m_cap": M_CAP,
        "tail_rel_tol": TAIL_REL_TOL,
        "tail_patience": TAIL_PATIENCE,
        "m_min": M_MIN,
        "infeas_penalty": INFEAS_PEN,
        "negz_penalty": NEGZ_PEN,
        "max_iter": MAX_ITER,
        "max_R_steps": MAX_R_STEPS,
        "eps_ll": EPS_LL,
    }

    return run_row, traj_df


# -----------------------------
# Summary builder (runs in main process)
# -----------------------------
def build_summary_from_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    # Use only successful fits with finite ll_true/ll_est for likelihood diagnostics
    df_good = df_runs[
        (df_runs["fit_success"] == True)
        & np.isfinite(df_runs["ll_true"].to_numpy(dtype=float))
        & np.isfinite(df_runs["ll_est"].to_numpy(dtype=float))
        & np.isfinite(df_runs["delta_ll_per_obs"].to_numpy(dtype=float))
    ].copy()

    out_rows: List[Dict[str, Any]] = []
    for N, g in df_good.groupby("N"):
        N = int(N)
        delta = g["delta_ll_per_obs"].astype(float).to_numpy()
        runt  = g["runtime_sec"].astype(float).to_numpy()
        iters = g["iters"].astype(float).to_numpy()
        conv  = g["converged_flag"].astype(float).to_numpy()  # bool -> {0,1}
        mono  = g["ll_monotone"].astype(float).to_numpy()

        out_rows.append({
            "N": N,
            "True_logLik_soft_mean": float(g["ll_true"].astype(float).mean()),
            "Est_logLik_soft_mean":  float(g["ll_est"].astype(float).mean()),
            "delta_ll_per_obs_mean": float(np.mean(delta)) if delta.size else np.nan,
            "delta_ll_per_obs_sd":   float(np.std(delta, ddof=1)) if delta.size > 1 else np.nan,
            "mean_time_sec":         float(np.mean(runt)) if runt.size else np.nan,
            "fit_ok":                int((df_runs[df_runs["N"] == N]["fit_success"] == True).sum()),
            "ll_ok":                 int(len(g)),
            "conv_rate":             float(np.mean(conv)) if conv.size else np.nan,
            "mono_rate":             float(np.mean(mono)) if mono.size else np.nan,
            "iters_median":          float(np.median(iters)) if iters.size else np.nan,
            "iters_p90":             float(np.quantile(iters, 0.90)) if iters.size else np.nan,
            "m_cap": M_CAP,
            "tail_rel_tol": TAIL_REL_TOL,
            "tail_patience": TAIL_PATIENCE,
            "m_min": M_MIN,
            "infeas_penalty": INFEAS_PEN,
            "negz_penalty": NEGZ_PEN,
            "max_iter": MAX_ITER,
            "max_R_steps": MAX_R_STEPS,
            "eps_ll": EPS_LL,
        })

    df_summary = pd.DataFrame(out_rows).sort_values("N").reset_index(drop=True)
    return df_summary


# -----------------------------
# Main driver
# -----------------------------
def main() -> None:
    # Optional: remove old outputs so you don't append to stale files
    # Comment out if you WANT to append/resume.
    for p in [RUNS_CSV, TRAJ_CSV, SUMMARY_CSV]:
        if os.path.exists(p):
            os.remove(p)

    # Build job list (optionally fewer for N=10000)
    jobs: List[Tuple[int, int]] = []
    for N in Ns:
        M_this = M
        if M_bigN is not None and int(N) == 10000:
            M_this = int(M_bigN)
        for m in range(M_this):
            jobs.append((int(N), int(m)))

    total = len(jobs)
    print(f"Launching {total} jobs with max_workers={MAX_WORKERS} ...")
    t_global0 = time.perf_counter()

    # Run parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one_job, N, m): (N, m) for (N, m) in jobs}

        done = 0
        for fut in as_completed(futures):
            N, m = futures[fut]
            try:
                run_row, traj_df = fut.result()
            except Exception as e:
                # catastrophic worker failure
                run_row = {
                    "N": N, "m": m, "seed": seed_for(N, m),
                    "fit_success": False,
                    "fit_error": f"worker_crash: {e}",
                    "runtime_sec": np.nan,
                    "lam0": np.nan, "rho0": np.nan, "mu0": np.nan,
                    "u0_1": np.nan, "u0_2": np.nan,
                    "lam_hat": np.nan, "rho_hat": np.nan, "mu_hat": np.nan,
                    "R11": np.nan, "R12": np.nan, "R21": np.nan, "R22": np.nan,
                    "ll_true": np.nan, "ll_est": np.nan,
                    "ll_true_per_obs": np.nan, "ll_est_per_obs": np.nan,
                    "delta_ll_per_obs": np.nan,
                    "iters": np.nan,
                    "ll_monotone": np.nan,
                    "converged_flag": np.nan,
                    "final_abs_dll_per_obs": np.nan,
                    "ll_hist_len": np.nan,
                    "m_cap": M_CAP,
                    "tail_rel_tol": TAIL_REL_TOL,
                    "tail_patience": TAIL_PATIENCE,
                    "m_min": M_MIN,
                    "infeas_penalty": INFEAS_PEN,
                    "negz_penalty": NEGZ_PEN,
                    "max_iter": MAX_ITER,
                    "max_R_steps": MAX_R_STEPS,
                    "eps_ll": EPS_LL,
                }
                traj_df = None

            # incremental write
            append_df_to_csv(RUNS_CSV, pd.DataFrame([run_row]))
            if WRITE_TRAJ and traj_df is not None and len(traj_df) > 0:
                append_df_to_csv(TRAJ_CSV, traj_df)

            done += 1
            status = "OK" if run_row.get("fit_success", False) else "FAIL"
            rt = run_row.get("runtime_sec", np.nan)
            iters = run_row.get("iters", np.nan)
            print(f"Done {done}/{total}: N={N}, m={m} | {status} | runtime={rt:.2f}s | iters={iters}")

    t_global1 = time.perf_counter()
    print(f"All jobs finished in {(t_global1 - t_global0)/60.0:.2f} minutes.")

    # Build summary from written runs CSV (robust)
    df_runs = pd.read_csv(RUNS_CSV)
    df_summary = build_summary_from_runs(df_runs)

    df_summary.to_csv(SUMMARY_CSV, index=False)
    print("Wrote:")
    print("  Runs   :", RUNS_CSV)
    if WRITE_TRAJ:
        print("  Traj   :", TRAJ_CSV)
    print("  Summary:", SUMMARY_CSV)

    # show summary in console
    print(df_summary)


if __name__ == "__main__":
    main()
