import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT_DIR = r"C:\Users\hhn54\OneDrive\Dokumenter\Speciale\PARARUNS"
os.makedirs(OUT_DIR, exist_ok=True)

Ns = [100, 500, 1000, 10000]
M = 7

# True parameters
lam_true   = 1.2
rho_true   = 0.7
mu_true    = 0.9
kappa_true = 0.6
lam3_true  = 1.5
r31_true   = 0.8
r32_true   = 1.1

init_sd_rate = 1e-1
init_sd_prob = 1e-1

rate_min = 1e-3
prob_min, prob_max = 1e-4, 1.0 - 1e-4

# Likelihood approximation controls
v_cap = 250
tail_rel_tol = 1e-08
tail_patience = 4
v_min = 8
n_grid = 60

# ECM controls
MAX_ITER = 200
EPS_LL = 1e-6
MIN_ITERS_LL = 5

MAX_R_INNER = 2
LR0 = 5e-2


def is_monotone_nondecreasing(x, tol=1e-10):
    x = np.asarray(x, float)
    if x.size < 2:
        return True
    return bool(np.all(np.diff(x) >= -tol))


def run_one_job(job):
    """
    One Monte Carlo job: (N, m).
    Returns:
      run_row, traj_rows, bits
    """
    import numpy as np
    import time
    import os

    # IMPORTANT: import inside worker (Windows spawn-safe)
    # CHANGE THIS to your actual module name (the .py file with run_ecm/simulate/loglik_adaptive)
    import EXAMPLE_5_PARRALEL_ALGO as ecm

    N, m = job
    pid = os.getpid()
    print(f"[PID {pid}] START  N={N}, m={m}", flush=True)   
    seed = 10_000 * m + N + 1234
    rng = np.random.default_rng(seed)

    # ---- Simulate ----
    try:
        Y, _, _, _ = ecm.simulate_noninv_unknown_dataset(
            lam_true, rho_true, mu_true, kappa_true, lam3_true,
            r31_true, r32_true,
            N=N, rng=rng
        )
    except Exception as e:
        return (
            {"N": N, "m": m, "seed": seed, "fit_success": False,
             "fit_error": f"simulate failed: {e}", "runtime_sec": np.nan},
            [],
            {"N": N, "ok_fit": 0, "ok_ll": 0}
        )

    if np.any(Y <= 0.0):
        err = f"Simulated Y has non-positive entries. min(Y)={Y.min(axis=0)}"
        run_row = {
            "N": N, "m": m, "seed": seed,
            "fit_success": False,
            "fit_error": err,
            "runtime_sec": np.nan,
            "v_cap": v_cap, "tail_rel_tol": tail_rel_tol, "tail_patience": tail_patience,
            "v_min": v_min, "n_grid": n_grid,
        }
        return run_row, [], {"N": N, "ok_fit": 0, "ok_ll": 0}

    # ---- Init ----
    lam0   = max(lam_true   + rng.normal(scale=init_sd_rate), rate_min)
    mu0    = max(mu_true    + rng.normal(scale=init_sd_rate), rate_min)
    kappa0 = max(kappa_true + rng.normal(scale=init_sd_rate), rate_min)
    lam3_0 = max(lam3_true  + rng.normal(scale=init_sd_rate), rate_min)
    r31_0  = max(r31_true   + rng.normal(scale=init_sd_rate), rate_min)
    r32_0  = max(r32_true   + rng.normal(scale=init_sd_rate), rate_min)
    rho0   = float(np.clip(rho_true + rng.normal(scale=init_sd_prob), prob_min, prob_max))

    # ---- Fit ----
    t0 = time.perf_counter()
    try:
        lam_hat, rho_hat, mu_hat, kappa_hat, lam3_hat, r31_hat, r32_hat, ll_hist = ecm.run_ecm(
            Y,
            lam0=lam0, rho0=rho0, mu0=mu0, kappa0=kappa0,
            lam3_0=lam3_0, r31_0=r31_0, r32_0=r32_0,
            max_iter=MAX_ITER,
            eps_ll=EPS_LL,
            min_iters_ll=MIN_ITERS_LL,
            v_cap=v_cap, tail_rel_tol=tail_rel_tol,
            tail_patience=tail_patience, v_min=v_min,
            n_grid=n_grid,
            max_R_inner_steps=MAX_R_INNER,
            lr0=LR0,
            verbose=False,
            track_ll=True
        )
        runtime = time.perf_counter() - t0
        print(f"[PID {pid}] FINISH N={N}, m={m} | {runtime:.2f}s", flush=True)  # <-- ADD
    except Exception as e:
        runtime = time.perf_counter() - t0
        print(f"[PID {pid}] FAIL(fit) N={N}, m={m} | {runtime:.2f}s: {e}", flush=True) 
        run_row = {
            "N": N, "m": m, "seed": seed,
            "fit_success": False,
            "fit_error": str(e),
            "runtime_sec": runtime,
            "lam0": lam0, "rho0": rho0, "mu0": mu0, "kappa0": kappa0, "lam3_0": lam3_0,
            "r31_0": r31_0, "r32_0": r32_0,
            "v_cap": v_cap, "tail_rel_tol": tail_rel_tol, "tail_patience": tail_patience,
            "v_min": v_min, "n_grid": n_grid,
        }
        return run_row, [], {"N": N, "ok_fit": 0, "ok_ll": 0, "runtime": runtime}

    # ---- LL hist ----
    ll_hist = np.asarray(ll_hist, dtype=float) if ll_hist is not None else np.array([], float)
    ll_hist = ll_hist[np.isfinite(ll_hist)]
    iters = int(ll_hist.size)
    ll_est_m = float(ll_hist[-1]) if iters > 0 else np.nan

    # true LL (same approximation settings so it is comparable)
    ll_true_m = ecm.loglik_adaptive(
        Y, lam_true, rho_true, mu_true, kappa_true, lam3_true, r31_true, r32_true,
        v_cap=v_cap, tail_rel_tol=tail_rel_tol, tail_patience=tail_patience,
        v_min=v_min, n_grid=n_grid
    )

    if not (np.isfinite(ll_true_m) and np.isfinite(ll_est_m)):
        run_row = {
            "N": N, "m": m, "seed": seed,
            "fit_success": True, "fit_error": "",
            "runtime_sec": runtime,
            "lam0": lam0, "rho0": rho0, "mu0": mu0, "kappa0": kappa0, "lam3_0": lam3_0,
            "r31_0": r31_0, "r32_0": r32_0,
            "lam_hat": float(lam_hat), "rho_hat": float(rho_hat), "mu_hat": float(mu_hat),
            "kappa_hat": float(kappa_hat), "lam3_hat": float(lam3_hat),
            "r31_hat": float(r31_hat), "r32_hat": float(r32_hat),
            "ll_true": float(ll_true_m) if np.isfinite(ll_true_m) else np.nan,
            "ll_est": float(ll_est_m) if np.isfinite(ll_est_m) else np.nan,
            "iters": iters,
            "v_cap": v_cap, "tail_rel_tol": tail_rel_tol, "tail_patience": tail_patience,
            "v_min": v_min, "n_grid": n_grid,
        }
        return run_row, [], {"N": N, "ok_fit": 1, "ok_ll": 0, "runtime": runtime}

    ll_hist_po = ll_hist / N
    mono = is_monotone_nondecreasing(ll_hist, tol=1e-10)
    final_abs_dll_po = float(abs(ll_hist_po[-1] - ll_hist_po[-2])) if iters >= 2 else np.nan
    converged_flag = bool(iters >= MIN_ITERS_LL and np.isfinite(final_abs_dll_po) and final_abs_dll_po <= EPS_LL)

    traj_rows = [
        {"N": N, "m": m, "seed": seed, "iter": k, "ll_per_obs": float(llk)}
        for k, llk in enumerate(ll_hist_po, start=1)
    ]

    ll_true_po = float(ll_true_m) / N
    ll_est_po  = float(ll_est_m) / N
    delta_po   = ll_est_po - ll_true_po

    run_row = {
        "N": N, "m": m, "seed": seed,
        "fit_success": True,
        "fit_error": "",
        "runtime_sec": runtime,
        "lam0": lam0, "rho0": rho0, "mu0": mu0, "kappa0": kappa0, "lam3_0": lam3_0,
        "r31_0": r31_0, "r32_0": r32_0,
        "lam_hat": float(lam_hat), "rho_hat": float(rho_hat), "mu_hat": float(mu_hat),
        "kappa_hat": float(kappa_hat), "lam3_hat": float(lam3_hat),
        "r31_hat": float(r31_hat), "r32_hat": float(r32_hat),
        "ll_true": float(ll_true_m), "ll_est": float(ll_est_m),
        "ll_true_per_obs": ll_true_po,
        "ll_est_per_obs": ll_est_po,
        "delta_ll_per_obs": float(delta_po),
        "iters": iters,
        "ll_monotone": mono,
        "converged_flag": converged_flag,
        "final_abs_dll_per_obs": final_abs_dll_po,
        "ll_hist_len": iters,
        "v_cap": v_cap, "tail_rel_tol": tail_rel_tol, "tail_patience": tail_patience,
        "v_min": v_min, "n_grid": n_grid,
    }

    bits = {
        "N": N, "ok_fit": 1, "ok_ll": 1,
        "runtime": runtime,
        "delta_po": float(delta_po),
        "mono": float(mono),
        "conv": float(converged_flag),
        "iters": iters,
        "ll_true": float(ll_true_m),
        "ll_est": float(ll_est_m),
    }

    return run_row, traj_rows, bits


def main():
    max_workers = 4
    jobs = [(N, m) for N in Ns for m in range(M)]

    rows_runs = []
    rows_ll_traj = []
    bits_all = []

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one_job, job): job for job in jobs}

        done = 0
        for fut in as_completed(futs):
            N, m = futs[fut]
            done += 1
            try:
                run_row, traj_rows, bits = fut.result()
            except Exception as e:
                run_row = {
                    "N": N, "m": m, "seed": 10_000*m + N + 1234,
                    "fit_success": False, "fit_error": f"Worker crashed: {e}",
                    "runtime_sec": np.nan
                }
                traj_rows = []
                bits = {"N": N, "ok_fit": 0, "ok_ll": 0}

            rows_runs.append(run_row)
            rows_ll_traj.extend(traj_rows)
            bits_all.append(bits)

            print(f"Done {done}/{len(jobs)}: N={N}, m={m} | fit_success={run_row.get('fit_success')}", flush=True)

    total_runtime = time.perf_counter() - t0
    print(f"\nAll jobs done in {total_runtime/60:.2f} minutes using max_workers={max_workers}.\n", flush=True)

    df_runs = pd.DataFrame(rows_runs)
    df_traj = pd.DataFrame(rows_ll_traj)

    # --- Summary per N (matches what your plotting code expects) ---
    rows_summary = []
    for N in Ns:
        bitsN = [b for b in bits_all if b.get("N") == N and b.get("ok_ll", 0) == 1]
        runtimes = [b["runtime"] for b in bitsN if "runtime" in b]
        delta_po = [b["delta_po"] for b in bitsN if "delta_po" in b]
        iters    = [b["iters"] for b in bitsN if "iters" in b]
        conv     = [b["conv"] for b in bitsN if "conv" in b]
        mono     = [b["mono"] for b in bitsN if "mono" in b]
        ll_true  = [b["ll_true"] for b in bitsN if "ll_true" in b]
        ll_est   = [b["ll_est"] for b in bitsN if "ll_est" in b]

        fit_ok = int(sum(1 for b in bits_all if b.get("N") == N and b.get("ok_fit", 0) == 1))
        ll_ok  = int(sum(1 for b in bits_all if b.get("N") == N and b.get("ok_ll", 0) == 1))

        rows_summary.append({
            "N": N,
            # NOTE: these names are chosen so they line up with your plot function
            # If your plot expects "True_logLik_mean", set ref_col accordingly (see below).
            "True_logLik_mean": float(np.mean(ll_true)) if ll_true else np.nan,
            "Est_logLik_mean":  float(np.mean(ll_est)) if ll_est else np.nan,
            "delta_ll_per_obs_mean": float(np.mean(delta_po)) if delta_po else np.nan,
            "delta_ll_per_obs_sd":   float(np.std(delta_po, ddof=1)) if len(delta_po) > 1 else np.nan,
            "mean_time_sec":         float(np.mean(runtimes)) if runtimes else np.nan,
            "fit_ok":                fit_ok,
            "ll_ok":                 ll_ok,
            "conv_rate":             float(np.mean(conv)) if conv else np.nan,
            "mono_rate":             float(np.mean(mono)) if mono else np.nan,
            "iters_median":          float(np.median(iters)) if iters else np.nan,
            "iters_p90":             float(np.quantile(iters, 0.90)) if iters else np.nan,
            "v_cap": v_cap, "tail_rel_tol": tail_rel_tol, "tail_patience": tail_patience,
            "v_min": v_min, "n_grid": n_grid,
        })

    df_summary = pd.DataFrame(rows_summary)

    # --- Save CSVs ---
    summary_path = os.path.join(OUT_DIR, "noninv_unknown_adaptive_summary_diagnostics_parallel.csv")
    runs_path    = os.path.join(OUT_DIR, "noninv_unknown_adaptive_runs_diagnostics_parallel.csv")
    traj_path    = os.path.join(OUT_DIR, "noninv_unknown_adaptive_ll_trajectory_long_parallel.csv")

    df_summary.to_csv(summary_path, index=False)
    df_runs.to_csv(runs_path, index=False)
    df_traj.to_csv(traj_path, index=False)

    print("Saved:")
    print(" ", summary_path)
    print(" ", runs_path)
    print(" ", traj_path)

    print("\nSummary:")
    print(df_summary)


if __name__ == "__main__":
    main()
