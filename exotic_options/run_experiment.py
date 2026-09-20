"""
Fair-budget comparison of plain MC vs antithetic vs control variate vs
both combined, for the barrier and Asian options.

"Fair budget" = every method uses the SAME number of underlying random
draws (n_draws_total independent standard normal path-increments), not
the same number of *paths*. Antithetic variates get n_draws_total/2
independent draws, each expanded into a +/- pair -> n_draws_total paths.
Plain MC gets n_draws_total independent draws -> n_draws_total paths.
This is the only fair comparison; comparing N antithetic paths to N
plain paths overstates the antithetic benefit by roughly 2x because it
silently uses twice the random numbers.

Market inputs: if market_params.json exists (written by
fetch_market_data.py after you run it locally), S0/r/sigma/T are loaded
from there instead of the synthetic placeholder defaults below, and the
run is labeled with the ticker/date it came from. Barrier level is set
as a fixed multiple of spot (not a fixed dollar level) so it's a
sensible barrier regardless of what ticker/spot you calibrate to.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from pricing import price_barrier_mc, price_asian_mc, RNG_SEED

# ---------------------------------------------------------------------
# Synthetic defaults, used only if market_params.json is not found.
# ---------------------------------------------------------------------
DEFAULT_PARAMS = dict(S0=100.0, K=100.0, r=0.03, q=0.0, sigma=0.25, T=1.0, n_steps=52)
BARRIER_MULTIPLIER = 1.15   # barrier set 15% above spot, up-and-out
K_MULTIPLIER = 1.0          # strike set at-the-money, i.e. K = S0

PARAMS_FILE = "market_params.json"
N_TOTAL_DRAWS = 200_000  # total random path-increment sets, fixed across methods
N_REPS = 20               # independent repetitions, to measure variance of the estimator itself


def load_params():
    """Use real calibrated inputs if fetch_market_data.py has been run
    locally and produced market_params.json; otherwise fall back to the
    synthetic defaults and say so explicitly, so nobody mistakes one
    for the other."""
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            data = json.load(f)
        S0 = data["S0"]
        r = data["r"]
        q = data.get("q", 0.0)  # older market_params.json files may predate the q field
        sigma = data["sigma"]
        T = data["T_years"]
        source_label = f"{data['ticker']} spot as of {data['fetched_at'][:10]}, expiry {data['expiry_used']}"
        print(f"Loaded CALIBRATED market params from {PARAMS_FILE}: {source_label}")
        return dict(S0=S0, K=S0 * K_MULTIPLIER, r=r, q=q, sigma=sigma, T=T, n_steps=52), source_label
    else:
        print(f"{PARAMS_FILE} not found -- using SYNTHETIC placeholder params. "
              f"Run fetch_market_data.py locally and drop market_params.json "
              f"next to this script to use real calibrated inputs instead.")
        p = dict(DEFAULT_PARAMS)
        return p, "synthetic placeholder values"


def run_method(pricer_fn, method, n_reps, **kwargs):
    prices, ses, variances, times = [], [], [], []
    for rep in range(n_reps):
        rng = np.random.default_rng(RNG_SEED + rep)
        n_draws = N_TOTAL_DRAWS // 2 if method.startswith("antithetic") else N_TOTAL_DRAWS
        t0 = time.perf_counter()
        price, se, var, n_paths = pricer_fn(n_draws=n_draws, method=method, rng=rng, **kwargs)
        elapsed = time.perf_counter() - t0
        prices.append(price)
        ses.append(se)
        variances.append(var)
        times.append(elapsed)
    return np.array(prices), np.array(ses), np.array(variances), np.array(times)


def summarize(name, prices, ses, variances, times, baseline_var=None, baseline_time=None):
    mean_price = prices.mean()
    mean_se = ses.mean()
    mean_var = variances.mean()
    mean_time = times.mean()
    ci_halfwidth = 1.96 * mean_se

    row = {
        "method": name,
        "price": round(mean_price, 4),
        "mc_std_error": round(mean_se, 5),
        "95%_CI_halfwidth": round(ci_halfwidth, 5),
        "per_path_variance": round(mean_var, 5),
        "avg_time_sec": round(mean_time, 4),
    }
    if baseline_var is not None:
        row["variance_reduction_%"] = round(100 * (1 - mean_var / baseline_var), 2)
    if baseline_time is not None and baseline_var is not None:
        eff = (baseline_var / mean_var) / (mean_time / baseline_time)
        row["efficiency_ratio_(var_speedup/time_cost)"] = round(eff, 3)
    return row


def run_suite(label, pricer_fn, **kwargs):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    methods = ["plain", "antithetic", "control", "antithetic_control"]
    results = {}
    for m in methods:
        results[m] = run_method(pricer_fn, m, N_REPS, **kwargs)

    baseline_var = results["plain"][2].mean()
    baseline_time = results["plain"][3].mean()

    rows = []
    for m in methods:
        prices, ses, variances, times = results[m]
        rows.append(summarize(m, prices, ses, variances, times, baseline_var, baseline_time))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    params, source_label = load_params()
    S0, K, r, q, sigma, T, n_steps = (
        params["S0"], params["K"], params["r"], params["q"],
        params["sigma"], params["T"], params["n_steps"]
    )
    B = S0 * BARRIER_MULTIPLIER

    barrier_df = run_suite(
        f"UP-AND-OUT BARRIER CALL  (S0={S0:.2f}, K={K:.2f}, B={B:.2f}, r={r*100:.2f}%, "
        f"q={q*100:.2f}%, sigma={sigma*100:.2f}%, T={T:.2f}y, weekly grid)  [{source_label}]",
        price_barrier_mc,
        S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        continuity_correction=True, q=q,
    )

    asian_df = run_suite(
        f"ARITHMETIC-AVERAGE ASIAN CALL  (S0={S0:.2f}, K={K:.2f}, r={r*100:.2f}%, "
        f"q={q*100:.2f}%, sigma={sigma*100:.2f}%, T={T:.2f}y, weekly grid)  [{source_label}]",
        price_asian_mc,
        S0=S0, K=K, r=r, sigma=sigma, T=T, n_steps=n_steps, q=q,
    )

    barrier_df.to_csv("barrier_results.csv", index=False)
    asian_df.to_csv("asian_results.csv", index=False)
    print("\nSaved barrier_results.csv and asian_results.csv")

