"""
Compares ALL barrier techniques attempted in this project, at the same
fixed compute budget, so the improvement from the domain-specific fixes
is visible against the naive baseline from run_experiment.py:

  1. Naive discrete indicator + continuity correction (original approach)
  2. Naive discrete indicator + vanilla-call control variate (original approach)
  3. Brownian bridge conditional expectation (Rao-Blackwellization)
  4. Brownian bridge + calibrated importance sampling

Uses market_params.json if present, else the same synthetic defaults
used elsewhere in this project.
"""
import os
import json
import time
import numpy as np
import pandas as pd

from pricing import price_barrier_mc
from barrier_advanced import price_barrier_bridge_mc, price_barrier_importance_sampling, calibrate_drift_shift
from barrier_state_dependent import (
    price_barrier_state_dependent_is, calibrate_tilt_strength,
    price_barrier_bridge_survival_cv,
)

DEFAULT_PARAMS = dict(S0=100.0, K=100.0, r=0.03, q=0.0, sigma=0.25, T=1.0, n_steps=52)
BARRIER_MULTIPLIER = 1.15
PARAMS_FILE = "market_params.json"
N_TOTAL_DRAWS = 200_000
N_REPS = 20
RNG_SEED = 12345


def load_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            data = json.load(f)
        S0, r, sigma, T = data["S0"], data["r"], data["sigma"], data["T_years"]
        q = data.get("q", 0.0)
        label = f"{data['ticker']}, {data['fetched_at'][:10]}"
        print(f"Loaded CALIBRATED market params: {label}")
        return dict(S0=S0, K=S0, r=r, q=q, sigma=sigma, T=T, n_steps=52), label
    print(f"{PARAMS_FILE} not found -- using SYNTHETIC placeholder params.")
    return dict(DEFAULT_PARAMS, K=DEFAULT_PARAMS["S0"]), "synthetic placeholder values"


def repeated_run(fn, n_reps, **kwargs):
    prices, variances, times = [], [], []
    for rep in range(n_reps):
        rng = np.random.default_rng(RNG_SEED + rep)
        t0 = time.perf_counter()
        price, se, var, n_paths = fn(rng=rng, **kwargs)
        elapsed = time.perf_counter() - t0
        prices.append(price)
        variances.append(var)
        times.append(elapsed)
    return np.mean(prices), np.mean(variances), np.mean(times)


if __name__ == "__main__":
    params, source_label = load_params()
    S0, K, r, q, sigma, T, n_steps = (
        params["S0"], params["K"], params["r"], params["q"],
        params["sigma"], params["T"], params["n_steps"]
    )
    B = S0 * BARRIER_MULTIPLIER

    print(f"\nUP-AND-OUT BARRIER CALL comparison  (S0={S0:.2f}, K={K:.2f}, B={B:.2f}, "
          f"r={r*100:.2f}%, q={q*100:.2f}%, sigma={sigma*100:.2f}%, T={T:.2f}y)  [{source_label}]\n")

    # 1. Naive + continuity correction (from run_experiment.py's "plain")
    price1, var1, time1 = repeated_run(
        price_barrier_mc, N_REPS, S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        n_draws=N_TOTAL_DRAWS, method="plain", continuity_correction=True, q=q)

    # 2. Naive + vanilla-call control variate (from run_experiment.py's "control")
    price2, var2, time2 = repeated_run(
        price_barrier_mc, N_REPS, S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        n_draws=N_TOTAL_DRAWS, method="control", continuity_correction=True, q=q)

    # 3. Brownian bridge conditional expectation (all four method variants,
    #    since the vanilla-call control variate may behave very differently
    #    now that it's being applied to a smooth payoff instead of a
    #    discontinuous one)
    price3a, var3a, time3a = repeated_run(
        price_barrier_bridge_mc, N_REPS, S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        n_draws=N_TOTAL_DRAWS, method="plain", q=q)
    price3b, var3b, time3b = repeated_run(
        price_barrier_bridge_mc, N_REPS, S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        n_draws=N_TOTAL_DRAWS, method="antithetic", q=q)
    price3c, var3c, time3c = repeated_run(
        price_barrier_bridge_mc, N_REPS, S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        n_draws=N_TOTAL_DRAWS, method="control", q=q)
    price3d, var3d, time3d = repeated_run(
        price_barrier_bridge_mc, N_REPS, S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        n_draws=N_TOTAL_DRAWS, method="antithetic_control", q=q)

    # 4. Brownian bridge + calibrated importance sampling
    best_shift, _ = calibrate_drift_shift(S0, K, B, r, sigma, T, n_steps, r_state=777,
                                           use_bridge=True, q=q, pilot_n=20_000)
    print(f"Calibrated drift shift for importance sampling: {best_shift:+.2f}\n")

    def bridge_is_fn(rng, **kw):
        return price_barrier_importance_sampling(drift_shift=best_shift, use_bridge=True,
                                                   rng=rng, **kw)

    price4, var4, time4 = repeated_run(
        bridge_is_fn, N_REPS, S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        n_draws=N_TOTAL_DRAWS, q=q)

    # 5. Survival-based control variate (on top of bridge, no IS)
    def survival_cv_fn(rng, **kw):
        price, se, var, n, corr = price_barrier_bridge_survival_cv(rng=rng, **kw)
        return price, se, var, n

    price5, var5, time5 = repeated_run(
        survival_cv_fn, N_REPS, S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        n_draws=N_TOTAL_DRAWS, q=q)

    rng_corr = np.random.default_rng(RNG_SEED)
    _, _, _, _, measured_corr = price_barrier_bridge_survival_cv(
        S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps, n_draws=N_TOTAL_DRAWS, rng=rng_corr, q=q)

    # 6. State-dependent (corridor-pinning) tilt -- calibrated strength
    best_strength, strength_sweep = calibrate_tilt_strength(
        S0, K, B, r, sigma, T, n_steps, r_state=888, q=q, pilot_n=20_000)
    print(f"Calibrated pinning strength for state-dependent tilt: {best_strength:.2f}\n")

    def state_dependent_fn(rng, **kw):
        return price_barrier_state_dependent_is(strength=best_strength, use_bridge=True, rng=rng, **kw)

    price6, var6, time6 = repeated_run(
        state_dependent_fn, N_REPS, S0=S0, K=K, B=B, r=r, sigma=sigma, T=T, n_steps=n_steps,
        n_draws=N_TOTAL_DRAWS, q=q)

    rows = [
        dict(method="1. Naive discrete + continuity correction (original)",
             price=round(price1, 4), variance=round(var1, 5),
             variance_reduction_pct=0.0, time_sec=round(time1, 3)),
        dict(method="2. Naive discrete + vanilla-call control variate (original)",
             price=round(price2, 4), variance=round(var2, 5),
             variance_reduction_pct=round(100 * (1 - var2 / var1), 2), time_sec=round(time2, 3)),
        dict(method="3a. Brownian bridge, plain",
             price=round(price3a, 4), variance=round(var3a, 5),
             variance_reduction_pct=round(100 * (1 - var3a / var1), 2), time_sec=round(time3a, 3)),
        dict(method="3b. Brownian bridge + antithetic",
             price=round(price3b, 4), variance=round(var3b, 5),
             variance_reduction_pct=round(100 * (1 - var3b / var1), 2), time_sec=round(time3b, 3)),
        dict(method="3c. Brownian bridge + vanilla-call control variate",
             price=round(price3c, 4), variance=round(var3c, 5),
             variance_reduction_pct=round(100 * (1 - var3c / var1), 2), time_sec=round(time3c, 3)),
        dict(method="3d. Brownian bridge + antithetic + control variate",
             price=round(price3d, 4), variance=round(var3d, 5),
             variance_reduction_pct=round(100 * (1 - var3d / var1), 2), time_sec=round(time3d, 3)),
        dict(method=f"4. Brownian bridge + importance sampling (shift={best_shift:+.2f})",
             price=round(price4, 4), variance=round(var4, 5),
             variance_reduction_pct=round(100 * (1 - var4 / var1), 2), time_sec=round(time4, 3)),
        dict(method=f"5. Brownian bridge + survival-based control variate (corr={measured_corr:.2f})",
             price=round(price5, 4), variance=round(var5, 5),
             variance_reduction_pct=round(100 * (1 - var5 / var1), 2), time_sec=round(time5, 3)),
        dict(method=f"6. Brownian bridge + state-dependent corridor-pinning tilt (strength={best_strength:.2f})",
             price=round(price6, 4), variance=round(var6, 5),
             variance_reduction_pct=round(100 * (1 - var6 / var1), 2), time_sec=round(time6, 3)),
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv("barrier_advanced_results.csv", index=False)
    print("\nSaved barrier_advanced_results.csv")
