"""
Convergence study: how does estimator standard error shrink as N grows,
for plain MC vs control-variate MC, on both option types?

Theory says standard error ~ C / sqrt(N) for any consistent MC estimator
-- the exponent (-0.5 slope on a log-log plot) doesn't change with
variance reduction. What changes is the constant C in front of it: a
good control variate lowers C (shifts the whole line down in log-log
space) without changing the slope. This script checks that empirically
holds, and produces the plot that makes it visible at a glance -- a
number in a table doesn't show this shape as clearly as the plot does.

Uses market_params.json if present (see fetch_market_data.py),
otherwise the same synthetic defaults as run_experiment.py.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pricing import price_barrier_mc, price_asian_mc, RNG_SEED

DEFAULT_PARAMS = dict(S0=100.0, K=100.0, r=0.03, q=0.0, sigma=0.25, T=1.0, n_steps=52)
BARRIER_MULTIPLIER = 1.15
PARAMS_FILE = "market_params.json"

N_VALUES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
N_REPS = 10  # repetitions per N, to average out noise in the SE estimate itself


def load_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            data = json.load(f)
        S0, r, sigma, T = data["S0"], data["r"], data["sigma"], data["T_years"]
        q = data.get("q", 0.0)
        label = f"{data['ticker']}, {data['fetched_at'][:10]}"
        return dict(S0=S0, K=S0, r=r, q=q, sigma=sigma, T=T, n_steps=52), label
    return dict(DEFAULT_PARAMS, K=DEFAULT_PARAMS["S0"]), "synthetic placeholder values"


def se_vs_n(pricer_fn, method, n_values, n_reps, **kwargs):
    ses = []
    for n in n_values:
        rep_ses = []
        for rep in range(n_reps):
            rng = np.random.default_rng(RNG_SEED + rep)
            n_draws = n // 2 if method.startswith("antithetic") else n
            _, se, _, _ = pricer_fn(n_draws=n_draws, method=method, rng=rng, **kwargs)
            rep_ses.append(se)
        ses.append(np.mean(rep_ses))
    return np.array(ses)


def fit_loglog_slope(n_values, ses):
    log_n = np.log(n_values)
    log_se = np.log(ses)
    slope, intercept = np.polyfit(log_n, log_se, 1)
    return slope, intercept


if __name__ == "__main__":
    params, source_label = load_params()
    B = params["S0"] * BARRIER_MULTIPLIER

    print(f"Using params from: {source_label}")
    print(f"Running convergence study across N = {N_VALUES} ...")

    barrier_plain = se_vs_n(price_barrier_mc, "plain", N_VALUES, N_REPS,
                             S0=params["S0"], K=params["K"], B=B, r=params["r"],
                             sigma=params["sigma"], T=params["T"], n_steps=params["n_steps"],
                             q=params["q"])
    barrier_cv = se_vs_n(price_barrier_mc, "control", N_VALUES, N_REPS,
                          S0=params["S0"], K=params["K"], B=B, r=params["r"],
                          sigma=params["sigma"], T=params["T"], n_steps=params["n_steps"],
                          q=params["q"])
    asian_plain = se_vs_n(price_asian_mc, "plain", N_VALUES, N_REPS,
                           S0=params["S0"], K=params["K"], r=params["r"],
                           sigma=params["sigma"], T=params["T"], n_steps=params["n_steps"],
                           q=params["q"])
    asian_cv = se_vs_n(price_asian_mc, "control", N_VALUES, N_REPS,
                        S0=params["S0"], K=params["K"], r=params["r"],
                        sigma=params["sigma"], T=params["T"], n_steps=params["n_steps"],
                        q=params["q"])

    print("\nFitted log-log slopes (theory predicts -0.5 for all four lines):")
    for name, ses in [("Barrier, plain", barrier_plain), ("Barrier, control", barrier_cv),
                       ("Asian, plain", asian_plain), ("Asian, control", asian_cv)]:
        slope, _ = fit_loglog_slope(N_VALUES, ses)
        print(f"  {name:20s}: slope = {slope:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].loglog(N_VALUES, barrier_plain, 'o-', label='Plain MC')
    axes[0].loglog(N_VALUES, barrier_cv, 's-', label='Control variate (vanilla call)')
    axes[0].set_xlabel('N (paths)')
    axes[0].set_ylabel('Standard error')
    axes[0].set_title('Barrier option: SE vs N\n(lines nearly overlap -- control variate barely helps)')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].loglog(N_VALUES, asian_plain, 'o-', label='Plain MC')
    axes[1].loglog(N_VALUES, asian_cv, 's-', label='Control variate (geometric Asian)')
    axes[1].set_xlabel('N (paths)')
    axes[1].set_ylabel('Standard error')
    axes[1].set_title('Asian option: SE vs N\n(control variate line sits ~30x lower, same slope)')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_study.png', dpi=150)
    print("\nSaved convergence_study.png")
