"""
Validation for barrier_state_dependent.py. Given the track record in
this project (two real bugs in the constant-shift IS code before it was
trusted), nothing here gets used in the main comparison until it passes
these checks.
"""
import numpy as np
from pricing import simulate_gbm_paths
from barrier_advanced import bridge_survival_probability, up_and_out_call_closed_form
from barrier_state_dependent import (
    no_touch_discounted_price,
    price_barrier_bridge_survival_cv,
    simulate_gbm_paths_state_dependent_tilt,
    price_barrier_state_dependent_is,
)

S0, K, r, sigma, T, n_steps, q = 100.0, 100.0, 0.03, 0.25, 1.0, 52, 0.0
B = 115.0

print("=== Check 1: no-touch closed form vs high-N survival-probability MC ===")
rng = np.random.default_rng(555)
paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, 500_000, False, rng, q=q)
dt = T / n_steps
survival = bridge_survival_probability(paths, B, sigma, dt)
mc_price = (np.exp(-r * T) * survival).mean()
mc_se = (np.exp(-r * T) * survival).std(ddof=1) / np.sqrt(len(survival))
cf_price = no_touch_discounted_price(S0, B, r, sigma, T, q=q)
print(f"  Closed form:        {cf_price:.5f}")
print(f"  Bridge MC (N=500k): {mc_price:.5f} +/- {1.96*mc_se:.5f}")
diff = abs(cf_price - mc_price)
within_ci = diff < 3 * mc_se
print(f"  Difference: {diff:.5f}  ({'within' if within_ci else 'OUTSIDE'} 3 std errors)")
if not within_ci:
    print("  WARNING: no-touch closed form does not match simulation -- NOT reliable, do not use.")
else:
    print("  PASS")

print("\n=== Check 2: survival-based control variate is unbiased, and its actual correlation ===")
rng = np.random.default_rng(1)
price_cv, se_cv, var_cv, n, corr = price_barrier_bridge_survival_cv(
    S0, K, B, r, sigma, T, n_steps, n_draws=100_000, rng=rng, q=q)
cf_call_price = up_and_out_call_closed_form(S0, K, B, r, sigma, T, q=q)
print(f"  Closed-form call price:         {cf_call_price:.4f}")
print(f"  Survival-CV estimate:           {price_cv:.4f} +/- {1.96*se_cv:.4f}")
agrees = abs(price_cv - cf_call_price) < 4 * se_cv
print(f"  Agrees with closed form: {agrees}")
print(f"  Empirical correlation(Y, survival): {corr:.4f}")
print(f"  (this is the number that actually matters -- not assumed, measured)")

print("\n=== Check 3: state-dependent tilt is unbiased, at several 'strength' settings ===")
rng_baseline = np.random.default_rng(2)
from barrier_advanced import price_barrier_bridge_mc
baseline_price, baseline_se, baseline_var, _ = price_barrier_bridge_mc(
    S0, K, B, r, sigma, T, n_steps, n_draws=200_000, method="plain", rng=rng_baseline, q=q)
print(f"  Baseline (plain bridge, N=200k): price={baseline_price:.4f} +/- {1.96*baseline_se:.4f}")

all_unbiased = True
for strength in [0.0, 0.3, 0.6, 1.0]:
    rng = np.random.default_rng(3)
    price, se, var, _ = price_barrier_state_dependent_is(
        S0, K, B, r, sigma, T, n_steps, n_draws=100_000, rng=rng, q=q, strength=strength)
    agrees = abs(price - baseline_price) < 4 * (se + baseline_se)
    all_unbiased = all_unbiased and agrees
    print(f"  strength={strength:.1f}: price={price:.4f} +/- {1.96*se:.4f}  "
          f"var={var:.4f}  unbiased={agrees}")
assert all_unbiased, "FAIL: state-dependent tilt should be unbiased at every strength"
print("  PASS: unbiased at every tested strength.")

print("\n=== Check 4: variance comparison, state-dependent tilt vs constant-shift IS ===")
from barrier_advanced import calibrate_drift_shift, price_barrier_importance_sampling
best_const_shift, _ = calibrate_drift_shift(S0, K, B, r, sigma, T, n_steps, r_state=777,
                                             use_bridge=True, q=q, pilot_n=20_000)
rng = np.random.default_rng(10)
_, _, var_const, _ = price_barrier_importance_sampling(
    S0, K, B, r, sigma, T, n_steps, n_draws=100_000, rng=rng,
    drift_shift=best_const_shift, use_bridge=True, q=q)

print(f"  Constant-shift IS (shift={best_const_shift:+.2f}): variance = {var_const:.4f}")
for strength in [0.3, 0.6, 1.0]:
    rng = np.random.default_rng(10)
    _, _, var_sd, _ = price_barrier_state_dependent_is(
        S0, K, B, r, sigma, T, n_steps, n_draws=100_000, rng=rng, q=q, strength=strength)
    print(f"  State-dependent tilt (strength={strength:.1f}):    variance = {var_sd:.4f}  "
          f"({'better' if var_sd < var_const else 'worse'} than constant-shift)")

print("\nAll checks complete.")
