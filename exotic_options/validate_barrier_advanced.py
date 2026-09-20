"""
Validation for barrier_advanced.py, BEFORE trusting any of it in the
main experiment. The closed-form formula in particular was reconstructed
from memory and needs to earn trust against simulation, not be assumed
correct because it looks like a textbook formula.
"""
import numpy as np
from pricing import bs_call_price
from barrier_advanced import (
    up_and_out_call_closed_form,
    price_barrier_bridge_mc,
    bridge_survival_probability,
    price_barrier_importance_sampling,
    simulate_gbm_paths_tilted,
    calibrate_drift_shift,
)
from pricing import simulate_gbm_paths

S0, K, r, sigma, T, n_steps, q = 100.0, 100.0, 0.03, 0.25, 1.0, 52, 0.0
B = 115.0

print("=== Check 1: closed-form up-and-out price vs high-N Brownian-bridge MC ===")
rng = np.random.default_rng(999)
price_mc, se_mc, _, _ = price_barrier_bridge_mc(S0, K, B, r, sigma, T, n_steps,
                                                 n_draws=500_000, method="plain", rng=rng, q=q)
price_cf = up_and_out_call_closed_form(S0, K, B, r, sigma, T, q=q)
print(f"  Closed form:          {price_cf:.4f}")
print(f"  Bridge MC (N=500k):   {price_mc:.4f} +/- {1.96*se_mc:.4f}")
diff = abs(price_cf - price_mc)
within_ci = diff < 3 * se_mc
print(f"  Difference: {diff:.4f}  ({'within' if within_ci else 'OUTSIDE'} 3 std errors)")
if not within_ci:
    print("  WARNING: closed-form formula does NOT match simulation. "
          "Treating the closed form as unreliable -- do not use it elsewhere "
          "until this is resolved.")
else:
    print("  PASS: closed-form formula validated against independent simulation.")

print("\n=== Check 2: closed-form up-and-out price vs high-N NAIVE discrete MC ===")
print("    (expect naive discrete to be slightly HIGHER than continuous closed form,")
print("     since discrete monitoring misses some crossings)")
rng = np.random.default_rng(998)
paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, 500_000, False, rng, q=q)
knocked_out = np.any(paths >= B, axis=1)
ST = paths[:, -1]
payoff = np.maximum(ST - K, 0.0)
payoff[knocked_out] = 0.0
naive_price = (np.exp(-r*T) * payoff).mean()
naive_se = (np.exp(-r*T) * payoff).std(ddof=1) / np.sqrt(len(payoff))
print(f"  Naive discrete MC:    {naive_price:.4f} +/- {1.96*naive_se:.4f}")
print(f"  Closed form (cont.):  {price_cf:.4f}")
print(f"  Naive > continuous?   {naive_price > price_cf}  (expected: True)")

print("\n=== Check 3: Brownian bridge survival prob is between 0 and 1, and 1 when far from barrier ===")
rng = np.random.default_rng(1)
paths = simulate_gbm_paths(50.0, r, sigma, T, n_steps, 1000, False, rng, q=q)  # spot far below B=115
surv = bridge_survival_probability(paths, B, sigma, T/n_steps)
print(f"  Spot=50 (far from B=115): min survival={surv.min():.6f}, max={surv.max():.6f}, mean={surv.mean():.6f}")
assert (surv >= 0).all() and (surv <= 1).all(), "FAIL: survival probability out of [0,1] range"
assert surv.mean() > 0.99, "FAIL: paths far from barrier should almost always survive"
print("  PASS")

print("\n=== Check 4: importance sampling stays unbiased regardless of drift shift sign/size ===")
print("    (this is the real invariant to test -- NOT a claimed direction for variance,")
print("     since the naive 'shift away from barrier' argument turned out to be wrong;")
print("     see module docstring and calibrate_drift_shift() for the honest version)")
rng1 = np.random.default_rng(42)
plain_price, plain_se, plain_var, _ = price_barrier_bridge_mc(
    S0, K, B, r, sigma, T, n_steps, n_draws=100_000, method="plain", rng=rng1, q=q)

all_unbiased = True
for shift in [-0.10, 0.05, 0.10, 0.20]:
    rng2 = np.random.default_rng(42)
    is_price, is_se, is_var, _ = price_barrier_importance_sampling(
        S0, K, B, r, sigma, T, n_steps, n_draws=100_000, rng=rng2,
        drift_shift=shift, use_bridge=True, q=q)
    agrees = abs(plain_price - is_price) < 4 * (plain_se + is_se)
    all_unbiased = all_unbiased and agrees
    print(f"  shift={shift:+.2f}: price={is_price:.4f} (plain={plain_price:.4f}), "
          f"var={is_var:.4f} (plain={plain_var:.4f}), unbiased={agrees}")
assert all_unbiased, "FAIL: importance sampling should be unbiased regardless of shift chosen"
print("  PASS: reweighted estimator matches plain bridge MC at every tested shift.")

print("\n=== Check 5: calibrate_drift_shift() finds a shift at least as good as no shift ===")
best_shift, sweep_results = calibrate_drift_shift(
    S0, K, B, r, sigma, T, n_steps, r_state=123, use_bridge=True, q=q, pilot_n=20_000)
print(f"  Sweep results (shift, variance): {[(s, round(v,3)) for s, v in sweep_results]}")
print(f"  Best shift found: {best_shift:+.2f}")
baseline_var = dict(sweep_results)[0.0]
best_var = dict(sweep_results)[best_shift]
print(f"  Best variance {best_var:.4f} <= baseline (shift=0) variance {baseline_var:.4f}: "
      f"{best_var <= baseline_var}")
assert best_var <= baseline_var, "FAIL: calibrated shift should be no worse than no shift"
print("  PASS")

print("\nAll barrier_advanced.py validation checks complete.")
