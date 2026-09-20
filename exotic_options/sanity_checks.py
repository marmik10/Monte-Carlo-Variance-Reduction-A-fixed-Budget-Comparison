"""
Before trusting any variance-reduction numbers, verify the simulator
against known closed-form limits. If these fail, the headline results
are meaningless.
"""
import numpy as np
from pricing import (
    price_barrier_mc, price_asian_mc, bs_call_price,
    geometric_asian_call_price, RNG_SEED
)

S0, K, r, sigma, T, n_steps = 100.0, 100.0, 0.03, 0.25, 1.0, 52

print("=== Sanity check 1: barrier -> infinity should equal vanilla BS call ===")
rng = np.random.default_rng(RNG_SEED)
bs_price = bs_call_price(S0, K, r, sigma, T)
price, se, _, _ = price_barrier_mc(S0, K, B=1e6, r=r, sigma=sigma, T=T,
                                    n_steps=n_steps, n_draws=100_000,
                                    method="plain", rng=rng,
                                    continuity_correction=False)
print(f"  BS closed form:      {bs_price:.4f}")
print(f"  MC (B=inf):           {price:.4f} +/- {1.96*se:.4f}")
assert abs(price - bs_price) < 3 * se, "FAIL: barrier limit does not match BS price"
print("  PASS\n")

print("=== Sanity check 2: barrier below spot should be worthless ===")
rng = np.random.default_rng(RNG_SEED)
price, se, _, _ = price_barrier_mc(S0, K, B=95.0, r=r, sigma=sigma, T=T,
                                    n_steps=n_steps, n_draws=50_000,
                                    method="plain", rng=rng)
print(f"  MC price (B < S0):    {price:.6f} +/- {1.96*se:.6f}")
assert price < 1e-6, "FAIL: sub-spot barrier should knock out on step 1"
print("  PASS\n")

print("=== Sanity check 3: geometric Asian closed form vs its own MC ===")
rng = np.random.default_rng(RNG_SEED)
geo_price = geometric_asian_call_price(S0, K, r, sigma, T, n_steps)
# reuse asian_payoff machinery directly via price_asian_mc's control path
from pricing import simulate_gbm_paths, asian_payoff
paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, 200_000, False, rng)
geo_mc = asian_payoff(paths, K, r, T, arithmetic=False)
print(f"  Kemna-Vorst closed form: {geo_price:.4f}")
print(f"  MC estimate:             {geo_mc.mean():.4f} +/- {1.96*geo_mc.std(ddof=1)/np.sqrt(len(geo_mc)):.4f}")
assert abs(geo_mc.mean() - geo_price) < 3 * geo_mc.std(ddof=1) / np.sqrt(len(geo_mc)), \
    "FAIL: geometric Asian MC does not match Kemna-Vorst"
print("  PASS\n")

print("=== Sanity check 4: continuity correction increases knock-out prob (lowers price) ===")
rng1 = np.random.default_rng(RNG_SEED)
rng2 = np.random.default_rng(RNG_SEED)
price_corrected, _, _, _ = price_barrier_mc(S0, K, B=115.0, r=r, sigma=sigma, T=T,
                                             n_steps=n_steps, n_draws=200_000,
                                             method="plain", rng=rng1,
                                             continuity_correction=True)
price_uncorrected, _, _, _ = price_barrier_mc(S0, K, B=115.0, r=r, sigma=sigma, T=T,
                                               n_steps=n_steps, n_draws=200_000,
                                               method="plain", rng=rng2,
                                               continuity_correction=False)
print(f"  Corrected price:   {price_corrected:.4f}")
print(f"  Uncorrected price: {price_uncorrected:.4f}")
assert price_corrected < price_uncorrected, "FAIL: correction should raise knock-out probability and lower price"
print("  PASS (uncorrected grid simulation is biased HIGH -- it misses within-step crossings,\n"
      "        so it understates knock-out probability relative to true continuous monitoring)\n")

print("All sanity checks passed.\n")

print("=== Sanity check 5: dividend yield reduces call value vs q=0, as it should ===")
q_test = 0.02
price_no_div = bs_call_price(S0, K, r, sigma, T, q=0.0)
price_with_div = bs_call_price(S0, K, r, sigma, T, q=q_test)
print(f"  BS call, q=0:     {price_no_div:.4f}")
print(f"  BS call, q={q_test}:  {price_with_div:.4f}")
assert price_with_div < price_no_div, "FAIL: dividends should reduce call value"

rng = np.random.default_rng(RNG_SEED)
mc_price, mc_se, _, _ = price_barrier_mc(S0, K, B=1e6, r=r, sigma=sigma, T=T,
                                          n_steps=n_steps, n_draws=100_000,
                                          method="plain", rng=rng,
                                          continuity_correction=False, q=q_test)
print(f"  MC (B=inf, q={q_test}): {mc_price:.4f} +/- {1.96*mc_se:.4f}  (should match BS call w/ dividends)")
assert abs(mc_price - price_with_div) < 3 * mc_se, "FAIL: dividend-adjusted MC does not match dividend-adjusted BS"
print("  PASS\n")

print("All sanity checks passed.")
