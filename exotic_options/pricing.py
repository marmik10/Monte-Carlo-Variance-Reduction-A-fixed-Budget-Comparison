"""
Exotic option pricing: continuously-monitored up-and-out barrier call
(priced via a finite grid, with a bias correction for that discretization),
and a discretely-monitored arithmetic-average Asian call (which is exactly
discrete by contract design -- no correction needed there), under
Black-Scholes/GBM.

Variance reduction techniques implemented:
  - Antithetic variates
  - Control variates (regression-estimated beta, not assumed beta=1)
  - Barrier discretization bias correction (Broadie-Glasserman-Kou
    continuity correction)

All comparisons are done at a FIXED total random-number budget, so
"plain MC" with N paths is compared against antithetic MC using N/2
independent draws (each expanded into a +/- pair), not N vs N.
"""

import numpy as np
from scipy.stats import norm

RNG_SEED = 12345
BGK_ETA = 0.5825972289  # -zeta(1/2)/sqrt(2*pi), Broadie-Glasserman-Kou constant


# ----------------------------------------------------------------------
# Closed-form benchmarks
# ----------------------------------------------------------------------

def bs_call_price(S0, K, r, sigma, T, q=0.0):
    """Black-Scholes European call price with continuous dividend yield q.
    Used as (a) a sanity check and (b) the analytic control-variate mean
    for the barrier option. q=0 recovers the standard non-dividend formula."""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def geometric_asian_call_price(S0, K, r, sigma, T, n_steps, q=0.0):
    """Kemna-Vorst (1990) closed form for a discretely-monitored
    geometric-average Asian call, with continuous dividend yield q. This
    is the control variate for the arithmetic-average Asian option below.

    The geometric average of n lognormal observations is itself
    lognormal, so this reduces to a Black-Scholes-style formula with
    adjusted volatility and drift.
    """
    n = n_steps
    sigma_hat = sigma * np.sqrt((n + 1) * (2 * n + 1) / (6 * n ** 2))
    rho = 0.5 * (r - q - 0.5 * sigma ** 2) * (n + 1) / n + 0.5 * sigma_hat ** 2
    d1 = (np.log(S0 / K) + (rho + 0.5 * sigma_hat ** 2) * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)
    return np.exp(-r * T) * (S0 * np.exp(rho * T) * norm.cdf(d1) - K * norm.cdf(d2))


# ----------------------------------------------------------------------
# Path simulation
# ----------------------------------------------------------------------

def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_draws, antithetic, rng, q=0.0):
    """Simulate GBM paths using the exact lognormal transition (no
    Euler discretization error in the marginal distribution). Drift is
    risk-neutral: (r - q - 0.5*sigma^2), so an underlying that pays a
    continuous dividend yield q grows more slowly under the pricing
    measure -- q=0 recovers the no-dividend case.

    If antithetic=True, n_draws independent Brownian increment sets are
    each expanded into a (+Z, -Z) pair, returning 2*n_draws paths total.
    This keeps the random-number budget comparable to plain MC with
    2*n_draws paths.
    """
    dt = T / n_steps
    Z = rng.standard_normal((n_draws, n_steps))

    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)

    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_paths)
    S = np.concatenate([np.full((S.shape[0], 1), S0), S], axis=1)
    return S  # shape (n_paths, n_steps + 1)


# ----------------------------------------------------------------------
# Barrier option: discretely-monitored up-and-out call
# ----------------------------------------------------------------------

def _corrected_barrier(B, sigma, dt, is_up):
    """Broadie-Glasserman-Kou continuity correction.

    This corrects for simulating a CONTINUOUSLY monitored barrier option
    on a finite time grid. Checking the barrier only at grid points misses
    within-step excursions, so knock-out probability is underestimated and
    the naive discretized price is biased HIGH relative to the true
    continuous-monitoring price.

    Fix: move the effective barrier TOWARD the spot price (easier to
    trigger in the coarse simulation), which compensates for the missed
    crossings. Up-barriers shift down, down-barriers shift up -- in both
    cases, toward S0.
    """
    sign = -1.0 if is_up else 1.0
    return B * np.exp(sign * BGK_ETA * sigma * np.sqrt(dt))


def barrier_payoff(paths, K, B, r, T, sigma, n_steps, continuity_correction):
    """Discounted payoff of an up-and-out call for each path."""
    dt = T / n_steps
    B_eff = _corrected_barrier(B, sigma, dt, is_up=True) if continuity_correction else B
    knocked_out = np.any(paths >= B_eff, axis=1)
    ST = paths[:, -1]
    payoff = np.maximum(ST - K, 0.0)
    payoff[knocked_out] = 0.0
    return np.exp(-r * T) * payoff


def price_barrier_mc(S0, K, B, r, sigma, T, n_steps, n_draws, method, rng,
                      continuity_correction=True, q=0.0):
    """
    method in {"plain", "antithetic", "control", "antithetic_control"}
    Returns (price, std_error, variance, n_paths_used).
    """
    antithetic = method in ("antithetic", "antithetic_control")
    use_cv = method in ("control", "antithetic_control")

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_draws, antithetic, rng, q=q)
    Y = barrier_payoff(paths, K, B, r, T, sigma, n_steps, continuity_correction)

    if not use_cv:
        price = Y.mean()
        var = Y.var(ddof=1)
        se = np.sqrt(var / len(Y))
        return price, se, var, len(Y)

    # Control variate: discounted vanilla call payoff on the SAME paths.
    ST = paths[:, -1]
    X = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    EX = bs_call_price(S0, K, r, sigma, T, q=q)

    cov = np.cov(Y, X, ddof=1)[0, 1]
    varX = X.var(ddof=1)
    beta = cov / varX if varX > 0 else 0.0

    Y_cv = Y - beta * (X - EX)
    price = Y_cv.mean()
    var = Y_cv.var(ddof=1)
    se = np.sqrt(var / len(Y_cv))
    return price, se, var, len(Y_cv)


# ----------------------------------------------------------------------
# Asian option: discretely-monitored arithmetic-average call
# ----------------------------------------------------------------------

def asian_payoff(paths, K, r, T, arithmetic=True):
    obs = paths[:, 1:]  # exclude S0 from the averaging window
    avg = obs.mean(axis=1) if arithmetic else np.exp(np.log(obs).mean(axis=1))
    payoff = np.maximum(avg - K, 0.0)
    return np.exp(-r * T) * payoff


def price_asian_mc(S0, K, r, sigma, T, n_steps, n_draws, method, rng, q=0.0):
    """
    method in {"plain", "antithetic", "control", "antithetic_control"}
    """
    antithetic = method in ("antithetic", "antithetic_control")
    use_cv = method in ("control", "antithetic_control")

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_draws, antithetic, rng, q=q)
    Y = asian_payoff(paths, K, r, T, arithmetic=True)

    if not use_cv:
        price = Y.mean()
        var = Y.var(ddof=1)
        se = np.sqrt(var / len(Y))
        return price, se, var, len(Y)

    X = asian_payoff(paths, K, r, T, arithmetic=False)
    EX = geometric_asian_call_price(S0, K, r, sigma, T, n_steps, q=q)

    cov = np.cov(Y, X, ddof=1)[0, 1]
    varX = X.var(ddof=1)
    beta = cov / varX if varX > 0 else 0.0

    Y_cv = Y - beta * (X - EX)
    price = Y_cv.mean()
    var = Y_cv.var(ddof=1)
    se = np.sqrt(var / len(Y_cv))
    return price, se, var, len(Y_cv)
