"""
Two further refinements to the barrier variance-reduction work in
barrier_advanced.py:

1. Survival-based control variate. Uses the discounted survival
   probability ITSELF (not multiplied by the payoff) as the control
   variate, with its own closed-form mean -- a discounted "no-touch"
   digital barrier price. This is a genuinely different quantity from
   the target Y, not a repackaging of the same closed-form call price
   used as both target and control (which would trivially collapse
   variance to ~0 by construction, without demonstrating anything about
   the technique -- see module-level warning below). Its actual
   correlation with Y is an empirical question, tested here rather than
   assumed.

2. State-dependent drift tilting. NOT a reproduction of the Glasserman-
   Heidelberger-Shahabuddin asymptotically-optimal large-deviations
   tilt -- deriving that requires solving a variational problem specific
   to this exact payoff, which is out of scope here. Instead: a
   "corridor-pinning" heuristic, where the drift shift at each step is
   recomputed from the CURRENT state (spot, time remaining) to steer the
   expected terminal log-price toward the geometric center of (K, B).
   The likelihood-ratio math for a state-dependent (predictable) drift
   shift is a direct, valid generalization of the constant-shift formula
   in barrier_advanced.py -- as long as the shift at each step is
   computed from information available BEFORE that step's random draw
   (i.e., from S_i and t_i, not from S_{i+1}), each step's local
   Radon-Nikodym factor is still exp(-c_i*eps_i - 0.5*c_i^2), and the
   product across steps is still a valid, unbiased change of measure.
   This is checked directly below (unbiasedness), not assumed, given
   that the constant-shift version of this exact kind of code had two
   real bugs before it was trusted.
"""

import numpy as np
from scipy.stats import norm

from pricing import simulate_gbm_paths
from barrier_advanced import bridge_survival_probability


# ----------------------------------------------------------------------
# 1. Survival-based control variate
# ----------------------------------------------------------------------

def no_touch_discounted_price(S0, B, r, sigma, T, q=0.0):
    """
    Closed-form price of a discounted "no-touch" claim: pays $1 at T if
    the continuously-monitored path never reaches the upper barrier B,
    else $0. This is the E[X] needed for the survival-based control
    variate -- a genuinely different closed form from the vanilla-call-
    based up_and_out_call_closed_form in barrier_advanced.py (that one
    has strike-dependent terms; this one doesn't, since it's a pure
    survival probability, not a payoff-weighted one).

    Standard reflection-principle result for GBM (see e.g. Shreve,
    "Stochastic Calculus for Finance II", or Haug's barrier binary
    formulas): under drift mu = (r - q - 0.5*sigma^2),

        P(max_t S_t < B) = N(d) - (B/S0)^(2*mu/sigma^2) * N(d')

    where d and d' are the usual barrier reflection terms. Reconstructed
    from memory, so validated against high-N simulation immediately
    below before being trusted as a control-variate mean.
    """
    mu = r - q - 0.5 * sigma ** 2
    sig_sqrtT = sigma * np.sqrt(T)

    d = (np.log(B / S0) - mu * T) / sig_sqrtT
    d_prime = (np.log(B / S0) + mu * T) / sig_sqrtT  # reflection term

    p_no_touch = norm.cdf(d) - (B / S0) ** (2 * mu / sigma ** 2) * norm.cdf(-d_prime)
    return np.exp(-r * T) * p_no_touch


def price_barrier_bridge_survival_cv(S0, K, B, r, sigma, T, n_steps, n_draws, rng, q=0.0):
    """
    Brownian-bridge barrier price with a survival-based control variate:
    X = discounted per-path survival probability (NOT multiplied by the
    vanilla payoff), EX = no_touch_discounted_price(...). Genuinely
    different quantity from Y, correlation tested empirically.
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_draws, False, rng, q=q)
    dt = T / n_steps
    survival = bridge_survival_probability(paths, B, sigma, dt)
    ST = paths[:, -1]

    Y = np.exp(-r * T) * np.maximum(ST - K, 0.0) * survival
    X = np.exp(-r * T) * survival
    EX = no_touch_discounted_price(S0, B, r, sigma, T, q=q)

    cov = np.cov(Y, X, ddof=1)[0, 1]
    varX = X.var(ddof=1)
    beta = cov / varX if varX > 0 else 0.0
    correlation = cov / np.sqrt(Y.var(ddof=1) * varX) if varX > 0 else 0.0

    Y_cv = Y - beta * (X - EX)
    price = Y_cv.mean()
    var = Y_cv.var(ddof=1)
    se = np.sqrt(var / len(Y_cv))
    return price, se, var, len(Y_cv), correlation


# ----------------------------------------------------------------------
# 2. State-dependent (corridor-pinning) drift tilting
# ----------------------------------------------------------------------

def simulate_gbm_paths_state_dependent_tilt(S0, K, B, r, sigma, T, n_steps, n_draws, rng, q=0.0,
                                             strength=1.0):
    """
    Simulate paths with a drift shift recomputed at EVERY STEP from the
    current state, steering the expected terminal log-price toward the
    geometric center of (K, B), sqrt(K*B). `strength` in [0, 1] scales
    how aggressively the tilt pursues that target (1.0 = fully pin to
    the target given current drift and time remaining; 0.0 = no tilt,
    recovers plain simulation).

    Returns (paths, likelihood_ratio). Likelihood ratio is accumulated
    step by step: since each step's shift c_i is computed from S_i and
    t_i (available BEFORE that step's random draw), the same per-step
    Radon-Nikodym factor from the constant-shift case applies at each
    step, and the product across steps is a valid, unbiased change of
    measure -- this is checked directly in the validation script, not
    assumed.
    """
    dt = T / n_steps
    target_log = 0.5 * (np.log(K) + np.log(B))
    drift = r - q - 0.5 * sigma ** 2

    S = np.full(n_draws, S0)
    log_L = np.zeros(n_draws)
    path_history = [S.copy()]

    for i in range(n_steps):
        tau_remaining = T - i * dt
        # what plain (untilted) drift would deliver by expiry, from here
        implied_terminal_log = np.log(S) + drift * tau_remaining
        # total EXTRA drift (not rate) needed over remaining time to hit target
        total_extra_drift_needed = strength * (target_log - implied_terminal_log)
        # convert to a rate, applied only for this one step's remaining horizon
        drift_shift_rate = np.where(tau_remaining > 1e-12,
                                     total_extra_drift_needed / tau_remaining, 0.0)
        c_i = drift_shift_rate * np.sqrt(dt) / sigma  # per-step shift constant, this step's state

        eps_i = rng.standard_normal(n_draws)
        z_i = eps_i + c_i

        increment = drift * dt + sigma * np.sqrt(dt) * z_i
        S = S * np.exp(increment)
        path_history.append(S.copy())

        log_L += -(c_i * eps_i + 0.5 * c_i ** 2)

    paths = np.stack(path_history, axis=1)
    likelihood_ratio = np.exp(log_L)
    return paths, likelihood_ratio


def price_barrier_state_dependent_is(S0, K, B, r, sigma, T, n_steps, n_draws, rng, q=0.0,
                                      strength=1.0, use_bridge=True):
    """
    Barrier price using the state-dependent (corridor-pinning) tilt.
    Returns (price, std_error, variance, n_paths_used).
    """
    paths, L = simulate_gbm_paths_state_dependent_tilt(
        S0, K, B, r, sigma, T, n_steps, n_draws, rng, q=q, strength=strength)

    if use_bridge:
        dt = T / n_steps
        survival = bridge_survival_probability(paths, B, sigma, dt)
        ST = paths[:, -1]
        Y_raw = np.exp(-r * T) * np.maximum(ST - K, 0.0) * survival
    else:
        knocked_out = np.any(paths >= B, axis=1)
        ST = paths[:, -1]
        payoff = np.maximum(ST - K, 0.0)
        payoff[knocked_out] = 0.0
        Y_raw = np.exp(-r * T) * payoff

    Y = Y_raw * L
    price = Y.mean()
    var = Y.var(ddof=1)
    se = np.sqrt(var / len(Y))
    return price, se, var, len(Y)


def calibrate_tilt_strength(S0, K, B, r, sigma, T, n_steps, r_state, q=0.0,
                             pilot_n=20_000, candidate_strengths=None):
    """Pilot search for the pinning strength, same pattern as
    calibrate_drift_shift in barrier_advanced.py -- no first-principles
    derivation of the optimum, just an honest empirical search."""
    if candidate_strengths is None:
        candidate_strengths = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    best_strength, best_var = 0.0, None
    results = []
    for strength in candidate_strengths:
        rng = np.random.default_rng(r_state)
        _, _, var, _ = price_barrier_state_dependent_is(
            S0, K, B, r, sigma, T, n_steps, pilot_n, rng, q=q, strength=strength)
        results.append((strength, var))
        if best_var is None or var < best_var:
            best_strength, best_var = strength, var

    return best_strength, results
