"""
Domain-specific variance reduction for the up-and-out barrier call,
addressing why plain antithetic/control-variate tricks failed (see
REPORT.md): the knock-out indicator is a discontinuous function of the
path, which breaks both techniques.

Three techniques, in order of what actually gets built:

1. Brownian bridge conditional expectation (Rao-Blackwellization).
   Instead of checking whether simulated GRID POINTS breach the barrier,
   compute the exact analytical probability that a continuous Brownian
   bridge between each pair of consecutive grid points crosses the
   barrier, and multiply survival probabilities across all intervals.
   This does two things at once: it removes the discretization bias
   (no more approximate BGK correction needed -- this IS the continuous
   monitoring probability, not an approximation of it), and it reduces
   variance by the law of total variance: Var(E[X|skeleton]) <= Var(X).
   This replaces the 0/1 knock-out indicator with a smooth number in
   [0, 1], which is why it helps -- discontinuities are what killed
   antithetic variates and the vanilla-call control variate in the
   original approach.

2. Reiner-Rubinstein closed-form continuous-barrier price. Used ONLY as
   an independent validation benchmark for the Brownian bridge
   estimator above -- NOT plugged in as a control-variate mean, because
   the naive discrete payoff and the continuous closed-form price
   differ by a bias term, not just noise, so using it that way would be
   silently wrong. This formula is reconstructed from memory and
   validated against high-N simulation below before being trusted for
   anything; if it doesn't match, the mismatch is reported rather than
   hidden.

3. Importance sampling via drift shift. NOTE: my first-pass reasoning
   here was wrong and worth recording rather than quietly fixing. The
   naive argument is "the barrier gets hit often, so surviving paths are
   the rare event -- shift drift away from the barrier to sample more of
   them." Empirically, that direction makes variance WORSE, not better,
   because it only accounts for one constraint (avoid knock-out) while
   ignoring the other (still needs to end above K). Since K = S0 here
   (at-the-money), shifting drift down to avoid the barrier also drags
   terminal price below the strike more often, killing the payoff
   through the exact channel the shift was meant to protect. This
   option's value lives in a narrow band of paths that rise enough to be
   ITM but not so much they hit the barrier -- a downward shift doesn't
   preserve that band, it destroys it. Empirically, a SMALL shift TOWARD
   the barrier gives a modest (~8-9%) variance reduction; large shifts in
   either direction make things worse. The calibrate_drift_shift()
   function below finds a reasonable value by pilot search rather than
   assuming a sign from theory alone -- a two-constraint rare-event
   problem (survive AND finish ITM) doesn't reduce to the single-event
   textbook case cleanly.
"""

import numpy as np
from scipy.stats import norm

from pricing import simulate_gbm_paths, bs_call_price


# ----------------------------------------------------------------------
# 1. Brownian bridge conditional expectation
# ----------------------------------------------------------------------

def bridge_survival_probability(paths, B, sigma, dt):
    """
    For each path, the probability that a Brownian bridge between each
    pair of consecutive observed points stays BELOW the upper barrier B
    throughout the interval, multiplied across all intervals.

    Standard result (see e.g. Glasserman, "Monte Carlo Methods in
    Financial Engineering", Sec 6.4): for a GBM path observed at S_i and
    S_{i+1} over an interval of length dt, conditional on those two
    endpoints, the probability the continuous path crosses an upper
    barrier B (with both S_i < B and S_{i+1} < B) is:

        p_cross = exp( -2 * ln(B/S_i) * ln(B/S_{i+1}) / (sigma^2 * dt) )

    If either endpoint is already >= B, the path has certainly crossed
    (crossing probability 1, survival 0).
    """
    S_i = paths[:, :-1]
    S_next = paths[:, 1:]

    already_breached = (S_i >= B) | (S_next >= B)

    # only compute the log-ratio where both endpoints are safely below B,
    # to avoid log(<=0) on entries that will be overridden anyway
    safe = ~already_breached
    log_ratio_i = np.where(safe, np.log(np.clip(B, 1e-300, None) / np.clip(S_i, 1e-300, None)), 0.0)
    log_ratio_next = np.where(safe, np.log(np.clip(B, 1e-300, None) / np.clip(S_next, 1e-300, None)), 0.0)

    p_cross = np.exp(-2.0 * log_ratio_i * log_ratio_next / (sigma ** 2 * dt))
    p_cross = np.where(already_breached, 1.0, p_cross)
    p_survive_interval = 1.0 - p_cross

    # product of per-interval survival probabilities = probability the
    # WHOLE path never crosses, conditional on the observed skeleton
    return np.prod(p_survive_interval, axis=1)


def barrier_payoff_bridge(paths, K, B, r, T, sigma, n_steps):
    """Rao-Blackwellized discounted payoff: discounted vanilla payoff at
    S_T, weighted by the exact conditional probability of never crossing
    the barrier given the simulated skeleton. Replaces the discontinuous
    0/1 knock-out indicator with a smooth number in [0, 1] -- this is
    what fixes the variance-reduction failure from the naive approach."""
    dt = T / n_steps
    survival = bridge_survival_probability(paths, B, sigma, dt)
    ST = paths[:, -1]
    vanilla_payoff = np.maximum(ST - K, 0.0)
    return np.exp(-r * T) * vanilla_payoff * survival


def price_barrier_bridge_mc(S0, K, B, r, sigma, T, n_steps, n_draws, method, rng, q=0.0):
    """
    Barrier pricer using the Brownian-bridge conditional payoff instead
    of the naive discrete indicator. method in
    {"plain", "antithetic", "control", "antithetic_control"} -- same
    method names as the original pricer, but the underlying payoff
    computation is fundamentally different (smooth, not discontinuous).
    """
    antithetic = method in ("antithetic", "antithetic_control")
    use_cv = method in ("control", "antithetic_control")

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_draws, antithetic, rng, q=q)
    Y = barrier_payoff_bridge(paths, K, B, r, T, sigma, n_steps)

    if not use_cv:
        price = Y.mean()
        var = Y.var(ddof=1)
        se = np.sqrt(var / len(Y))
        return price, se, var, len(Y)

    # Control variate: discounted vanilla call payoff, same as the
    # original approach. Weaker justification here than technique (1)
    # itself, but included for a like-for-like comparison.
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
# 2. Closed-form continuous-barrier price (Reiner-Rubinstein / Merton)
#    -- validation benchmark ONLY, not a control-variate mean.
# ----------------------------------------------------------------------

def up_and_out_call_closed_form(S0, K, B, r, sigma, T, q=0.0):
    """
    Closed-form price of a CONTINUOUSLY monitored up-and-out European
    call, for the case B > K (barrier above strike), no rebate.
    Reiner & Rubinstein (1991) / Merton (1973) formula, as tabulated in
    Haug's "Complete Guide to Option Pricing Formulas".

    This is reconstructed from memory, not copied from a live reference,
    so it is validated against high-N simulation in sanity_checks.py
    before being trusted anywhere else in this project. If validation
    fails, that failure is reported, not hidden.
    """
    if B <= K:
        raise ValueError("This formula assumes B > K (barrier above strike); "
                          "a different formula branch is needed otherwise.")

    b = r - q
    mu = (b - 0.5 * sigma ** 2) / sigma ** 2
    sig_sqrtT = sigma * np.sqrt(T)

    x1 = np.log(S0 / K) / sig_sqrtT + (1 + mu) * sig_sqrtT
    x2 = np.log(S0 / B) / sig_sqrtT + (1 + mu) * sig_sqrtT
    y1 = np.log(B ** 2 / (S0 * K)) / sig_sqrtT + (1 + mu) * sig_sqrtT
    y2 = np.log(B / S0) / sig_sqrtT + (1 + mu) * sig_sqrtT

    term1 = S0 * np.exp(-q * T) * (norm.cdf(x1) - norm.cdf(x2))
    term2 = -K * np.exp(-r * T) * (norm.cdf(x1 - sig_sqrtT) - norm.cdf(x2 - sig_sqrtT))
    term3 = -S0 * np.exp(-q * T) * (B / S0) ** (2 * (mu + 1)) * (norm.cdf(y1) - norm.cdf(y2))
    term4 = K * np.exp(-r * T) * (B / S0) ** (2 * mu) * (norm.cdf(y1 - sig_sqrtT) - norm.cdf(y2 - sig_sqrtT))

    return term1 + term2 + term3 + term4


# ----------------------------------------------------------------------
# 3. Importance sampling via drift shift
# ----------------------------------------------------------------------

def simulate_gbm_paths_tilted(S0, r, sigma, T, n_steps, n_draws, rng, drift_shift, q=0.0):
    """
    Simulate paths under a drift-SHIFTED measure Q, and return both the
    paths and the likelihood ratio L = dP/dQ needed to convert a
    Q-expectation back into the correct (unbiased) P-expectation.

    drift_shift is added to the risk-neutral drift (r - q - 0.5*sigma^2)
    for the PURPOSE OF SAMPLING ONLY -- it changes which paths get drawn,
    not the model's actual dynamics. Positive drift_shift pushes paths
    UP (toward an upper barrier); negative pushes them down (away from
    it). For an up-and-out call where knock-outs are already common,
    drift_shift should be NEGATIVE (push away from the barrier, generate
    more surviving/ITM paths) -- see module docstring.

    Derivation: raw standard normals eps ~ N(0,1) are shifted by a
    constant c = drift_shift*sqrt(dt)/sigma before being used to build
    the path (this adds exactly drift_shift*dt of extra drift per step,
    drift_shift*T total, since summing n_steps identical per-step shifts
    of size sigma*c*sqrt(dt) = drift_shift*dt gives drift_shift*T over
    the whole period). The likelihood ratio for n_steps i.i.d. shocks
    mean-shifted by c is L = exp(-c*sum(eps) - 0.5*n_steps*c^2), which
    reweights a Q-average back to the correct P-expectation.
    """
    dt = T / n_steps
    c = drift_shift * np.sqrt(dt) / sigma

    eps = rng.standard_normal((n_draws, n_steps))
    z_tilted = eps + c

    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z_tilted
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_paths)
    S = np.concatenate([np.full((S.shape[0], 1), S0), S], axis=1)

    likelihood_ratio = np.exp(-c * eps.sum(axis=1) - 0.5 * n_steps * c ** 2)
    return S, likelihood_ratio


def price_barrier_importance_sampling(S0, K, B, r, sigma, T, n_steps, n_draws, rng,
                                       drift_shift, use_bridge=True, q=0.0):
    """
    Importance-sampled barrier price. use_bridge=True combines this with
    the Brownian-bridge payoff (1) for the strongest available combo;
    use_bridge=False uses the naive discrete indicator, to isolate how
    much importance sampling contributes on its own.

    Returns (price, std_error, variance, n_paths_used).
    """
    paths, L = simulate_gbm_paths_tilted(S0, r, sigma, T, n_steps, n_draws, rng, drift_shift, q=q)

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


def calibrate_drift_shift(S0, K, B, r, sigma, T, n_steps, r_state, use_bridge=True,
                           q=0.0, pilot_n=20_000, candidate_shifts=None):
    """
    Pick a drift shift by pilot search rather than assuming a sign from
    theory -- see the module docstring for why the naive "shift away
    from the barrier" argument doesn't hold here. Runs a small pilot at
    each candidate shift (shared random numbers across candidates, for a
    fair comparison) and returns the shift with the lowest observed
    variance.

    r_state: an integer seed (not an np.random.Generator) so each
    candidate can be evaluated on the SAME underlying draws.
    """
    if candidate_shifts is None:
        candidate_shifts = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]

    best_shift, best_var = 0.0, None
    results = []
    for shift in candidate_shifts:
        rng = np.random.default_rng(r_state)
        if shift == 0.0:
            paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, pilot_n, False, rng, q=q)
            if use_bridge:
                dt = T / n_steps
                survival = bridge_survival_probability(paths, B, sigma, dt)
                ST = paths[:, -1]
                Y = np.exp(-r * T) * np.maximum(ST - K, 0.0) * survival
            else:
                knocked_out = np.any(paths >= B, axis=1)
                ST = paths[:, -1]
                payoff = np.maximum(ST - K, 0.0)
                payoff[knocked_out] = 0.0
                Y = np.exp(-r * T) * payoff
            var = Y.var(ddof=1)
        else:
            _, _, var, _ = price_barrier_importance_sampling(
                S0, K, B, r, sigma, T, n_steps, pilot_n, rng, shift, use_bridge=use_bridge, q=q)
        results.append((shift, var))
        if best_var is None or var < best_var:
            best_shift, best_var = shift, var

    return best_shift, results
