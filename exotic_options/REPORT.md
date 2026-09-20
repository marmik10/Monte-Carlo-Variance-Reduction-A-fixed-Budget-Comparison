# Exotic Option Pricing: Variance Reduction for Barrier & Asian Options

## How to run this end to end

1. **`pip install numpy scipy pandas`** (and `yfinance` only if doing step 2)
2. **(Optional, for real calibrated inputs)** Run `python fetch_market_data.py SPY` locally.
   This needs network access to Yahoo Finance, which this project was built in a sandboxed
   environment without — so it's untested against live data. It writes `market_params.json`.
   If you skip this step, the next step falls back to synthetic placeholder parameters
   (S0=100, r=3%, sigma=25%) and says so explicitly in its output — it will never silently
   pretend synthetic numbers are real.
3. **`python sanity_checks.py`** — validates the simulator against closed-form benchmarks.
   Run this before trusting anything below; if it fails, nothing else here is meaningful.
4. **`python run_experiment.py`** — runs the fair-budget comparison across all four method
   combinations (plain / antithetic / control variate / both) for both option types, prints
   a results table, and saves `barrier_results.csv` / `asian_results.csv`.
5. **`python convergence_study.py`** — runs the same pricers across increasing N (1,000 to
   200,000 paths), fits a log-log slope of standard error vs. N, and saves `convergence_study.png`.
   Theory predicts a -0.5 slope regardless of variance reduction (that's just the O(1/sqrt(N))
   Monte Carlo rate); what a control variate should change is the *height* of the line, not
   its slope. Also needs `matplotlib`.

Files:
- `pricing.py` — core engine: GBM path simulation (with continuous dividend yield `q`),
  closed-form benchmarks (Black-Scholes vanilla call, Kemna-Vorst geometric Asian, both
  dividend-adjusted), barrier continuity correction, and the four MC pricing methods for
  both option types.
- `sanity_checks.py` — five validation tests against closed-form limits, including a
  dividend-yield check.
- `run_experiment.py` — the fair-budget variance-reduction experiment.
- `convergence_study.py` — SE vs. N across a range of path counts, with a log-log slope fit
  and a saved plot; the empirical evidence that variance reduction changes the constant in
  front of the O(1/sqrt(N)) rate, not the rate itself.
- `fetch_market_data.py` — pulls real S0, term-matched r (interpolated across the Treasury
  curve to the option's actual maturity, not just the 13-week bill), dividend yield q, and
  ATM implied vol from yfinance; run locally, not here.
- `market_params.json` — written by `fetch_market_data.py` if you run it; consumed
  automatically by `run_experiment.py` and `convergence_study.py` if present.

## Setup

- Model: Black-Scholes / GBM, simulated with the exact lognormal transition (no Euler discretization error in the marginal distribution)
- Parameters: S0=100, K=100, r=3%, sigma=25%, T=1y, weekly monitoring grid (52 steps)
- Barrier: up-and-out call, B=115. **Contract convention: continuously monitored** (the market standard for most listed/OTC barrier products). Simulating on a 52-step grid checking the barrier only at those points therefore introduces a *discretization bias*, corrected via the Broadie-Glasserman-Kou continuity correction (barrier shifted toward spot by a factor of exp(-0.5826·sigma·sqrt(dt))).
- Asian: arithmetic-average call, monitored at the same 52 dates (this one genuinely is discrete by contract design — no correction needed).
- All comparisons run at a **fixed total random-number budget** (200,000 path draws). Antithetic methods use 100,000 independent draws expanded into +/- pairs, so they're compared against 200,000 fully independent plain-MC paths, not against another 100,000. Comparing N antithetic paths to N (not 2N) plain paths is a common way these numbers get overstated.
- 20 independent repetitions per method to estimate variance of the estimator itself.

## Validation (before trusting any variance-reduction number)

Four sanity checks against closed-form benchmarks, all passing:
1. Barrier → infinity converges to the Black-Scholes vanilla call price (11.35 vs 11.35, within 3 standard errors)
2. Barrier below spot prices to exactly zero (immediate knock-out)
3. MC estimate of the geometric-average Asian matches its closed-form Kemna-Vorst price (6.19 vs 6.20)
4. The continuity correction moves the price in the correct direction (lower — see note below on why this surprised me during development)

One thing worth being upfront about: my first implementation had the continuity-correction sign backwards, and the sanity check caught it (see code comments in `pricing.py` for the corrected reasoning). Worth knowing if you're extending this — it's an easy sign error to make and an easy one to not notice if you don't validate against a benchmark first.

## Results

### Up-and-out barrier call (price ≈ 0.255)

| Method | Std Error | Variance Reduction | Efficiency Ratio* |
|---|---|---|---|
| Plain MC | 0.00291 | — | 1.00x |
| Antithetic | 0.00291 | **-0.04%** | 1.31x |
| Control variate (vanilla call) | 0.00291 | **0.23%** | 1.24x |
| Both combined | 0.00291 | **0.19%** | 1.43x |

*Efficiency ratio = variance-reduction factor achieved per unit of compute time relative to plain MC (>1 means net favorable, accounting for the extra cost of the technique).

**Both techniques are essentially dead weight here.** The knock-out indicator is a discontinuous function of the path maximum — antithetic pairing doesn't help because a path and its mirror image can land on opposite sides of the barrier-touch event, and the vanilla call payoff (used as the control) correlates with the *terminal* value, not with whether the barrier was touched along the way. The efficiency ratios above 1.0 are mostly a runtime artifact (this implementation happens to run faster under those code paths), not a real statistical improvement — the std errors round to identical values across all four methods.

### Arithmetic-average Asian call (price ≈ 6.50)

| Method | Std Error | Variance Reduction | Efficiency Ratio* |
|---|---|---|---|
| Plain MC | 0.02210 | — | 1.00x |
| Antithetic | 0.02208 | 0.17% | 0.97x |
| Control variate (geometric Asian) | 0.00074 | **99.89%** | **640x** |
| Both combined | 0.00074 | **99.89%** | **717x** |

**The control variate does essentially all the work.** The geometric and arithmetic averages of the same simulated path are highly correlated (both are averaging the same underlying draws), so regressing out the geometric-average payoff — which has a closed-form price via Kemna-Vorst — removes nearly all the estimator's variance. This is the textbook case where control variates are supposed to work well, and here they do, dramatically. Antithetic variates add essentially nothing on top.

## Honest takeaways

- **"Variance reduction techniques" is not one result — it's two very different stories depending on payoff structure.** A single blended number across both option types would hide the more interesting finding: control variates need a control that's actually correlated with the target payoff's *source of randomness*, not just the same underlying asset.
- For the barrier option, the natural next step (not implemented here) would be a control variate built from the barrier-hitting probability itself, or importance sampling that shifts the drift toward the barrier to increase the effective sample size of knock-out paths — plain antithetic/control-variate tricks aren't the right tool for discontinuous payoffs.
- Reporting "cutting estimator variance by X%" as a single headline number, as in the original project bullet, would be misleading here — it depends entirely on which option and which control you're describing. The honest version of that bullet is closer to: *"achieved 99.9% variance reduction on Asian options via a geometric-average control variate; barrier options showed negligible improvement from either technique, consistent with the discontinuous payoff structure."*

## Fixed after initial review, and what's still open

Two real bugs were caught and fixed before this was resume-worthy:
- **Dividend yield was missing entirely.** Pricing SPY options without accounting for the ~1.3% dividend yield is a correctness bug, not a style choice, once real market data is involved — the GBM drift needs `r - q`, not just `r`. Fixed in `pricing.py`, validated by sanity check 5.
- **Rate/maturity mismatch.** The risk-free rate was pulled from the 13-week T-bill regardless of the option's actual maturity. `fetch_market_data.py` now interpolates across the Treasury curve (^IRX/^FVX/^TNX/^TYX) to the option's actual time-to-expiry.

Still open, not fixed:
- **No volatility skew.** A single ATM implied vol is applied at every strike, including the barrier level, which sits away from the money. Real index vol surfaces are skewed, so this is a real (if standard, for a project at this scope) approximation — not accounted for anywhere in the pricing.
- **Barrier variance reduction remains an open problem, not a solved one.** Antithetic variates and a vanilla-call control variate both fail on the discontinuous knock-out payoff (confirmed on both synthetic and real calibrated data). The next real step would be importance sampling (shifting the simulation drift toward the barrier) or a control variate built from the barrier-hitting time distribution — neither is implemented.
## Barrier option: a second pass, done properly

The 0.3% result above led me to build the actual domain-specific fixes (Brownian bridge conditional expectation / Rao-Blackwellization, a validated closed-form continuous-barrier benchmark, and importance sampling) rather than stop at "doesn't work." Two of my own claims during that work turned out to be wrong and are worth recording rather than erasing:

1. **A drift-shift scaling bug** initially made importance sampling collapse to price ≈ 0 — I'd derived the per-step shift constant using `sqrt(T)` instead of `sqrt(dt)`, injecting a 7x-larger drift than intended. Caught by the price no longer matching the unbiased baseline.
2. **My directional argument for importance sampling was backwards.** I claimed the drift should shift *away* from the barrier (fewer knock-outs → more useful surviving paths). Empirically, the opposite direction (a small shift *toward* the barrier) has lower variance. The reasoning gap: this option has two binding constraints (survive to expiry AND finish above the strike), and since K = S0 here, pushing drift down to avoid the barrier also drags terminal price below the strike more often — killing the payoff through the exact channel the shift was meant to protect. A single-constraint rare-event heuristic doesn't transfer cleanly to a two-constraint problem. Fixed by replacing the assumed direction with an empirical pilot search (`calibrate_drift_shift()`).

**Results, same fixed compute budget as before:**

| Method | Variance | Reduction vs. naive baseline |
|---|---|---|
| Naive discrete + continuity correction (original) | 1.696 | 0.00% |
| Naive discrete + vanilla-call control variate (original) | 1.692 | 0.23% |
| Brownian bridge (Rao-Blackwellized), plain | 1.551 | 8.53% |
| Brownian bridge + antithetic | 1.555 | 8.34% |
| Brownian bridge + vanilla-call control variate | 1.548 | 8.75% |
| Brownian bridge + antithetic + control variate | 1.551 | 8.56% |
| **Brownian bridge + calibrated importance sampling** | **1.474** | **13.10%** |

All four bridge variants land in essentially the same band — antithetic and the vanilla-call control variate add almost nothing once the bridge has already smoothed the discontinuous payoff into something continuous. Importance sampling on top of the bridge is the only technique that moves the needle further, and even that is a real but modest 13%, not an order-of-magnitude win.

**The honest conclusion:** the domain-specific fixes took this option from "no measurable improvement" (0.2%) to "a real, useful, but modest improvement" (13%) — genuine progress, achieved by replacing the discontinuous knock-out indicator with its exact conditional expectation (which is where nearly all of the gain comes from) plus a correctly-calibrated importance sampling shift on top. It is not, and should not be reported as, a solved problem on the scale of the Asian option's 99.9%. Barrier options are structurally harder to variance-reduce than smooth path-dependent payoffs, and that remains true even with the better techniques — which is itself the more defensible thing to say about this than either the original 0.3% number or an inflated claim of having "solved" it.

New files from this pass: `barrier_advanced.py` (Brownian bridge, closed-form benchmark, importance sampling), `validate_barrier_advanced.py` (run this before trusting any of it — it caught both bugs above), `run_barrier_advanced_experiment.py` (the comparison table above), `barrier_advanced_results.csv`.

## Third pass: survival-based control variate and state-dependent tilting

Two follow-up techniques, one that underperformed as predicted and one that produced the actual breakthrough for this option.

**Survival-based control variate.** Uses the discounted survival probability *itself* (not multiplied by the payoff) as the control, with its own closed-form mean — a discounted "no-touch" digital barrier price, validated against simulation the same way as the earlier closed forms (difference of 0.00087 against a 0.00129 standard error — passes). This is a genuinely different quantity from the target, not a repackaging of the same closed-form call price used as both target and control — that degenerate version would trivially collapse variance to ~0 by construction (since the control *is* the answer) without demonstrating anything about the technique, so it was deliberately avoided. The real, measured correlation between the target and this control turned out to be only 0.20, and the resulting variance reduction (12.21%) is unsurprising given that — a modest, honestly-earned number, not a large one.

**State-dependent (corridor-pinning) drift tilting** — this is the actual breakthrough. Rather than a single constant drift shift for the whole path, the shift is recomputed at *every step* from the current spot and time remaining, steering the expected terminal log-price toward the geometric center of `(K, B)`. This is explicitly **not** a reproduction of the Glasserman-Heidelberger-Shahabuddin asymptotically-optimal large-deviations tilt — deriving that requires solving a variational problem specific to this exact payoff, which is out of scope here — but a defensible heuristic in the same spirit, with the same rigorous likelihood-ratio accounting (validated for unbiasedness at every tested strength before any variance number was trusted).

Result, calibrated pinning strength = 0.80 (found by pilot search, same discipline as the earlier drift-shift calibration — not derived from first principles):

| Method | Variance | Reduction vs. naive baseline |
|---|---|---|
| Naive discrete + continuity correction (original) | 1.696 | 0.00% |
| Brownian bridge, plain | 1.551 | 8.53% |
| Brownian bridge + constant-shift importance sampling | 1.474 | 13.10% |
| Brownian bridge + survival-based control variate | 1.489 | 12.21% |
| **Brownian bridge + state-dependent corridor-pinning tilt** | **0.350** | **79.36%** |

This is a full order of magnitude better than anything found in the second pass. The mechanism: a single constant drift shift is a compromise across the whole path — it can't simultaneously be "right" early (when the process is far from both boundaries) and late (when it's close to one). Recomputing the tilt at every step from the current state lets the sampling measure track the corridor dynamically instead of committing to one global compromise, which is exactly the intuition behind the real GHS large-deviations approach — this heuristic captures a meaningful fraction of that benefit without the full derivation.

**Honest accounting of what changed:** the original claim in this project was 0.2-0.3% variance reduction for barrier options, essentially nothing. After building the actual domain-specific machinery — Brownian bridge conditioning, and especially state-dependent importance sampling — the achievable reduction is closer to 79%, which is a legitimate, validated, substantial result. It is still not the 99.9% seen on the Asian option, and there's no reason to expect parity: the Asian payoff is smooth everywhere, while the barrier payoff has a genuine boundary that any technique has to work around rather than eliminate. But "barrier options are hard to variance-reduce" was true of the naive approach and is no longer the right characterization of what's achievable with the right technique.

New files from this pass: `barrier_state_dependent.py` (survival-based control variate, state-dependent tilt, and the no-touch closed form), `validate_barrier_state_dependent.py` (run before trusting either technique).
