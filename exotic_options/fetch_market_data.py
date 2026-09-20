"""
Run this locally: pip install yfinance
Then: python fetch_market_data.py [TICKER]  (default TICKER = SPY)

Pulls the four inputs the pricing model needs:
  - S0: current spot price
  - sigma: at-the-money implied volatility, from the options expiry
           closest to ~1 year out (falls back to nearest available
           expiry and prints a warning if none are that far out)
  - r: risk-free rate, TERM-MATCHED to the option's actual maturity via
       linear interpolation across the Treasury curve -- not just the
       13-week bill rate, which would be a real mismatch for a ~1-year
       option if the curve isn't flat
  - q: trailing dividend yield (continuous-yield approximation)

Writes market_params.json in the current directory, which
run_experiment.py auto-detects and uses instead of its synthetic
placeholder defaults.
"""

import sys
import json
from datetime import datetime

import numpy as np
import yfinance as yf


def get_spot(ticker: str) -> float:
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No price history returned for {ticker}")
    return float(hist["Close"].iloc[-1])


def get_atm_implied_vol(ticker: str, spot: float, target_days: int = 365):
    """
    Find the options expiry closest to target_days out, then take the
    implied vol of the call whose strike is closest to the current spot
    (i.e. the ATM call). Returns (sigma, expiry_used, days_to_expiry, strike).
    """
    t = yf.Ticker(ticker)
    expiries = t.options
    if not expiries:
        raise RuntimeError(f"No options chain available for {ticker}")

    today = datetime.now()
    expiry_days = [(e, (datetime.strptime(e, "%Y-%m-%d") - today).days) for e in expiries]
    best_expiry, best_days = min(expiry_days, key=lambda x: abs(x[1] - target_days))

    chain = t.option_chain(best_expiry)
    calls = chain.calls
    if calls.empty:
        raise RuntimeError(f"No call quotes for {ticker} expiry {best_expiry}")

    calls = calls.copy()
    calls["strike_dist"] = (calls["strike"] - spot).abs()
    atm_row = calls.sort_values("strike_dist").iloc[0]

    iv = float(atm_row["impliedVolatility"])
    return iv, best_expiry, best_days, float(atm_row["strike"])


def get_risk_free_rate(target_years: float) -> float:
    """
    Term-matched risk-free rate via linear interpolation between Treasury
    yield curve points. Using the 13-week bill to discount a ~1-year
    option is a real mismatch if the curve isn't flat -- this fixes that.

    Free Yahoo tickers available: ^IRX (13-week bill, ~0.25y), ^FVX
    (5-year note), ^TNX (10-year note), ^TYX (30-year bond). There's no
    clean 1-2y point this way, so for target maturities between 0.25y
    and 5y this linearly interpolates between ^IRX and ^FVX. That's a
    real approximation -- the actual curve isn't linear there, and
    short-to-intermediate curvature is often where the curve bends the
    most -- but it's a better estimate than pinning everything to the
    3-month rate regardless of option maturity.
    """
    tenors = {"^IRX": 0.25, "^FVX": 5.0, "^TNX": 10.0, "^TYX": 30.0}
    points = []
    for ticker, years in tenors.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                points.append((years, float(hist["Close"].iloc[-1]) / 100.0))
        except Exception:
            continue

    if not points:
        raise RuntimeError("Could not fetch any Treasury yield curve points")

    points.sort()
    years_arr = [p[0] for p in points]
    rates_arr = [p[1] for p in points]

    if target_years <= years_arr[0]:
        return rates_arr[0]
    if target_years >= years_arr[-1]:
        return rates_arr[-1]
    return float(np.interp(target_years, years_arr, rates_arr))


def get_dividend_yield(ticker: str) -> float:
    """Trailing dividend yield as a decimal (e.g. 0.013 for 1.3%).
    Falls back to 0.0 with a warning if unavailable -- some tickers
    (individual growth stocks, non-dividend payers) legitimately have
    none, and yfinance's info dict is not always populated or consistent
    in units across versions."""
    try:
        info = yf.Ticker(ticker).info
        y = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0.0
        y = float(y)
        # yfinance has changed units across versions (fraction vs percent).
        # A dividend yield above 25% is implausible for a broad index or
        # large-cap stock -- treat that as a signal the value is already
        # in percent form and needs dividing down.
        if y > 0.25:
            y = y / 100.0
        return y
    except Exception as e:
        print(f"  WARNING: could not fetch dividend yield ({e}); using q=0.0")
        return 0.0


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"

    print(f"Fetching data for {ticker}...")
    spot = get_spot(ticker)
    print(f"  Spot price: {spot:.2f}")

    sigma, expiry, days, atm_strike = get_atm_implied_vol(ticker, spot)
    T_years = days / 365.0
    print(f"  ATM implied vol: {sigma*100:.2f}%  (expiry {expiry}, {days} days out, strike {atm_strike})")

    r = get_risk_free_rate(target_years=T_years)
    print(f"  Risk-free rate, interpolated to {T_years:.2f}y: {r*100:.3f}%")

    q = get_dividend_yield(ticker)
    print(f"  Dividend yield: {q*100:.3f}%")

    if abs(days - 365) > 60:
        print(f"  WARNING: closest available expiry is {days} days out, not ~365. "
              f"Consider adjusting T in the pricing model to match, "
              f"or re-run pointing at a ticker with longer-dated options (e.g. SPY, QQQ).")

    result = {
        "ticker": ticker,
        "S0": round(spot, 4),
        "r": round(r, 5),
        "q": round(q, 5),
        "sigma": round(sigma, 5),
        "T_years": round(T_years, 4),
        "expiry_used": expiry,
        "atm_strike_reference": atm_strike,
        "fetched_at": datetime.now().isoformat(),
    }

    print("\n--- Result (already saved, no need to paste it back) ---")
    print(json.dumps(result, indent=2))

    with open("market_params.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to market_params.json in the current directory.")
    print("Copy that file next to pricing.py / run_experiment.py, then just run:")
    print("    python run_experiment.py")
    print("It will auto-detect market_params.json and use these real inputs,")
    print("including the dividend yield and term-matched rate, instead of")
    print("the synthetic placeholder defaults.")


if __name__ == "__main__":
    main()
