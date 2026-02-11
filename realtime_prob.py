#!/usr/bin/env python3
"""
realtime_prob.py

Purpose:
- Compute P(S_T > K) (digital / cash-or-nothing) using Black-Scholes (N(d2)).
- Supports manual inputs or best-effort auto-fetch of spot + IV using yfinance (SPY proxy).
- Optional small gamma tilt (manual or numeric regime) to model your gamma-positioning overlay.

How to run (examples):
1) Manual inputs (recommended if you have IV):
   python realtime_prob.py --spot 695.0 --strike 700.0 --expiry "2026-02-11 16:00" --iv 0.18

2) Use yfinance to auto fetch SPY spot and nearest IV (best-effort, may be delayed):
   python realtime_prob.py --auto --index SPY --strike 700.0 --expiry "2026-02-11 16:00"

3) Quick intraday minutes to expiry (T given in minutes):
   python realtime_prob.py --spot 695.0 --strike 700.0 --minutes 45 --iv 0.20

4) Apply gamma tilt (e.g. tilt = -0.03 means reduce model probability by 3% relative):
   python realtime_prob.py --spot 695.0 --strike 700.0 --expiry "2026-02-11 16:00" --iv 0.18 --tilt -0.03

Notes:
- IV should be annualized decimal (e.g. 18% -> 0.18).
- If you use SPX markets but rely on SPY chain, use mapping SPX->SPY by dividing by ~10 (configurable).
- yfinance option IVs can be delayed or incomplete; manual IV from a data provider is best for live trading.
"""

import argparse
import math
from datetime import datetime, timedelta
import pytz
from scipy.stats import norm

# Optional: only needed for auto mode; may be slower in GitHub Actions
try:
    import yfinance as yf
except Exception:
    yf = None

ET = pytz.timezone("America/New_York")

# ----------------------------
# Helper math
# ----------------------------
def years_between(now_dt: datetime, future_dt: datetime) -> float:
    """Return T in years between now (aware dt) and future (aware dt). If negative, return 0."""
    diff = future_dt - now_dt
    secs = diff.total_seconds()
    return max(secs / (365.0 * 24.0 * 3600.0), 0.0)

def parse_datetime_et(s: str) -> datetime:
    """
    Parse a datetime string. Accepts:
      - "YYYY-MM-DD HH:MM" (treated as ET)
      - "YYYY-MM-DD" (assumed 16:00 ET)
    """
    s = s.strip()
    if len(s) == 10:
        # date only
        dt = datetime.strptime(s, "%Y-%m-%d")
        dt = ET.localize(datetime(dt.year, dt.month, dt.day, 16, 0, 0))
        return dt
    # try full date+time
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
        return ET.localize(dt)
    except ValueError:
        # try with seconds
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return ET.localize(dt)

def digital_prob_log_normal(S: float, K: float, sigma: float, T_years: float, r: float = 0.0) -> dict:
    """
    Compute P(S_T > K) under BS lognormal (digital = N(d2)).
    Returns dict with p, d1, d2.
    """
    if any(x <= 0 for x in (S, K, sigma)) or T_years <= 0:
        return {"p": float("nan"), "d1": float("nan"), "d2": float("nan")}
    vol_sqrt = sigma * math.sqrt(T_years)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_years) / vol_sqrt
    d2 = d1 - vol_sqrt
    p = norm.cdf(d2)
    return {"p": p, "d1": d1, "d2": d2}

# ----------------------------
# yfinance helpers (best-effort)
# ----------------------------
def fetch_spot_and_iv_spy(strike_spy: float, expiry_date: str):
    """
    Best-effort: fetch SPY last price and nearest-call implied vol near strike from yfinance.
    expiry_date is a string "YYYY-MM-DD" corresponding to an options expiry to search (or leave None to pick nearest).
    Returns (spot, iv) where iv is decimal (e.g. 0.18) or (None, None) if unavailable.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install it or run in manual mode with --iv.")

    ticker = yf.Ticker("SPY")
    # spot
    spot = None
    try:
        spot = ticker.fast_info.get("lastPrice", None)
    except Exception:
        spot = None
    if spot is None:
        hist = ticker.history(period="1d", interval="1m")
        spot = float(hist["Close"].iloc[-1])

    # find expiries
    options = ticker.options
    if not options:
        return float(spot), None

    chosen_exp = None
    if expiry_date:
        # pick the expiry matching the date (if available)
        for e in options:
            if e.startswith(expiry_date):
                chosen_exp = e
                break
    if chosen_exp is None:
        # pick the nearest expiry
        chosen_exp = options[0]

    # Pull option chain at chosen_exp
    try:
        chain = ticker.option_chain(chosen_exp)
    except Exception:
        return float(spot), None

    calls = chain.calls.copy()
    # find nearest strike
    calls["dist"] = (calls["strike"] - strike_spy).abs()
    calls = calls.sort_values("dist").reset_index(drop=True)
    if calls.shape[0] == 0:
        return float(spot), None

    iv = calls.loc[0, "impliedVolatility"]
    # yfinance may present IV as NaN or 0
    if iv is None or iv != iv or iv <= 0:
        return float(spot), None

    return float(spot), float(iv)

# ----------------------------
# Main CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--auto", action="store_true", help="Auto-fetch spot + IV using yfinance (SPY).")
    p.add_argument("--index", choices=["SPY", "SPX"], default="SPY", help="Index to use for auto fetch (SPY proxy for SPX).")
    p.add_argument("--spot", type=float, help="Current spot price (e.g. SPY price ~ 695).")
    p.add_argument("--strike", type=float, required=True, help="Strike price (same units as spot).")
    p.add_argument("--expiry", type=str, help='Expiry datetime in ET e.g. "2026-02-11 16:00" or "2026-02-11" (assumes 16:00 ET).')
    p.add_argument("--minutes", type=float, help="Alternative: minutes from now to expiry (overrides expiry if provided).")
    p.add_argument("--iv", type=float, help="Implied vol (annual decimal, e.g. 0.18). If omitted and --auto used, script will try to fetch it.")
    p.add_argument("--tilt", type=float, default=0.0, help="Relative gamma tilt applied to model probability (e.g. -0.03 reduces prob by 3%).")
    p.add_argument("--spx_to_spy", type=float, default=10.0, help="Mapping ratio for SPX->SPY (if using SPY as proxy).")
    args = p.parse_args()

    # Determine now
    now = datetime.now(tz=ET)

    # Determine T_years
    if args.minutes is not None:
        T_years = max(args.minutes * 60.0, 0.0) / (365.0 * 24.0 * 3600.0)
        expiry_dt = now + timedelta(seconds=args.minutes * 60.0)
    elif args.expiry:
        expiry_dt = parse_datetime_et(args.expiry)
        T_years = years_between(now, expiry_dt)
    else:
        raise SystemExit("Either --minutes or --expiry must be provided (or use --auto with expiry).")

    if T_years <= 0:
        print("Target time is in the past or too near; set a future expiry/minutes.")
        return

    # Spot / iv resolution
    spot = args.spot
    iv = args.iv
    # If user wants auto fetch, attempt it
    if args.auto:
        # If user gave SPX strike (they may), convert to SPY proxy
        strike_for_fetch = args.strike
        if args.index == "SPX":
            strike_for_fetch = args.strike / args.spx_to_spy
            # We'll assume spot must also be SPY-scale
        try:
            fetched_spot, fetched_iv = fetch_spot_and_iv_spy(strike_for_fetch, expiry_dt.strftime("%Y-%m-%d") if args.expiry else None)
            if spot is None:
                spot = fetched_spot
            if iv is None:
                iv = fetched_iv
            # If user passed SPX index, convert strike/spot back to SPY scale for calculation
            if args.index == "SPX":
                # Convert all values to SPY scale (approx)
                spot = spot
                # strike_for_calc below will use args.strike/spx_to_spy
        except Exception as exc:
            print(f"[auto-fetch] error fetching spot/iv via yfinance: {exc}")
    # At this point: spot and iv may still be None if unavailable
    if spot is None:
        raise SystemExit("Spot price not provided and auto-fetch failed. Provide --spot.")
    if iv is None:
        raise SystemExit("IV not provided and auto-fetch failed. Provide --iv (annual decimal, e.g. 0.18).")

    # If index=SPX and we are using SPY proxy, map strike to SPY units (divide by ratio)
    strike_for_calc = args.strike
    spot_for_calc = spot
    if args.index == "SPX":
        strike_for_calc = args.strike / args.spx_to_spy
        # If spot given in SPX units, convert to SPY
        if args.spot is not None:
            spot_for_calc = args.spot / args.spx_to_spy
        # If auto-fetched spot was SPY lastPrice, spot_for_calc already correct

    # Now compute probability
    res = digital_prob_log_normal(spot_for_calc, strike_for_calc, iv, T_years, r=0.0)
    p_model = res["p"]

    # Apply tilt (relative)
    tilt = args.tilt or 0.0
    p_adj = p_model * (1.0 + tilt)
    p_adj = max(0.0001, min(0.9999, p_adj))

    # Output nicely
    print("=== Digital Probability Calculator ===")
    print(f"Run time (ET):        {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Index mode:           {args.index} (SPX->SPY ratio: {args.spx_to_spy})")
    print(f"Spot (calc units):    {spot_for_calc:.6g}")
    print(f"Strike (calc units):  {strike_for_calc:.6g}")
    print(f"Time to expiry (yrs): {T_years:.8f}")
    print(f"Implied vol (ann):    {iv:.4f}")
    print(f"d1: {res['d1']:.4f}   d2: {res['d2']:.4f}")
    print(f"Raw model prob P(S_T>K):      {p_model:.4%}")
    if abs(tilt) > 0:
        print(f"Applied relative tilt: {tilt:+.2%} -> Adjusted prob: {p_adj:.4%}")
    else:
        print("No tilt applied.")
    print("\nHow to use:")
    print("- Compare 'Adjusted prob' to Kalshi YES mid price (e.g. 0.65).")
    print("- If model prob 70% and Kalshi 50% you have ~20% absolute edge (ignoring fees & slippage).")
    print("- IV from yfinance may be delayed; use a real-time IV source for trading.")
    print("\nCaveats:")
    print("- This is a pure risk-neutral digital probability under BS assumptions.")
    print("- For short times (minutes) or heavy skew/tails, consider MC sims, realized vol blending, or jump models.")
    print("- Gamma tilt is a crude relative adjustment; build a calibrated mapping from GEX -> tilt over time.")
    print("=====================================")

if __name__ == "__main__":
    main()
