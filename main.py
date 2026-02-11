import math
from datetime import datetime
import pytz
import pandas as pd
import yfinance as yf

from config import SETTINGS
from model import digital_prob_log_normal, kalshi_fee_per_contract, choose_bias
from gex import compute_gex_from_chain, gex_regime_score
from kalshi_adapter import load_mock_markets

ET = pytz.timezone("America/New_York")

def now_et() -> datetime:
    return datetime.now(tz=ET)

def time_to_target_years(target_hhmm: str) -> float:
    """
    Compute T in years from now to today's target time (ET).
    If target already passed today, returns 0.
    """
    hh, mm = map(int, target_hhmm.split(":"))
    n = now_et()
    target_dt = n.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target_dt <= n:
        return 0.0
    dt = (target_dt - n).total_seconds()
    return dt / (365.0 * 24.0 * 3600.0)

def fetch_spy_spot() -> float:
    t = yf.Ticker("SPY")
    spot = None
    try:
        spot = t.fast_info.get("lastPrice", None)
    except Exception:
        spot = None
    if spot is None:
        hist = t.history(period="1d", interval="1m")
        spot = float(hist["Close"].iloc[-1])
    return float(spot)

def pick_near_expiry(ticker: yf.Ticker, max_days: int) -> list:
    """
    Choose expirations within max_days of today.
    """
    n = now_et().date()
    expiries = []
    for e in ticker.options:
        d = datetime.strptime(e, "%Y-%m-%d").date()
        if 0 <= (d - n).days <= max_days:
            expiries.append(e)
    return expiries

def estimate_iv_near_strike(ticker: yf.Ticker, expiry: str, K: float) -> float:
    """
    Estimate IV near K using the nearest strike call IV.
    (Prototype: can be improved with interpolation & skew.)
    """
    chain = ticker.option_chain(expiry)
    calls = chain.calls.copy()
    calls["dist"] = (calls["strike"] - K).abs()
    calls = calls.sort_values("dist")
    iv = float(calls.iloc[0].get("impliedVolatility", 0.0) or 0.0)
    return iv if iv > 0 else float("nan")

def build_gex_proxy(ticker: yf.Ticker, expiries: list, spot: float) -> float:
    """
    Aggregate GEX proxy across selected expiries (calls + puts).
    Uses each expiry's time-to-expiry as T.
    """
    total_gex = 0.0
    n = now_et()
    for e in expiries:
        exp_dt = ET.localize(datetime.strptime(e, "%Y-%m-%d"))
        exp_dt = exp_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        T = max((exp_dt - n).total_seconds(), 0) / (365.0 * 24.0 * 3600.0)
        if T <= 0:
            continue
        chain = ticker.option_chain(e)
        total_gex += compute_gex_from_chain(spot, chain.calls.copy(), T)
        total_gex += compute_gex_from_chain(spot, chain.puts.copy(), T)
    return float(total_gex)

def main():
    markets = load_mock_markets("mock_kalshi_markets.json")

    spy = yf.Ticker("SPY")
    spot_spy = fetch_spy_spot()
    expiries = pick_near_expiry(spy, SETTINGS.gex_lookahead_days)

    if not expiries:
        print("No near expiries found (market closed/holiday?). Try increasing gex_lookahead_days.")
        return

    # Gamma positioning proxy
    gex_raw = build_gex_proxy(spy, expiries, spot_spy)
    regime = gex_regime_score(gex_raw)  # [-1, +1]

    rows = []
    for m in markets:
        T = time_to_target_years(m.target_time_et)
        if T <= 0:
            continue

        # SPX -> SPY proxy
        K_spy = m.threshold_spx / SETTINGS.spx_to_spy_ratio

        iv = estimate_iv_near_strike(spy, expiries[0], K_spy)
        if not (iv and iv > 0 and math.isfinite(iv)):
            continue

        # Base probability
        p = digital_prob_log_normal(spot_spy, K_spy, iv, T).p
        p = min(max(p, SETTINGS.prob_clip[0]), SETTINGS.prob_clip[1])

        # Gamma tilt (small relative adjustment)
        # long gamma => reduce breakout odds; short gamma => increase breakout odds
        tilt = -regime * SETTINGS.gex_tilt_max_abs
        p_adj = p * (1.0 + tilt)
        p_adj = min(max(p_adj, SETTINGS.prob_clip[0]), SETTINGS.prob_clip[1])

        fee = kalshi_fee_per_contract(m.mid, k=SETTINGS.kalshi_fee_k)

        edge_yes = p_adj - m.mid - fee
        edge_no  = (1.0 - p_adj) - (1.0 - m.mid) - fee

        signal = choose_bias(p_adj, m.mid, fee, SETTINGS.min_edge_net)

        rows.append({
            "market_id": m.market_id,
            "target_time_et": m.target_time_et,
            "threshold_spx": m.threshold_spx,
            "spot_spy": spot_spy,
            "K_spy_proxy": K_spy,
            "iv_proxy": iv,
            "T_years": T,
            "gex_raw": gex_raw,
            "gex_regime": regime,
            "p_model": p,
            "p_adj": p_adj,
            "kalshi_mid": m.mid,
            "fee_est": fee,
            "edge_yes": edge_yes,
            "edge_no": edge_no,
            "best_edge": max(edge_yes, edge_no),
            "signal": signal,
            "title": m.title
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No markets evaluated. Check that target times are in the future and mock markets exist.")
        return

    df = df.sort_values("best_edge", ascending=False).head(SETTINGS.max_trades)

    print("\n=== Gamma-Aware Digital Mispricing Blotter (Prototype / Dry-run) ===")
    print(f"Now (ET): {now_et().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"SPY spot: {spot_spy:.2f}")
    print(f"GEX raw:  {gex_raw:,.0f}")
    print(f"GEX regime score [-1..1]: {regime:.3f}")
    print("\nTop opportunities (fee-adjusted EV thresholds applied):\n")

    show_cols = [
        "target_time_et",
        "threshold_spx",
        "kalshi_mid",
        "p_adj",
        "fee_est",
        "edge_yes",
        "edge_no",
        "signal",
        "title"
    ]
    print(df[show_cols].to_string(index=False))

    print("\nInterpretation tips:")
    print("- signal is based on fee-adjusted EV and a minimum edge threshold.")
    print("- This uses SPY as a proxy for SPX and yfinance IV as a proxy for 0DTE IV; refine for production.")
    print("- Gamma overlay is heuristic; the tilt is intentionally small and bounded.")

if __name__ == "__main__":
    main()
