import math
import pandas as pd
from scipy.stats import norm

def bs_gamma(S: float, K: float, sigma: float, T: float, r: float = 0.0) -> float:
    """
    Black–Scholes gamma for a European option (call/put share same gamma).
    """
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

def compute_gex_from_chain(spot: float, chain: pd.DataFrame, T_years: float, multiplier: float = 100.0) -> float:
    """
    Rough gamma-dollar exposure proxy:
      sum(OI * gamma * spot^2 * multiplier)
    """
    if chain is None or chain.empty:
        return 0.0

    gex = 0.0
    for _, row in chain.iterrows():
        K = float(row.get("strike", 0.0) or 0.0)
        oi = float(row.get("openInterest", 0.0) or 0.0)
        iv = float(row.get("impliedVolatility", 0.0) or 0.0)

        if K <= 0 or oi <= 0 or iv <= 0:
            continue

        gamma = bs_gamma(spot, K, iv, T_years)
        if math.isnan(gamma) or gamma <= 0:
            continue

        gex += oi * gamma * (spot ** 2) * multiplier

    return float(gex)

def gex_regime_score(gex_value: float) -> float:
    """
    Map raw GEX into bounded regime score in [-1, +1] using tanh.
    Positive score = treat as "long gamma" vibe (pinning/mean reversion)
    Negative score = treat as "short gamma" vibe (trend/expansion)

    NOTE: This is a heuristic mapping; refine sign logic later.
    """
    scale = 1e9
    return math.tanh(gex_value / scale)
