import math
from dataclasses import dataclass
from scipy.stats import norm

@dataclass
class DigitalProbResult:
    p: float
    d2: float
    sigma: float
    T: float

def digital_prob_log_normal(S: float, K: float, sigma: float, T_years: float, r: float = 0.0) -> DigitalProbResult:
    """
    Risk-neutral digital probability under Black–Scholes lognormal assumption:
      P(S_T > K) = N(d2)
    """
    if S <= 0 or K <= 0 or sigma <= 0 or T_years <= 0:
        return DigitalProbResult(p=float("nan"), d2=float("nan"), sigma=sigma, T=T_years)

    vol_sqrt = sigma * math.sqrt(T_years)
    d2 = (math.log(S / K) + (r - 0.5 * sigma * sigma) * T_years) / vol_sqrt
    p = norm.cdf(d2)
    return DigitalProbResult(p=p, d2=d2, sigma=sigma, T=T_years)

def kalshi_fee_per_contract(price_dollars: float, k: float = 0.07) -> float:
    """
    Smooth fee approximation for 1 contract:
      fee ≈ k * P * (1-P), with P in dollars (0..1)
    (Avoids ceil() to keep EV continuous.)
    """
    P = min(max(price_dollars, 0.0), 1.0)
    return k * P * (1.0 - P)

def expected_value_yes(p_true: float, price_yes: float, fee_per_contract: float = 0.0) -> float:
    """
    EV for buying YES at price_yes:
      EV = p_true - price_yes - fee
    """
    return p_true - price_yes - fee_per_contract

def choose_bias(p_true: float, kalshi_mid: float, fee: float, min_edge: float) -> str:
    """
    Generic action label:
      BUY_YES if EV_yes >= min_edge
      BUY_NO  if EV_no  >= min_edge
      else NO_TRADE
    """
    ev_yes = expected_value_yes(p_true, kalshi_mid, fee)
    ev_no  = expected_value_yes(1.0 - p_true, 1.0 - kalshi_mid, fee)
    if ev_yes >= min_edge:
        return "BUY_YES"
    if ev_no >= min_edge:
        return "BUY_NO"
    return "NO_TRADE"
