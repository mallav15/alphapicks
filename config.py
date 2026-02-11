from dataclasses import dataclass

@dataclass
class Settings:
    # Trade selection controls (research only)
    min_edge_net: float = 0.04          # 4% minimum EV edge after fee estimate
    max_trades: int = 12                # show top-N opportunities
    prob_clip: tuple = (0.01, 0.99)

    # Gamma tilt controls (small & bounded to avoid overfitting)
    gex_tilt_max_abs: float = 0.06      # max +/- 6% relative tilt
    gex_lookahead_days: int = 2         # include expiries within N days

    # SPX->SPY mapping (prototype)
    spx_to_spy_ratio: float = 10.0

    # Fee approximation constant (see Kalshi fee schedule form)
    kalshi_fee_k: float = 0.07

SETTINGS = Settings()
