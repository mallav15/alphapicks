import json
from dataclasses import dataclass
from typing import List

@dataclass
class KalshiMarket:
    market_id: str
    title: str
    target_time_et: str     # "14:00", "15:00", "16:00"
    threshold_spx: float
    mid: float              # 0..1
    yes_bid: float
    yes_ask: float

def load_mock_markets(path: str) -> List[KalshiMarket]:
    with open(path, "r") as f:
        data = json.load(f)

    out = []
    for m in data.get("markets", []):
        out.append(KalshiMarket(
            market_id=m["market_id"],
            title=m["title"],
            target_time_et=m["target_time_et"],
            threshold_spx=float(m["threshold_spx"]),
            mid=float(m["mid"]),
            yes_bid=float(m["yes_bid"]),
            yes_ask=float(m["yes_ask"]),
        ))
    return out
