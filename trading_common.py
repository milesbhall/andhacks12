"""
trading_common.py
=================
Shared pieces so Kalshi and Polymarket trade exactly the same way:

  - one set of risk limits (per order, per day across BOTH venues)
  - one kill switch (create a file named STOP_TRADING to block live orders)
  - one trade log (trades.jsonl), every line tagged with its venue
  - the same fee formula shape (both venues charge theta * C * p * (1 - p))

Both venues quote prices from the YES side:
  buy YES at p  -> costs p per contract
  buy NO        -> sell YES at p, costs (1 - p) per contract
"""

import json
import os
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Risk limits (shared by every venue) ----
MAX_CONTRACTS_PER_ORDER = 25
MAX_DOLLARS_PER_ORDER = 10.00
MAX_DOLLARS_PER_DAY = 50.00       # total live spend across Kalshi + Polymarket
MAX_SLIPPAGE = 0.02               # pay at most 2 cents worse than the best price

KILL_SWITCH_FILE = os.path.join(SCRIPT_DIR, "STOP_TRADING")
TRADE_LOG_PATH = os.path.join(SCRIPT_DIR, "trades.jsonl")

# Taker fee coefficients: fee = theta * contracts * p * (1 - p)
TAKER_FEE_THETA = {
    "kalshi": 0.07,
    "polymarket": 0.0695,
}


def taker_fee(venue: str, contracts: float, price: float) -> float:
    return TAKER_FEE_THETA[venue] * contracts * price * (1.0 - price)


def limit_price(side: str, best_bid: float, best_ask: float) -> tuple:
    """Returns (yes_limit_price, cost_per_contract) for buying `side`.

    Buying YES lifts the ask; buying NO sells YES into the bid. Either way we
    allow at most MAX_SLIPPAGE worse than the current best price.
    """
    if side == "yes":
        if best_ask is None:
            raise RuntimeError("No YES offers on the book.")
        yes_limit = round(min(best_ask + MAX_SLIPPAGE, 0.99), 2)
        return yes_limit, yes_limit
    if best_bid is None or best_bid <= 0:
        raise RuntimeError("No YES bids on the book (can't buy NO).")
    yes_limit = round(max(best_bid - MAX_SLIPPAGE, 0.01), 2)
    return yes_limit, 1.0 - yes_limit


def spent_today() -> float:
    """Worst-case cost of LIVE orders actually sent today (UTC), all venues."""
    if not os.path.isfile(TRADE_LOG_PATH):
        return 0.0
    today = datetime.now(timezone.utc).date().isoformat()
    total = 0.0
    with open(TRADE_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("live") and entry.get("sent") and entry.get("time", "").startswith(today):
                total += entry.get("max_cost", 0.0)
    return total


def risk_check(qty: float, max_cost: float, live: bool) -> list:
    """Returns a list of problems. Empty list means the order is allowed."""
    problems = []
    if qty > MAX_CONTRACTS_PER_ORDER:
        problems.append(f"qty {qty} > MAX_CONTRACTS_PER_ORDER {MAX_CONTRACTS_PER_ORDER}")
    if max_cost > MAX_DOLLARS_PER_ORDER:
        problems.append(f"max cost ${max_cost} > MAX_DOLLARS_PER_ORDER ${MAX_DOLLARS_PER_ORDER}")
    if live:
        spent = spent_today()
        if spent + max_cost > MAX_DOLLARS_PER_DAY:
            problems.append(f"daily cap: ${spent:.2f} spent + ${max_cost} > ${MAX_DOLLARS_PER_DAY}")
        if os.path.exists(KILL_SWITCH_FILE):
            problems.append("STOP_TRADING file exists (kill switch on)")
    return problems


def log_trade(entry: dict):
    entry["time"] = datetime.now(timezone.utc).isoformat()
    with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def trade_status(result: dict) -> str:
    if result.get("blocked"):
        return "BLOCKED: " + "; ".join(result["blocked"])
    if result.get("sent"):
        return "SENT"
    return "DRY RUN"
