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
import math
from contextlib import contextmanager
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Risk limits (shared by every venue) ----
MAX_CONTRACTS_PER_ORDER = 25
MAX_DOLLARS_PER_ORDER = 10.00
MAX_DOLLARS_PER_DAY = 50.00       # total live spend across Kalshi + Polymarket
MAX_SLIPPAGE = 0.02               # pay at most 2 cents worse than the best price

KILL_SWITCH_FILE = os.path.join(SCRIPT_DIR, "STOP_TRADING")
TRADE_LOG_PATH = os.path.join(SCRIPT_DIR, "trades.jsonl")
TRADE_LOCK_PATH = os.path.join(SCRIPT_DIR, "trade_risk.lock")

# Taker fee coefficients: fee = theta * contracts * p * (1 - p)
TAKER_FEE_THETA = {
    "kalshi": 0.07,
    "polymarket": 0.0695,
}


# Trading modes, shared by every entry point:
#   "dry"  : nothing sent (priced against real order books)
#   "demo" : Kalshi orders go to Kalshi's demo exchange (fake money); Polymarket stays dry
#   "live" : real orders on both venues
MODES = ("dry", "demo", "live")


def mode_of(value) -> str:
    """Accepts the old live=True/False or a mode string."""
    if value is True:
        return "live"
    if value in (False, None, ""):
        return "dry"
    value = str(value).lower()
    if value not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {value!r}")
    return value


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
    attempts = {}
    with open(TRADE_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("env") == "demo":
                continue  # fake money doesn't count toward the real daily cap
            if entry.get("live") and entry.get("time", "").startswith(today):
                request_id = entry.get("request_id")
                if request_id:
                    # A durable pending record reserves the budget even when the
                    # exchange response is lost. Later receipts replace it.
                    attempts[request_id] = entry
                elif entry.get("sent"):
                    total += entry.get("max_cost", 0.0)
    total += sum(float(e.get("max_cost", 0)) for e in attempts.values()
                 if e.get("status") not in ("rejected_before_send", "blocked"))
    return total


def risk_check(qty: float, max_cost: float, live: bool) -> list:
    """Returns a list of problems. Empty list means the order is allowed."""
    problems = []
    if not all(math.isfinite(float(x)) and float(x) > 0 for x in (qty, max_cost)):
        return ["quantity and cost must be finite and positive"]
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
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def trade_lock():
    """Serialize every risk check, reservation, exchange send and receipt."""
    with open(TRADE_LOCK_PATH, "a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def trade_status(result: dict) -> str:
    if result.get("blocked"):
        return "BLOCKED: " + "; ".join(result["blocked"])
    if result.get("sent"):
        return "SENT (demo)" if result.get("env") == "demo" else "SENT"
    return "DRY RUN"
