"""
polymarket_client.py
====================
Polymarket US piece of the pipeline. It does three jobs:

  1. Market data (public, no key needed)
       - search events/markets, read best bid/ask
  2. Market finder (uses Gemini)
       - given a speaker statement, find the Polymarket US markets it could
         move and which way (YES more likely / less likely)
  3. Trading (needs Polymarket US API keys)
       - balances, positions, and order placement with risk limits.
         Orders are DRY RUN by default. Nothing is sent unless you pass --live.

Polymarket US basics (from docs.polymarket.us):
  - Public data:  https://gateway.polymarket.us   (no auth)
  - Trading:      https://api.polymarket.us       (Ed25519-signed requests)
  - Each market has ONE instrument, the YES side. Prices always refer to YES.
    Buying NO = shorting YES. Buying NO "at 0.40" means selling YES at 0.60.
  - Taker fee = 0.0695 * contracts * p * (1 - p)
  - Rate limit: 20 requests/second

------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------
1. pip install requests cryptography google-genai

2. Put these files beside this script (they are in .gitignore):
     polymarketkey.txt      Polymarket US Key ID
     polymarketsecret.txt   Polymarket US Secret Key
     gemapi.txt             Gemini API key (already used by the rest of the repo)
   or set env vars POLYMARKET_KEY_ID, POLYMARKET_SECRET_KEY, GEMINI_API_KEY.

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  Public data (no keys):
    python polymarket_client.py --search "fed decision"
    python polymarket_client.py --bbo rdc-usfed-fomc-2026-10-28-hike25

  Find markets a statement could move (Gemini key needed):
    python polymarket_client.py --find "If inflation is not moving to 2 percent, the Fed still has work to do." --speaker kevin_warsh

  Account (Polymarket keys needed):
    python polymarket_client.py --balance
    python polymarket_client.py --positions

  Trade (dry run unless --live):
    python polymarket_client.py --trade rdc-usfed-fomc-2026-10-28-hike25 --side yes --qty 2
    python polymarket_client.py --trade rdc-usfed-fomc-2026-10-28-hike25 --side yes --qty 2 --live

  Emergency stop: create an empty file named STOP_TRADING in this folder.
  While it exists, no live orders are sent.
------------------------------------------------------------------------
"""

import argparse
import base64
import json
import os
import re
import time
from datetime import datetime, timezone

import requests

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PUBLIC_BASE_URL = "https://gateway.polymarket.us"
TRADING_BASE_URL = "https://api.polymarket.us"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

TAKER_FEE_THETA = 0.0695

# ---- Risk limits (change these on purpose, not by accident) ----
MAX_CONTRACTS_PER_ORDER = 25       # hard cap on contracts in one order
MAX_DOLLARS_PER_ORDER = 10.00      # hard cap on worst-case cost of one order
MAX_DOLLARS_PER_DAY = 50.00        # hard cap on total live spend per UTC day
MAX_SLIPPAGE = 0.02                # pay at most 2 cents worse than best price
KILL_SWITCH_FILE = os.path.join(SCRIPT_DIR, "STOP_TRADING")
TRADE_LOG_PATH = os.path.join(SCRIPT_DIR, "polymarket_trades.jsonl")


def _read_secret_file(filename: str) -> str:
    try:
        with open(os.path.join(SCRIPT_DIR, filename), encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


POLYMARKET_KEY_ID = os.environ.get("POLYMARKET_KEY_ID") or _read_secret_file("polymarketkey.txt")
POLYMARKET_SECRET_KEY = os.environ.get("POLYMARKET_SECRET_KEY") or _read_secret_file("polymarketsecret.txt")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or _read_secret_file("gemapi.txt")


def _amount(value) -> float:
    """Polymarket returns prices as {"value": "0.63", "currency": "USD"}."""
    if isinstance(value, dict):
        value = value.get("value")
    if value is None or value == "":
        return None
    return float(value)


def taker_fee(contracts: float, price: float) -> float:
    return TAKER_FEE_THETA * contracts * price * (1.0 - price)


# ------------------------------------------------------------------ #
# 1. PUBLIC MARKET DATA
# ------------------------------------------------------------------ #

class PolymarketPublic:
    def __init__(self):
        self.session = requests.Session()

    def _get(self, path: str, params: dict = None) -> dict:
        for attempt in range(4):
            resp = self.session.get(PUBLIC_BASE_URL + path, params=params, timeout=20)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Rate limited on {path}")

    def search(self, query: str, limit: int = 10) -> list:
        """Full-text search. Returns a list of events, each with nested markets."""
        data = self._get("/v1/search", {"query": query, "limit": limit})
        return data.get("events", [])

    def market(self, slug: str) -> dict:
        data = self._get(f"/v1/market/slug/{slug}")
        return data.get("market", data)

    def bbo(self, slug: str) -> dict:
        """Best bid/ask for the YES side, as plain floats."""
        data = self._get(f"/v1/markets/{slug}/bbo").get("marketData", {})
        return {
            "slug": slug,
            "best_bid": _amount(data.get("bestBid")),
            "best_ask": _amount(data.get("bestAsk")),
            "last": _amount(data.get("lastTradePx")),
            "open_interest": data.get("openInterest"),
            "state": data.get("state"),
        }

    def list_markets(self, category: str = None, limit: int = 50) -> list:
        params = {"limit": limit, "active": "true", "closed": "false"}
        if category:
            params["categories"] = category
        return self._get("/v1/markets", params).get("markets", [])


# ------------------------------------------------------------------ #
# 2. AUTHENTICATED CLIENT (balances, positions, orders)
# ------------------------------------------------------------------ #

class PolymarketTrader:
    def __init__(self, key_id: str = POLYMARKET_KEY_ID, secret_key: str = POLYMARKET_SECRET_KEY):
        if not key_id or not secret_key:
            raise RuntimeError(
                "Missing Polymarket keys. Put them in polymarketkey.txt and "
                "polymarketsecret.txt, or set POLYMARKET_KEY_ID / POLYMARKET_SECRET_KEY."
            )
        from cryptography.hazmat.primitives.asymmetric import ed25519

        self.key_id = key_id
        self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
            base64.b64decode(secret_key)[:32]
        )
        self.session = requests.Session()

    def _headers(self, method: str, path: str) -> dict:
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path}"
        signature = base64.b64encode(self.private_key.sign(message.encode())).decode()
        return {
            "X-PM-Access-Key": self.key_id,
            "X-PM-Timestamp": timestamp,
            "X-PM-Signature": signature,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, body: dict = None) -> dict:
        url = TRADING_BASE_URL + path
        resp = self.session.request(
            method, url, headers=self._headers(method, path),
            data=json.dumps(body) if body is not None else None, timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Polymarket {method} {path} failed ({resp.status_code}): {resp.text}")
        return resp.json() if resp.text else {}

    def balances(self) -> dict:
        return self._request("GET", "/v1/account/balances")

    def positions(self) -> dict:
        return self._request("GET", "/v1/portfolio/positions")

    def open_orders(self) -> dict:
        return self._request("GET", "/v1/orders/open")

    def preview_order(self, order: dict) -> dict:
        return self._request("POST", "/v1/order/preview", order)

    def create_order(self, order: dict) -> dict:
        return self._request("POST", "/v1/orders", order)

    def cancel_all(self) -> dict:
        return self._request("POST", "/v1/orders/open/cancel", {})


# ------------------------------------------------------------------ #
# 3. RISK-CHECKED TRADE
# ------------------------------------------------------------------ #

def _spent_today() -> float:
    """Sum of worst-case cost of LIVE orders logged today (UTC)."""
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


def _log_trade(entry: dict):
    entry["time"] = datetime.now(timezone.utc).isoformat()
    with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def place_trade(slug: str, side: str, qty: float, live: bool = False, reason: str = "") -> dict:
    """Buy YES or NO on a market with an immediate-or-cancel limit order.

    The limit is the current best price plus MAX_SLIPPAGE, so we never pay
    more than that. Dry run unless live=True.
    """
    side = side.lower()
    if side not in ("yes", "no"):
        raise ValueError("side must be 'yes' or 'no'")

    public = PolymarketPublic()
    quote = public.bbo(slug)
    if quote["state"] != "MARKET_STATE_OPEN":
        raise RuntimeError(f"{slug} is not open for trading (state={quote['state']}).")

    if side == "yes":
        if quote["best_ask"] is None:
            raise RuntimeError("No YES offers on the book.")
        # Buy YES: price is the YES price we are willing to pay.
        yes_limit = round(min(quote["best_ask"] + MAX_SLIPPAGE, 0.99), 2)
        cost_per_contract = yes_limit
        intent = "ORDER_INTENT_BUY_LONG"
    else:
        if quote["best_bid"] is None:
            raise RuntimeError("No YES bids on the book (can't buy NO).")
        # Buy NO = sell YES. Price is the YES price we sell at; lower is worse for us.
        yes_limit = round(max(quote["best_bid"] - MAX_SLIPPAGE, 0.01), 2)
        cost_per_contract = 1.0 - yes_limit
        intent = "ORDER_INTENT_BUY_SHORT"

    qty = float(qty)
    fee = taker_fee(qty, yes_limit)
    max_cost = round(qty * cost_per_contract + fee, 2)

    order = {
        "marketSlug": slug,
        "intent": intent,
        "type": "ORDER_TYPE_LIMIT",
        "price": {"value": f"{yes_limit:.2f}", "currency": "USD"},
        "quantity": qty,
        "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
    }

    # ---- risk checks ----
    problems = []
    if qty > MAX_CONTRACTS_PER_ORDER:
        problems.append(f"qty {qty} > MAX_CONTRACTS_PER_ORDER {MAX_CONTRACTS_PER_ORDER}")
    if max_cost > MAX_DOLLARS_PER_ORDER:
        problems.append(f"max cost ${max_cost} > MAX_DOLLARS_PER_ORDER ${MAX_DOLLARS_PER_ORDER}")
    spent = _spent_today()
    if live and spent + max_cost > MAX_DOLLARS_PER_DAY:
        problems.append(f"daily cap: ${spent:.2f} spent + ${max_cost} > ${MAX_DOLLARS_PER_DAY}")
    if live and os.path.exists(KILL_SWITCH_FILE):
        problems.append("STOP_TRADING file exists (kill switch on)")

    result = {
        "slug": slug, "side": side, "qty": qty, "yes_limit": yes_limit,
        "quote": quote, "est_fee": round(fee, 4), "max_cost": max_cost,
        "live": live, "sent": False, "reason": reason, "order": order,
    }

    if problems:
        result["blocked"] = problems
        _log_trade(result)
        return result

    if not live:
        result["note"] = "DRY RUN: order not sent. Re-run with --live to send."
        _log_trade(result)
        return result

    trader = PolymarketTrader()
    try:
        result["preview"] = trader.preview_order(order)
    except RuntimeError as e:
        result["preview_error"] = str(e)
    response = trader.create_order(order)
    result["sent"] = True
    result["response"] = response
    _log_trade(result)
    return result


# ------------------------------------------------------------------ #
# 4. MARKET FINDER (Gemini)
# ------------------------------------------------------------------ #

def _ask_gemini_json(prompt: str) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY or create gemapi.txt.")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    )
    text = response.text.strip()
    text = re.sub(r"^```(json)?|```$", "", text).strip()
    return json.loads(text)


def find_markets(statement: str, speaker: str = "", top_n: int = 5) -> list:
    """Statement -> list of relevant open Polymarket US markets with a
    direction ("YES more likely" / "YES less likely") and current prices.
    """
    public = PolymarketPublic()

    # Step 1: Gemini turns the statement into short search queries.
    queries_json = _ask_gemini_json(
        "You map public statements to prediction markets.\n"
        f"Speaker: {speaker or 'unknown'}\nStatement: \"{statement}\"\n\n"
        "Return JSON {\"queries\": [...]} with 3 to 6 short search queries "
        "(1-3 words each) for prediction markets this statement could move, "
        "e.g. \"fed decision\", \"rate hike\", \"inflation\", \"recession\"."
    )
    queries = queries_json.get("queries", [])[:6]

    # Step 2: search Polymarket and collect open markets (dedupe by slug).
    candidates = {}
    for query in queries:
        try:
            events = public.search(query, limit=8)
        except requests.RequestException:
            continue
        for event in events:
            if not event.get("active") or event.get("closed"):
                continue
            for market in event.get("markets") or []:
                if market.get("closed") or not market.get("active"):
                    continue
                slug = market.get("slug")
                if slug and slug not in candidates:
                    candidates[slug] = {
                        "slug": slug,
                        "event": event.get("title"),
                        "category": event.get("category"),
                        "rules": (market.get("description") or "")[:400],
                    }
        time.sleep(0.1)

    if not candidates:
        return []

    # Step 3: Gemini picks the relevant ones and the direction.
    listing = list(candidates.values())[:60]
    ranked = _ask_gemini_json(
        f"Speaker: {speaker or 'unknown'}\nStatement: \"{statement}\"\n\n"
        "Below are open prediction markets (slug, event, rules). Pick the ones "
        "this statement plausibly moves. For each, say whether it makes YES more "
        "or less likely, a relevance score 0-1, and a one-sentence reason.\n"
        "Return JSON {\"markets\": [{\"slug\": str, \"direction\": \"YES_UP\" or "
        "\"YES_DOWN\", \"relevance\": float, \"reason\": str}]}. Only use slugs "
        "from the list. Omit anything with relevance below 0.5.\n\n"
        + json.dumps(listing)
    )

    results = []
    for item in ranked.get("markets", []):
        slug = item.get("slug")
        if slug not in candidates:
            continue  # ignore anything Gemini invented
        try:
            quote = public.bbo(slug)
        except requests.RequestException:
            continue
        results.append({**candidates[slug], **item, "quote": quote})

    results.sort(key=lambda r: r.get("relevance", 0), reverse=True)
    return results[:top_n]


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _print_json(data):
    print(json.dumps(data, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Polymarket US client")
    parser.add_argument("--search", help="Search events/markets by text")
    parser.add_argument("--bbo", help="Best bid/ask for a market slug")
    parser.add_argument("--find", help="Statement to map to relevant markets (uses Gemini)")
    parser.add_argument("--speaker", default="", help="Speaker for --find, e.g. kevin_warsh")
    parser.add_argument("--balance", action="store_true", help="Show account balances")
    parser.add_argument("--positions", action="store_true", help="Show open positions")
    parser.add_argument("--trade", help="Market slug to trade")
    parser.add_argument("--side", choices=["yes", "no"], help="Side to buy for --trade")
    parser.add_argument("--qty", type=float, default=1, help="Contracts for --trade")
    parser.add_argument("--live", action="store_true", help="Actually send the order")
    parser.add_argument("--cancel-all", action="store_true", help="Cancel all open orders")
    args = parser.parse_args()

    if args.search:
        for event in PolymarketPublic().search(args.search):
            print(f"\nEVENT: {event.get('slug')} -- {event.get('title')} [{event.get('category')}]")
            for market in (event.get("markets") or [])[:10]:
                state = "closed" if market.get("closed") else "open"
                print(f"   MARKET: {market.get('slug')} -- {market.get('question')} [{state}]")
    elif args.bbo:
        _print_json(PolymarketPublic().bbo(args.bbo))
    elif args.find:
        _print_json(find_markets(args.find, args.speaker))
    elif args.balance:
        _print_json(PolymarketTrader().balances())
    elif args.positions:
        _print_json(PolymarketTrader().positions())
    elif args.cancel_all:
        _print_json(PolymarketTrader().cancel_all())
    elif args.trade:
        if not args.side:
            parser.error("--trade needs --side yes|no")
        _print_json(place_trade(args.trade, args.side, args.qty, live=args.live))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
