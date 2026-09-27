"""
kalshi_trader.py
================
Kalshi order placement that behaves exactly like polymarket_client.place_trade:
same risk limits, kill switch, trade log, YES-side pricing, and dry run by
default. Uses Kalshi's V2 order endpoint (POST /portfolio/events/orders),
which quotes everything from the YES side:
    side "bid" = buy YES,  side "ask" = sell YES (= buy NO).

Environment:
    KALSHI_ENV=prod  (default)  -> https://external-api.kalshi.com       (real money; dry run unless --live)
    KALSHI_ENV=demo             -> https://external-api.demo.kalshi.co   (fake money; needs a demo key)
Dry runs are priced against PRODUCTION order books (demo books are mostly
empty). Live orders are priced against the environment they are sent to.

Keys (same files the rest of the repo uses, all gitignored):
    kalshikey.txt   Kalshi API key ID          (or env KALSHI_API_KEY_ID)
    privkey.txt     Kalshi RSA private key PEM (or env KALSHI_PRIVATE_KEY_PATH)
Demo and prod keys are different; point these at the right ones.

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  python kalshi_trader.py --quote KXFEDHIKE-2-27DEC31
  python kalshi_trader.py --balance
  python kalshi_trader.py --balance --env demo      # demo account (kalshikey_demo.txt + privkey_demo.txt)
  python kalshi_trader.py --trade KXFEDHIKE-2-27DEC31 --side yes --qty 2          # dry run
  python kalshi_trader.py --trade KXFEDHIKE-2-27DEC31 --side yes --qty 2 --live   # sends it
------------------------------------------------------------------------
"""

import argparse
import base64
import json
import os
import time
import uuid

import requests

import trading_common as tc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_URLS = {
    "demo": "https://external-api.demo.kalshi.co",
    "prod": "https://external-api.kalshi.com",
}
API_PREFIX = "/trade-api/v2"
KALSHI_ENV = os.environ.get("KALSHI_ENV", "prod").lower()   # our key is a kalshi.com (prod) key; orders are still dry runs unless --live


def _read_secret_file(filename: str) -> str:
    try:
        with open(os.path.join(SCRIPT_DIR, filename), encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _base_url(env: str = None) -> str:
    env = env or KALSHI_ENV
    if env not in BASE_URLS:
        raise RuntimeError(f"KALSHI_ENV must be 'demo' or 'prod', not {env!r}")
    return BASE_URLS[env]


def _float(value):
    if value is None or value == "":
        return None
    return float(value)


# ------------------------------------------------------------------ #
# PUBLIC DATA
# ------------------------------------------------------------------ #

def quote(ticker: str, env: str = "prod") -> dict:
    """Best YES bid/ask for a market, as floats (public, no key needed).

    Defaults to PRODUCTION prices: the demo exchange's order books are mostly
    empty, so real prices are what a dry run should be judged against.
    """
    resp = requests.get(f"{_base_url(env)}{API_PREFIX}/markets/{ticker}", timeout=20)
    resp.raise_for_status()
    m = resp.json()["market"]
    bid = _float(m.get("yes_bid_dollars"))
    ask = _float(m.get("yes_ask_dollars"))
    return {
        "ticker": ticker,
        "title": m.get("title"),
        "subtitle": m.get("yes_sub_title"),
        "best_bid": bid if bid else None,           # 0 bid means nobody is bidding
        "best_ask": ask if ask and ask < 1 else None,  # 1.00 ask means nobody is offering
        "last": _float(m.get("last_price_dollars")),
        "status": m.get("status"),
        "env": env,
    }


# ------------------------------------------------------------------ #
# AUTHENTICATED CLIENT
# ------------------------------------------------------------------ #

# Demo and real Kalshi accounts have different keys.
KEY_FILES = {
    "prod": ("kalshikey.txt", "privkey.txt"),
    "demo": ("kalshikey_demo.txt", "privkey_demo.txt"),
}


class KalshiTrader:
    def __init__(self, env: str = None):
        from cryptography.hazmat.primitives import serialization

        self.env = env or KALSHI_ENV
        key_file, pem_file = KEY_FILES[self.env]
        env_ok = self.env == KALSHI_ENV   # env-var keys belong to the default environment
        self.key_id = (env_ok and os.environ.get("KALSHI_API_KEY_ID")) or _read_secret_file(key_file)
        key_value = (env_ok and os.environ.get("KALSHI_PRIVATE_KEY_PATH")) or ""
        key_path = key_value if key_value and os.path.isfile(key_value) else os.path.join(SCRIPT_DIR, pem_file)
        if not self.key_id or (not key_value and not os.path.isfile(key_path)):
            raise RuntimeError(f"Missing Kalshi {self.env} keys: need {key_file} and {pem_file}.")
        if "BEGIN" in key_value and "PRIVATE KEY" in key_value:
            key_bytes = key_value.encode()
        else:
            with open(key_path, "rb") as f:
                key_bytes = f.read()
        if not key_bytes:
            raise RuntimeError(f"Missing Kalshi {self.env} private key.")
        self.private_key = serialization.load_pem_private_key(key_bytes, password=None)
        self.session = requests.Session()

    def _headers(self, method: str, path: str) -> dict:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        from cryptography.hazmat.primitives.asymmetric import ed25519

        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path.split('?')[0]}".encode()
        if isinstance(self.private_key, ed25519.Ed25519PrivateKey):
            signature = self.private_key.sign(message)      # Ed25519 keys sign the message directly
        else:
            signature = self.private_key.sign(               # RSA keys: RSA-PSS with SHA-256
                message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, body: dict = None) -> dict:
        path = API_PREFIX + endpoint
        resp = self.session.request(
            method, _base_url(self.env) + path, headers=self._headers(method, path),
            data=json.dumps(body) if body is not None else None, timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Kalshi {method} {endpoint} failed ({resp.status_code}): {resp.text}")
        return resp.json() if resp.text else {}

    def balance(self) -> dict:
        return self._request("GET", "/portfolio/balance")

    def positions(self) -> dict:
        return self._request("GET", "/portfolio/positions")

    def create_order(self, order: dict) -> dict:
        return self._request("POST", "/portfolio/events/orders", order)


# ------------------------------------------------------------------ #
# RISK-CHECKED TRADE (same behavior as polymarket_client.place_trade)
# ------------------------------------------------------------------ #

def place_trade(ticker: str, side: str, qty: float, live: bool = False, reason: str = "",
                env: str = None) -> dict:
    """env overrides KALSHI_ENV for this order ("demo" = Kalshi's fake-money exchange)."""
    env = env or KALSHI_ENV
    side = side.lower()
    if side not in ("yes", "no"):
        raise ValueError("side must be 'yes' or 'no'")

    # Dry runs price against real (prod) books; live orders use the book they'll hit.
    q = quote(ticker, env=env if live else "prod")
    if q["status"] not in ("active", "open"):
        raise RuntimeError(f"{ticker} is not open for trading (status={q['status']}).")

    yes_limit, cost_per_contract = tc.limit_price(side, q["best_bid"], q["best_ask"])
    qty = int(qty)  # Kalshi event markets trade whole contracts
    fee = tc.taker_fee("kalshi", qty, yes_limit)
    max_cost = round(qty * cost_per_contract + fee, 2)

    order = {
        "ticker": ticker,
        "side": "bid" if side == "yes" else "ask",
        "count": f"{qty:.2f}",
        "price": f"{yes_limit:.4f}",
        "time_in_force": "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": str(uuid.uuid4()),
    }

    result = {
        "venue": "kalshi", "env": env, "market": ticker, "side": side, "qty": qty,
        "yes_limit": yes_limit, "quote": q, "est_fee": round(fee, 4), "max_cost": max_cost,
        "live": live, "sent": False, "reason": reason, "order": order,
    }

    problems = tc.risk_check(qty, max_cost, live)
    if problems:
        result["blocked"] = problems
    elif not live:
        result["note"] = "DRY RUN: order not sent. Re-run with --live to send."
    else:
        result["response"] = KalshiTrader(env).create_order(order)
        result["sent"] = True

    tc.log_trade(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=f"Kalshi trader (KALSHI_ENV={KALSHI_ENV})")
    parser.add_argument("--quote", help="Best bid/ask for a ticker")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--positions", action="store_true")
    parser.add_argument("--trade", help="Ticker to trade")
    parser.add_argument("--side", choices=["yes", "no"])
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--live", action="store_true", help="Actually send the order")
    parser.add_argument("--env", choices=["prod", "demo"], help="Override KALSHI_ENV for this command")
    args = parser.parse_args()
    env = args.env or KALSHI_ENV

    if args.quote:
        print(json.dumps(quote(args.quote), indent=2))
    elif args.balance:
        print(json.dumps(KalshiTrader(env).balance(), indent=2))
    elif args.positions:
        print(json.dumps(KalshiTrader(env).positions(), indent=2))
    elif args.trade:
        if not args.side:
            parser.error("--trade needs --side yes|no")
        print(json.dumps(place_trade(args.trade, args.side, args.qty, live=args.live, env=env), indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
