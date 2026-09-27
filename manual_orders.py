"""Private operator order flow. Only the local worker imports this module.

The hosted PHP endpoint queues requests and never receives exchange credentials.
No speaker-derived signal can invoke these functions.
"""

import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import trading_common as tc

PREVIEWS = Path(__file__).resolve().parent / "manual_order_previews.json"
PREVIEW_TTL = 90
MARKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,159}$")


def _number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _load_previews():
    try:
        data = json.loads(PREVIEWS.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if v.get("expires_at", 0) > time.time()}
    except (OSError, ValueError, AttributeError):
        return {}


def _save_previews(data):
    tmp = PREVIEWS.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, PREVIEWS)


def _quote(venue, env, market):
    if venue == "kalshi":
        import kalshi_trader
        return kalshi_trader.quote(market, env=env)
    import polymarket_client
    # Bypass the recommendation cache for a manual order decision.
    return polymarket_client.PolymarketPublic()._bbo_uncached(market)


def _validate(raw):
    venue = raw.get("venue")
    env = raw.get("env") or "prod"
    market = raw.get("market")
    outcome = raw.get("outcome")
    side = raw.get("side")
    if venue not in ("kalshi", "polymarket") or env not in ("prod", "demo") or \
            (venue == "polymarket" and env != "prod"):
        raise ValueError("Unsupported venue or environment")
    if not isinstance(market, str) or not MARKET_RE.fullmatch(market):
        raise ValueError("Invalid market")
    if outcome not in ("yes", "no") or side not in ("buy", "sell"):
        raise ValueError("Choose YES or NO and Buy or Sell")
    qty = _number(raw.get("qty"), "Quantity")
    if venue == "kalshi" and qty != int(qty):
        raise ValueError("Kalshi quantity must be a whole contract")
    price = _number(raw.get("limit_price"), "Limit price")
    if not 0.01 <= price <= 0.99 or round(price, 2) != price:
        raise ValueError("Use a limit price from 0.01 to 0.99 in cents")
    if qty > tc.MAX_CONTRACTS_PER_ORDER:
        raise ValueError("Quantity exceeds the per-order cap")
    return dict(venue=venue, env=env, market=market, outcome=outcome,
                side=side, qty=qty, limit_price=price)


def _position_qty(venue, raw, market, outcome):
    if venue == "kalshi":
        rows = raw.get("market_positions")
        if not isinstance(rows, list):
            raise RuntimeError("Could not verify Kalshi positions")
        net = sum(float(row.get("position_fp", 0)) for row in rows if row.get("ticker") == market)
        return max(0.0, net if outcome == "yes" else -net)
    positions = raw.get("positions")
    if isinstance(positions, dict):
        rows = [dict(value, marketSlug=slug) for slug, value in positions.items() if isinstance(value, dict)]
    else:
        rows = positions
    if not isinstance(rows, list):
        raise RuntimeError("Could not verify Polymarket positions")
    total = 0.0
    for row in rows:
        if row.get("marketSlug") != market:
            continue
        # The account response must identify side and quantity unambiguously.
        side = row.get("outcomeSide") or row.get("side")
        if side not in ("OUTCOME_SIDE_YES", "OUTCOME_SIDE_NO", "YES", "NO"):
            raise RuntimeError("Could not verify Polymarket position side")
        if side.lower().endswith(outcome):
            total += _number(row.get("quantity"), "Position quantity")
    return total


def _trader(venue, env):
    if venue == "kalshi":
        import kalshi_trader
        return kalshi_trader.KalshiTrader(env)
    import polymarket_client
    return polymarket_client.PolymarketTrader()


def _unresolved_attempt(venue, market):
    try:
        with open(tc.TRADE_LOG_PATH, encoding="utf-8") as handle:
            latest = {}
            for line in handle:
                row = json.loads(line)
                if row.get("request_id"):
                    latest[row["request_id"]] = row
        return any(row.get("venue") == venue and row.get("market") == market and
                   row.get("status") in ("pending", "unknown") for row in latest.values())
    except FileNotFoundError:
        return False


def _ensure_quote(p, quote):
    if p["venue"] == "kalshi":
        if quote.get("status") not in ("active", "open"):
            raise RuntimeError("Market is not open")
    elif quote.get("state") != "MARKET_STATE_OPEN":
        raise RuntimeError("Market is not open")
    bid, ask = quote.get("best_bid"), quote.get("best_ask")
    if p["side"] == "buy":
        raw_reference = ask if p["outcome"] == "yes" else bid
        reference = raw_reference if p["outcome"] == "yes" else 1 - _number(raw_reference, "Current executable quote")
    else:
        raw_reference = bid if p["outcome"] == "yes" else ask
        reference = raw_reference if p["outcome"] == "yes" else 1 - _number(raw_reference, "Current executable quote")
    reference = _number(reference, "Current executable quote")
    price = p["limit_price"]
    if p["side"] == "buy" and price > reference + tc.MAX_SLIPPAGE + 1e-8:
        raise ValueError("Limit is more than 2 cents above the current quote")
    if p["side"] == "sell" and price < reference - tc.MAX_SLIPPAGE - 1e-8:
        raise ValueError("Limit is more than 2 cents below the current quote")
    return reference


def preview(raw, owner):
    if not owner or len(owner) > 128:
        raise ValueError("Missing order owner")
    p = _validate(raw)
    quote = _quote(p["venue"], p["env"], p["market"])
    reference = _ensure_quote(p, quote)
    fee = tc.taker_fee(p["venue"], p["qty"], p["limit_price"])
    cost = round(p["qty"] * p["limit_price"] + fee, 4) if p["side"] == "buy" else round(fee, 4)
    if p["qty"] * p["limit_price"] + fee > tc.MAX_DOLLARS_PER_ORDER:
        raise ValueError("Order value including fees exceeds the per-order cap")
    if p["side"] == "sell":
        trader = _trader(p["venue"], p["env"])
        held = _position_qty(p["venue"], trader.positions(), p["market"], p["outcome"])
        if held < p["qty"]:
            raise ValueError("Sell quantity exceeds the verified position")
    problems = tc.risk_check(p["qty"], max(cost, fee, 0.0001), p["env"] == "prod")
    if problems:
        raise ValueError("; ".join(problems))
    token = uuid.uuid4().hex
    expires = time.time() + PREVIEW_TTL
    saved = {**p, "owner": owner, "expires_at": expires}
    with tc.trade_lock():
        data = _load_previews()
        data[token] = saved
        _save_previews(data)
    return {**p, "preview_id": token, "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat(),
            "estimated_fee": round(fee, 4), "max_cost": cost,
            "quote": {"best_bid": quote.get("best_bid"), "best_ask": quote.get("best_ask"),
                      "reference_price": reference}}


def submit(preview_id, owner):
    if not isinstance(preview_id, str) or not re.fullmatch(r"[a-f0-9]{32}", preview_id):
        raise ValueError("Invalid preview")
    if os.environ.get("MARKETPULSE_ALLOW_LIVE") != "1":
        raise RuntimeError("Local MARKETPULSE_ALLOW_LIVE=1 opt-in is required")
    with tc.trade_lock():
        previews = _load_previews()
        p = previews.pop(preview_id, None)
        if p is None or p["owner"] != owner:
            raise ValueError("Preview expired, was used, or belongs to another account")
        _save_previews(previews)  # consume before any network access
        if _unresolved_attempt(p["venue"], p["market"]):
            raise RuntimeError("An earlier order on this market has an unknown outcome; check the exchange")
        quote = _quote(p["venue"], p["env"], p["market"])
        _ensure_quote(p, quote)
        trader = _trader(p["venue"], p["env"])
        if p["side"] == "sell":
            held = _position_qty(p["venue"], trader.positions(), p["market"], p["outcome"])
            if held < p["qty"]:
                raise ValueError("Sell quantity exceeds the verified position")
        fee = tc.taker_fee(p["venue"], p["qty"], p["limit_price"])
        cost = round(p["qty"] * p["limit_price"] + fee, 4) if p["side"] == "buy" else round(fee, 4)
        if p["qty"] * p["limit_price"] + fee > tc.MAX_DOLLARS_PER_ORDER:
            raise ValueError("Order value including fees exceeds the per-order cap")
        problems = tc.risk_check(p["qty"], max(cost, fee, 0.0001), p["env"] == "prod")
        if problems:
            raise ValueError("; ".join(problems))
        request_id = str(uuid.uuid4())
        if p["venue"] == "kalshi":
            book_side = "bid" if (p["outcome"] == "yes") == (p["side"] == "buy") else "ask"
            yes_price = p["limit_price"] if p["outcome"] == "yes" else 1 - p["limit_price"]
            order = {"ticker": p["market"], "side": book_side, "count": f"{p['qty']:.2f}",
                     "price": f"{yes_price:.4f}", "time_in_force": "immediate_or_cancel",
                     "self_trade_prevention_type": "taker_at_cross", "reduce_only": p["side"] == "sell",
                     "client_order_id": request_id}
        else:
            intent = ("ORDER_INTENT_BUY_" if p["side"] == "buy" else "ORDER_INTENT_SELL_") + \
                     ("LONG" if p["outcome"] == "yes" else "SHORT")
            yes_price = p["limit_price"] if p["outcome"] == "yes" else 1 - p["limit_price"]
            order = {"marketSlug": p["market"], "intent": intent, "type": "ORDER_TYPE_LIMIT",
                     "price": {"value": f"{yes_price:.2f}", "currency": "USD"}, "quantity": p["qty"],
                     "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
                     "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_MANUAL"}
        entry = {**{k: p[k] for k in ("venue", "env", "market", "outcome", "side", "qty", "limit_price")},
                 "request_id": request_id, "max_cost": cost, "live": True,
                 "status": "pending", "sent": False, "reason": "manual order", "owner": owner}
        tc.log_trade(entry)  # durable reservation before the only send attempt
        try:
            response = trader.create_order(order)
        except Exception:
            entry["status"] = "unknown"
            tc.log_trade(entry)
            raise RuntimeError("Order outcome is unknown. Check the exchange before placing another order") from None
        entry["status"] = "submitted"
        entry["sent"] = True
        tc.log_trade(entry)
        return {"status": "submitted", "request_id": request_id,
                "order_id": response.get("order_id") or response.get("id") or (response.get("order") or {}).get("id"),
                "venue": p["venue"], "env": p["env"], "market": p["market"],
                "side": p["side"], "outcome": p["outcome"], "qty": p["qty"],
                "limit_price": p["limit_price"], "fill_count": response.get("fill_count"),
                "remaining_count": response.get("remaining_count")}


def snapshot():
    result = {"as_of": datetime.now(timezone.utc).isoformat(), "connected": False,
              "orders": [], "positions": [], "recent_attempts": [], "errors": []}
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    accounts = [("kalshi", "prod"), ("polymarket", "prod")]
    if os.path.isfile(os.path.join(here, "kalshikey_demo.txt")):
        accounts.insert(1, ("kalshi", "demo"))     # the Kalshi demo account (fake money) too
    for venue, env in accounts:
        try:
            trader = _trader(venue, env)
            if venue == "kalshi":
                cursor = ""
                for _ in range(5):
                    endpoint = "/portfolio/orders?status=resting&limit=100" + ("&cursor=" + cursor if cursor else "")
                    page = trader._request("GET", endpoint)
                    for row in page.get("orders", []):
                        result["orders"].append({"venue": venue, "env": env, "id": row.get("order_id"),
                            "market": row.get("ticker"), "side": row.get("action"), "outcome": row.get("side"),
                            "qty": row.get("remaining_count_fp"), "limit_price": row.get("yes_price_dollars"),
                            "status": row.get("status")})
                    cursor = page.get("cursor") or ""
                    if not cursor: break
                cursor = ""
                for _ in range(5):
                    endpoint = "/portfolio/positions?limit=100" + ("&cursor=" + cursor if cursor else "")
                    page = trader._request("GET", endpoint)
                    for row in page.get("market_positions", []):
                        net = float(row.get("position_fp", 0))
                        if net:
                            result["positions"].append({"venue": venue, "env": env, "market": row.get("ticker"),
                                "outcome": "yes" if net > 0 else "no", "qty": abs(net)})
                    cursor = page.get("cursor") or ""
                    if not cursor: break
            else:
                for row in trader.open_orders().get("orders", []):
                    result["orders"].append({"venue": venue, "env": "prod", "id": row.get("id"),
                        "market": row.get("marketSlug"), "side": row.get("action"), "outcome": row.get("outcomeSide"),
                        "qty": row.get("leavesQuantity"), "limit_price": (row.get("price") or {}).get("value"),
                        "status": row.get("state")})
                cursor = ""
                for _ in range(5):
                    endpoint = "/v1/portfolio/positions?limit=100" + ("&cursor=" + cursor if cursor else "")
                    page = trader._request("GET", endpoint)
                    positions = page.get("positions", {})
                    if isinstance(positions, dict):
                        positions = [dict(value, marketSlug=slug) for slug, value in positions.items() if isinstance(value, dict)]
                    for row in positions:
                        result["positions"].append({"venue": venue, "env": "prod", "market": row.get("marketSlug"),
                            "outcome": row.get("outcomeSide") or row.get("side"), "qty": row.get("quantity")})
                    cursor = page.get("nextCursor") or ""
                    if page.get("eof", True) or not cursor: break
            result["connected"] = True
        except Exception as exc:
            result["errors"].append(f"{venue}{' demo' if env == 'demo' else ''}: account data unavailable ({type(exc).__name__})")
    try:
        with open(tc.TRADE_LOG_PATH, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        latest = {}
        cutoff = datetime.now(timezone.utc).timestamp() - 30 * 60   # attempts clear after 30 minutes
        for n, row in enumerate(rows):
            try:
                when = datetime.fromisoformat(str(row.get("time")).replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            if when >= cutoff:   # manual tickets (request_id) and desk-triggered orders alike
                latest[row.get("request_id") or f"desk{n}"] = row
        result["recent_attempts"] = [{**{k: row.get(k) for k in
            ("time", "venue", "env", "market", "side", "outcome", "qty", "limit_price", "request_id", "reason")},
            "outcome": row.get("outcome") or row.get("side"), "side": row.get("action") or "buy",
            "status": row.get("status") or tc.trade_status(row)}
            for row in list(latest.values())[-20:][::-1]]
    except (OSError, ValueError):
        pass
    return result
