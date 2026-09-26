"""
market_router.py
================
One interface over both venues, so Kalshi and Polymarket are found, priced
and traded the same way:

    find_all(statement, speaker, context)  -> list of matches
    trade_all(matches, live=False)          -> list of trade results

Every match looks the same regardless of venue:
    {"venue": "kalshi" | "polymarket", "market": <ticker or slug>,
     "title": str, "direction": "YES_UP" | "YES_DOWN", "side": "yes" | "no",
     "relevance": 0-1, "reason": str,
     "quote": {"best_bid": float, "best_ask": float}}

Kalshi markets come from the teammate's kalshi_ticker2 (used as-is,
not modified). Its full-catalog download takes ~2.5 minutes, so this module
caches the catalog for CATALOG_TTL_SECONDS. The first call is slow; every
call after that is fast.

Polymarket markets come from polymarket_client.find_markets.

Direction (does the statement make YES more or less likely?) is decided by
Gemini for both venues, with the stance result passed in as context.
------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  python market_router.py --statement "The committee raised rates; inflation is still too high." \
      --context "Speaker sounded HAWKISH vs usual (z=+3.0)"
  python market_router.py --statement "..." --trade          # dry-run trades
------------------------------------------------------------------------
"""

import argparse
import gzip
import json
import os
import pickle
import re
import time

# Use the fast, high-quota model everywhere unless the caller overrides it.
# (Must be set before importing modules that read GEMINI_MODEL at import time.)
os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash-lite")

import polymarket_client
import kalshi_trader
import trading_common as tc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_CACHE_PATH = os.path.join(SCRIPT_DIR, "kalshi_catalog_cache.pkl.gz")
CATALOG_TTL_SECONDS = 30 * 60

MIN_TRADE_RELEVANCE = 0.7
MAX_TRADES_PER_VENUE = 2
DEFAULT_QTY = 2

_catalog_memory = {"time": 0, "events": None}


# ------------------------------------------------------------------ #
# KALSHI: teammate's finder with a cached catalog
# ------------------------------------------------------------------ #

def _load_kalshi_finder():
    import kalshi_ticker2 as finder
    # Use the same high-quota Gemini model as the rest of the pipeline.
    finder.GEMINI_MODEL = os.environ["GEMINI_MODEL"]

    if getattr(finder, "_router_cache_installed", False):
        return finder
    original_fetch = finder.fetch_live_kalshi_events

    def cached_fetch(*args, **kwargs):
        now = time.time()
        if _catalog_memory["events"] is not None and now - _catalog_memory["time"] < CATALOG_TTL_SECONDS:
            return _catalog_memory["events"]
        if os.path.isfile(CATALOG_CACHE_PATH) and now - os.path.getmtime(CATALOG_CACHE_PATH) < CATALOG_TTL_SECONDS:
            with gzip.open(CATALOG_CACHE_PATH, "rb") as f:
                events = pickle.load(f)
        else:
            print("Downloading full Kalshi catalog (first run, ~2-3 min; cached for 30 min)...")
            events = original_fetch(*args, **kwargs)
            with gzip.open(CATALOG_CACHE_PATH, "wb") as f:
                pickle.dump(events, f)
        _catalog_memory.update(time=now, events=events)
        return events

    finder.fetch_live_kalshi_events = cached_fetch
    finder._router_cache_installed = True
    return finder


def _gemini_json(prompt: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=polymarket_client.GEMINI_API_KEY)
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=os.environ["GEMINI_MODEL"], contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
            )
            return json.loads(re.sub(r"^```(json)?|```$", "", response.text.strip()).strip())
        except Exception as e:
            if any(code in str(e) for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                time.sleep(5 + 5 * attempt)
                continue
            raise
    raise RuntimeError("Gemini unavailable.")


# Who the speaker is, in the words market titles use. The Kalshi search keys
# on topic words ("Federal Reserve", "interest rates"), which a Chair's answer
# often never says out loud.
SPEAKER_CONTEXT = {
    "kevin_warsh": "Federal Reserve Chair on interest rates and inflation",
    "jerome_powell": "Federal Reserve Chair on interest rates and inflation",
    "donald_trump": "President Donald Trump",
}


# Series whose outcome doesn't follow from stance: which words get said, and
# how individual officials vote. A hawkish surprise says nothing about either.
KALSHI_SKIP_SERIES = ("KXFEDMENTION", "KXFEDDISSENT")


def find_kalshi(statement: str, speaker: str = "", context: str = "", top_n: int = 5) -> list:
    finder = _load_kalshi_finder()
    who = SPEAKER_CONTEXT.get(speaker, speaker.replace("_", " ").title())
    query = f"{who}: {statement}" + (f"\n{context}" if context else "")
    tickers = [t for t in finder.find_relevant_tickers(query, top_n=top_n + 3)
               if not str(t.get("ticker", "")).startswith(KALSHI_SKIP_SERIES)][:top_n]
    if not tickers:
        return []

    # The finder gives relevance but not direction. Ask Gemini once for all of them.
    listing = []
    for t in tickers:
        listing.append({"ticker": t.get("ticker"), "event": t.get("event_title"),
                        "market": t.get("market_title")})
    directions = _gemini_json(
        f"Speaker: {speaker or 'unknown'}\nStatement: \"{statement}\"\n"
        + (f"Context: {context}\n" if context else "")
        + "\nFor each prediction market below, does this statement make YES more likely "
        "(YES_UP) or less likely (YES_DOWN)? Return JSON {\"markets\": [{\"ticker\": str, "
        "\"direction\": \"YES_UP\" or \"YES_DOWN\", \"reason\": \"<=15 words\"}]}.\n\n"
        + json.dumps(listing)
    )
    by_ticker = {d.get("ticker"): d for d in directions.get("markets", [])}

    matches = []
    for t in tickers:
        ticker = t.get("ticker")
        d = by_ticker.get(ticker)
        if not d:
            continue
        try:
            q = kalshi_trader.quote(ticker)
        except Exception:
            continue  # e.g. ticker not listed in the demo environment
        matches.append({
            "venue": "kalshi", "market": ticker,
            "title": f"{t.get('event_title')} -- {t.get('market_title')}",
            "direction": d["direction"], "side": "yes" if d["direction"] == "YES_UP" else "no",
            "relevance": float(t.get("relevance_score", 0)),
            "reason": d.get("reason") or t.get("reasoning", ""),
            "quote": {"best_bid": q["best_bid"], "best_ask": q["best_ask"]},
        })
    return matches


# ------------------------------------------------------------------ #
# POLYMARKET
# ------------------------------------------------------------------ #

def find_polymarket(statement: str, speaker: str = "", context: str = "", top_n: int = 5) -> list:
    matches = []
    for m in polymarket_client.find_markets(statement, speaker, top_n=top_n, context=context):
        matches.append({
            "venue": "polymarket", "market": m["slug"], "title": m.get("event") or m["slug"],
            "direction": m["direction"], "side": "yes" if m["direction"] == "YES_UP" else "no",
            "relevance": float(m.get("relevance", 0)), "reason": m.get("reason", ""),
            "quote": {"best_bid": m["quote"]["best_bid"], "best_ask": m["quote"]["best_ask"]},
        })
    return matches


# ------------------------------------------------------------------ #
# BOTH VENUES
# ------------------------------------------------------------------ #

def find_all(statement: str, speaker: str = "", context: str = "",
             venues=("kalshi", "polymarket"), top_n: int = 5) -> list:
    matches = []
    if "kalshi" in venues:
        try:
            matches += find_kalshi(statement, speaker, context, top_n)
        except Exception as e:
            print(f"Kalshi search failed: {e}")
    if "polymarket" in venues:
        try:
            matches += find_polymarket(statement, speaker, context, top_n)
        except Exception as e:
            print(f"Polymarket search failed: {e}")
    matches.sort(key=lambda m: m["relevance"], reverse=True)
    return matches


def priced_in(match: dict, edge: float = 0.03) -> bool:
    """True if the market already reflects the move: the side we'd buy costs
    ~$1 or has no offers. E.g. YES_UP on a market already bid at 0.99."""
    q = match["quote"]
    if match["side"] == "yes":
        return q["best_ask"] is None or q["best_ask"] >= 1 - edge
    return q["best_bid"] is None or q["best_bid"] <= edge


def trade_all(matches: list, live: bool = False, qty: int = DEFAULT_QTY,
              min_relevance: float = MIN_TRADE_RELEVANCE, reason: str = "") -> list:
    """Trade the most relevant matches on each venue with the shared risk limits."""
    results = []
    per_venue = {}
    for m in matches:
        if m["relevance"] < min_relevance:
            continue
        if per_venue.get(m["venue"], 0) >= MAX_TRADES_PER_VENUE:
            continue
        if priced_in(m):
            results.append({"venue": m["venue"], "market": m["market"], "side": m["side"],
                            "error": "already priced in (nothing left to buy on that side)"})
            continue  # doesn't use up a trade slot; try the next match
        place = kalshi_trader.place_trade if m["venue"] == "kalshi" else polymarket_client.place_trade
        try:
            r = place(m["market"], m["side"], qty, live=live, reason=reason)
        except Exception as e:
            r = {"venue": m["venue"], "market": m["market"], "side": m["side"], "error": str(e)}
        results.append(r)
        per_venue[m["venue"]] = per_venue.get(m["venue"], 0) + 1
    return results


def print_matches(matches: list):
    for m in matches:
        q = m["quote"]
        print(f"  [{m['venue']:<10}] {m['market']:<42} {m['direction']:<8} rel {m['relevance']:.2f}  "
              f"bid {q['best_bid']} / ask {q['best_ask']}  -- {m['reason']}")


def print_trades(results: list):
    for r in results:
        if r.get("error"):
            print(f"  [{r['venue']:<10}] {r['market']}: skipped ({r['error']})")
            continue
        print(f"  [{r['venue']:<10}] buy {r['side'].upper()} x{r['qty']} {r['market']} "
              f"limit {r['yes_limit']} (max ${r['max_cost']}): {tc.trade_status(r)}")


def main():
    parser = argparse.ArgumentParser(description="Find and trade related markets on Kalshi + Polymarket")
    parser.add_argument("--statement", required=True)
    parser.add_argument("--speaker", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--venues", nargs="+", default=["kalshi", "polymarket"])
    parser.add_argument("--trade", action="store_true", help="Place trades (dry run unless --live)")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    matches = find_all(args.statement, args.speaker, args.context, args.venues)
    print_matches(matches)
    if args.trade:
        print("\nTrades:")
        print_trades(trade_all(matches, live=args.live))


if __name__ == "__main__":
    main()
