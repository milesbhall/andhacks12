"""
publish.py
==========
Sends the live desk's state to the hosted dashboard (web/index.html on Hostinger)
about once a second, so anyone with the link can watch in real time.

  python publish.py                 # uses hostinger_url.txt + hostinger_token.txt
  python publish.py --once          # send one update and exit (test)

What leaves the laptop: the live state (scores, surprises, orders, watchlist)
and the recommenders' output. Anything that isn't the speaker (reporter
questions, room chatter picked up by the mic) is sent WITHOUT its text.

Files (gitignored): hostinger_url.txt  e.g. https://<site>.hostingersite.com
                    hostinger_token.txt  upload token that matches web/update.php
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(SCRIPT_DIR, "live_state.json")
RECOMMENDERS = {"Kalshi": os.path.join(SCRIPT_DIR, "live_recommendations.json"),
                "Polymarket": os.path.join(SCRIPT_DIR, "live_polymarket_recommendations.json")}
HEARTBEAT_SECONDS = 5


def _read(name: str) -> str:
    path = os.path.join(SCRIPT_DIR, name)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _fields(value, names):
    """Explicit public schema; never mirror arbitrary local state or API payloads."""
    return {key: value[key] for key in names if isinstance(value, dict) and key in value}


def _items(value, names, limit):
    if not isinstance(value, list):
        return []
    out = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        public = _fields(item, names)
        if 'quote' in public:
            public['quote'] = _fields(public['quote'], ('best_bid', 'best_ask'))
        out.append(public)
    return out


def build_payload() -> dict:
    raw = _load(STATE_PATH) or {}
    live = _fields(raw, ("speaker", "source", "started", "status", "live_orders", "mode"))
    live["baseline"] = _fields(raw.get("baseline"), ("mean", "stdev", "n"))
    watch = raw.get("watchlist") or {}
    market_fields = ("venue", "market", "title", "side", "direction", "relevance", "quote")
    live["watchlist"] = {direction: _items(watch.get(direction), market_fields, 20)
                         for direction in ("HAWKISH", "DOVISH")}
    live["chunks"] = _items(raw.get("chunks"),
        ("time", "role", "text", "stance", "z", "direction", "summary", "score_ms", "latency_ms"), 500)
    live["alerts"] = _items(raw.get("alerts"),
        ("time", "direction", "z", "summary", "statement", "latency_ms", "orders"), 100)
    live["trades"] = _items(raw.get("trades"),
        ("time", "venue", "market", "title", "side", "qty", "yes_limit", "max_cost", "trigger", "skipped", "status"), 200)
    try:
        live_updated_at = datetime.fromtimestamp(os.path.getmtime(STATE_PATH), timezone.utc).isoformat()
    except OSError:
        live_updated_at = None
    for chunk in live["chunks"]:
        if chunk.get("role") != "speaker":
            chunk["text"] = ""          # never publish what someone else in the room said
            chunk["summary"] = ""
    recs = {}
    for venue, path in RECOMMENDERS.items():
        data = _load(path)
        if data:
            rec_fields = ("venue", "ticker", "market_id", "event_title", "market_title",
                          "relevance_score", "side", "quote")
            recs[venue] = {
                "baseline": _fields(data.get("baseline"), ("surprise_direction", "z")),
                "candidates": _items(data.get("candidates"), rec_fields, 20),
                "recommendations": _items(data.get("recommendations"), rec_fields, 20),
            }
    return {"published_at": datetime.now(timezone.utc).isoformat(),
            "live_updated_at": live_updated_at, "live": live, "recommenders": recs}


def send(url: str, token: str, payload: dict) -> bool:
    r = requests.post(url.rstrip("/") + "/update.php", data=json.dumps(payload, default=str),
                      headers={"X-Upload-Token": token, "Content-Type": "application/json"}, timeout=10)
    if r.status_code != 200:
        print(f"  upload failed ({r.status_code}): {r.text[:120]}")
    return r.status_code == 200


def main():
    parser = argparse.ArgumentParser(description="Publish the live desk to the hosted dashboard")
    parser.add_argument("--url", default=os.environ.get("HOSTINGER_URL") or _read("hostinger_url.txt"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("HOSTINGER_TOKEN") or _read("hostinger_token.txt")
    if not args.url or not token:
        raise SystemExit("Need hostinger_url.txt and hostinger_token.txt (or HOSTINGER_URL / HOSTINGER_TOKEN).")

    last_hash, last_sent = None, 0.0
    print(f"Publishing to {args.url} (Ctrl+C to stop)")
    while True:
        payload = build_payload()
        body = json.dumps({k: v for k, v in payload.items() if k != "published_at"}, sort_keys=True, default=str)
        digest = hashlib.sha256(body.encode()).hexdigest()
        if digest != last_hash or time.time() - last_sent > HEARTBEAT_SECONDS:
            try:
                if send(args.url, token, payload):
                    last_hash, last_sent = digest, time.time()
            except requests.RequestException as e:
                print(f"  upload error: {e}")
        if args.once:
            print("sent" if last_hash else "not sent")
            return
        time.sleep(1.0)


if __name__ == "__main__":
    main()
