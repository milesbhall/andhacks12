"""
tiger_store.py
==============
Time-series storage on Tiger Data (TimescaleDB / Postgres).

Three hypertables, all keyed by time, so the dashboard can line up
"what the speaker said" with "what the market did" on one timeline:

  signals   one row per scored statement (stance, z-score, direction)
  ticks     market prices (venue, market, bid, ask) sampled around signals
  trades    every order attempt from trading_common (dry run or live)

Connection string (gitignored): tigerdb.txt, or env TIGER_DB_URL.
Get it from the Tiger Cloud console: service -> Connect -> "Connection string".
Looks like: postgres://tsdbadmin:<password>@<host>:<port>/tsdb?sslmode=require

Everything here is best effort: if no connection string is set, calls are
silently skipped so the pipeline still runs.

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  pip install psycopg2-binary
  python tiger_store.py --init                   # create tables (once)
  python tiger_store.py --recent                 # last 10 signals
  python tiger_store.py --sample KXFEDHIKE-2-27DEC31 --venue kalshi   # log one price tick
------------------------------------------------------------------------
"""

import argparse
import json
import os
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    time        TIMESTAMPTZ NOT NULL,
    speaker     TEXT NOT NULL,
    statement   TEXT,
    stance      DOUBLE PRECISION,
    baseline    DOUBLE PRECISION,
    z           DOUBLE PRECISION,
    direction   TEXT,
    summary     TEXT,
    source      TEXT,
    solana_sig  TEXT
);
CREATE TABLE IF NOT EXISTS ticks (
    time    TIMESTAMPTZ NOT NULL,
    venue   TEXT NOT NULL,
    market  TEXT NOT NULL,
    bid     DOUBLE PRECISION,
    ask     DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS trades (
    time       TIMESTAMPTZ NOT NULL,
    venue      TEXT NOT NULL,
    market     TEXT NOT NULL,
    side       TEXT,
    qty        DOUBLE PRECISION,
    yes_limit  DOUBLE PRECISION,
    max_cost   DOUBLE PRECISION,
    live       BOOLEAN,
    status     TEXT,
    reason     TEXT
);
SELECT create_hypertable('signals', 'time', if_not_exists => TRUE);
SELECT create_hypertable('ticks',   'time', if_not_exists => TRUE);
SELECT create_hypertable('trades',  'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS ticks_market_time ON ticks (market, time DESC);
"""

_conn = None


def db_url() -> str:
    url = os.environ.get("TIGER_DB_URL", "")
    path = os.path.join(SCRIPT_DIR, "tigerdb.txt")
    if not url and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            url = f.read().strip()
    return url


def enabled() -> bool:
    return bool(db_url())


def connect():
    global _conn
    if _conn is None or _conn.closed:
        import psycopg2
        _conn = psycopg2.connect(db_url())
        _conn.autocommit = True
    return _conn


def _now():
    return datetime.now(timezone.utc)


def init():
    with connect().cursor() as cur:
        cur.execute(SCHEMA)


def _safe(fn):
    """Storage must never break trading: log and move on."""
    def wrapper(*args, **kwargs):
        if not enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"  (Tiger Data write skipped: {e})")
            return None
    return wrapper


@_safe
def log_signal(record: dict, source: str = "", solana_sig: str = None):
    with connect().cursor() as cur:
        cur.execute(
            "INSERT INTO signals VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (_now(), record.get("speaker"), record.get("statement"), record.get("stance"),
             record.get("baseline_mean"), record.get("z"), record.get("direction"),
             record.get("summary"), source, solana_sig))


@_safe
def log_ticks(matches: list):
    rows = [(_now(), m["venue"], m["market"], m["quote"].get("best_bid"), m["quote"].get("best_ask"))
            for m in matches]
    with connect().cursor() as cur:
        cur.executemany("INSERT INTO ticks VALUES (%s,%s,%s,%s,%s)", rows)


@_safe
def log_trades(results: list):
    import trading_common as tc
    rows = []
    for r in results:
        status = r.get("error") and f"SKIPPED: {r['error']}" or tc.trade_status(r)
        rows.append((_now(), r.get("venue"), r.get("market"), r.get("side"), r.get("qty"),
                     r.get("yes_limit"), r.get("max_cost"), r.get("live"), status, r.get("reason")))
    with connect().cursor() as cur:
        cur.executemany("INSERT INTO trades VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)


def query(sql: str, params=None) -> list:
    with connect().cursor() as cur:
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def main():
    parser = argparse.ArgumentParser(description="Tiger Data (TimescaleDB) storage")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--recent", action="store_true")
    parser.add_argument("--sample", help="Log one price tick for this market")
    parser.add_argument("--venue", choices=["kalshi", "polymarket"], default="kalshi")
    args = parser.parse_args()

    if not enabled():
        raise SystemExit("No connection string: put it in tigerdb.txt or TIGER_DB_URL.")
    if args.init:
        init()
        print("Tables ready: signals, ticks, trades (hypertables).")
    elif args.recent:
        for row in query("SELECT time, speaker, direction, z, summary FROM signals ORDER BY time DESC LIMIT 10"):
            print(json.dumps(row, default=str))
    elif args.sample:
        if args.venue == "kalshi":
            import kalshi_trader
            q = kalshi_trader.quote(args.sample)
        else:
            import polymarket_client
            q = polymarket_client.PolymarketPublic().bbo(args.sample)
        log_ticks([{"venue": args.venue, "market": args.sample, "quote": q}])
        print("logged", q.get("best_bid"), q.get("best_ask"))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
