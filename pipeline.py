"""
pipeline.py
===========
End-to-end: statement -> stance surprise -> markets on Kalshi + Polymarket
-> risk-checked trades -> log.

    speech text
       |
    stance_scorer   : how hawkish/dovish is this, vs. THIS speaker's usual? (z-score)
       |  (only if |z| >= 2)
    market_router   : related markets on both venues + which side the surprise favors
       |
    trade_all       : same risk limits on both venues (dry run unless --live)
       |
    results/<run>.json  (+ Backboard log if a key is present)

Optional sponsor hooks (each skipped if not set up):
    solana_proof   : SHA-256 of each surprise signal written to Solana devnet
    tiger_store    : signals / price ticks / trades as Tiger Data hypertables

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  # One statement
  python pipeline.py --speaker kevin_warsh --statement "We will keep raising rates until inflation is beaten."

  # Replay a held-out press conference answer by answer (the demo)
  python pipeline.py --replay 20260916

  # Faster replay: score stance only, skip market search and trades
  python pipeline.py --replay 20260916 --stance-only

  Add --live to send real orders. Kalshi uses KALSHI_ENV (demo by default).
------------------------------------------------------------------------
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash-lite")

import stance_scorer
import market_router
import trading_common as tc
import tiger_store

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _backboard_log(question: str, answer: str):
    """Best effort: log to Backboard using the teammate's client if keys exist."""
    try:
        import kalshigeminibackboard as kgb
        if not kgb.BACKBOARD_API_KEY:
            return None
        return kgb.BackboardClient(kgb.BACKBOARD_API_KEY).log_exchange(kgb.BACKBOARD_THREAD_ID, question, answer)
    except Exception as e:
        print(f"  (Backboard log skipped: {e})")
        return None


def _solana_proof(record: dict):
    """Best effort: hash of the signal on Solana devnet, if a funded wallet exists."""
    try:
        import solana_proof
        if not os.path.isfile(solana_proof.KEYPAIR_PATH):
            return None
        return solana_proof.prove_signal({k: record[k] for k in (
            "speaker", "statement", "stance", "baseline_mean", "z", "direction", "summary")})
    except Exception as e:
        print(f"  (Solana receipt skipped: {e})")
        return None


def act_on(result, speaker: str, venues, live: bool, qty: int, stance_only: bool,
           source: str = "") -> dict:
    """Given a StanceResult, find markets and trade if it's a surprise."""
    record = {
        "speaker": speaker, "statement": result.statement, "stance": result.stance,
        "baseline_mean": result.baseline_mean, "baseline_stdev": result.baseline_stdev,
        "z": round(result.z, 2), "direction": result.direction, "summary": result.summary,
        "matches": [], "trades": [],
    }
    if not result.is_surprising or stance_only:
        tiger_store.log_signal(record, source=source)
        return record

    # Public receipt first: proves the call was made before we looked at prices.
    proof = _solana_proof(record)
    if proof:
        record["solana"] = proof
        print(f"  Solana receipt: {proof['explorer']}")
    tiger_store.log_signal(record, source=source, solana_sig=(proof or {}).get("signature"))

    context = (f"{speaker} sounded {result.direction} relative to their own usual stance "
               f"(stance {result.stance:+.2f} vs usual {result.baseline_mean:+.2f}, z={result.z:+.1f}). "
               f"Key point: {result.summary}")
    matches = market_router.find_all(result.statement, speaker, context, venues)
    record["matches"] = matches
    tiger_store.log_ticks(matches)
    print("  Markets:")
    market_router.print_matches(matches)

    trades = market_router.trade_all(matches, live=live, qty=qty,
                                     reason=f"{speaker} {result.direction} z={result.z:+.1f}")
    record["trades"] = trades
    tiger_store.log_trades(trades)
    if trades:
        print("  Trades:")
        market_router.print_trades(trades)

    lines = [f"{t.get('venue')} {t.get('side', '').upper()} {t.get('market')}: "
             f"{t.get('error') or tc.trade_status(t)}" for t in trades]
    _backboard_log(
        f"[{speaker}] {result.direction} z={result.z:+.1f} :: {result.statement[:300]}",
        f"Stance {result.stance:+.2f} vs usual {result.baseline_mean:+.2f}. {result.summary}\n"
        + "\n".join(lines),
    )
    return record


def save(run_name: str, records: list) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{run_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"run": run_name, "created": datetime.now(timezone.utc).isoformat(),
                   "records": records}, f, indent=1, default=str, ensure_ascii=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Speech -> stance surprise -> Kalshi + Polymarket trades")
    parser.add_argument("--speaker", default="kevin_warsh")
    parser.add_argument("--statement")
    parser.add_argument("--replay", help="Press conference date to replay, e.g. 20260916")
    parser.add_argument("--venues", nargs="+", default=["kalshi", "polymarket"])
    parser.add_argument("--qty", type=int, default=market_router.DEFAULT_QTY)
    parser.add_argument("--live", action="store_true", help="Send real orders (default: dry run)")
    parser.add_argument("--stance-only", action="store_true", help="Skip market search and trades")
    args = parser.parse_args()

    records = []
    if args.statement:
        r = stance_scorer.score_statement(args.speaker, args.statement)
        print(f"stance {r.stance:+.2f}  usual {r.baseline_mean:+.2f}  z {r.z:+.1f}  -> {r.direction}  | {r.summary}")
        records.append(act_on(r, args.speaker, args.venues, args.live, args.qty, args.stance_only,
                              source="statement"))
        run_name = f"statement_{int(time.time())}"
    elif args.replay:
        # stance_scorer.replay scores every answer in a few batched calls and prints them.
        results = stance_scorer.replay(args.speaker, args.replay)
        surprising = [r for r in results if r.is_surprising]
        print(f"\n{len(surprising)} of {len(results)} answers are surprises.")
        for i, r in enumerate(results, start=1):
            if r.is_surprising and not args.stance_only:
                print(f"\n#{i} {r.direction} z={r.z:+.1f}: {r.summary}")
            records.append(act_on(r, args.speaker, args.venues, args.live, args.qty, args.stance_only,
                                  source=f"replay_{args.replay}"))
        run_name = f"replay_{args.replay}"
    else:
        parser.error("Give --statement or --replay")

    print(f"\nSaved {save(run_name, records)}")


if __name__ == "__main__":
    main()
