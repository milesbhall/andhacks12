"""
live.py
=======
Trade while the speech is still happening.

    audio ──► speechtxt.py (ElevenLabs Scribe v2 Realtime) ──► live_transcript.json
                                                                   │  (new sentences every few seconds)
                                                                   ▼
    live.py:  buffer ~25+ words ─► Gemini stance score (+ is this the speaker or a reporter?)
              ─► z vs. the speaker's usual stance ─► |z| >= 2 ─► orders on the pre-picked markets
              ─► Solana receipt, Tiger Data rows, Backboard memory (in the background)
              ─► live_state.json  (the dashboard's Live tab reads this every second)

Why pre-pick markets: searching 130k Kalshi markets takes minutes, which is
too slow mid-speech. Before the speech starts we find the markets a HAWKISH
surprise should move and the ones a DOVISH surprise should move (the
"watchlist"). When a surprise hits, we only refresh prices and send orders.

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  # Live: run Dylan's transcriber in one terminal ...
  python speechtxt.py --url https://www.federalreserve.gov/live-broadcast.htm
  # ... and this in another
  python live.py --speaker kevin_warsh

  # Demo without audio: feed the Sept 16 transcript at 10x speaking speed
  python live.py --simulate 20260916 --speed 10

  Options: --venues kalshi polymarket, --qty 2, --live (real orders), --no-prewarm
------------------------------------------------------------------------
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash-lite")

import stance_scorer
import trading_common as tc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_PATH = os.path.join(SCRIPT_DIR, "live_transcript.json")
STATE_PATH = os.path.join(SCRIPT_DIR, "live_state.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MIN_WORDS = 25          # don't score fragments shorter than this
MAX_WORDS = 90          # score even without a sentence end once the buffer is this long
POLL_SECONDS = 0.3
WORDS_PER_SECOND = 2.5  # normal speaking pace, used by --simulate

# What a clearly hawkish / dovish remark from this speaker would say. Used
# before the speech to find which markets each kind of surprise should move.
PROTOTYPES = {
    "HAWKISH": "Inflation is still too high. The committee is prepared to raise interest rates "
               "further and keep policy tight for longer.",
    "DOVISH": "The labor market is weakening and inflation is coming down. The committee is "
              "prepared to cut interest rates soon to support the economy.",
}

LIVE_PROMPT = stance_scorer.RUBRIC + """
This passage comes from a live transcript of a press conference or speech. It has no
speaker labels, so also decide who is talking:
  "speaker"  : the official giving remarks or answering
  "reporter" : a journalist asking a question
  "other"    : moderator, filler, or unclear
Return JSON {"role": "speaker"|"reporter"|"other", "stance": float, "summary": "<=12 words"}.
Score stance only from what the official says; for a reporter's question use 0.
Score the NEW PASSAGE only. The earlier context is there so you read it as part of the
answer it belongs to (e.g. "hard pressed to call conditions restrictive" argues for
TIGHTER policy). A remark that is not about the policy path (anecdotes, household
examples, institutional points) is 0.
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ------------------------------------------------------------------ #
# SHARED STATE (written for the dashboard)
# ------------------------------------------------------------------ #

class State:
    def __init__(self, speaker: str, source: str, base: dict, live: bool):
        self.lock = threading.Lock()
        self.data = {
            "speaker": speaker, "source": source, "started": now_iso(), "status": "starting",
            "live_orders": live, "baseline": {"mean": base["mean"], "stdev": base["stdev"], "n": base["n"]},
            "watchlist": {"HAWKISH": [], "DOVISH": []}, "chunks": [], "alerts": [], "trades": [],
        }
        self.save()

    def update(self, **kw):
        with self.lock:
            self.data.update(kw)
        self.save()

    def append(self, key: str, item: dict):
        with self.lock:
            self.data[key].append(item)
        self.save()

    def save(self):
        with self.lock:
            text = json.dumps(self.data, indent=1, default=str, ensure_ascii=False)
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, STATE_PATH)


# ------------------------------------------------------------------ #
# BEFORE THE SPEECH: WATCHLIST
# ------------------------------------------------------------------ #

def build_watchlist(speaker: str, venues, top_n: int = 6) -> dict:
    import market_router
    watch = {}
    for direction, text in PROTOTYPES.items():
        print(f"Finding markets a {direction} surprise should move...")
        matches = market_router.find_all(text, speaker, f"{speaker} sounds {direction} vs. usual", venues, top_n)
        watch[direction] = [m for m in matches if m["relevance"] >= market_router.MIN_TRADE_RELEVANCE]
        for m in watch[direction]:
            print(f"   {direction:<8} {m['venue']:<10} {m['side'].upper():<3} {m['market']}  ({m['title'][:60]})")
    return watch


def fresh_quote(m: dict) -> dict:
    try:
        if m["venue"] == "kalshi":
            import kalshi_trader
            q = kalshi_trader.quote(m["market"])
        else:
            import polymarket_client
            q = polymarket_client.PolymarketPublic().bbo(m["market"])
        return {**m, "quote": {"best_bid": q["best_bid"], "best_ask": q["best_ask"]}}
    except Exception:
        return m


# ------------------------------------------------------------------ #
# DURING THE SPEECH
# ------------------------------------------------------------------ #

class LiveDesk:
    def __init__(self, speaker, venues, qty, live, watchlist, state):
        self.speaker, self.venues, self.qty, self.live = speaker, venues, qty, live
        self.watch = watchlist
        self.state = state
        self.base = stance_scorer.load_store()[speaker]
        self.traded = set()
        self.recent = []            # last few passages, given to the scorer as context
        self.held_direction = None  # direction of positions already taken this session
        self.pending_flip = None    # a first opposite-direction surprise waiting for confirmation
        self.background = ThreadPoolExecutor(max_workers=4)
        self.records = []

    def score(self, text: str) -> dict:
        context = " ".join(self.recent[-2:])
        prompt = LIVE_PROMPT + (f"\nEarlier context: {context[-1500:]}\n" if context else "") + \
            f"\nNEW PASSAGE: {text[:2500]}"
        data = stance_scorer._gemini_json(prompt)
        stance = max(-1.0, min(1.0, float(data.get("stance", 0.0))))
        return {"role": data.get("role", "other"), "stance": stance, "summary": data.get("summary", "")}

    def handle(self, text: str, heard_at: float):
        t0 = time.time()
        scored = self.score(text)
        scored_at = time.time()
        self.recent = (self.recent + [text])[-3:]
        chunk = {"time": now_iso(), "text": text, "role": scored["role"], "stance": scored["stance"],
                 "summary": scored["summary"], "z": None, "direction": "—",
                 "score_ms": int((scored_at - t0) * 1000)}
        if scored["role"] != "speaker":
            chunk["direction"] = "QUESTION" if scored["role"] == "reporter" else "—"
            self.state.append("chunks", chunk)
            print(f"   [{scored['role']}] {text[:90]}")
            return

        r = stance_scorer._result(self.speaker, text, scored, self.base)
        chunk.update(z=round(r.z, 2), direction=r.direction)
        flag = f"<< {r.direction}" if r.is_surprising else ""
        print(f"   stance {r.stance:+.2f}  z {r.z:+.1f} {flag:<11} {r.summary}   ({chunk['score_ms']} ms)")

        record = {"speaker": self.speaker, "statement": text, "stance": r.stance, "baseline_mean": r.baseline_mean,
                  "baseline_stdev": r.baseline_stdev, "z": round(r.z, 2), "direction": r.direction,
                  "summary": r.summary, "matches": [], "trades": []}
        if r.is_surprising:
            flip = self.held_direction and r.direction != self.held_direction
            if flip and self.pending_flip != r.direction:
                # Opposite to what we hold: wait for a second surprise in a row before trading.
                self.pending_flip = r.direction
                chunk["note"] = "opposite to positions held; waiting for confirmation"
                print(f"   (first {r.direction} signal against our {self.held_direction} positions; "
                      f"need one more in a row to trade)")
            else:
                self.pending_flip = None
                self.act(record, heard_at, t0)
                chunk["latency_ms"] = record.get("latency_ms")
                if any(not t.get("error") and not t.get("blocked") for t in record["trades"]):
                    self.held_direction = r.direction
        else:
            self.pending_flip = None
        self.state.append("chunks", chunk)
        self.records.append(record)
        self.background.submit(self.persist, record)

    def act(self, record: dict, heard_at: float, scoring_started: float):
        import market_router
        candidates = [m for m in self.watch.get(record["direction"], [])
                      if (m["venue"], m["market"]) not in self.traded and m["venue"] in self.venues]
        with ThreadPoolExecutor(max_workers=8) as pool:            # refresh prices in parallel
            fresh = list(pool.map(fresh_quote, candidates))
        trades = market_router.trade_all(fresh, live=self.live, qty=self.qty,
                                         reason=f"LIVE {self.speaker} {record['direction']} z={record['z']:+.1f}")
        done = time.time()
        for t in trades:
            if not t.get("error") and not t.get("blocked"):
                self.traded.add((t.get("venue"), t.get("market")))
        record.update(matches=fresh, trades=trades,
                      latency_ms=int((done - heard_at) * 1000),
                      decision_ms=int((done - scoring_started) * 1000))
        alert = {"time": now_iso(), "direction": record["direction"], "z": record["z"], "summary": record["summary"],
                 "statement": record["statement"][:300], "latency_ms": record["latency_ms"],
                 "orders": sum(1 for t in trades if not t.get("error") and not t.get("blocked"))}
        self.state.append("alerts", alert)
        for t in trades:
            self.state.append("trades", {"time": now_iso(), "venue": t.get("venue"), "market": t.get("market"),
                                         "side": t.get("side"), "qty": t.get("qty"), "yes_limit": t.get("yes_limit"),
                                         "max_cost": t.get("max_cost"),
                                         "status": t.get("error") and f"skipped: {t['error']}" or tc.trade_status(t)})
        print(f"   >>> {record['direction']} surprise: {alert['orders']} order(s) in "
              f"{record['decision_ms']} ms after scoring started ({record['latency_ms']} ms after the words arrived)")
        market_router.print_trades(trades)

    def persist(self, record: dict):
        """Receipts and storage, off the hot path."""
        import pipeline
        import tiger_store
        if record["direction"] in ("HAWKISH", "DOVISH"):
            proof = pipeline._solana_proof(record)
            if proof:
                record["solana"] = proof
            tiger_store.log_signal(record, source="live", solana_sig=(proof or {}).get("signature"))
            tiger_store.log_ticks(record["matches"])
            tiger_store.log_trades(record["trades"])
            pipeline._backboard_record(record, "live_" + self.state.data["started"][:10])
        else:
            tiger_store.log_signal(record, source="live")


# ------------------------------------------------------------------ #
# SOURCES
# ------------------------------------------------------------------ #

def watch_file(path: str):
    """Yield new committed sentences as speechtxt.py appends them."""
    seen = 0
    print(f"Watching {os.path.basename(path)} for new speech (Ctrl+C to stop)...")
    while True:
        try:
            with open(path, encoding="utf-8") as f:
                segments = json.load(f).get("segments", [])
        except (FileNotFoundError, json.JSONDecodeError):
            segments = []
        if len(segments) < seen:          # transcriber restarted
            seen = 0
        for seg in segments[seen:]:
            yield seg.get("text", "")
        seen = len(segments)
        time.sleep(POLL_SECONDS)


def simulate(date: str, speed: float):
    """Feed a saved transcript sentence by sentence at speaking pace (reporters included, no labels)."""
    path = os.path.join(stance_scorer.TRANSCRIPT_DIR, f"{date}.json")
    if not os.path.isfile(path):
        import fed_transcripts
        fed_transcripts.load_or_parse(date, fed_transcripts.DEFAULT_CHAIR_LABEL)
    with open(path, encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    print(f"Simulating {date} at {speed:g}x speaking speed ({len(segments)} turns)...")
    for seg in segments:
        for sentence in re.split(r"(?<=[.?!])\s+", seg["text"]):
            if sentence.strip():
                time.sleep(len(sentence.split()) / (WORDS_PER_SECOND * speed))
                yield sentence


def chunks(sentences):
    """Group sentences into passages long enough to score."""
    buf, started = [], None
    for sentence in sentences:
        if not sentence.strip():
            continue
        if not buf:
            started = time.time()
        buf.append(sentence.strip())
        words = sum(len(s.split()) for s in buf)
        ends = buf[-1].endswith((".", "?", "!"))
        if (words >= MIN_WORDS and ends) or words >= MAX_WORDS or buf[-1].endswith("?"):
            yield " ".join(buf), time.time()
            buf = []
    if buf:
        yield " ".join(buf), time.time()


def main():
    parser = argparse.ArgumentParser(description="Score and trade a speech while it is happening")
    parser.add_argument("--speaker", default="kevin_warsh")
    parser.add_argument("--watch", default=TRANSCRIPT_PATH, help="Transcript file speechtxt.py writes")
    parser.add_argument("--simulate", metavar="DATE", help="Feed transcripts/DATE.json instead of live audio")
    parser.add_argument("--speed", type=float, default=10.0, help="Simulation speed-up (1 = real time)")
    parser.add_argument("--venues", nargs="+", default=["kalshi", "polymarket"])
    parser.add_argument("--qty", type=int, default=2)
    parser.add_argument("--live", action="store_true", help="Send real orders (default: dry run)")
    parser.add_argument("--no-prewarm", action="store_true", help="Skip the market watchlist (score only)")
    args = parser.parse_args()

    store = stance_scorer.load_store()
    if args.speaker not in store:
        raise SystemExit(f"No stance baseline for {args.speaker}. Run: python stance_scorer.py --seed")
    source = f"simulate {args.simulate} x{args.speed:g}" if args.simulate else os.path.basename(args.watch)
    state = State(args.speaker, source, store[args.speaker], args.live)

    watchlist = {"HAWKISH": [], "DOVISH": []}
    if not args.no_prewarm:
        state.update(status="finding markets before the speech")
        watchlist = build_watchlist(args.speaker, args.venues)
    state.update(watchlist=watchlist, status="listening")

    desk = LiveDesk(args.speaker, args.venues, args.qty, args.live, watchlist, state)
    sentences = simulate(args.simulate, args.speed) if args.simulate else watch_file(args.watch)
    try:
        for text, heard_at in chunks(sentences):
            desk.handle(text, heard_at)
    except KeyboardInterrupt:
        pass
    finally:
        state.update(status="finished")
        desk.background.shutdown(wait=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        name = f"live_{state.data['started'][:19].replace(':', '').replace('-', '')}"
        with open(os.path.join(RESULTS_DIR, name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"run": name, "created": state.data["started"], "records": desk.records}, f,
                      indent=1, default=str, ensure_ascii=False)
        print(f"\nDone. {len(state.data['alerts'])} surprise(s), {len(state.data['trades'])} order attempt(s). "
              f"Saved results/{name}.json")


if __name__ == "__main__":
    main()
