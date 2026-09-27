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

  # Talk into the laptop mic yourself (or use the dashboard's Live audio panel)
  python mic.py          (terminal 1)
  python live.py         (terminal 2)

  # Demo without audio: feed the Sept 16 transcript at 10x speaking speed
  python live.py --simulate 20260916 --speed 10

  Options: --venues kalshi polymarket, --qty 2, --mode dry|demo|live, --no-prewarm
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
# Output of the live recommenders (run_kalshi_ticker2.py / run_polymarket.py --watch)
RECOMMENDER_FILES = {"kalshi": os.path.join(SCRIPT_DIR, "live_recommendations.json"),
                     "polymarket": os.path.join(SCRIPT_DIR, "live_polymarket_recommendations.json")}
RECOMMENDER_MAX_AGE = 90        # seconds; ignore stale recommender output
RECOMMENDER_MAX_EXTRA = 3       # extra markets per surprise taken from the recommenders
STATE_PATH = os.path.join(SCRIPT_DIR, "live_state.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MIN_WORDS = 25          # don't score fragments shorter than this
MAX_WORDS = 90          # score even without a sentence end once the buffer is this long
POLL_SECONDS = 0.3
PAUSE_SECONDS = 2.0     # live speech: score what we have after this much silence ...
MIN_PAUSE_WORDS = 6     # ... as long as it's at least this many words
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
            "live_orders": tc.mode_of(live) != "dry", "mode": tc.mode_of(live), "baseline": {"mean": base["mean"], "stdev": base["stdev"], "n": base["n"]},
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
        for attempt in range(20):          # Windows: a reader (dashboard, uploader) may hold the file briefly
            try:
                os.replace(tmp, STATE_PATH)
                return
            except PermissionError:
                time.sleep(0.05)
        print("   (couldn't update live_state.json; will retry on the next change)")


# ------------------------------------------------------------------ #
# BEFORE THE SPEECH: WATCHLIST
# ------------------------------------------------------------------ #

FED_SPEAKERS = {"kevin_warsh", "jerome_powell"}

# Which side of each next-meeting outcome a surprise favors.
# Kalshi suffixes: H0 = hold, H25/H26 = hike, C25/C26 = cut.  Polymarket: nochng, hike25/50, cut25/50.
FED_OUTCOME_SIDES = {
    "hold":       {"HAWKISH": "no",  "DOVISH": "yes", "relevance": 0.97},
    "hike_small": {"HAWKISH": "yes", "DOVISH": "no",  "relevance": 0.98},
    "hike_big":   {"HAWKISH": "yes", "DOVISH": "no",  "relevance": 0.80},
    "cut_small":  {"HAWKISH": "no",  "DOVISH": "yes", "relevance": 0.90},
    "cut_big":    {"HAWKISH": "no",  "DOVISH": "yes", "relevance": 0.75},
}


def _kalshi_next_fed_decision() -> list:
    """Outcomes of the next FOMC decision on Kalshi: [(ticker, title, outcome_kind)]."""
    import requests
    resp = requests.get("https://api.elections.kalshi.com/trade-api/v2/markets",
                        params={"series_ticker": "KXFEDDECISION", "status": "open", "limit": 200}, timeout=20)
    resp.raise_for_status()
    markets = resp.json().get("markets", [])
    if not markets:
        return []
    next_event = min(markets, key=lambda m: m.get("close_time") or "9999")["event_ticker"]
    kinds = {"H0": "hold", "H25": "hike_small", "H26": "hike_big", "C25": "cut_small", "C26": "cut_big"}
    out = []
    for m in markets:
        suffix = m["ticker"].rsplit("-", 1)[-1]
        if m["event_ticker"] == next_event and suffix in kinds:
            out.append((m["ticker"], f"{m.get('title', '')} -- {m.get('yes_sub_title', '')}", kinds[suffix]))
    return out


def _polymarket_next_fed_decision() -> list:
    """Outcomes of the next FOMC decision on Polymarket US: [(slug, title, outcome_kind)]."""
    import polymarket_client
    kinds = {"nochng": "hold", "hike25": "hike_small", "hike50": "hike_big", "cut25": "cut_small", "cut50": "cut_big"}
    found = {}
    for event in polymarket_client.PolymarketPublic().search("Fed Decision", limit=20):
        for m in event.get("markets") or []:
            slug = m.get("slug") or ""
            if slug.startswith("rdc-usfed-fomc-") and m.get("active") and not m.get("closed"):
                found[slug] = (slug, f"{event.get('title', '')} -- {m.get('title', '')}")
    if not found:
        return []
    next_date = min(slug[len("rdc-usfed-fomc-"):len("rdc-usfed-fomc-") + 10] for slug in found)
    return [(slug, title, kinds[slug.rsplit("-", 1)[-1]]) for slug, title in found.values()
            if next_date in slug and slug.rsplit("-", 1)[-1] in kinds]


def fed_decision_watchlist(venues) -> dict:
    """The markets a Fed surprise should move first: the next meeting's decision."""
    watch = {"HAWKISH": [], "DOVISH": []}
    sources = []
    if "kalshi" in venues:
        sources.append(("kalshi", _kalshi_next_fed_decision))
    if "polymarket" in venues:
        sources.append(("polymarket", _polymarket_next_fed_decision))
    for venue, fetch in sources:
        try:
            outcomes = fetch()
        except Exception as e:
            print(f"   (couldn't load next Fed decision on {venue}: {e})")
            continue
        for market, title, kind in outcomes:
            rule = FED_OUTCOME_SIDES[kind]
            for direction in ("HAWKISH", "DOVISH"):
                side = rule[direction]
                base = fresh_quote({"venue": venue, "market": market,
                                    "quote": {"best_bid": None, "best_ask": None}})
                watch[direction].append({
                    "venue": venue, "market": market, "title": title,
                    "direction": "YES_UP" if side == "yes" else "YES_DOWN", "side": side,
                    "relevance": rule["relevance"],
                    "reason": f"next FOMC decision: a {direction.lower()} surprise favors {side.upper()}",
                    "quote": base["quote"],
                })
    for direction in watch:
        watch[direction].sort(key=lambda m: m["relevance"], reverse=True)
    return watch


def build_watchlist(speaker: str, venues, top_n: int = 6) -> dict:
    import market_router
    core = fed_decision_watchlist(venues) if speaker in FED_SPEAKERS else {"HAWKISH": [], "DOVISH": []}
    watch = {}
    for direction, text in PROTOTYPES.items():
        print(f"Finding markets a {direction} surprise should move...")
        matches = market_router.find_all(text, speaker, f"{speaker} sounds {direction} vs. usual", venues, top_n)
        found = [m for m in matches if m["relevance"] >= market_router.MIN_TRADE_RELEVANCE]
        seen = {(m["venue"], m["market"]) for m in core[direction]}
        watch[direction] = core[direction] + [m for m in found if (m["venue"], m["market"]) not in seen]
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

def recommender_markets(record: dict, venues, already: set) -> list:
    """Markets the live recommenders found that aren't on the watchlist yet, with a buy side.
    Uses their side when they set one; otherwise one Gemini call decides YES/NO for all."""
    from datetime import datetime, timezone
    found = []
    for venue, path in RECOMMENDER_FILES.items():
        if venue not in venues or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(data["updated_at"])).total_seconds()
        except Exception:
            continue
        if age > RECOMMENDER_MAX_AGE:
            continue
        for r in (data.get("recommendations") or data.get("candidates") or []):
            key = (venue, r.get("market_id") or r.get("ticker"))
            if key[1] and key not in already and float(r.get("relevance_score", 0)) >= 0.6:
                found.append((venue, r))
                already.add(key)
    found = found[:RECOMMENDER_MAX_EXTRA]
    if not found:
        return []

    unsided = [r for _, r in found if not r.get("side")]
    sides = {}
    if unsided:
        listing = [{"id": r.get("market_id") or r.get("ticker"),
                    "market": f"{r.get('event_title', '')} {r.get('market_title', '')}".strip()} for r in unsided]
        try:
            data = stance_scorer._gemini_json(
                f"A {record['direction'].lower()} surprise from {record['speaker']}: \"{record['statement'][:600]}\"\n"
                "For each prediction market, does this make YES more likely (YES_UP), less likely (YES_DOWN), "
                "or is it unrelated (NONE)? Return JSON {\"markets\": [{\"id\": str, \"direction\": str}]}.\n"
                + json.dumps(listing))
            sides = {m["id"]: m["direction"] for m in data.get("markets", [])}
        except Exception:
            sides = {}
    out = []
    for venue, r in found:
        market = r.get("market_id") or r.get("ticker")
        side = r.get("side") or {"YES_UP": "yes", "YES_DOWN": "no"}.get(sides.get(market))
        if not side:
            continue          # unrelated or undecided: don't trade it
        out.append({"venue": venue, "market": market,
                    "title": f"{r.get('event_title', '')} -- {r.get('market_title', '')}".strip(" -"),
                    "direction": "YES_UP" if side == "yes" else "YES_DOWN", "side": side,
                    "relevance": float(r.get("relevance_score", 0.7)) if r.get("side") else 0.7,
                    "reason": "from the live recommender", "quote": r.get("quote") or {}})
    return out


class LiveDesk:
    def __init__(self, speaker, venues, qty, live, watchlist, state, source="live", surprises_only=False):
        self.speaker, self.venues, self.qty, self.live = speaker, venues, qty, live
        self.source = source                  # "live", "mic" or "simulate": how rows are tagged in Tiger Data
        self.surprises_only = surprises_only  # mic demo: don't store neutral chatter
        self.watch = watchlist
        self.state = state
        self.base = stance_scorer.macro_adjusted(stance_scorer.load_store()[speaker])   # FRED-adjusted
        self.traded = set()
        self.recent = []            # last few passages, given to the scorer as context
        self.held_direction = None  # direction of positions already taken this session
        self.pending_flip = None    # a first opposite-direction surprise waiting for confirmation
        self.background = ThreadPoolExecutor(max_workers=4)
        self.records = []
        try:   # macro backdrop from FRED, read once from the hourly cache (never blocks scoring)
            import fred_client
            self.macro = fred_client.context_line()
        except Exception:
            self.macro = ""

    def score(self, text: str) -> dict:
        context = " ".join(self.recent[-2:])
        prompt = LIVE_PROMPT + (f"\nCurrent data (FRED): {self.macro}\n" if self.macro else "") + \
            (f"\nEarlier context: {context[-1500:]}\n" if context else "") + \
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
        # Add what the live recommenders are pointing at right now (new topics mid-speech).
        known = {(m["venue"], m["market"]) for m in self.watch.get(record["direction"], [])} | self.traded
        extra = recommender_markets(record, self.venues, known)
        if extra:
            print(f"   + {len(extra)} market(s) from the live recommenders: "
                  + ", ".join(f"{m['side'].upper()} {m['market']}" for m in extra))
        candidates += extra
        with ThreadPoolExecutor(max_workers=8) as pool:            # refresh prices in parallel
            fresh = list(pool.map(fresh_quote, candidates))
        trades = market_router.trade_all(fresh, live=self.live, qty=self.qty, max_per_venue=3,
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
        try:   # compare with the crowd (Bluesky + Reddit), cached file only
            import social_sentiment
            with open(social_sentiment.CACHE_PATH, encoding="utf-8") as f:
                alert["crowd"] = social_sentiment.crowd_note(record["direction"], json.load(f))
        except (OSError, ValueError, ImportError):
            pass
        self.state.append("alerts", alert)
        titles = {(m["venue"], m["market"]): m.get("title", "") for m in fresh}
        for t in trades:
            self.state.append("trades", {
                "time": now_iso(), "venue": t.get("venue"), "market": t.get("market"),
                "title": titles.get((t.get("venue"), t.get("market")), ""),
                "side": t.get("side"), "qty": t.get("qty"), "yes_limit": t.get("yes_limit"),
                "max_cost": t.get("max_cost"), "trigger": f"{record['direction']} z {record['z']:+.1f}",
                "skipped": bool(t.get("error") or t.get("blocked")),
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
            tiger_store.log_signal(record, source=self.source, solana_sig=(proof or {}).get("signature"))
            tiger_store.log_ticks(record["matches"])
            tiger_store.log_trades(record["trades"])
            pipeline._backboard_record(record, f"{self.source}_" + self.state.data["started"][:10])
        elif not self.surprises_only:
            tiger_store.log_signal(record, source=self.source)


# ------------------------------------------------------------------ #
# SOURCES
# ------------------------------------------------------------------ #

def watch_file(path: str):
    """Yield new committed sentences as speechtxt.py / mic.py append them.
    Yields "" on quiet polls so the chunker can notice a pause."""
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
        new = segments[seen:]
        for seg in new:
            yield seg.get("text", "")
        seen = len(segments)
        if not new:
            yield ""
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


def chunks(sentences, per_utterance: bool = False):
    """Group sentences into passages long enough to score. A pause of PAUSE_SECONDS
    flushes a shorter passage (someone talking into the mic says one or two sentences).
    per_utterance=True (microphone): every committed utterance of MIN_PAUSE_WORDS+ words
    is scored right away instead of waiting for 25 words or a 2-second pause."""
    buf, last = [], None
    for sentence in sentences:
        if not sentence.strip():
            if buf and time.time() - last >= PAUSE_SECONDS and \
                    sum(len(s.split()) for s in buf) >= MIN_PAUSE_WORDS:
                yield " ".join(buf), last
                buf = []
            continue
        buf.append(sentence.strip())
        last = time.time()
        words = sum(len(s.split()) for s in buf)
        ends = buf[-1].endswith((".", "?", "!"))
        if (per_utterance and words >= MIN_PAUSE_WORDS) or \
                (words >= MIN_WORDS and ends) or words >= MAX_WORDS or buf[-1].endswith("?"):
            yield " ".join(buf), last
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
    parser.add_argument("--live", action="store_true", help="Send real orders (same as --mode live)")
    parser.add_argument("--mode", choices=tc.MODES, default="dry",
                        help="dry (default), demo (Kalshi demo exchange), live (real money)")
    parser.add_argument("--no-prewarm", action="store_true", help="Skip the market watchlist (score only)")
    parser.add_argument("--source", choices=["live", "mic"], default="live",
                        help="Tag stored rows as coming from a real speech or the mic demo")
    parser.add_argument("--source-label", default="",
                        help="Display label for this audio source in live_state.json")
    parser.add_argument("--surprises-only", action="store_true",
                        help="Only store surprises (use for the mic demo so room chatter isn't saved)")
    parser.add_argument("--fast", action="store_true",
                        help="Only watch the next Fed decision's markets (starts in seconds; good for the mic demo)")
    args = parser.parse_args()
    args.live = "live" if args.live else args.mode

    store = stance_scorer.load_store()
    if args.speaker not in store:
        raise SystemExit(f"No stance baseline for {args.speaker}. Run: python stance_scorer.py --seed")
    source = (args.source_label or
              (f"simulate {args.simulate} x{args.speed:g}" if args.simulate else os.path.basename(args.watch)))
    state = State(args.speaker, source, stance_scorer.macro_adjusted(store[args.speaker]), args.live)

    watchlist = {"HAWKISH": [], "DOVISH": []}
    if args.fast:
        state.update(status="loading the next Fed decision markets")
        watchlist = fed_decision_watchlist(args.venues)
    elif not args.no_prewarm:
        state.update(status="finding markets before the speech")
        watchlist = build_watchlist(args.speaker, args.venues)
    state.update(watchlist=watchlist, status="listening")

    desk = LiveDesk(args.speaker, args.venues, args.qty, args.live, watchlist, state,
                    source="simulate" if args.simulate else args.source, surprises_only=args.surprises_only)
    sentences = simulate(args.simulate, args.speed) if args.simulate else watch_file(args.watch)
    try:
        for text, heard_at in chunks(sentences, per_utterance=(args.source == "mic" and not args.simulate)):
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
