"""
run_polymarket.py
=================
Polymarket twin of run_kalshi_ticker2.py: same input, same output shape.

  python run_polymarket.py                          # default transcript (20260916.json)
  python run_polymarket.py live_transcript.json     # the file speechtxt.py writes live
  python run_polymarket.py transcripts/20260916.json

Reads the transcript JSON exactly the way run_kalshi_ticker2.py does (the
"text" field if present, otherwise the Chair's segments, otherwise every
segment), and emits the shared venue output schema with a speaker-baseline
snapshot, market candidates, direction where available, and best bid/ask.
Only baseline surprises (|z| >= 2) are promoted from candidates to recommendations.

--watch mirrors `run_kalshi_ticker2.py --watch`: download the Polymarket
catalog once, index it with Dylan's MarketCatalogIndex (same TF-IDF ranking
code), and re-rank on every new sentence of live_transcript.json. Each
20+-word statement is scored against the selected speaker's stance baseline.
Results go to live_polymarket_recommendations.json using the same schema as
live_recommendations.json.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash-lite")

from polymarket_client import PUBLIC_BASE_URL, PolymarketPublic, find_markets
from recommendation_schema import baseline_snapshot, output_payload, recommendation


INPUT_PATH = Path(__file__).with_name("20260916.json")
LIVE_INPUT_PATH = Path(__file__).with_name("live_transcript.json")
LIVE_RESULTS_PATH = Path(__file__).with_name("live_polymarket_recommendations.json")
# Sports is ~90% of Polymarket US and never relevant to a Fed speech.
CATEGORIES = ["politics", "macro", "finance", "geopolitics", "economics", "technology",
              "crypto", "culture", "climate", "science"]
MIN_RELEVANCE_SCORE = 0.60   # same cutoff as kalshi_ticker2
TOP_N = 3                    # same as run_kalshi_ticker2
MAX_TEXT_CHARS = 6000        # keep the Gemini prompt small on long live transcripts (uses the latest part)


def load_transcript_text(path: Path) -> str:
    """Same rules as run_kalshi_ticker2.load_transcript_text."""
    with path.open(encoding="utf-8") as transcript_file:
        payload = json.load(transcript_file)

    direct_text = payload.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"Expected a 'segments' list in {path}")

    text_segments = [
        segment["text"].strip()
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("text"), str)
        and segment["text"].strip()
    ]

    chair_segments = [
        segment["text"].strip()
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("role"), str)
        and segment["role"].strip().lower() == "chair"
        and isinstance(segment.get("text"), str)
        and segment["text"].strip()
    ]
    selected_segments = chair_segments or text_segments
    if not selected_segments:
        raise ValueError(f"No non-empty transcript segments found in {path}")

    return "\n\n".join(selected_segments)


def find_relevant_markets(speech_text: str, top_n: int = TOP_N) -> list:
    """Polymarket matches in the same shape kalshi_ticker2.find_relevant_tickers returns."""
    text = speech_text[-MAX_TEXT_CHARS:]
    results = []
    for m in find_markets(text, speaker="Federal Reserve Chair", top_n=top_n * 2):
        score = float(m.get("relevance", 0))
        if score < MIN_RELEVANCE_SCORE:
            continue
        results.append(recommendation(
            venue="polymarket",
            market_id=m["slug"],
            event_title=m.get("event") or "",
            market_title=m.get("question") or m.get("title") or m["slug"],
            relevance_score=score,
            reasoning=m.get("reason", ""),
            quote=m.get("quote"),
            market_direction=m.get("direction"),
        ))
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_n]


def fetch_polymarket_catalog(categories=CATEGORIES) -> list:
    """All open Polymarket US markets in kalshi_ticker2's catalog format."""
    public = PolymarketPublic()
    catalog, seen = [], set()
    for category in categories:
        offset = 0
        while True:
            markets = public._get("/v1/markets", {"limit": 500, "offset": offset, "active": "true",
                                                  "closed": "false", "categories": category}).get("markets", [])
            for m in markets:
                slug = m.get("slug")
                if not slug or slug in seen or m.get("category") == "sports":
                    continue
                seen.add(slug)
                description = (m.get("description") or "").strip()
                # Kalshi-style question from the rules: "This market will settle to Yes if X." -> "Will X?"
                first = re.split(r"(?<=\.)\s", description, maxsplit=1)[0]
                rule = re.sub(r"^This market will (settle|resolve) to [\"']?Yes[\"']? if\s+", "", first, flags=re.I)
                question = f"Will {rule.rstrip('.')}?" if rule != first else (first or m.get("question", ""))
                outcome = (m.get("title") or "").strip()
                side = (m.get("marketSides") or [{}])[0]
                catalog.append({
                    "market_ticker": slug,
                    "market_title": question,
                    # Outcomes of one question share a slug prefix (…-2026-10-28-hike25 / -cut25),
                    # like a Kalshi event; the ranker keeps one market per event.
                    "event_ticker": slug.rsplit("-", 1)[0],
                    "event_title": f"{m.get('question', '')}: {outcome}" if outcome else m.get("question", ""),
                    "expected_expiration_time": m.get("endDate"),
                    "subtitle": m.get("category", ""),
                    "yes_sub_title": outcome or (side.get("team") or {}).get("name") or "",
                    "no_sub_title": "",
                })
            offset += len(markets)
            if len(markets) < 500:
                break
    return catalog


def _recommendations_with_prices(ranked: list) -> list:
    """Same fields as run_kalshi_ticker2._recommendations_from_local_rank, plus live prices."""
    public = PolymarketPublic()
    out = []
    for market in ranked:
        try:
            q = public.bbo(market["market_ticker"])
        except Exception:
            q = {"best_bid": None, "best_ask": None}
        out.append(recommendation(
            venue="polymarket",
            market_id=market["market_ticker"],
            event_title=market.get("event_title", ""),
            market_title=market.get("market_title", ""),
            relevance_score=market["_hybrid_score"],
            reasoning="; ".join(market.get("_local_reasons", []))
            or "Selected by cached local relevance ranking.",
            quote=q,
        ))
    return out


def _save_live_recommendations(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def watch_live_transcript(transcript_path: Path, poll_interval: float = 0.5,
                          context_characters: int = 3000, speaker: str = "kevin_warsh") -> None:
    """Same loop as run_kalshi_ticker2.watch_live_transcript, over the Polymarket catalog."""
    from kalshi_ticker2 import MarketCatalogIndex
    from run_kalshi_ticker2 import _read_transcript_payload, _split_complete_sentences

    print("Fetching the Polymarket market catalog once...")
    market_catalog = fetch_polymarket_catalog()
    if not market_catalog:
        raise RuntimeError("No Polymarket markets were fetched; cannot start live ranking.")
    market_index = MarketCatalogIndex(market_catalog)
    if market_index.vectorizer is None:
        raise RuntimeError("Live watch mode requires scikit-learn so the market index can be cached.")
    print(f"Indexed {len(market_catalog):,} markets. Watching {transcript_path} for committed speech.")

    payload = _read_transcript_payload(transcript_path)
    segments = payload.get("segments", []) if payload else []
    seen_segments = len(segments) if isinstance(segments, list) else 0
    context, pending_sentence, pending_since, update_number = "", "", None, 0

    def rank_context(text: str, statement: str) -> None:
        nonlocal update_number
        window = text[-context_characters:]
        baseline = baseline_snapshot(speaker, statement)
        ranked = market_index.rank_live(window, max_candidates=TOP_N)
        recommendations = _recommendations_with_prices(ranked)
        update_number += 1
        payload = output_payload("polymarket", speaker, window, baseline,
                                 recommendations, update_number)
        _save_live_recommendations(LIVE_RESULTS_PATH, payload)
        print(f"\n--- Live Polymarket recommendations {update_number} ---")
        if baseline["status"] == "ERROR":
            print(f"Baseline score unavailable: {baseline['summary']}")
        print(json.dumps(payload, indent=2, ensure_ascii=False) if recommendations
              else "No markets currently meet the relevance threshold.")

    # Catalog fetching can take longer than the first replay sentences. Rank the
    # already committed speaker text once before following new segments.
    if isinstance(segments, list) and segments and payload:
        try:
            stamp = datetime.fromisoformat(str(payload.get("updated_at", "")).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            recent = 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= 120
        except ValueError:
            recent = False
        if recent:
            context = " ".join(str(s.get("text", "")).strip() for s in segments
                               if isinstance(s, dict) and s.get("role", "speaker") in ("chair", "speaker", "president"))[-context_characters:]
            if context:
                rank_context(context, context)

    while True:
        payload = _read_transcript_payload(transcript_path)
        segments = payload.get("segments", []) if payload else []
        if not isinstance(segments, list):
            segments = []
        if len(segments) < seen_segments:          # transcriber restarted
            seen_segments, context, pending_sentence, pending_since = 0, "", "", None
        new_segments = segments[seen_segments:]
        seen_segments = len(segments)
        for segment in new_segments:
            if isinstance(segment, dict) and segment.get("role", "speaker") not in ("chair", "speaker", "president"):
                continue
            text = segment.get("text") if isinstance(segment, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            pending_sentence = f"{pending_sentence} {text.strip()}".strip()
            if pending_since is None:
                pending_since = time.monotonic()
            completed, pending_sentence = _split_complete_sentences(pending_sentence)
            for sentence in completed:
                context = f"{context} {sentence}".strip()[-context_characters:]
                rank_context(context, sentence)
            if completed:
                pending_since = time.monotonic() if pending_sentence else None
        if pending_sentence and pending_since is not None and time.monotonic() - pending_since >= 3.0:
            statement = pending_sentence
            context = f"{context} {statement}".strip()[-context_characters:]
            pending_sentence, pending_since = "", None
            rank_context(context, statement)
        time.sleep(poll_interval)


def run_once(transcript_path: Path, speaker: str = "kevin_warsh") -> dict:
    speech_text = load_transcript_text(transcript_path)
    matches = find_relevant_markets(speech_text, top_n=TOP_N)
    baseline = baseline_snapshot(speaker, speech_text)
    payload = output_payload("polymarket", speaker, speech_text, baseline, matches)
    print("\nRELEVANT POLYMARKET MARKETS")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Polymarket markets against a transcript JSON file.")
    parser.add_argument("transcript", nargs="?", type=Path, default=None,
                        help="Transcript JSON (defaults to the live transcript in --watch mode).")
    parser.add_argument("--watch", action="store_true",
                        help="Cache the Polymarket catalog once and rerank as new transcript sentences arrive.")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--context-characters", type=int, default=3000)
    parser.add_argument("--speaker", default="kevin_warsh",
                        help="Speaker key in stance_baselines.json.")
    args = parser.parse_args()

    default_path = LIVE_INPUT_PATH if args.watch else INPUT_PATH
    transcript_path = (args.transcript or default_path).expanduser().resolve()
    if args.watch:
        watch_live_transcript(transcript_path, poll_interval=max(args.poll_interval, 0.1),
                              context_characters=max(args.context_characters, 500),
                              speaker=args.speaker)
        return
    run_once(transcript_path, speaker=args.speaker)


if __name__ == "__main__":
    main()
