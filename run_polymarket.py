"""
run_polymarket.py
=================
Polymarket twin of run_kalshi_ticker2.py: same input, same output shape.

  python run_polymarket.py                          # default transcript (20260916.json)
  python run_polymarket.py live_transcript.json     # the file speechtxt.py writes live
  python run_polymarket.py transcripts/20260916.json

Reads the transcript JSON exactly the way run_kalshi_ticker2.py does (the
"text" field if present, otherwise the Chair's segments, otherwise every
segment), asks polymarket_client.find_markets for the best matches, and
prints the top 3 with the same keys as the Kalshi runner:
    ticker, event_title, market_title, relevance_score, reasoning
plus Polymarket extras: direction (YES_UP / YES_DOWN), side, best_bid, best_ask.

Add --watch to re-run every time the transcript file changes (live speech).
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash-lite")

from polymarket_client import find_markets


INPUT_PATH = Path(__file__).with_name("20260916.json")
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
        results.append({
            "ticker": m["slug"],
            "event_title": m.get("event") or "",
            "market_title": m.get("question") or m.get("title") or m["slug"],
            "relevance_score": score,
            "reasoning": m.get("reason", ""),
            "direction": m.get("direction"),
            "side": "yes" if m.get("direction") == "YES_UP" else "no",
            "best_bid": m["quote"].get("best_bid"),
            "best_ask": m["quote"].get("best_ask"),
        })
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_n]


def run_once(transcript_path: Path) -> list:
    speech_text = load_transcript_text(transcript_path)
    matches = find_relevant_markets(speech_text, top_n=TOP_N)
    print("\nRELEVANT POLYMARKET MARKETS")
    print(json.dumps(matches, indent=2, ensure_ascii=False))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank Polymarket markets against a transcript JSON file."
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        type=Path,
        default=INPUT_PATH,
        help=f"Transcript JSON (default: {INPUT_PATH.name}).",
    )
    parser.add_argument("--watch", action="store_true",
                        help="Re-run whenever the transcript file changes (live speech).")
    parser.add_argument("--interval", type=float, default=15.0,
                        help="With --watch: minimum seconds between runs.")
    args = parser.parse_args()

    transcript_path = args.transcript.expanduser().resolve()
    if not args.watch:
        run_once(transcript_path)
        return

    last_mtime = None
    print(f"Watching {transcript_path.name} (Ctrl+C to stop)...")
    while True:
        try:
            mtime = transcript_path.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        if mtime and mtime != last_mtime:
            last_mtime = mtime
            try:
                run_once(transcript_path)
            except Exception as e:
                print(f"(skipped this update: {e})")
            time.sleep(args.interval)
        else:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
