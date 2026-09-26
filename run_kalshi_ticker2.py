import json
import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from kalshi_ticker2 import (
    MarketCatalogIndex,
    build_market_catalog,
    fetch_live_kalshi_events,
    find_relevant_tickers,
)


INPUT_PATH = Path(__file__).with_name("20260916.json")
LIVE_INPUT_PATH = Path(__file__).with_name("live_transcript.json")
LIVE_RESULTS_PATH = Path(__file__).with_name("live_recommendations.json")
SENTENCE_END = re.compile(r"[.!?](?:[\"'\u2019\u201d)]*)(?=\s|$)")


def load_transcript_text(path: Path) -> str:
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


def _read_transcript_payload(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as transcript_file:
            payload = json.load(transcript_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _split_complete_sentences(text: str) -> tuple[list[str], str]:
    sentences = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        sentence = text[start:match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
    return sentences, text[start:].strip()


def _recommendations_from_local_rank(ranked: list[dict]) -> list[dict]:
    return [
        {
            "ticker": market["market_ticker"],
            "event_title": market.get("event_title", ""),
            "market_title": market.get("market_title", ""),
            "relevance_score": float(market["_hybrid_score"]),
            "reasoning": "; ".join(market.get("_local_reasons", []))
            or "Selected by cached local relevance ranking.",
        }
        for market in ranked
    ]


def _save_live_recommendations(
    path: Path,
    recommendations: list[dict],
    context: str,
    update_number: int,
) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "transcript_update": update_number,
        "context": context,
        "recommendations": recommendations,
    }
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def watch_live_transcript(
    transcript_path: Path,
    poll_interval: float = 0.5,
    context_characters: int = 3000,
) -> None:
    print("Fetching the Kalshi market catalog once...")
    events = fetch_live_kalshi_events(limit=200)
    market_catalog = build_market_catalog(events)
    if not market_catalog:
        raise RuntimeError("No Kalshi markets were fetched; cannot start live ranking.")

    market_index = MarketCatalogIndex(market_catalog)
    if market_index.vectorizer is None:
        raise RuntimeError(
            "Live watch mode requires scikit-learn so the market index can be cached."
        )
    print(
        f"Indexed {len(market_catalog):,} markets. "
        f"Watching {transcript_path} for committed speech."
    )

    payload = _read_transcript_payload(transcript_path)
    existing_segments = payload.get("segments", []) if payload else []
    if not isinstance(existing_segments, list):
        existing_segments = []
    seen_segments = len(existing_segments)
    transcript_is_recent = False
    if payload and isinstance(payload.get("updated_at"), str):
        try:
            updated_at = datetime.fromisoformat(
                payload["updated_at"].replace("Z", "+00:00")
            )
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
            transcript_is_recent = -30 <= age_seconds <= 120
        except ValueError:
            pass

    context = ""
    if transcript_is_recent:
        context = " ".join(
            str(segment.get("text", "")).strip()
            for segment in existing_segments
            if isinstance(segment, dict) and segment.get("text")
        )[-context_characters:]
    pending_sentence = ""
    pending_since = None
    update_number = 0

    def rank_context(text: str) -> None:
        nonlocal update_number
        context = text[-context_characters:]
        ranked = market_index.rank_live(context, max_candidates=3)
        recommendations = _recommendations_from_local_rank(ranked)
        update_number += 1
        _save_live_recommendations(
            LIVE_RESULTS_PATH,
            recommendations,
            context,
            update_number,
        )
        print(f"\n--- Live recommendations {update_number} ---")
        if recommendations:
            print(json.dumps(recommendations, indent=2, ensure_ascii=False))
        else:
            print("No markets currently meet the relevance threshold.")

    if context:
        rank_context(context)

    while True:
        payload = _read_transcript_payload(transcript_path)
        segments = payload.get("segments", []) if payload else []
        if not isinstance(segments, list):
            segments = []

        if len(segments) < seen_segments:
            seen_segments = 0
            context = ""
            pending_sentence = ""
            pending_since = None

        new_segments = segments[seen_segments:]
        seen_segments = len(segments)
        for segment in new_segments:
            if not isinstance(segment, dict):
                continue
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            pending_sentence = f"{pending_sentence} {text.strip()}".strip()
            if pending_since is None:
                pending_since = time.monotonic()
            completed, pending_sentence = _split_complete_sentences(pending_sentence)
            for sentence in completed:
                context = f"{context} {sentence}".strip()[-context_characters:]
                rank_context(context)
            if completed:
                pending_since = time.monotonic() if pending_sentence else None

        if (
            pending_sentence
            and pending_since is not None
            and time.monotonic() - pending_since >= 3.0
        ):
            context = f"{context} {pending_sentence}".strip()[-context_characters:]
            pending_sentence = ""
            pending_since = None
            rank_context(context)

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank Kalshi markets against a transcript JSON file."
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        type=Path,
        default=None,
        help="Transcript JSON (defaults to the live transcript in --watch mode).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Cache the Kalshi catalog once and rerank as new transcript sentences arrive.",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--context-characters", type=int, default=3000)
    args = parser.parse_args()

    default_path = LIVE_INPUT_PATH if args.watch else INPUT_PATH
    transcript_path = (args.transcript or default_path).expanduser().resolve()
    if args.watch:
        watch_live_transcript(
            transcript_path,
            poll_interval=max(args.poll_interval, 0.1),
            context_characters=max(args.context_characters, 500),
        )
        return

    speech_text = load_transcript_text(transcript_path)
    matches = find_relevant_tickers(speech_text, top_n=3)

    print("\nRELEVANT KALSHI MARKETS")
    print(json.dumps(matches, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
