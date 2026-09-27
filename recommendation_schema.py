"""Shared output and speaker-baseline fields for venue recommendations."""

from datetime import datetime, timezone

import stance_scorer


BASELINE_MIN_WORDS = stance_scorer.MIN_ANSWER_WORDS


def baseline_snapshot(speaker: str, statement: str) -> dict:
    excerpt = statement.strip()[:2500]
    stored = stance_scorer.load_store().get(speaker)
    if stored is None:
        raise RuntimeError(
            f"No stance baseline for {speaker}. Run: python stance_scorer.py --seed --speaker {speaker}"
        )

    snapshot = {
        "speaker": speaker,
        "statement": excerpt,
        "stance": None,
        "baseline_mean": stored["mean"],
        "baseline_stdev": stored["stdev"],
        "z": None,
        "is_surprising": None,
        "surprise_direction": "UNSCORED",
        "summary": "",
        "status": "UNSCORED",
    }
    if len(excerpt.split()) < BASELINE_MIN_WORDS:
        snapshot["summary"] = f"Passage has fewer than {BASELINE_MIN_WORDS} words."
        return snapshot

    try:
        result = stance_scorer.score_statement(speaker, excerpt)
    except Exception as error:
        snapshot["status"] = "ERROR"
        snapshot["summary"] = str(error)
        return snapshot

    snapshot.update({
        "stance": result.stance,
        "baseline_mean": result.baseline_mean,
        "baseline_stdev": result.baseline_stdev,
        "z": round(result.z, 3),
        "is_surprising": result.is_surprising,
        "surprise_direction": result.direction,
        "summary": result.summary,
        "status": "SCORED",
    })
    return snapshot


def recommendation(
    venue: str,
    market_id: str,
    event_title: str,
    market_title: str,
    relevance_score: float,
    reasoning: str,
    quote: dict | None = None,
    market_direction: str | None = None,
) -> dict:
    quote = quote or {}
    side = None
    if market_direction == "YES_UP":
        side = "yes"
    elif market_direction == "YES_DOWN":
        side = "no"

    return {
        "venue": venue,
        "market_id": market_id,
        "ticker": market_id,
        "event_title": event_title,
        "market_title": market_title,
        "relevance_score": float(relevance_score),
        "reasoning": reasoning,
        "market_direction": market_direction,
        "side": side,
        "quote": {
            "best_bid": quote.get("best_bid"),
            "best_ask": quote.get("best_ask"),
        },
    }


def output_payload(
    venue: str,
    speaker: str,
    context: str,
    baseline: dict,
    candidates: list[dict],
    transcript_update: int | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "venue": venue,
        "speaker": speaker,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "transcript_update": transcript_update,
        "context": context,
        "baseline": baseline,
        "candidates": candidates,
        "recommendations": candidates if baseline.get("is_surprising") is True else [],
    }