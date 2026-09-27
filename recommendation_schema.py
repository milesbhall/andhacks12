"""Shared output and speaker-baseline fields for venue recommendations."""

from datetime import datetime, timezone

import stance_scorer


# Live speech arrives one sentence at a time, so use a small floor (like live.py)
# rather than the 20-word filter used when building baselines from full answers.
BASELINE_MIN_WORDS = 6


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


_FED_CACHE = {"time": 0.0, "watch": None}
FED_CACHE_SECONDS = 60


def fed_decision_recommendations(venue: str, speaker: str, direction: str) -> list[dict]:
    """The next FOMC decision outcomes a surprise in `direction` favors (from live.py),
    already carrying a buy side. Cached for a minute so every sentence doesn't refetch."""
    import time
    import live
    if speaker not in live.FED_SPEAKERS or direction not in ("HAWKISH", "DOVISH"):
        return []
    if _FED_CACHE["watch"] is None or time.time() - _FED_CACHE["time"] > FED_CACHE_SECONDS:
        try:
            _FED_CACHE["watch"] = live.fed_decision_watchlist(["kalshi", "polymarket"])
            _FED_CACHE["time"] = time.time()
        except Exception:
            return []
    out = []
    for m in _FED_CACHE["watch"].get(direction, []):
        if m["venue"] != venue:
            continue
        event, _, outcome = (m.get("title") or "").partition(" -- ")
        out.append(recommendation(venue=venue, market_id=m["market"], event_title=event,
                                  market_title=outcome or m["market"], relevance_score=m["relevance"],
                                  reasoning=m.get("reason", ""), quote=m.get("quote"),
                                  market_direction=m.get("direction")))
    return out


def output_payload(
    venue: str,
    speaker: str,
    context: str,
    baseline: dict,
    candidates: list[dict],
    transcript_update: int | None = None,
) -> dict:
    recommendations = []
    if baseline.get("is_surprising") is True:
        # On a Fed surprise, the next meeting's decision markets come first, then the ranked ones.
        fed = fed_decision_recommendations(venue, speaker, baseline.get("surprise_direction"))
        seen = {r["market_id"] for r in fed}
        recommendations = fed + [c for c in candidates if c["market_id"] not in seen]
    return {
        "schema_version": 1,
        "venue": venue,
        "speaker": speaker,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "transcript_update": transcript_update,
        "context": context,
        "baseline": baseline,
        "candidates": candidates,
        "recommendations": recommendations,
    }
