"""
stance_scorer.py
================
Speaker-relative STANCE surprise. This replaces "is the topic unusual?"
(embedding distance, which barely moves because every Fed answer is about
rates and inflation) with "is the stance unusual for this person?"

How it works:
  1. Gemini scores each answer's policy stance from -1 (very dovish) to
     +1 (very hawkish), using a fixed rubric.
  2. A speaker's baseline is the mean and spread of their past answers'
     stance scores (built from transcripts/ by seed_baselines.py).
  3. A new answer's surprise = z-score = (stance - mean) / stdev.
     |z| >= 2 is flagged. The sign says which way: + hawkish, - dovish.

This is the "baseline hawkishness 0.10; this answer 0.65 = 2.3 sigma"
number from the README.

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  # Build Warsh's stance baseline from June + July press conferences
  python stance_scorer.py --seed

  # Score one statement
  python stance_scorer.py --score "The Fed still has work to do."

  # Replay a held-out press conference answer by answer (the demo)
  python stance_scorer.py --replay 20260916
------------------------------------------------------------------------
"""

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import dataclass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_DIR = os.path.join(SCRIPT_DIR, "transcripts")
STANCE_STORE_PATH = os.path.join(SCRIPT_DIR, "stance_baselines.json")

GEMINI_MODEL = os.environ.get("STANCE_MODEL", "gemini-3.8-flash")
# Latest model first. The free tier only allows ~20 requests/day on
# gemini-3.8-flash, so fall back down this list if it's out of quota.
# Enabling billing on the Gemini API project removes the problem.
MODEL_CHAIN = [GEMINI_MODEL, "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
Z_THRESHOLD = 2.0
MIN_ANSWER_WORDS = 20      # skip "Thank you." / "Sure." answers
BATCH_SIZE = 8             # answers scored per Gemini call (saves quota)

RUBRIC = """You score what a central banker says for monetary-policy stance.

Score each passage from -1.0 to +1.0:
  +1.0  very hawkish: signals tighter policy, rate hikes, balance-sheet
        shrinkage, strong worry that inflation is too high or persistent,
        tolerance for weaker growth or jobs to get inflation down.
  +0.5  leaning hawkish.
   0.0  neutral: procedural, institutional, or balanced two-sided remarks,
        or not about policy at all (e.g. Fed independence, communication).
  -0.5  leaning dovish.
  -1.0  very dovish: signals easier policy, rate cuts, concern about jobs,
        growth, or financial stress over inflation.

Judge the policy signal, not the tone or politeness. Use the full range.
"""


def _read_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    path = os.path.join(SCRIPT_DIR, "gemapi.txt")
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
    return key


def _gemini_json(prompt: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_read_gemini_key())
    for attempt in range(8):
        # Fall back to a second model if the main one keeps failing.
        model = MODEL_CHAIN[min(attempt, len(MODEL_CHAIN) - 1)]
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.0
                ),
            )
            text = re.sub(r"^```(json)?|```$", "", response.text.strip()).strip()
            return json.loads(text)
        except Exception as e:
            message = str(e)
            if any(code in message for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500")):
                quota = re.findall(r"quotaMetric': '([^']+)'.*?quotaValue': '([^']+)'", message)
                print(f"  Gemini error on {model} ({quota or message[:90]}); retrying in 5s...")
                time.sleep(5)
                continue
            raise
    raise RuntimeError("Gemini kept rate limiting.")


def score_stances(passages: list) -> list:
    """Returns [{"stance": float, "summary": str}] in the same order as passages."""
    results = []
    for start in range(0, len(passages), BATCH_SIZE):
        batch = passages[start:start + BATCH_SIZE]
        numbered = []
        for i, text in enumerate(batch):
            numbered.append({"id": i, "text": text[:2500]})
        data = _gemini_json(
            RUBRIC
            + "\nReturn JSON {\"scores\": [{\"id\": int, \"stance\": float, "
            "\"summary\": \"<=12 words on the policy signal\"}]} with one entry per passage.\n\n"
            + json.dumps(numbered, ensure_ascii=False)
        )
        by_id = {}
        for item in data.get("scores", []):
            by_id[int(item["id"])] = item
        for i in range(len(batch)):
            item = by_id.get(i, {"stance": 0.0, "summary": "(no score returned)"})
            stance = max(-1.0, min(1.0, float(item.get("stance", 0.0))))
            results.append({"stance": stance, "summary": item.get("summary", "")})
    return results


# ------------------------------------------------------------------ #
# BASELINE STORE
# ------------------------------------------------------------------ #

def load_store() -> dict:
    if os.path.isfile(STANCE_STORE_PATH):
        with open(STANCE_STORE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_store(store: dict):
    with open(STANCE_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=1, ensure_ascii=False)


def chair_answers(date: str) -> list:
    path = os.path.join(TRANSCRIPT_DIR, f"{date}.json")
    if not os.path.isfile(path):
        raise RuntimeError(f"{path} missing. Run: python seed_baselines.py --parse-only --dates {date}")
    with open(path, encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    answers = []
    for seg in segments:
        if seg["role"] == "chair" and len(seg["text"].split()) >= MIN_ANSWER_WORDS:
            answers.append(seg["text"])
    return answers


def seed(speaker: str, dates: list):
    store = load_store()
    entries = []
    for date in dates:
        answers = chair_answers(date)
        print(f"{date}: scoring {len(answers)} answers...")
        for text, scored in zip(answers, score_stances(answers)):
            entries.append({"date": date, "stance": scored["stance"],
                            "summary": scored["summary"], "text": text[:300]})
    stances = [e["stance"] for e in entries]
    store[speaker] = {
        "dates": dates,
        "n": len(stances),
        "mean": statistics.mean(stances),
        "stdev": max(statistics.pstdev(stances), 0.05),  # floor so z can't explode
        "entries": entries,
    }
    save_store(store)
    b = store[speaker]
    print(f"\n{speaker} stance baseline: n={b['n']}  mean={b['mean']:+.3f}  stdev={b['stdev']:.3f}")


# ------------------------------------------------------------------ #
# SCORING
# ------------------------------------------------------------------ #

@dataclass
class StanceResult:
    speaker: str
    statement: str
    stance: float
    baseline_mean: float
    baseline_stdev: float
    z: float
    is_surprising: bool
    direction: str      # "HAWKISH" / "DOVISH" / "IN LINE"
    summary: str


def _result(speaker, statement, scored, base) -> StanceResult:
    z = (scored["stance"] - base["mean"]) / base["stdev"]
    surprising = abs(z) >= Z_THRESHOLD
    if not surprising:
        direction = "IN LINE"
    elif z > 0:
        direction = "HAWKISH"
    else:
        direction = "DOVISH"
    return StanceResult(speaker, statement, scored["stance"], base["mean"],
                        base["stdev"], z, surprising, direction, scored["summary"])


def score_statement(speaker: str, statement: str) -> StanceResult:
    store = load_store()
    if speaker not in store:
        raise RuntimeError(f"No stance baseline for {speaker}. Run: python stance_scorer.py --seed")
    return _result(speaker, statement, score_stances([statement])[0], store[speaker])


def replay(speaker: str, date: str) -> list:
    store = load_store()
    if speaker not in store:
        raise RuntimeError(f"No stance baseline for {speaker}. Run: python stance_scorer.py --seed")
    base = store[speaker]
    answers = chair_answers(date)
    print(f"Replaying {date}: {len(answers)} answers vs {speaker} baseline "
          f"(mean {base['mean']:+.2f}, sd {base['stdev']:.2f})\n")
    results = []
    for i, (text, scored) in enumerate(zip(answers, score_stances(answers)), start=1):
        r = _result(speaker, text, scored, base)
        results.append(r)
        flag = f"<< {r.direction}" if r.is_surprising else ""
        print(f"{i:>2}. stance {r.stance:+.2f}  z {r.z:+.1f}  {flag:<12} {r.summary}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Speaker-relative stance surprise")
    parser.add_argument("--speaker", default="kevin_warsh")
    parser.add_argument("--seed", action="store_true", help="Build the stance baseline")
    parser.add_argument("--dates", nargs="+", default=["20260617", "20260729"])
    parser.add_argument("--score", help="Score one statement")
    parser.add_argument("--replay", help="Replay a press conference date, e.g. 20260916")
    args = parser.parse_args()

    if args.seed:
        seed(args.speaker, args.dates)
    elif args.score:
        r = score_statement(args.speaker, args.score)
        print(f"stance {r.stance:+.2f}  baseline {r.baseline_mean:+.2f} (sd {r.baseline_stdev:.2f})  "
              f"z {r.z:+.1f}  -> {r.direction}\n{r.summary}")
    elif args.replay:
        replay(args.speaker, args.replay)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
