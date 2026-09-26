"""
speaker_baseline.py
====================
The core novel piece of the pipeline: scores a new statement for how much
it deviates ("surprises") relative to a speaker's own historical baseline,
rather than scoring generic sentiment.

Method: maintain a rolling set of embeddings for everything a speaker has
previously said. A new statement's "surprise score" is 1 minus its cosine
similarity to that speaker's baseline centroid. Low similarity (= high
surprise) means the speaker said something meaningfully different from
their established pattern -- which is the signal, not raw sentiment.

This mirrors the text-similarity approach used in the Chicago Fed's 2026
working paper on FOMC speech alignment, applied live instead of after the
fact.

------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------
    pip install google-genai --break-system-packages
    export GEMINI_API_KEY="your-gemini-key"

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
    from speaker_baseline import SurpriseScorer

    scorer = SurpriseScorer(store_path="baselines.json")

    # Seed a speaker's baseline with past statements (do this once,
    # up front, with a backlog of real past speeches/posts)
    scorer.add_to_baseline("kevin_warsh", "We remain data dependent...")
    scorer.add_to_baseline("kevin_warsh", "Inflation risks are two-sided...")

    # Score a new, live statement against that baseline
    result = scorer.score("kevin_warsh", "The Fed still has work to do.")
    print(result.surprise_score, result.is_surprising)
------------------------------------------------------------------------
"""

import json
import os
from dataclasses import dataclass, field


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "gemini-embedding-001"  # text-embedding-004 was retired (404)

# Anything above this is flagged as "surprising" and worth acting on.
# Start conservative; tune this against your backtest set.
SURPRISE_THRESHOLD = 0.35

# Only keep the most recent N statements per speaker in the baseline,
# so a speaker's baseline drifts to reflect their recent pattern rather
# than being dominated by things they said years ago.
MAX_BASELINE_SIZE = 300  # two press conferences are ~120 chunks

# seed_baselines.py writes a per-speaker threshold here (mean + 2 sd of the
# speaker's own past answers). If present, it overrides SURPRISE_THRESHOLD.
CALIBRATION_FILENAME = "baseline_calibration.json"


@dataclass
class ScoreResult:
    speaker: str
    statement: str
    surprise_score: float
    is_surprising: bool
    baseline_size: int


class EmbeddingClient:
    """Thin wrapper around Gemini's embedding endpoint."""

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY.")
        from google import genai  # pip install google-genai

        self.client = genai.Client(api_key=api_key)

    def embed(self, text: str) -> list:
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        # google-genai returns a list of embedding objects; we send one
        # piece of text, so we want the first (and only) result.
        embedding_values = response.embeddings[0].values
        return list(embedding_values)


def cosine_similarity(vector_a: list, vector_b: list) -> float:
    """Explicit loop-based cosine similarity (no list comprehensions),
    so this reads step by step rather than as a one-liner.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must be the same length to compare.")

    dot_product = 0.0
    index = 0
    while index < len(vector_a):
        dot_product = dot_product + (vector_a[index] * vector_b[index])
        index = index + 1

    norm_a = 0.0
    index = 0
    while index < len(vector_a):
        norm_a = norm_a + (vector_a[index] * vector_a[index])
        index = index + 1
    norm_a = norm_a ** 0.5

    norm_b = 0.0
    index = 0
    while index < len(vector_b):
        norm_b = norm_b + (vector_b[index] * vector_b[index])
        index = index + 1
    norm_b = norm_b ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return similarity


def centroid(vectors: list) -> list:
    """Average a list of equal-length embedding vectors into one vector
    representing the speaker's overall baseline.
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute a centroid with no vectors.")

    vector_length = len(vectors[0])
    summed = []
    position = 0
    while position < vector_length:
        summed.append(0.0)
        position = position + 1

    for vector in vectors:
        position = 0
        while position < vector_length:
            summed[position] = summed[position] + vector[position]
            position = position + 1

    averaged = []
    position = 0
    while position < vector_length:
        averaged.append(summed[position] / len(vectors))
        position = position + 1

    return averaged


class BaselineStore:
    """Persists each speaker's statement history + embeddings to a local
    JSON file, so baselines survive between runs. Swap this for a
    Backboard-backed store later if you want it shared/remote.
    """

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.data = {}
        if os.path.isfile(store_path):
            with open(store_path, "r") as f:
                self.data = json.load(f)

    def get_embeddings(self, speaker: str) -> list:
        if speaker in self.data:
            entries = self.data[speaker]
            embeddings = []
            for entry in entries:
                embeddings.append(entry["embedding"])
            return embeddings
        return []

    def add(self, speaker: str, statement: str, embedding: list):
        if speaker not in self.data:
            self.data[speaker] = []

        self.data[speaker].append({"statement": statement, "embedding": embedding})

        # Trim to the most recent MAX_BASELINE_SIZE entries.
        if len(self.data[speaker]) > MAX_BASELINE_SIZE:
            overflow = len(self.data[speaker]) - MAX_BASELINE_SIZE
            self.data[speaker] = self.data[speaker][overflow:]

        self._save()

    def _save(self):
        with open(self.store_path, "w") as f:
            json.dump(self.data, f)


class SurpriseScorer:
    def __init__(self, store_path: str = "baselines.json", api_key: str = None):
        key = api_key if api_key else GEMINI_API_KEY
        self.embedder = EmbeddingClient(key)
        self.store = BaselineStore(store_path)
        self.calibration_path = os.path.join(
            os.path.dirname(os.path.abspath(store_path)), CALIBRATION_FILENAME
        )

    def threshold_for(self, speaker: str) -> float:
        """Per-speaker threshold from seed_baselines.py calibration, if it
        exists; otherwise the global SURPRISE_THRESHOLD.
        """
        if os.path.isfile(self.calibration_path):
            with open(self.calibration_path, "r") as f:
                calibration = json.load(f)
            if speaker in calibration:
                return calibration[speaker]["suggested_threshold"]
        return SURPRISE_THRESHOLD

    def add_to_baseline(self, speaker: str, statement: str):
        """Call this to seed or update a speaker's baseline with a
        statement they're known to have made (backlog or live, once
        it's been scored).
        """
        embedding = self.embedder.embed(statement)
        self.store.add(speaker, statement, embedding)

    def score(self, speaker: str, statement: str) -> ScoreResult:
        """Scores a new statement against the speaker's existing
        baseline. Does NOT automatically add it to the baseline --
        call add_to_baseline separately once you've decided to.
        """
        past_embeddings = self.store.get_embeddings(speaker)

        if len(past_embeddings) == 0:
            # No baseline yet -- nothing to compare against, so we
            # can't call it surprising or not.
            return ScoreResult(
                speaker=speaker,
                statement=statement,
                surprise_score=0.0,
                is_surprising=False,
                baseline_size=0,
            )

        new_embedding = self.embedder.embed(statement)
        baseline_vector = centroid(past_embeddings)
        similarity = cosine_similarity(new_embedding, baseline_vector)
        surprise_score = 1.0 - similarity

        threshold = self.threshold_for(speaker)
        is_surprising = False
        if surprise_score >= threshold:
            is_surprising = True

        return ScoreResult(
            speaker=speaker,
            statement=statement,
            surprise_score=surprise_score,
            is_surprising=is_surprising,
            baseline_size=len(past_embeddings),
        )


# ------------------------------------------------------------------ #
# DEMO / SMOKE TEST
# ------------------------------------------------------------------ #

def main():
    scorer = SurpriseScorer(store_path="baselines.json")

    # Seed a small baseline for Kevin Warsh with a few typical
    # data-dependent, measured statements.
    seed_statements = [
        "We remain data dependent and will assess incoming information carefully.",
        "Inflation risks are two-sided and policy should remain patient.",
        "The committee will continue to monitor labor market conditions closely.",
    ]
    for statement in seed_statements:
        scorer.add_to_baseline("kevin_warsh", statement)

    # Score a statement close to the baseline (should be low surprise).
    result_normal = scorer.score(
        "kevin_warsh",
        "We continue to watch the data and remain patient on policy.",
    )
    print("Normal statement:")
    print(f"  surprise_score = {result_normal.surprise_score:.3f}")
    print(f"  is_surprising  = {result_normal.is_surprising}")

    # Score a statement that breaks from the pattern (should be higher).
    result_surprising = scorer.score(
        "kevin_warsh",
        "If underlying inflation is not moving to 2 percent clearly and "
        "at sufficient speed, the Fed still has work to do.",
    )
    print("\nPotentially surprising statement:")
    print(f"  surprise_score = {result_surprising.surprise_score:.3f}")
    print(f"  is_surprising  = {result_surprising.is_surprising}")


if __name__ == "__main__":
    main()