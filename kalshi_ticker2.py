"""
kalshi_ticker_finder.py
========================

Finds active Kalshi prediction markets that are relevant to a
speech or statement.

Pipeline:

    Speech
       ↓
    Extract important topics/keywords
       ↓
    Retrieve active Kalshi markets
       ↓
    Local keyword filtering
       ↓
    Gemini semantic relevance analysis
       ↓
    Validate and rank results
       ↓
    Relevant Kalshi tickers
"""

import os
import json
import time
import re
import requests
from datetime import datetime, timezone

from typing import List, Dict, Any, Optional, Tuple

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    np = None
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from google import genai
from google.genai import types


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or ""
)

if not GEMINI_API_KEY:
    try:
        with open(
            os.path.join(SCRIPT_DIR, "gemapi.txt"),
            encoding="utf-8"
        ) as f:
            GEMINI_API_KEY = f.read().strip()
    except FileNotFoundError:
        GEMINI_API_KEY = ""


# Gemini model
GEMINI_MODEL = "gemini-3.8-flash"

# Kalshi API
KALSHI_BASE_URL = (
    "https://api.elections.kalshi.com/trade-api/v2"
)

# Number of Gemini candidates
MAX_GEMINI_CANDIDATES = 40

# Hybrid retrieval settings.
# TF-IDF is used as a very fast first-stage vector search across
# the entire Kalshi catalog. Semantic embeddings are then used
# only on the strongest lexical candidates.
VECTOR_RETRIEVAL_CANDIDATES = 1500
SEMANTIC_RETRIEVAL_CANDIDATES = 300
SEMANTIC_MODEL_NAME = os.environ.get(
    "KALSHI_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)
SEMANTIC_BATCH_SIZE = 128
SEMANTIC_WEIGHT = 0.55
LEXICAL_WEIGHT = 0.25
RULE_WEIGHT = 0.20
TFIDF_FALLBACK_LEXICAL_WEIGHT = 0.25
TFIDF_FALLBACK_RULE_WEIGHT = 0.75

# Maximum markets from the same event during local selection.
MAX_MARKETS_PER_EVENT = 1

# Minimum relevance score for returned markets
MIN_RELEVANCE_SCORE = 0.60
MIN_HYBRID_SCORE = 0.55

# Gemini retry settings
MAX_RETRIES = 5


# ============================================================
# Kalshi API
# ============================================================

def fetch_live_kalshi_events(
    limit: int = 200
) -> List[Dict[str, Any]]:
    """
    Fetch active/open Kalshi events and their nested markets.
    """

    url = f"{KALSHI_BASE_URL}/events"

    events = []
    cursor = None
    seen_cursors = set()

    while True:
        params = {
            "limit": limit,
            "with_nested_markets": True,
            "status": "open",
        }
        if cursor:
            params["cursor"] = cursor

        try:
            response = requests.get(
                url,
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Warning: Failed to fetch Kalshi events: {e}")
            break

        events.extend(data.get("events", []))
        next_cursor = data.get("cursor")
        if not next_cursor or next_cursor in seen_cursors:
            break

        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return events


# ============================================================
# Convert Kalshi events into market records
# ============================================================

def build_market_catalog(
    events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert nested Kalshi event data into a flat list of markets.
    """

    catalog = []

    for event in events:

        event_title = event.get(
            "title",
            ""
        )

        event_ticker = event.get(
            "event_ticker",
            ""
        )

        markets = event.get(
            "markets",
            []
        )

        for market in markets:

            ticker = market.get(
                "ticker"
            )

            if not ticker:
                continue

            catalog.append({
                "market_ticker": ticker,

                "market_title": market.get(
                    "title",
                    ""
                ),

                "event_ticker": event_ticker,

                "event_title": event_title,

                "expected_expiration_time": (
                    market.get("expected_expiration_time")
                    or market.get("expiration_time")
                ),

                "subtitle": market.get(
                    "subtitle",
                    ""
                ),

                "yes_sub_title": market.get(
                    "yes_sub_title",
                    ""
                ),

                "no_sub_title": market.get(
                    "no_sub_title",
                    ""
                ),
            })

    return catalog


def _days_until_expiration(market: Dict[str, Any]) -> Optional[float]:
    expiration_time = market.get("expected_expiration_time")
    if not expiration_time:
        return None

    try:
        expiration = datetime.fromisoformat(
            str(expiration_time).replace("Z", "+00:00")
        )
    except ValueError:
        return None

    if expiration.tzinfo is None:
        expiration = expiration.replace(tzinfo=timezone.utc)

    return (expiration - datetime.now(timezone.utc)).total_seconds() / 86400


def _expiration_hybrid_adjustment(market: Dict[str, Any]) -> float:
    days_until_expiration = _days_until_expiration(market)
    if days_until_expiration is None:
        return 0.0
    if days_until_expiration <= 30:
        return 0.20
    if days_until_expiration <= 90:
        return 0.12
    if days_until_expiration <= 180:
        return 0.02
    if days_until_expiration <= 365:
        return -0.08
    if days_until_expiration <= 730:
        return -0.16
    if days_until_expiration <= 1095:
        return -0.24
    if days_until_expiration <= 1460:
        return -0.30
    return -0.35


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for simple keyword matching.
    """

    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def contains_keyword(
    text: str,
    keyword: str
) -> bool:
    """
    Match a keyword using word boundaries.
    """

    normalized_text = normalize_text(text)
    normalized_keyword = normalize_text(keyword)

    if not normalized_keyword:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(normalized_keyword)
        + r"(?![a-z0-9])"
    )

    return re.search(
        pattern,
        normalized_text
    ) is not None


# ============================================================
# Topic extraction
# ============================================================

def get_topic_keywords(
    speech_text: str
) -> List[str]:
    """
    Identify broad economic/political/financial topics using
    simple keyword rules.

    This is intentionally deterministic so we do not need to
    spend a Gemini request just figuring out what the speech
    is about.
    """

    text = normalize_text(
        speech_text
    )

    keyword_groups = {

        "federal reserve": [
            "federal reserve",
            "fed",
            "fomc",
            "monetary policy",
            "central bank",
        ],

        "interest rates": [
            "interest rate",
            "interest rates",
            "rate cut",
            "rate cuts",
            "rate hike",
            "rate hikes",
            "policy rate",
            "fed funds",
            "federal funds",
        ],

        "inflation": [
            "inflation",
            "cpi",
            "consumer price",
            "prices",
            "price stability",
            "pce",
            "core inflation",
        ],

        "labor market": [
            "labor market",
            "labour market",
            "employment",
            "unemployment",
            "unemployed",
            "jobs",
            "job growth",
            "payroll",
            "nonfarm payroll",
            "nonfarm",
            "wages",
            "wage growth",
        ],

        "economic growth": [
            "gdp",
            "economic growth",
            "growth",
            "recession",
            "output",
            "economic activity",
        ],

        "housing": [
            "housing",
            "home prices",
            "house prices",
            "mortgage",
            "mortgages",
            "rent",
            "rents",
            "real estate",
        ],

        "stocks": [
            "stock market",
            "stocks",
            "equity market",
            "s&p",
            "nasdaq",
            "dow",
        ],

        "treasury": [
            "treasury",
            "treasuries",
            "bond yields",
            "yield",
            "yield curve",
        ],

        "government": [
            "government",
            "congress",
            "senate",
            "house of representatives",
            "federal government",
        ],

        "elections": [
            "election",
            "elections",
            "vote",
            "voting",
            "ballot",
            "president",
            "presidential",
        ],

        "tariffs": [
            "tariff",
            "tariffs",
            "trade war",
            "imports",
            "exports",
            "trade",
        ],

        "oil": [
            "oil",
            "crude",
            "opec",
            "gas prices",
            "gasoline",
            "petroleum",
        ],

        "crypto": [
            "bitcoin",
            "ethereum",
            "crypto",
            "cryptocurrency",
            "digital asset",
        ],

        "technology": [
            "artificial intelligence",
            "ai",
            "technology",
            "tech",
            "semiconductor",
            "chips",
        ],
    }

    found_topics = []

    for topic, keywords in keyword_groups.items():

        for keyword in keywords:

            if contains_keyword(text, keyword):

                found_topics.append(topic)

                break

    return found_topics


# ============================================================
# Hybrid vector retrieval
# ============================================================

def _market_vector_text(market: Dict[str, Any]) -> str:
    """Text used by the vector retrievers."""
    market_title = str(market.get("market_title", ""))
    return normalize_text(" ".join((
        market_title,
        market_title,
        market_title,
        str(market.get("event_title", "")),
        str(market.get("market_ticker", "")),
        str(market.get("event_ticker", "")),
        str(market.get("event_subtitle", "")),
        str(market.get("subtitle", "")),
        str(market.get("yes_sub_title", "")),
        str(market.get("no_sub_title", "")),
    )))


_SEMANTIC_MODEL = None


def _get_semantic_model():
    """Load the embedding model once per process."""
    global _SEMANTIC_MODEL

    if SentenceTransformer is None:
        return None

    if _SEMANTIC_MODEL is None:
        print(
            f"Loading semantic embedding model: {SEMANTIC_MODEL_NAME}..."
        )
        _SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)

    return _SEMANTIC_MODEL


def score_market_locally(
    speech_text: str,
    market: Dict[str, Any],
    topics: List[str],
) -> Tuple[float, List[str]]:
    """Fast deterministic rule score used as one component of the hybrid rank."""

    speech = normalize_text(speech_text)
    market_title = normalize_text(str(market.get("market_title", "")))
    market_text = _market_vector_text(market)
    score = 0.0
    reasons = []

    topic_terms = {
        "federal reserve": [
            "federal reserve", "fed", "fomc",
            "federal funds", "fed funds", "monetary policy",
            "central bank", "target range", "policy rate",
        ],
        "interest rates": [
            "interest rate", "interest rates", "rate cut", "rate cuts",
            "rate hike", "rate hikes", "cut rates", "hike rates",
            "cut interest rates", "raise rates", "lower rates",
            "policy rate", "fed funds", "federal funds", "target range",
        ],
        "inflation": [
            "inflation", "cpi", "consumer price", "consumer prices",
            "pce", "core inflation", "price stability", "prices",
        ],
        "labor market": [
            "labor market", "labour market", "employment", "unemployment",
            "unemployed", "jobs", "job growth", "payroll", "payrolls",
            "nonfarm payrolls", "nonfarm",
            "wages", "wage growth", "jobless",
        ],
        "economic growth": [
            "gdp", "economic growth", "growth", "recession", "output",
            "economic activity", "economy",
        ],
        "housing": [
            "housing", "home prices", "house prices", "mortgage",
            "mortgages", "rent", "rents", "real estate",
        ],
        "stocks": [
            "stock market", "stocks", "equity market", "equities",
            "s&p", "nasdaq", "dow",
        ],
        "treasury": [
            "treasury", "treasuries", "bond yield", "bond yields",
            "yield curve", "10 year", "10-year", "2 year", "2-year",
        ],
        "government": [
            "government", "congress", "senate", "house of representatives",
            "federal government",
        ],
        "elections": [
            "election", "elections", "vote", "voting", "ballot",
            "president", "presidential",
        ],
        "tariffs": [
            "tariff", "tariffs", "trade war", "imports", "exports", "trade",
        ],
        "oil": [
            "oil", "crude", "opec", "gas prices", "gasoline", "petroleum",
        ],
        "crypto": [
            "bitcoin", "ethereum", "crypto", "cryptocurrency", "digital asset",
        ],
        "technology": [
            "artificial intelligence", "ai", "technology", "tech",
            "semiconductor", "chips",
        ],
    }

    for topic in topics:
        matches = [
            term for term in topic_terms.get(topic, [])
            if contains_keyword(market_text, term)
        ]
        if matches:
            score += 8 + min(len(matches) - 1, 5)
            reasons.append(f"{topic}: {', '.join(matches[:4])}")

    speech_words = set(speech.split())
    market_words = set(market_text.split())
    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "than",
        "that", "this", "these", "those", "will", "would", "could",
        "should", "have", "has", "had", "are", "were", "was", "been",
        "being", "from", "with", "for", "into", "about", "above", "below",
        "before", "after", "what", "which", "who", "when", "where", "how",
        "its", "their", "our", "your", "they", "them", "we", "you", "i",
        "to", "of", "in", "on", "at", "by", "as", "is", "be",
    }
    overlap = (speech_words & market_words) - stop_words
    score += min(len(overlap), 6)

    if "federal reserve" in topics:
        fed_matches = [
            term for term in (
                "federal reserve", "federal funds", "fed funds", "fomc",
                "target range", "policy rate", "monetary policy",
            )
            if contains_keyword(market_title, term)
        ]
        if fed_matches:
            score += 8
            reasons.append("Fed-specific: " + ", ".join(fed_matches[:4]))

    macro_outcomes = {
        "inflation outcomes": (
            ("inflation", "cpi", "consumer price", "pce"),
            ("inflation", "cpi", "consumer price", "pce"),
        ),
        "labor outcomes": (
            ("labor market", "labour market", "employment", "unemployment",
             "jobs", "payroll", "wages"),
            ("unemployment", "employment", "jobs", "payroll", "labor force",
             "payrolls", "nonfarm payrolls", "wages"),
        ),
        "Fed policy outcomes": (
            ("federal reserve", "fed", "fomc", "interest rate", "monetary policy",
             "rate cut", "rate hike"),
            ("fed decision", "fomc", "federal funds rate", "fed funds rate",
             "rate cut", "rate hike", "cut rates", "hike rates",
             "cut interest rates", "raise rates", "lower rates",
             "interest rate", "target range"),
        ),
    }

    for outcome, (speech_terms, market_terms) in macro_outcomes.items():
        if (
            any(contains_keyword(speech, term) for term in speech_terms)
            and any(contains_keyword(market_title, term) for term in market_terms)
        ):
            score += 22
            reasons.append(outcome)

    fed_personnel_terms = (
        "next president", "next fed president", "next federal reserve president",
        "president of the federal reserve", "next chair", "next fed chair",
        "chairman", "chairperson", "governor", "governors", "fed chair nominee",
        "nominee", "successor", "appointment", "appoint", "appointed",
        "nomination", "nominate", "replacement for", "who will lead",
    )
    fed_institution_terms = (
        "federal reserve", "fed chair", "fed president", "fomc",
        "board of governors", "central bank",
    )
    if (
        any(contains_keyword(market_title, term) for term in fed_personnel_terms)
        and any(contains_keyword(market_title, term) for term in fed_institution_terms)
    ):
        score -= 35
        reasons.append("Fed personnel or appointment market")

    policy_speech_terms = (
        "federal reserve", "fed", "fomc", "interest rate", "monetary policy",
        "rate cut", "rate hike",
    )
    policy_market_terms = (
        "fed decision", "fomc", "federal funds rate", "fed funds rate",
        "rate cut", "rate hike", "cut rates", "hike rates",
        "cut interest rates", "raise rates", "lower rates", "interest rate",
        "target range",
    )
    near_term_terms = (
        "2026", "2027", "next meeting", "next fed meeting", "next rate decision",
        "upcoming meeting", "this year", "by year end", "before year end",
        "next 12 months",
    )
    if (
        any(contains_keyword(speech, term) for term in policy_speech_terms)
        and any(contains_keyword(market_title, term) for term in policy_market_terms)
        and any(contains_keyword(market_title, term) for term in near_term_terms)
    ):
        score += 14
        reasons.append("Near-term Fed policy contract")

    contract_text = " ".join((
        market_title,
        normalize_text(str(market.get("market_ticker", ""))),
    ))
    contract_years = [
        int(year) for year in re.findall(r"\b20\d{2}\b", contract_text)
    ]
    if contract_years:
        farthest_year = max(contract_years)
        if farthest_year >= 2035:
            score -= 45
            reasons.append("Very long-dated contract")
        elif farthest_year >= 2030:
            score -= 30
            reasons.append("Long-dated contract")
        elif farthest_year >= 2029:
            score -= 5
            reasons.append("Medium-term contract")

    unrelated_terms = [
        "nfl", "nba", "mlb", "nhl", "super bowl", "touchdown",
        "game winner", "oscar", "grammy", "movie", "box office", "celebrity",
    ]
    if any(contains_keyword(market_text, term) for term in unrelated_terms):
        score -= 15

    return score, reasons


def hybrid_vector_retrieval(
    speech_text: str,
    market_catalog: List[Dict[str, Any]],
    max_candidates: int = MAX_GEMINI_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    Fast two-stage vector retrieval.

    Stage 1:
        TF-IDF cosine similarity over the entire market catalog.

    Stage 2:
        Sentence-transformer semantic similarity on only the strongest
        lexical candidates.

    The final ranking combines semantic similarity, lexical similarity,
    and the existing deterministic rule score. This avoids running a
    Python regex/scoring loop over every market for every speech while
    preserving the existing topic-specific logic.
    """

    if not market_catalog:
        return []

    speech_vector_text = normalize_text(speech_text)
    market_texts = [_market_vector_text(m) for m in market_catalog]

    # --------------------------------------------------------
    # Stage 1: very fast sparse-vector retrieval
    # --------------------------------------------------------
    if TfidfVectorizer is None or cosine_similarity is None:
        print(
            "scikit-learn is not installed; falling back to keyword filtering."
        )
        return filter_markets_locally_legacy(
            speech_text,
            market_catalog,
            max_candidates=max_candidates,
        )

    print(
        f"Vector stage 1: indexing {len(market_catalog):,} markets with TF-IDF..."
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=150000,
    )

    matrix = vectorizer.fit_transform(market_texts)
    query_matrix = vectorizer.transform([speech_vector_text])
    lexical_scores = cosine_similarity(query_matrix, matrix).ravel()

    stage1_count = min(
        VECTOR_RETRIEVAL_CANDIDATES,
        len(market_catalog),
    )

    if stage1_count == len(market_catalog):
        stage1_indices = np.argsort(-lexical_scores)
    else:
        stage1_indices = np.argpartition(
            -lexical_scores,
            stage1_count - 1,
        )[:stage1_count]
        stage1_indices = stage1_indices[
            np.argsort(-lexical_scores[stage1_indices])
        ]

    # --------------------------------------------------------
    # Stage 2: semantic vectors on only the top lexical matches
    # --------------------------------------------------------
    semantic_model = _get_semantic_model()
    has_semantic_scores = False

    if semantic_model is not None:
        semantic_indices = stage1_indices[
            :min(
                SEMANTIC_RETRIEVAL_CANDIDATES,
                len(stage1_indices),
            )
        ]

        print(
            f"Vector stage 2: embedding {len(semantic_indices):,} strongest markets..."
        )

        try:
            query_embedding = semantic_model.encode(
                [speech_vector_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            market_embeddings = semantic_model.encode(
                [market_texts[i] for i in semantic_indices],
                batch_size=SEMANTIC_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

            semantic_scores = np.dot(
                market_embeddings,
                query_embedding[0],
            )
            has_semantic_scores = True

        except Exception as e:
            print(
                f"Semantic embedding stage failed: {e}"
            )
            print(
                "Continuing with TF-IDF + rule scoring."
            )
            semantic_indices = stage1_indices
            semantic_scores = np.zeros(len(semantic_indices))
    else:
        print(
            "sentence-transformers is not installed; using TF-IDF + rules."
        )
        semantic_indices = stage1_indices
        semantic_scores = np.zeros(len(semantic_indices))

    # --------------------------------------------------------
    # Existing deterministic rules, evaluated only on the vector
    # shortlist rather than all 131k+ markets.
    # --------------------------------------------------------
    topics = get_topic_keywords(speech_text)
    scored = []

    semantic_lookup = {
        int(index): float(score)
        for index, score in zip(
            semantic_indices,
            semantic_scores,
        )
    }

    max_lexical = max(
        float(lexical_scores[i]) for i in semantic_lookup
    ) if semantic_lookup else 1.0

    for index, semantic_score in semantic_lookup.items():
        market = market_catalog[index]
        local_score, reasons = score_market_locally(
            speech_text,
            market,
            topics,
        )

        if has_semantic_scores:
            rule_score = min(max(local_score / 50.0, -1.0), 1.0)
            lexical_score = min(
                max(float(lexical_scores[index]) / max(max_lexical, 1e-9), 0.0),
                1.0,
            )
            semantic_score_normalized = min(
                max((float(semantic_score) + 1.0) / 2.0, 0.0),
                1.0,
            )
            hybrid_score = (
                SEMANTIC_WEIGHT * semantic_score_normalized
                + LEXICAL_WEIGHT * lexical_score
                + RULE_WEIGHT * rule_score
            )
        else:
            lexical_score = min(max(float(lexical_scores[index]), 0.0), 1.0)
            rule_score = local_score / (abs(local_score) + 10.0)
            semantic_score_normalized = 0.0
            hybrid_score = (
                TFIDF_FALLBACK_LEXICAL_WEIGHT * lexical_score
                + TFIDF_FALLBACK_RULE_WEIGHT * rule_score
            )

        if "Fed personnel or appointment market" in reasons:
            hybrid_score -= 0.35
        if "Very long-dated contract" in reasons:
            hybrid_score -= 0.35
        elif "Long-dated contract" in reasons:
            hybrid_score -= 0.25

        hybrid_score += _expiration_hybrid_adjustment(market)
        hybrid_score = max(0.0, min(hybrid_score, 1.0))

        if hybrid_score < MIN_HYBRID_SCORE:
            continue

        market_copy = dict(market)
        market_copy["_local_score"] = local_score
        market_copy["_local_reasons"] = reasons
        market_copy["_lexical_similarity"] = float(lexical_scores[index])
        market_copy["_semantic_similarity"] = float(semantic_score)
        market_copy["_hybrid_score"] = float(hybrid_score)
        scored.append(market_copy)

    scored.sort(
        key=lambda x: x["_hybrid_score"],
        reverse=True,
    )

    # Preserve event diversity so Gemini does not receive many nearly
    # identical contracts from one event.
    selected = []
    event_counts = {}

    for market in scored:
        event_ticker = market.get("event_ticker", "")
        count = event_counts.get(event_ticker, 0)

        if count >= MAX_MARKETS_PER_EVENT:
            continue

        selected.append(market)
        event_counts[event_ticker] = count + 1

        if len(selected) >= max_candidates:
            break

    return selected


# ============================================================
# Legacy deterministic filter
# ============================================================

def filter_markets_locally_legacy(
    speech_text: str,
    market_catalog: List[Dict[str, Any]],
    max_candidates: int = MAX_GEMINI_CANDIDATES,
) -> List[Dict[str, Any]]:
    """Original deterministic filter retained as a fallback."""

    topics = get_topic_keywords(speech_text)
    scored = []

    for market in market_catalog:
        local_score, reasons = score_market_locally(
            speech_text,
            market,
            topics,
        )
        if local_score <= 0:
            continue

        hybrid_score = local_score / (local_score + 10.0)
        if "Fed personnel or appointment market" in reasons:
            hybrid_score -= 0.35
        if "Very long-dated contract" in reasons:
            hybrid_score -= 0.35
        elif "Long-dated contract" in reasons:
            hybrid_score -= 0.30
        elif "Medium-term contract" in reasons:
            hybrid_score -= 0.05
        hybrid_score += _expiration_hybrid_adjustment(market)
        hybrid_score = max(0.0, min(hybrid_score, 1.0))
        if hybrid_score < MIN_HYBRID_SCORE:
            continue

        copy = dict(market)
        copy["_local_score"] = local_score
        copy["_local_reasons"] = reasons
        copy["_hybrid_score"] = hybrid_score
        scored.append(copy)

    scored.sort(key=lambda x: x["_hybrid_score"], reverse=True)

    selected = []
    event_counts = {}
    for market in scored:
        event_ticker = market.get("event_ticker", "")
        count = event_counts.get(event_ticker, 0)
        if count >= MAX_MARKETS_PER_EVENT:
            continue

        selected.append(market)
        event_counts[event_ticker] = count + 1
        if len(selected) >= max_candidates:
            break

    return selected


# ============================================================
# Local market filtering
# ============================================================

def filter_markets_locally(
    speech_text: str,
    market_catalog: List[Dict[str, Any]],
    max_candidates: int = MAX_GEMINI_CANDIDATES
) -> List[Dict[str, Any]]:
    """Use hybrid vector retrieval instead of scanning every market with regex."""
    return hybrid_vector_retrieval(
        speech_text,
        market_catalog,
        max_candidates=max_candidates,
    )

# ============================================================
# Gemini request with retry
# ============================================================

def ask_gemini(
    client: genai.Client,
    prompt: str
):
    """
    Ask Gemini for market relevance analysis.

    Automatically retries temporary 503/unavailable errors.
    """

    config = types.GenerateContentConfig(
        response_mime_type="application/json"
    )

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )

        except Exception as e:

            error_text = str(e)

            normalized_error = error_text.upper()
            is_quota_error = any(
                marker in normalized_error
                for marker in ("429", "RESOURCE_EXHAUSTED", "QUOTA")
            )

            if is_quota_error:
                raise

            is_temporary_error = (
                "503" in normalized_error
                or "UNAVAILABLE" in normalized_error
            )

            if not is_temporary_error:

                raise

            if attempt == MAX_RETRIES - 1:

                print(
                    "\nGemini remained unavailable "
                    f"after {MAX_RETRIES} attempts."
                )

                raise

            wait_time = 2 ** attempt

            print(
                f"\nGemini temporarily unavailable. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(
                wait_time
            )


# ============================================================
# Gemini relevance analysis
# ============================================================

def find_relevant_tickers(
    speech_text: str,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Match a speech against active Kalshi markets.
    """

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Put your API key in gemapi.txt."
        )

    # --------------------------------------------------------
    # Fetch markets
    # --------------------------------------------------------

    print(
        "Fetching active Kalshi markets..."
    )

    events = fetch_live_kalshi_events(
        limit=200
    )

    market_catalog = build_market_catalog(
        events
    )

    if not market_catalog:

        print(
            "No active markets retrieved from Kalshi."
        )

        return []

    print(
        f"Retrieved {len(market_catalog)} active markets."
    )

    # --------------------------------------------------------
    # Identify topics
    # --------------------------------------------------------

    topics = get_topic_keywords(
        speech_text
    )

    if topics:

        print(
            "Detected topics: "
            + ", ".join(topics)
        )

    else:

        print(
            "No predefined topics detected."
        )

    # --------------------------------------------------------
    # Local filtering
    # --------------------------------------------------------

    candidates = filter_markets_locally(
        speech_text,
        market_catalog,
        max_candidates=MAX_GEMINI_CANDIDATES
    )

    print(
        f"Reduced market candidates to "
        f"{len(candidates)} for Gemini."
    )

    if not candidates:

        print(
            "No markets passed the preliminary filter."
        )

        return []

    def local_fallback_results(reasoning: str) -> List[Dict[str, Any]]:
        fallback = []
        for market in candidates:
            score = float(market.get("_hybrid_score", 0.0))
            if score < MIN_HYBRID_SCORE:
                continue

            fallback.append({
                "ticker": market.get("market_ticker"),
                "event_title": market.get("event_title"),
                "market_title": market.get("market_title"),
                "relevance_score": score,
                "reasoning": reasoning,
            })

        return fallback[:top_n]

    # --------------------------------------------------------
    # Build Gemini prompt
    # --------------------------------------------------------

    prompt = f"""
You are an expert financial and prediction-market analyst.

Your task is to identify ACTIVE Kalshi prediction markets that
are genuinely relevant to the speech below.

============================================================
SPEECH
============================================================

{speech_text}

============================================================
DETECTED TOPICS
============================================================

{json.dumps(topics)}

============================================================
KALSHI MARKET CANDIDATES
============================================================

{json.dumps(candidates, indent=2)}

============================================================
TASK
============================================================

Select up to {top_n} markets that are genuinely relevant to
what the speaker is discussing.

The purpose of this system is to identify prediction markets
whose outcomes could reasonably be affected by the information
or policy discussion contained in the speech.

============================================================
IMPORTANT RELEVANCE RULES
============================================================

1. DIRECT RELEVANCE

A market is highly relevant if the speaker explicitly discusses
the subject measured by the market.

Example:

Speech:
"The Federal Reserve remains focused on bringing inflation
back to 2 percent."

Market:
"Will CPI inflation be above X%?"

This is highly relevant.

------------------------------------------------------------

2. STRONG ECONOMIC CONNECTIONS

A market may be relevant when the speech discusses a variable
that directly determines or strongly informs the market's
outcome.

For example:

Speech:
"The labor market has weakened."

Market:
"Will the unemployment rate be above X%?"

This is relevant.

------------------------------------------------------------

3. AVOID LONG CHAINS OF INDIRECT EFFECTS

Do NOT select a market merely because the speech could
theoretically affect another variable that could then affect
the market.

Example:

Fed interest rates
    ->
technology valuations
    ->
IPO activity
    ->
specific company IPO

That is too indirect.

A Fed speech should therefore NOT automatically make an
OpenAI IPO market relevant.

------------------------------------------------------------

4. DISTINGUISH SUBJECT MATTER FROM GENERAL MACRO EFFECTS

If a market concerns a completely different subject from the
speech, do not select it merely because macroeconomic conditions
could theoretically affect it.

------------------------------------------------------------

5. BE CONSERVATIVE

If only one or two markets are genuinely relevant, return only
those markets.

Do not fill all {top_n} positions with weak relationships.

------------------------------------------------------------

6. RELEVANCE SCORE

Use this scale:

0.90 - 1.00
Directly discussed or extremely closely connected.

0.75 - 0.89
Strong direct connection.

0.60 - 0.74
Plausible but somewhat indirect.

Below 0.60
Weak connection.

Do NOT return markets below 0.60.

------------------------------------------------------------

7. DO NOT INVENT TICKERS

Only return markets that appear in the candidate list.

The ticker must exactly match one of the supplied market tickers.

------------------------------------------------------------

8. RETURN VALID JSON ONLY

Return ONLY a JSON array.

Do not use markdown.

Do not include any explanation outside the JSON.

Format:

[
    {{
        "ticker": "MARKET_TICKER",
        "event_title": "Event Title",
        "market_title": "Market Title",
        "relevance_score": 0.95,
        "reasoning": "Brief explanation of the direct connection."
    }}
]

If there are no sufficiently relevant markets, return:

[]
"""

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    print(
        "\nAnalyzing speech for relevant "
        "Kalshi tickers...\n"
    )

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    try:

        response = ask_gemini(
            client,
            prompt
        )

    except Exception as e:

        print(
            "\nGemini unavailable or quota exhausted."
        )

        print(
            f"Gemini error: {e}"
        )

        print(
            "\nFalling back to local vector/rule ranking."
        )

        return local_fallback_results(
            "Selected by local hybrid relevance ranking after Gemini failed."
        )

    # --------------------------------------------------------
    # Parse Gemini response
    # --------------------------------------------------------

    try:

        results = json.loads(
            response.text
        )

    except Exception as e:

        print(
            f"Error parsing Gemini JSON response: {e}"
        )

        print(
            "\nRaw Gemini output:"
        )

        print(
            response.text
        )

        print(
            "\nFalling back to local ranking."
        )

        return local_fallback_results(
            "Selected by local hybrid relevance ranking because Gemini returned invalid JSON."
        )

    if not isinstance(
        results,
        list
    ):

        print(
            "Gemini did not return a JSON list."
        )

        return []

    # --------------------------------------------------------
    # Validate tickers
    # --------------------------------------------------------

    valid_tickers = {
        market["market_ticker"]
        for market in candidates
        if market.get("market_ticker")
    }

    validated_results = []

    for result in results:

        if not isinstance(
            result,
            dict
        ):
            continue

        ticker = result.get(
            "ticker"
        )

        # Prevent hallucinated tickers.
        if ticker not in valid_tickers:
            continue

        score = result.get(
            "relevance_score",
            0
        )

        try:

            score = float(
                score
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if score < MIN_RELEVANCE_SCORE:
            continue

        validated_results.append({

            "ticker": ticker,

            "event_title": result.get(
                "event_title",
                ""
            ),

            "market_title": result.get(
                "market_title",
                ""
            ),

            "relevance_score": score,

            "reasoning": result.get(
                "reasoning",
                ""
            ),

        })

    # --------------------------------------------------------
    # Sort by Gemini relevance
    # --------------------------------------------------------

    validated_results.sort(
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    return validated_results[:top_n]


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    sample_speech = (
        "We remain deeply committed to bringing inflation back "
        "down to our 2% target. The labor market remains tight, "
        "and while we've seen progress, the Federal Reserve "
        "will not hesitate to adjust interest rate policy if "
        "macroeconomic indicators demand it."
    )

    try:

        matches = find_relevant_tickers(
            sample_speech,
            top_n=3
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "RELEVANT KALSHI MARKETS"
        )

        print(
            "=" * 60
        )

        if matches:

            print(
                json.dumps(
                    matches,
                    indent=2
                )
            )

        else:

            print(
                "No sufficiently relevant "
                "Kalshi markets found."
            )

    except KeyboardInterrupt:

        print(
            "\nProgram interrupted."
        )

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(
            e
        )