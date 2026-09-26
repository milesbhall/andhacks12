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
MAX_GEMINI_CANDIDATES = 100

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

# Maximum markets from the same event during local selection.
MAX_MARKETS_PER_EVENT = 4

# Minimum relevance score for returned markets
MIN_RELEVANCE_SCORE = 0.60

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

    params = {
        "limit": limit,
        "with_nested_markets": True,
        "status": "open",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        return data.get("events", [])

    except Exception as e:

        print(
            f"Warning: Failed to fetch Kalshi events: {e}"
        )

        return []


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

            if keyword in text:

                found_topics.append(topic)

                break

    return found_topics


# ============================================================
# Hybrid vector retrieval
# ============================================================

def _market_vector_text(market: Dict[str, Any]) -> str:
    """Text used by the vector retrievers."""
    return normalize_text(" ".join(
        str(market.get(field, ""))
        for field in (
            "market_ticker",
            "event_ticker",
            "market_title",
            "event_title",
            "event_subtitle",
            "subtitle",
            "yes_sub_title",
            "no_sub_title",
        )
    ))


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
    market_text = _market_vector_text(market)
    score = 0.0
    reasons = []

    topic_terms = {
        "federal reserve": [
            "federal reserve", "fed", "fomc", "fed chair",
            "federal funds", "fed funds", "monetary policy",
            "central bank", "target range", "policy rate",
        ],
        "interest rates": [
            "interest rate", "interest rates", "rate cut", "rate cuts",
            "rate hike", "rate hikes", "policy rate", "fed funds",
            "federal funds", "target range",
        ],
        "inflation": [
            "inflation", "cpi", "consumer price", "consumer prices",
            "pce", "core inflation", "price stability", "prices",
        ],
        "labor market": [
            "labor market", "labour market", "employment", "unemployment",
            "unemployed", "jobs", "job growth", "payroll", "nonfarm",
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
            if contains_keyword(market_text, term)
        ]
        if fed_matches:
            score += 15
            reasons.append("Fed-specific: " + ", ".join(fed_matches[:4]))

    if "federal reserve" in topics or "interest rates" in topics:
        near_term_terms = [
            "fed decision", "rate cut before", "rate hike", "cut rates",
            "hike rates", "next fed rate cut", "before 2027", "2026",
            "2027",
        ]
        matches = [
            term for term in near_term_terms
            if contains_keyword(market_text, term)
        ]
        if matches:
            score += 8
            reasons.append("Near-term monetary policy: " + ", ".join(matches[:3]))

        if re.search(r"\b20(?:29|3[0-9])\b", market_text):
            score -= 8
            reasons.append("Long-term contract")

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

        except Exception as e:
            print(
                f"Semantic embedding stage failed: {e}"
            )
            print(
                "Continuing with TF-IDF + rule scoring."
            )
            semantic_indices = stage1_indices
            semantic_scores = lexical_scores[semantic_indices]
    else:
        print(
            "sentence-transformers is not installed; using TF-IDF + rules."
        )
        semantic_indices = stage1_indices
        semantic_scores = lexical_scores[semantic_indices]

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

        # Normalize the old rule score into roughly 0-1. The exact
        # value is less important than preserving its relative effect.
        rule_score = min(max(local_score / 50.0, 0.0), 1.0)
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

        if local_score <= 0 and semantic_score_normalized < 0.55:
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

    speech_normalized = normalize_text(speech_text)
    topics = get_topic_keywords(speech_text)

    topic_keywords = {
        "federal reserve": ["fed", "federal reserve", "fomc", "central bank", "monetary policy"],
        "interest rates": ["interest", "rate", "rates", "fed funds", "policy rate"],
        "inflation": ["inflation", "cpi", "pce", "prices"],
        "labor market": ["unemployment", "employment", "jobs", "payroll", "wages", "labor", "labour"],
        "economic growth": ["gdp", "growth", "recession", "economy", "economic"],
        "housing": ["housing", "home", "house", "mortgage", "rent", "real estate"],
        "stocks": ["stock", "stocks", "s&p", "nasdaq", "dow", "equity"],
        "treasury": ["treasury", "bond", "yield", "yields"],
        "government": ["government", "congress", "senate", "house"],
        "elections": ["election", "vote", "voting", "ballot", "president"],
        "tariffs": ["tariff", "tariffs", "trade", "imports", "exports"],
        "oil": ["oil", "crude", "opec", "gas", "gasoline"],
        "crypto": ["bitcoin", "ethereum", "crypto", "cryptocurrency"],
        "technology": ["artificial intelligence", "ai", "technology", "tech", "semiconductor", "chips"],
    }

    speech_words = set(speech_normalized.split())
    stop_words = {
        "the", "a", "an", "will", "be", "is", "to", "of", "in", "for",
        "on", "and", "or", "by", "at", "from", "this", "that", "it",
        "with", "as", "are", "was", "were", "above", "below", "we",
        "our", "i", "you", "your", "they", "their",
    }
    speech_words -= stop_words

    scored = []

    for market in market_catalog:
        market_text = _market_vector_text(market)
        market_words = set(market_text.split())
        score = min(len(speech_words & market_words), 4)
        reasons = []

        for topic in topics:
            matches = [
                kw for kw in topic_keywords.get(topic, [])
                if contains_keyword(market_text, kw)
            ]
            if matches:
                score += 8 + min(len(matches) - 1, 4)
                reasons.append(f"{topic}: {', '.join(matches[:4])}")

        if score > 0:
            copy = dict(market)
            copy["_local_score"] = score
            copy["_local_reasons"] = reasons
            scored.append(copy)

    scored.sort(key=lambda x: x["_local_score"], reverse=True)
    return scored[:max_candidates]


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

            is_temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
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

    response = ask_gemini(
        client,
        prompt
    )

    # --------------------------------------------------------
    # Parse response
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

        return []

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

        # Minimum relevance threshold.
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
    # Sort by relevance
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