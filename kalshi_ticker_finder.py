"""
kalshi_ticker_finder.py
=======================

Find active Kalshi prediction markets relevant to a speech or statement.

This file is designed to work in two ways:

1. Directly from the command line:

       python kalshi_ticker_finder.py --speech "The Federal Reserve..."

2. From another Python program:

       from kalshi_ticker_finder import find_relevant_tickers

       results = find_relevant_tickers(transcript)

The second method is intended for integration with a separate
speech transcription program.

Pipeline:

    Speech transcript
          ↓
    Detect topics
          ↓
    Fetch active Kalshi events
          ↓
    Build market catalog
          ↓
    Local candidate filtering
          ↓
    Gemini semantic analysis
          ↓
    Validate tickers
          ↓
    Return relevant Kalshi markets


Environment:

    GEMINI_API_KEY
    GEMINI_MODEL

Alternatively, put the Gemini API key in:

    gemapi.txt
"""

import os
import json
import time
import re
import argparse
import random
from typing import List, Dict, Any, Optional, Tuple

import requests

from google import genai
from google.genai import types


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    ""
).strip()

if not GEMINI_API_KEY:
    try:
        with open(
            os.path.join(
                SCRIPT_DIR,
                "gemapi.txt"
            ),
            encoding="utf-8"
        ) as f:
            GEMINI_API_KEY = f.read().strip()

    except FileNotFoundError:
        GEMINI_API_KEY = ""


GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

KALSHI_BASE_URL = (
    "https://api.elections.kalshi.com/trade-api/v2"
)

EVENT_PAGE_SIZE = 200

# Number of candidates sent to Gemini.
MAX_GEMINI_CANDIDATES = 100

# Number of candidates displayed in --no-gemini mode.
MAX_LOCAL_DISPLAY = 30

# Maximum markets from the same event during local selection.
MAX_MARKETS_PER_EVENT = 4

# Minimum Gemini relevance score.
MIN_RELEVANCE_SCORE = 0.60

# Maximum Gemini attempts for temporary failures.
MAX_GEMINI_RETRIES = 6

REQUEST_TIMEOUT = 30

DEBUG_FILE = os.path.join(
    SCRIPT_DIR,
    "kalshi_candidates_debug.json"
)


# ============================================================
# Topic definitions
# ============================================================

TOPIC_KEYWORDS = {

    "federal reserve": [
        "federal reserve",
        "fed",
        "fomc",
        "fed chair",
        "federal funds",
        "fed funds",
        "monetary policy",
        "central bank",
        "target range",
        "policy rate",
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
        "target range",
        "borrowing costs",
    ],

    "inflation": [
        "inflation",
        "cpi",
        "consumer price",
        "consumer prices",
        "pce",
        "core inflation",
        "price stability",
        "prices",
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
        "jobless",
    ],

    "economic growth": [
        "gdp",
        "economic growth",
        "growth",
        "recession",
        "output",
        "economic activity",
        "economy",
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
        "equities",
        "s&p",
        "nasdaq",
        "dow",
    ],

    "treasury": [
        "treasury",
        "treasuries",
        "bond yield",
        "bond yields",
        "yield curve",
        "10 year",
        "10-year",
        "2 year",
        "2-year",
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


# ============================================================
# Text utilities
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for keyword matching.
    """

    text = str(text or "").lower()

    text = (
        text
        .replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
    )

    text = re.sub(
        r"[^a-z0-9\s\-.%]",
        " ",
        text
    )

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

    This prevents:

        "ai" from matching "remain"
        "fed" from matching unrelated substrings
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
# Topic detection
# ============================================================

def get_topic_keywords(
    speech_text: str
) -> List[str]:

    text = normalize_text(
        speech_text
    )

    topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():

        for keyword in keywords:

            if contains_keyword(
                text,
                keyword
            ):

                topics.append(
                    topic
                )

                break

    return topics


# ============================================================
# Kalshi API
# ============================================================

def fetch_live_kalshi_events() -> List[Dict[str, Any]]:
    """
    Fetch all open Kalshi events using cursor pagination.
    """

    url = (
        f"{KALSHI_BASE_URL}/events"
    )

    all_events = []

    cursor = None
    page = 0

    while True:

        page += 1

        params = {
            "limit": EVENT_PAGE_SIZE,
            "status": "open",
            "with_nested_markets": "true",
        }

        if cursor:
            params["cursor"] = cursor

        try:

            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

        except Exception as e:

            print(
                f"\nWarning: Failed to fetch Kalshi "
                f"page {page}: {e}"
            )

            break

        events = data.get(
            "events",
            []
        )

        all_events.extend(
            events
        )

        print(
            f"  Kalshi page {page}: "
            f"{len(events)} events "
            f"(total events: {len(all_events)})"
        )

        cursor = data.get(
            "cursor"
        )

        if not cursor:
            break

        if not events:
            break

    return all_events


# ============================================================
# Market catalog
# ============================================================

def build_market_catalog(
    events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    catalog = []

    skipped_inactive = 0

    seen_tickers = set()

    for event in events:

        event_title = event.get(
            "title",
            ""
        )

        event_ticker = event.get(
            "event_ticker",
            ""
        )

        event_subtitle = event.get(
            "sub_title",
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

            if ticker in seen_tickers:
                continue

            seen_tickers.add(
                ticker
            )

            status = str(
                market.get(
                    "status",
                    ""
                )
            ).lower()

            if status and status not in {
                "open",
                "active"
            }:

                skipped_inactive += 1

                continue

            catalog.append({

                "market_ticker": ticker,

                "market_title": market.get(
                    "title",
                    ""
                ),

                "event_ticker": event_ticker,

                "event_title": event_title,

                "event_subtitle": event_subtitle,

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

                "status": status,

                "yes_bid": market.get(
                    "yes_bid"
                ),

                "yes_ask": market.get(
                    "yes_ask"
                ),

                "no_bid": market.get(
                    "no_bid"
                ),

                "no_ask": market.get(
                    "no_ask"
                ),

                "last_price": market.get(
                    "last_price"
                ),

                "volume": market.get(
                    "volume"
                ),

                "open_interest": market.get(
                    "open_interest"
                ),

                "close_time": market.get(
                    "close_time"
                ),

                "expiration_time": market.get(
                    "expiration_time"
                ),
            })

    print(
        f"Skipped {skipped_inactive} non-active markets."
    )

    return catalog


# ============================================================
# Market text
# ============================================================

def market_search_text(
    market: Dict[str, Any]
) -> str:

    pieces = [

        market.get(
            "market_ticker",
            ""
        ),

        market.get(
            "event_ticker",
            ""
        ),

        market.get(
            "market_title",
            ""
        ),

        market.get(
            "event_title",
            ""
        ),

        market.get(
            "event_subtitle",
            ""
        ),

        market.get(
            "subtitle",
            ""
        ),

        market.get(
            "yes_sub_title",
            ""
        ),

        market.get(
            "no_sub_title",
            ""
        ),
    ]

    return normalize_text(
        " ".join(
            str(x)
            for x in pieces
            if x
        )
    )


# ============================================================
# Local relevance scoring
# ============================================================

def score_market_locally(
    speech_text: str,
    market: Dict[str, Any],
    topics: List[str]
) -> Tuple[float, List[str]]:

    speech = normalize_text(
        speech_text
    )

    market_text = market_search_text(
        market
    )

    score = 0.0
    reasons = []

    # --------------------------------------------------------
    # Topic-specific matches
    # --------------------------------------------------------

    for topic in topics:

        keywords = TOPIC_KEYWORDS.get(
            topic,
            []
        )

        matches = []

        for keyword in keywords:

            if contains_keyword(
                market_text,
                keyword
            ):

                matches.append(
                    keyword
                )

        if matches:

            score += 10

            score += min(
                len(matches) - 1,
                5
            )

            reasons.append(
                f"{topic}: "
                + ", ".join(
                    matches[:4]
                )
            )

    # --------------------------------------------------------
    # Strong Fed-specific language
    # --------------------------------------------------------

    fed_terms = [
        "federal reserve",
        "federal funds",
        "fed funds",
        "fomc",
        "target range",
        "policy rate",
        "monetary policy",
        "fed chair",
    ]

    fed_matches = [
        term
        for term in fed_terms
        if contains_keyword(
            market_text,
            term
        )
    ]

    if fed_matches:

        score += 12

        reasons.append(
            "Fed-specific: "
            + ", ".join(
                fed_matches[:4]
            )
        )

    # --------------------------------------------------------
    # Strong macro-variable matches
    # --------------------------------------------------------

    variable_terms = [
        "inflation",
        "cpi",
        "pce",
        "unemployment",
        "employment",
        "payroll",
        "gdp",
        "interest rate",
        "rate cut",
        "rate hike",
    ]

    variable_matches = [
        term
        for term in variable_terms
        if contains_keyword(
            market_text,
            term
        )
    ]

    if variable_matches:

        score += min(
            len(variable_matches) * 3,
            12
        )

        reasons.append(
            "Macro variables: "
            + ", ".join(
                variable_matches[:5]
            )
        )

    # --------------------------------------------------------
    # Exact meaningful word overlap
    # --------------------------------------------------------

    speech_words = set(
        speech.split()
    )

    market_words = set(
        market_text.split()
    )

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "will",
        "would",
        "could",
        "should",
        "have",
        "has",
        "had",
        "are",
        "were",
        "was",
        "been",
        "being",
        "from",
        "with",
        "for",
        "into",
        "about",
        "above",
        "below",
        "before",
        "after",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "its",
        "their",
        "our",
        "your",
        "they",
        "them",
        "we",
        "you",
        "i",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "as",
        "is",
        "be",
    }

    overlap = (
        speech_words &
        market_words
    ) - stop_words

    if overlap:

        score += min(
            len(overlap),
            6
        )

    # --------------------------------------------------------
    # Fed context boost
    # --------------------------------------------------------

    if "federal reserve" in topics:

        if (
            contains_keyword(
                market_text,
                "federal reserve"
            )
            or contains_keyword(
                market_text,
                "federal funds"
            )
            or contains_keyword(
                market_text,
                "fed funds"
            )
            or contains_keyword(
                market_text,
                "fomc"
            )
        ):

            score += 15

    # --------------------------------------------------------
    # Near-term monetary-policy boost
    # --------------------------------------------------------

    if (
        "federal reserve" in topics
        or "interest rates" in topics
    ):

        near_term_terms = [
            "fed decision",
            "fed decision in",
            "rate cut before",
            "rate hike",
            "cut rates",
            "hike rates",
            "next fed rate cut",
        ]

        near_term_matches = [
            term
            for term in near_term_terms
            if contains_keyword(
                market_text,
                term
            )
        ]

        if near_term_matches:

            score += 8

            reasons.append(
                "Near-term monetary policy: "
                + ", ".join(
                    near_term_matches[:3]
                )
            )

    # --------------------------------------------------------
    # Long-term penalty
    # --------------------------------------------------------

    if (
        "federal reserve" in topics
        or "interest rates" in topics
    ):

        long_term_years = re.findall(
            r"\b20(2[9]|3[0-9])\b",
            market_text
        )

        if long_term_years:

            score -= 8

            reasons.append(
                "Long-term contract"
            )

    # --------------------------------------------------------
    # Obviously unrelated categories
    # --------------------------------------------------------

    unrelated_terms = [
        "nfl",
        "nba",
        "mlb",
        "nhl",
        "super bowl",
        "touchdown",
        "game winner",
        "oscar",
        "grammy",
        "movie",
        "box office",
        "celebrity",
    ]

    if any(
        contains_keyword(
            market_text,
            term
        )
        for term in unrelated_terms
    ):

        score -= 15

    return score, reasons


# ============================================================
# Diversified local filtering
# ============================================================

def filter_markets_locally(
    speech_text: str,
    market_catalog: List[Dict[str, Any]],
    max_candidates: int = MAX_GEMINI_CANDIDATES
) -> List[Dict[str, Any]]:

    topics = get_topic_keywords(
        speech_text
    )

    scored = []

    for market in market_catalog:

        score, reasons = score_market_locally(
            speech_text,
            market,
            topics
        )

        if score <= 0:
            continue

        market_copy = dict(
            market
        )

        market_copy["_local_score"] = score

        market_copy["_local_reasons"] = reasons

        scored.append(
            market_copy
        )

    scored.sort(
        key=lambda x: x[
            "_local_score"
        ],
        reverse=True
    )

    selected = []

    event_counts = {}

    # First pass: diversity.
    for market in scored:

        event_ticker = market.get(
            "event_ticker",
            ""
        )

        count = event_counts.get(
            event_ticker,
            0
        )

        if count >= MAX_MARKETS_PER_EVENT:
            continue

        selected.append(
            market
        )

        event_counts[event_ticker] = (
            count + 1
        )

        if len(selected) >= max_candidates:
            break

    return selected


# ============================================================
# Debug output
# ============================================================

def clean_for_output(
    market: Dict[str, Any]
) -> Dict[str, Any]:

    result = dict(
        market
    )

    result.pop(
        "_local_score",
        None
    )

    result.pop(
        "_local_reasons",
        None
    )

    return result


def save_debug_candidates(
    candidates: List[Dict[str, Any]]
) -> None:

    try:

        with open(
            DEBUG_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                [
                    clean_for_output(
                        market
                    )
                    for market in candidates
                ],
                f,
                indent=2
            )

        print(
            f"\nSaved candidate markets to "
            f"{DEBUG_FILE}"
        )

    except Exception as e:

        print(
            f"Warning: could not save debug file: {e}"
        )


# ============================================================
# Gemini
# ============================================================

def is_temporary_gemini_error(
    error: Exception
) -> bool:

    error_text = str(
        error
    ).lower()

    temporary_terms = [
        "503",
        "unavailable",
        "429",
        "resource_exhausted",
        "high demand",
        "temporarily unavailable",
        "deadline exceeded",
        "timeout",
        "timed out",
    ]

    return any(
        term in error_text
        for term in temporary_terms
    )


def ask_gemini(
    client: genai.Client,
    prompt: str
):
    """
    Send the Gemini request using the Chat API.

    Using client.chats.create() avoids the automatic-function-
    calling warning associated with directly using
    client.models.generate_content().
    """

    config = types.GenerateContentConfig(
        response_mime_type="application/json"
    )

    last_error = None

    for attempt in range(
        1,
        MAX_GEMINI_RETRIES + 1
    ):

        try:

            print(
                f"Gemini attempt "
                f"{attempt}/{MAX_GEMINI_RETRIES}..."
            )

            chat = client.chats.create(
                model=GEMINI_MODEL,
                config=config
            )

            response = chat.send_message(
                prompt
            )

            return response

        except Exception as e:

            last_error = e

            if not is_temporary_gemini_error(
                e
            ):

                raise

            if attempt >= MAX_GEMINI_RETRIES:

                print(
                    "\nGemini remained unavailable "
                    f"after {MAX_GEMINI_RETRIES} attempts."
                )

                raise last_error

            base_wait = min(
                2 ** attempt,
                30
            )

            jitter = random.uniform(
                0,
                1.5
            )

            wait_time = (
                base_wait +
                jitter
            )

            print(
                "\nGemini temporarily unavailable."
            )

            print(
                f"Retrying in {wait_time:.1f} seconds..."
            )

            time.sleep(
                wait_time
            )

    raise last_error


# ============================================================
# Gemini JSON extraction
# ============================================================

def extract_json_array(
    text: str
) -> Optional[List[Any]]:

    if not text:
        return None

    text = text.strip()

    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            list
        ):

            return parsed

    except Exception:
        pass

    fenced = re.search(
        r"```(?:json)?\s*(\[.*?\])\s*```",
        text,
        re.DOTALL
    )

    if fenced:

        try:

            parsed = json.loads(
                fenced.group(1)
            )

            if isinstance(
                parsed,
                list
            ):

                return parsed

        except Exception:
            pass

    start = text.find(
        "["
    )

    end = text.rfind(
        "]"
    )

    if start >= 0 and end > start:

        candidate = text[
            start:end + 1
        ]

        try:

            parsed = json.loads(
                candidate
            )

            if isinstance(
                parsed,
                list
            ):

                return parsed

        except Exception:
            pass

    return None


# ============================================================
# Gemini semantic analysis
# ============================================================

def analyze_with_gemini(
    speech_text: str,
    topics: List[str],
    candidates: List[Dict[str, Any]],
    top_n: int
) -> Tuple[Optional[List[Dict[str, Any]]], str]:

    if not GEMINI_API_KEY:

        print(
            "\nGemini API key not found."
        )

        print(
            "Use --no-gemini or add your key "
            "to gemapi.txt."
        )

        return None, "unavailable"

    clean_candidates = [
        clean_for_output(market)
        for market in candidates
    ]

    prompt = f"""
You are an expert prediction-market analyst.

Identify ACTIVE Kalshi prediction markets that are genuinely
relevant to the speech below.

SPEECH
============================================================
{speech_text}

DETECTED TOPICS
============================================================
{json.dumps(topics)}

CANDIDATE MARKETS
============================================================
{json.dumps(clean_candidates, indent=2)}

TASK
============================================================

Select up to {top_n} markets.

Only select a market if the speech has a direct or strong
economic, policy, financial, or factual connection to the
contract's actual resolution.

RELEVANCE STANDARD
============================================================

1. DIRECT RELEVANCE

Prefer contracts whose outcome is directly about something
the speaker discusses.

2. STRONG CONNECTION

A market can be relevant when the speech discusses a variable
that directly informs the contract.

3. AVOID LONG INDIRECT CHAINS

Do not select a market merely because the speech could
eventually affect it through several intermediate variables.

4. READ THE ACTUAL CONTRACT

Pay attention to the exact wording of the market.

5. FEDERAL RESERVE SPEECHES

For Federal Reserve speeches, distinguish carefully between:

- upcoming Fed decisions
- near-term rate cuts
- near-term rate hikes
- federal funds rate levels
- inflation
- unemployment/labor market
- GDP/economic growth

A Fed speech mentioning monetary policy does NOT make every
Fed-related contract relevant.

6. TIME HORIZON MATTERS

Prefer markets whose resolution horizon matches the subject
of the speech.

For a current Fed speech, an upcoming Fed decision can be
substantially more relevant than a contract about the federal
funds rate many years in the future.

7. DO NOT FILL THE LIST

If only one or two markets are genuinely relevant, return
only those.

It is completely acceptable to return [].

8. SCORE

0.90-1.00 = extremely direct
0.75-0.89 = strong connection
0.60-0.74 = plausible but somewhat indirect

Do not return anything below 0.60.

9. NEVER INVENT A TICKER

The ticker must exactly match one of the candidate markets.

10. RETURN JSON ONLY

Return exactly:

[
  {{
    "ticker": "EXACT_TICKER",
    "event_title": "Event title",
    "market_title": "Market title",
    "relevance_score": 0.95,
    "reasoning": "Brief explanation."
  }}
]

If there are no sufficiently relevant markets:

[]
"""

    print(
        "\nAnalyzing speech for relevant "
        "Kalshi tickers...\n"
    )

    try:

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        response = ask_gemini(
            client,
            prompt
        )

    except Exception as e:

        print(
            "\nGemini unavailable:"
        )

        print(
            e
        )

        return None, "unavailable"

    response_text = getattr(
        response,
        "text",
        ""
    )

    results = extract_json_array(
        response_text
    )

    if results is None:

        print(
            "\nCould not parse Gemini response."
        )

        print(
            "\nRaw Gemini output:"
        )

        print(
            response_text
        )

        return None, "parse_error"

    if not results:

        return [], "no_relevant_markets"

    valid_markets = {
        market["market_ticker"]: market
        for market in candidates
        if market.get(
            "market_ticker"
        )
    }

    validated = []

    for result in results:

        if not isinstance(
            result,
            dict
        ):

            continue

        ticker = result.get(
            "ticker"
        )

        if ticker not in valid_markets:

            print(
                f"Warning: Gemini returned ticker "
                f"not in candidate list: {ticker}"
            )

            continue

        try:

            score = float(
                result.get(
                    "relevance_score",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if score < MIN_RELEVANCE_SCORE:

            continue

        original = valid_markets[
            ticker
        ]

        validated.append({

            "ticker": ticker,

            "event_title": (
                result.get(
                    "event_title"
                )
                or original.get(
                    "event_title",
                    ""
                )
            ),

            "market_title": (
                result.get(
                    "market_title"
                )
                or original.get(
                    "market_title",
                    ""
                )
            ),

            "relevance_score": score,

            "reasoning": result.get(
                "reasoning",
                ""
            ),

        })

    validated.sort(
        key=lambda x: x[
            "relevance_score"
        ],
        reverse=True
    )

    if not validated:

        return [], "no_relevant_markets"

    return validated[:top_n], "success"


# ============================================================
# Main finder
# ============================================================

def find_relevant_tickers(
    speech_text: str,
    top_n: int = 5,
    use_gemini: bool = True
) -> List[Dict[str, Any]]:

    """
    Main function for other Python programs.

    Example:

        transcript = "The Federal Reserve..."
        results = find_relevant_tickers(transcript)

    Returns a list of dictionaries such as:

        [
            {
                "ticker": "KXRATECUT-26DEC31",
                "event_title": "Fed rate cut before 2027?",
                "market_title": "...",
                "relevance_score": 0.85,
                "reasoning": "..."
            }
        ]
    """

    if not speech_text or not speech_text.strip():

        print(
            "No speech text was provided."
        )

        return []

    topics = get_topic_keywords(
        speech_text
    )

    if topics:

        print(
            "Detected topics: "
            + ", ".join(
                topics
            )
        )

    else:

        print(
            "No predefined topics detected."
        )

    # --------------------------------------------------------
    # Fetch active events
    # --------------------------------------------------------

    print(
        "\nFetching active Kalshi markets...\n"
    )

    events = fetch_live_kalshi_events()

    print(
        f"\nRetrieved {len(events)} active events."
    )

    if not events:

        print(
            "No active events retrieved from Kalshi."
        )

        return []

    # --------------------------------------------------------
    # Build catalog
    # --------------------------------------------------------

    market_catalog = build_market_catalog(
        events
    )

    print(
        f"Retrieved {len(market_catalog)} "
        f"currently active markets."
    )

    if not market_catalog:

        return []

    # --------------------------------------------------------
    # Local filtering
    # --------------------------------------------------------

    candidates = filter_markets_locally(
        speech_text,
        market_catalog,
        max_candidates=MAX_GEMINI_CANDIDATES
    )

    print(
        f"Local filtering produced "
        f"{len(candidates)} candidates."
    )

    save_debug_candidates(
        candidates
    )

    if not candidates:

        print(
            "\nNo local candidates found."
        )

        return []

    # --------------------------------------------------------
    # Local-only mode
    # --------------------------------------------------------

    if not use_gemini:

        print(
            "\nGemini disabled with --no-gemini."
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "LOCAL CANDIDATES"
        )

        print(
            "=" * 60
        )

        display = []

        for market in candidates[
            :MAX_LOCAL_DISPLAY
        ]:

            display.append({

                "ticker": market.get(
                    "market_ticker"
                ),

                "event_title": market.get(
                    "event_title"
                ),

                "market_title": market.get(
                    "market_title"
                ),

                "local_score": market.get(
                    "_local_score"
                ),

                "reasons": market.get(
                    "_local_reasons"
                ),

            })

        print(
            json.dumps(
                display,
                indent=2
            )
        )

        return display[:top_n]

    # --------------------------------------------------------
    # Gemini analysis
    # --------------------------------------------------------

    results, status = analyze_with_gemini(
        speech_text,
        topics,
        candidates,
        top_n
    )

    # --------------------------------------------------------
    # Gemini found nothing
    # --------------------------------------------------------

    if status == "no_relevant_markets":

        print(
            "\nGemini found no sufficiently relevant "
            "Kalshi markets."
        )

        return []

    # --------------------------------------------------------
    # Gemini unavailable
    # --------------------------------------------------------

    if status in {
        "unavailable",
        "parse_error",
        "error"
    }:

        print(
            "\nGemini could not validate the candidates."
        )

        print(
            "Returning top local candidates as a "
            "fallback."
        )

        fallback = []

        for market in candidates[
            :top_n
        ]:

            fallback.append({

                "ticker": market.get(
                    "market_ticker"
                ),

                "event_title": market.get(
                    "event_title"
                ),

                "market_title": market.get(
                    "market_title"
                ),

                "relevance_score": None,

                "reasoning": (
                    "Local candidate only; "
                    "Gemini validation was unavailable."
                ),

            })

        return fallback

    return results or []


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Find Kalshi markets relevant to a speech."
        )
    )

    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help=(
            "Only run local filtering."
        )
    )

    parser.add_argument(
        "--speech",
        type=str,
        default=None,
        help=(
            "Speech text to analyze."
        )
    )

    parser.add_argument(
        "--speech-file",
        type=str,
        default=None,
        help=(
            "Path to a text file containing "
            "a speech transcript."
        )
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help=(
            "Number of relevant markets to return."
        )
    )

    return parser.parse_args()


# ============================================================
# Program entry
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    print(
        "\n"
        + "=" * 60
    )

    print(
        "KALSHI TICKER FINDER"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Get speech
    # --------------------------------------------------------

    sample_speech = None

    if args.speech:

        sample_speech = args.speech

    elif args.speech_file:

        try:

            with open(
                args.speech_file,
                "r",
                encoding="utf-8"
            ) as f:

                sample_speech = f.read()

        except Exception as e:

            print(
                f"\nERROR reading speech file: {e}"
            )

            raise SystemExit(1)

    else:

        # Built-in test speech.
        sample_speech = (
            "The Federal Reserve remains committed to "
            "bringing inflation back to our 2 percent target. "
            "The labor market has remained resilient, but we "
            "are closely watching the balance between inflation "
            "and employment. If economic conditions require it, "
            "the Federal Reserve will adjust the target range "
            "for the federal funds rate."
        )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    try:

        matches = find_relevant_tickers(
            sample_speech,
            top_n=args.top,
            use_gemini=not args.no_gemini
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
            "\nProgram interrupted.")

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(
            e
        )