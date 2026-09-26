"""
kalshi_ticker_finder.py
=======================

Finds active Kalshi prediction markets that are relevant to
a speech or statement.

Pipeline:

    Speech
       ↓
    Detect topics
       ↓
    Retrieve active Kalshi events using pagination
       ↓
    Deduplicate markets
       ↓
    Local relevance filtering
       ↓
    Gemini semantic relevance analysis
       ↓
    Validate + rank results
       ↓
    Relevant Kalshi tickers
"""

import os
import json
import time
import random
import re

from typing import List, Dict, Any

import requests

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

GEMINI_MODEL = "gemini-3.8-flash"

KALSHI_BASE_URL = (
    "https://api.elections.kalshi.com/trade-api/v2"
)

# Kalshi page size.
KALSHI_PAGE_SIZE = 200

# Number of locally filtered markets sent to Gemini.
MAX_GEMINI_CANDIDATES = 60

# Number of results returned.
DEFAULT_TOP_N = 5

# Minimum relevance score accepted from Gemini.
MIN_RELEVANCE_SCORE = 0.60

# Gemini retries.
MAX_GEMINI_RETRIES = 5

# Give Gemini plenty of room to finish the JSON.
GEMINI_MAX_OUTPUT_TOKENS = 4000


# ============================================================
# GEMINI API KEY
# ============================================================

def read_gemini_api_key() -> str:
    """
    Read the Gemini API key from either:

        GEMINI_API_KEY environment variable

    or:

        gemapi.txt
    """

    key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if key:
        return key.strip()

    key_file = os.path.join(
        SCRIPT_DIR,
        "gemapi.txt"
    )

    try:

        with open(
            key_file,
            encoding="utf-8"
        ) as f:

            return f.read().strip()

    except FileNotFoundError:

        return ""


GEMINI_API_KEY = read_gemini_api_key()


# ============================================================
# KALSHI: FETCH ALL EVENTS
# ============================================================

def fetch_live_kalshi_events(
    max_pages: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch all open Kalshi events using cursor pagination.

    max_pages is a safety limit so a malformed API response
    cannot cause an infinite loop.
    """

    all_events = []

    cursor = None

    page_number = 0

    while page_number < max_pages:

        page_number += 1

        url = (
            f"{KALSHI_BASE_URL}/events"
        )

        params = {
            "limit": KALSHI_PAGE_SIZE,
            "with_nested_markets": "true",
            "status": "open",
        }

        if cursor:
            params["cursor"] = cursor

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

        except requests.exceptions.HTTPError as e:

            print(
                f"\nKalshi HTTP error on page "
                f"{page_number}: {e}"
            )

            try:

                print(
                    "Response:"
                )

                print(
                    response.text[:1000]
                )

            except Exception:
                pass

            break

        except requests.exceptions.RequestException as e:

            print(
                f"\nKalshi request failed on page "
                f"{page_number}: {e}"
            )

            break

        except Exception as e:

            print(
                f"\nUnexpected Kalshi error: {e}"
            )

            break

        page_events = data.get(
            "events",
            []
        )

        all_events.extend(
            page_events
        )

        print(
            f"  Kalshi page {page_number}: "
            f"{len(page_events)} events "
            f"(total events: {len(all_events)})"
        )

        next_cursor = data.get(
            "cursor"
        )

        if not next_cursor:
            break

        if next_cursor == cursor:

            print(
                "Warning: Kalshi returned "
                "the same cursor twice."
            )

            break

        cursor = next_cursor

    return all_events


# ============================================================
# BUILD UNIQUE MARKET CATALOG
# ============================================================

def build_market_catalog(
    events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Flatten nested Kalshi markets into a unique market list.

    A market is uniquely identified by its ticker.
    """

    markets_by_ticker = {}

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

            # Keep only one copy of each market.
            markets_by_ticker[ticker] = {
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
            }

    return list(
        markets_by_ticker.values()
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str
) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TOPIC DETECTION
# ============================================================

def get_topic_keywords(
    speech_text: str
) -> List[str]:

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
            "technology",
            "tech",
            "semiconductor",
            "chips",
        ],
    }

    topics = []

    for topic, keywords in keyword_groups.items():

        if any(
            keyword in text
            for keyword in keywords
        ):

            topics.append(
                topic
            )

    return topics


# ============================================================
# LOCAL MARKET FILTER
# ============================================================

def filter_markets_locally(
    speech_text: str,
    market_catalog: List[Dict[str, Any]],
    max_candidates: int = MAX_GEMINI_CANDIDATES
) -> List[Dict[str, Any]]:
    """
    Rank markets using deterministic keyword/topic matching.

    This is only a pre-filter. Gemini makes the final relevance
    determination.
    """

    speech_normalized = normalize_text(
        speech_text
    )

    speech_words = set(
        speech_normalized.split()
    )

    topics = get_topic_keywords(
        speech_text
    )

    topic_keywords = {

        "federal reserve": [
            "fed",
            "federal reserve",
            "fomc",
            "central bank",
            "monetary policy",
        ],

        "interest rates": [
            "interest",
            "rate",
            "rates",
            "fed funds",
            "federal funds",
            "policy rate",
            "rate cut",
            "rate hike",
        ],

        "inflation": [
            "inflation",
            "cpi",
            "pce",
            "prices",
        ],

        "labor market": [
            "unemployment",
            "employment",
            "jobs",
            "payroll",
            "wages",
            "labor",
            "labour",
        ],

        "economic growth": [
            "gdp",
            "growth",
            "recession",
            "economy",
            "economic",
        ],

        "housing": [
            "housing",
            "home",
            "house",
            "mortgage",
            "rent",
            "real estate",
        ],

        "stocks": [
            "stock",
            "stocks",
            "s&p",
            "nasdaq",
            "dow",
            "equity",
        ],

        "treasury": [
            "treasury",
            "treasuries",
            "bond",
            "yield",
            "yields",
        ],

        "government": [
            "government",
            "congress",
            "senate",
            "house",
        ],

        "elections": [
            "election",
            "elections",
            "vote",
            "voting",
            "ballot",
            "president",
        ],

        "tariffs": [
            "tariff",
            "tariffs",
            "trade",
            "imports",
            "exports",
        ],

        "oil": [
            "oil",
            "crude",
            "opec",
            "gas",
            "gasoline",
        ],

        "crypto": [
            "bitcoin",
            "ethereum",
            "crypto",
            "cryptocurrency",
        ],

        "technology": [
            "artificial intelligence",
            "technology",
            "tech",
            "semiconductor",
            "chips",
        ],
    }

    stop_words = {
        "the",
        "a",
        "an",
        "will",
        "be",
        "is",
        "to",
        "of",
        "in",
        "for",
        "on",
        "and",
        "or",
        "by",
        "at",
        "from",
        "this",
        "that",
        "it",
        "with",
        "as",
        "are",
        "was",
        "were",
        "above",
        "below",
    }

    scored_markets = []

    for market in market_catalog:

        combined_text = " ".join([
            str(
                market.get(
                    "market_title",
                    ""
                )
            ),
            str(
                market.get(
                    "event_title",
                    ""
                )
            ),
            str(
                market.get(
                    "subtitle",
                    ""
                )
            ),
            str(
                market.get(
                    "yes_sub_title",
                    ""
                )
            ),
            str(
                market.get(
                    "no_sub_title",
                    ""
                )
            ),
        ])

        market_normalized = normalize_text(
            combined_text
        )

        market_words = set(
            market_normalized.split()
        )

        score = 0

        # ----------------------------------------------------
        # Exact phrase matches
        # ----------------------------------------------------

        for topic in topics:

            for keyword in topic_keywords.get(
                topic,
                []
            ):

                if keyword in market_normalized:

                    # Multi-word phrases are stronger evidence.
                    if " " in keyword:
                        score += 6
                    else:
                        score += 3

        # ----------------------------------------------------
        # Individual word overlap
        # ----------------------------------------------------

        common_words = (
            speech_words &
            market_words
        )

        meaningful_overlap = (
            common_words -
            stop_words
        )

        score += min(
            len(meaningful_overlap),
            5
        )

        if score > 0:

            scored_markets.append(
                (
                    score,
                    market
                )
            )

    scored_markets.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        market
        for score, market
        in scored_markets[
            :max_candidates
        ]
    ]


# ============================================================
# GEMINI REQUEST
# ============================================================

def ask_gemini_with_retry(
    client: genai.Client,
    prompt: str
):
    """
    Send the Gemini request with exponential backoff.
    """

    config = types.GenerateContentConfig(

        response_mime_type="application/json",

        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    )

    for attempt in range(
        MAX_GEMINI_RETRIES
    ):

        try:

            response = client.models.generate_content(

                model=GEMINI_MODEL,

                contents=prompt,

                config=config,
            )

            return response

        except Exception as e:

            error_text = str(e)

            temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "500" in error_text
                or "INTERNAL" in error_text
                or "504" in error_text
                or "DEADLINE_EXCEEDED" in error_text
            )

            if not temporary_error:

                raise

            if attempt >= (
                MAX_GEMINI_RETRIES - 1
            ):

                raise

            delay = (
                2 ** attempt
                + random.uniform(
                    0.5,
                    1.5
                )
            )

            print(
                f"\nGemini temporarily unavailable."
            )

            print(
                f"Retrying in {delay:.1f} seconds..."
            )

            time.sleep(
                delay
            )


# ============================================================
# REPAIR TRUNCATED JSON
# ============================================================

def parse_gemini_json(
    response_text: str
):
    """
    Parse Gemini JSON.

    If Gemini accidentally truncates the response near the end,
    attempt to recover complete objects from the JSON array.
    """

    text = response_text.strip()

    # --------------------------------------------------------
    # Normal JSON parsing
    # --------------------------------------------------------

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Attempt recovery from truncated array
    # --------------------------------------------------------

    if not text.startswith("["):

        raise ValueError(
            "Gemini response did not begin with a JSON array."
        )

    # Find complete JSON objects in the response.
    objects = []

    decoder = json.JSONDecoder()

    position = 1

    while position < len(text):

        # Skip whitespace and commas.
        while (
            position < len(text)
            and text[position] in " \n\r\t,"
        ):

            position += 1

        if position >= len(text):
            break

        if text[position] == "]":
            break

        try:

            obj, end_position = decoder.raw_decode(
                text,
                position
            )

            if isinstance(
                obj,
                dict
            ):

                objects.append(
                    obj
                )

            position = end_position

        except json.JSONDecodeError:

            # We reached an incomplete final object.
            break

    if objects:

        print(
            "Warning: Gemini returned truncated JSON. "
            f"Recovered {len(objects)} complete objects."
        )

        return objects

    raise ValueError(
        "Could not recover any complete JSON objects."
    )


# ============================================================
# FIND RELEVANT TICKERS
# ============================================================

def find_relevant_tickers(
    speech_text: str,
    top_n: int = DEFAULT_TOP_N
) -> List[Dict[str, Any]]:

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Put your Gemini API key in gemapi.txt."
        )

    # --------------------------------------------------------
    # Fetch Kalshi
    # --------------------------------------------------------

    print(
        "\nFetching active Kalshi markets...\n"
    )

    events = fetch_live_kalshi_events()

    if not events:

        print(
            "No active events retrieved from Kalshi."
        )

        return []

    market_catalog = build_market_catalog(
        events
    )

    print(
        f"\nRetrieved {len(events)} active events."
    )

    print(
        f"Retrieved {len(market_catalog)} unique active markets."
    )

    if not market_catalog:
        return []

    # --------------------------------------------------------
    # Topics
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
    # Local filter
    # --------------------------------------------------------

    candidates = filter_markets_locally(
        speech_text,
        market_catalog,
        MAX_GEMINI_CANDIDATES
    )

    print(
        f"Local filtering reduced the market universe "
        f"to {len(candidates)} candidates."
    )

    if not candidates:

        print(
            "No markets passed the local filter."
        )

        return []

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an expert prediction-market analyst.

Analyze the speech below and identify the active Kalshi
prediction markets that are genuinely relevant to the speech.

SPEECH:
{speech_text}

DETECTED TOPICS:
{json.dumps(topics)}

CANDIDATE MARKETS:
{json.dumps(candidates, indent=2)}

TASK:

Select up to {top_n} genuinely relevant markets.

A market is highly relevant when:

1. The speaker explicitly discusses the subject measured by
   the market.

2. The speech directly concerns a variable that determines
   the market outcome.

3. The speech changes or provides information relevant to the
   probability of the market outcome.

Avoid weak chains of indirect economic effects.

For example:

Fed policy
→ interest rates
→ technology valuations
→ IPO activity
→ specific company IPO

is too indirect.

Therefore, a Fed speech should not automatically make a
specific company IPO market relevant.

Be conservative.

If only two markets are genuinely relevant, return two.

If none are sufficiently relevant, return [].

SCORING:

0.90 - 1.00:
Directly discussed or extremely closely connected.

0.75 - 0.89:
Strong direct connection.

0.60 - 0.74:
Plausible but somewhat indirect.

Below 0.60:
Do not return.

IMPORTANT:

Only return tickers appearing in the candidate list.

Never invent a ticker.

Return ONLY valid JSON.

Required format:

[
  {{
    "ticker": "MARKET_TICKER",
    "event_title": "Event Title",
    "market_title": "Market Title",
    "relevance_score": 0.95,
    "reasoning": "Brief explanation."
  }}
]

Return [] if no market qualifies.
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

    response = ask_gemini_with_retry(
        client,
        prompt
    )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    try:

        results = parse_gemini_json(
            response.text
        )

    except Exception as e:

        print(
            "\nERROR parsing Gemini response:"
        )

        print(
            e
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
            "Gemini did not return a JSON array."
        )

        return []

    # --------------------------------------------------------
    # Validate tickers
    # --------------------------------------------------------

    valid_tickers = {
        market[
            "market_ticker"
        ]
        for market in candidates
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

        if ticker not in valid_tickers:
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

        validated.append({

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
    # Sort
    # --------------------------------------------------------

    validated.sort(
        key=lambda item: item[
            "relevance_score"
        ],
        reverse=True
    )

    return validated[:top_n]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    sample_speech = (
        "We remain deeply committed to bringing inflation "
        "back down to our 2% target. The labor market remains "
        "tight, and while we've seen progress, the Federal "
        "Reserve will not hesitate to adjust interest rate "
        "policy if macroeconomic indicators demand it."
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