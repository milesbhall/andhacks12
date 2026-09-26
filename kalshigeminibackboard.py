"""
kalshi_gemini_backboard.py
===========================
Integrates four pieces using three API credentials:

  1. Kalshi           -> pulls live prediction-market data (signed REST calls)
  2. Gemini            -> analyzes that data directly via Google's Generative
                          AI API, and also powers the speaker-baseline
                          surprise scorer (see speaker_baseline.py)
  3. Backboard          -> persists the exchange (question + Gemini's answer)
                          to a Backboard thread, so it's remembered across
                          sessions and can be pulled into future conversations
  4. SurpriseScorer     -> (speaker_baseline.py) flags when a new statement
                          deviates from a speaker's own historical baseline

Two modes:

  A) Direct market lookup (original behavior):
       python kalshi_gemini_backboard.py "How is KXWTAOPEN doing?"

  B) Speaker-statement mode (new -- the actual pipeline):
       python kalshi_gemini_backboard.py --speaker kevin_warsh \
           --statement "The Fed still has work to do." \
           --ticker KXWTAOPEN-26

     This scores the statement against kevin_warsh's baseline first.
     If it's surprising, it pulls live Kalshi data for --ticker, asks
     Gemini to interpret the statement in light of that market data,
     and logs the whole exchange to Backboard. If it's NOT surprising,
     it says so and stops (no wasted Kalshi/Backboard calls).

  Seed a speaker's baseline without scoring/trading:
       python kalshi_gemini_backboard.py --speaker kevin_warsh \
           --statement "We remain data dependent." --seed

------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------
1. Install dependencies (RSA signing shells out to the `openssl` CLI, so
   no `cryptography` package is needed -- just make sure `openssl` is on
   your PATH, which it is on virtually every Mac/Linux/WSL box):
       pip install requests google-genai --break-system-packages

2. Put these files beside this script:

    privkey.txt       Kalshi RSA private key in PEM format
    kalshikey.txt     Kalshi API key ID
    gemapi.txt        Gemini API key
    backboardapi.txt  Backboard API key

    Environment variables with the names below override these files.

   KALSHI_API_KEY_ID        Your Kalshi API key ID (a UUID, from Kalshi
                             account settings -> API Keys)
   KALSHI_PRIVATE_KEY_PATH  Path to the RSA private key .pem file Kalshi
                             gave you when you created that API key
                             (Kalshi auth is API-key-ID + RSA key pair,
                             not a single bearer token)

   GEMINI_API_KEY            Your Google AI Studio / Gemini API key

   BACKBOARD_API_KEY         Your Backboard API key (Settings -> API Keys
                             in the Backboard dashboard)

3. Also keep speaker_baseline.py in the same folder -- this script
   imports SurpriseScorer from it.
------------------------------------------------------------------------
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

from speaker_baseline import SurpriseScorer

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_secret_file(filename: str) -> str:
    try:
        with open(os.path.join(SCRIPT_DIR, filename), encoding="utf-8") as secret_file:
            return secret_file.read().strip()
    except FileNotFoundError:
        return ""


KALSHI_API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID") or _read_secret_file("kalshikey.txt")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH") or os.path.join(SCRIPT_DIR, "privkey.txt")
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or _read_secret_file("gemapi.txt")
GEMINI_MODEL = "gemini-2.5-flash"

BACKBOARD_API_KEY = os.environ.get("BACKBOARD_API_KEY") or _read_secret_file("backboardapi.txt")
BACKBOARD_BASE_URL = "https://app.backboard.io/api"
# Optional: reuse an existing Backboard thread id across runs so the
# conversation history accumulates. Leave blank to start a new thread
# every run.
BACKBOARD_THREAD_ID = os.environ.get("BACKBOARD_THREAD_ID", "")

# Default market to look up if no ticker is given.
MARKET_TICKER = "KXWTAOPEN-26"

# Where the speaker-baseline embeddings are persisted between runs.
BASELINE_STORE_PATH = os.path.join(SCRIPT_DIR, "baselines.json")


# ------------------------------------------------------------------ #
# 1. KALSHI  -- signed REST client
# ------------------------------------------------------------------ #

class KalshiClient:
    def __init__(self, api_key_id: str, private_key_path: str):
        if not api_key_id or not private_key_path:
            raise RuntimeError(
                "Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH."
            )
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        if not os.path.isfile(private_key_path):
            raise RuntimeError(f"No private key file at {private_key_path}")
        # Fail fast if openssl isn't available, with a clear message.
        try:
            subprocess.run(
                ["openssl", "version"], capture_output=True, check=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "openssl CLI not found on PATH. Install it (e.g. "
                "`apt install openssl` / `brew install openssl`) or ask "
                "for a pure-Python fallback (needs pycryptodome instead)."
            )

    def _sign(self, method: str, path: str) -> tuple[str, str]:
        """Returns (timestamp_ms, base64_signature) per Kalshi's RSA-PSS scheme.

        Kalshi signs: timestamp_ms + METHOD + path (no query string, no host).
        Signing is delegated to the `openssl` CLI so no extra pip packages
        are required.
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}".encode("utf-8")

        proc = subprocess.run(
            [
                "openssl", "dgst", "-sha256",
                "-sign", self.private_key_path,
                "-sigopt", "rsa_padding_mode:pss",
                "-sigopt", "rsa_pss_saltlen:-1",  # salt length == digest length
            ],
            input=message,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"openssl signing failed: {proc.stderr.decode()}")

        return timestamp_ms, base64.b64encode(proc.stdout).decode("utf-8")

    def _headers(self, method: str, path: str) -> dict:
        timestamp_ms, sig = self._sign(method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "Content-Type": "application/json",
        }

    def get_market(self, ticker: str) -> dict:
        path = f"/trade-api/v2/markets/{ticker}"
        url = f"{KALSHI_BASE_URL}/markets/{ticker}"
        resp = requests.get(url, headers=self._headers("GET", path))
        resp.raise_for_status()
        return resp.json()

    def list_markets(self, series_ticker: str = None, limit: int = 10) -> dict:
        path = "/trade-api/v2/markets"
        url = f"{KALSHI_BASE_URL}/markets"
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        resp = requests.get(url, headers=self._headers("GET", path), params=params)
        resp.raise_for_status()
        return resp.json()

    def list_events(self, series_ticker: str = None, limit: int = 20) -> dict:
        """Events group related markets under one series -- this is usually
        the easier way to browse what's currently live and find real
        tickers, rather than guessing market tickers directly.
        """
        path = "/trade-api/v2/events"
        url = f"{KALSHI_BASE_URL}/events"
        params = {"limit": limit, "with_nested_markets": True}
        if series_ticker:
            params["series_ticker"] = series_ticker
        resp = requests.get(url, headers=self._headers("GET", path), params=params)
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------------------ #
# 2. GEMINI -- direct call with the user's own Gemini API key
# ------------------------------------------------------------------ #

def ask_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY.")

    from google import genai  # pip install google-genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


# ------------------------------------------------------------------ #
# 3. BACKBOARD -- persist the exchange to a thread (memory/history)
# ------------------------------------------------------------------ #

class BackboardClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("Set BACKBOARD_API_KEY.")
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def log_exchange(self, thread_id: str, question: str, answer: str) -> str:
        """Sends a message to a Backboard thread recording what was asked and
        what Gemini answered, so it's retrievable later. Returns the thread_id
        (a new one is created if none was given).
        """
        content = (
            f"[Kalshi/Gemini integration log -- {datetime.now(timezone.utc).isoformat()}]\n\n"
            f"Question: {question}\n\n"
            f"Gemini's analysis:\n{answer}"
        )
        payload = {"content": content, "stream": False}
        if thread_id:
            payload["thread_id"] = thread_id

        resp = requests.post(
            f"{BACKBOARD_BASE_URL}/threads/messages",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("thread_id", thread_id)


# ------------------------------------------------------------------ #
# 4. PIPELINE -- speaker statement -> surprise score -> (maybe) trade signal
# ------------------------------------------------------------------ #

def handle_statement(speaker: str, statement: str, ticker: str, seed_only: bool):
    scorer = SurpriseScorer(store_path=BASELINE_STORE_PATH, api_key=GEMINI_API_KEY)

    if seed_only:
        scorer.add_to_baseline(speaker, statement)
        print(f"Added to {speaker}'s baseline (not scored). Baseline updated.")
        return

    result = scorer.score(speaker, statement)
    print(f"Speaker:        {speaker}")
    print(f"Statement:      {statement}")
    print(f"Baseline size:  {result.baseline_size}")
    print(f"Surprise score: {result.surprise_score:.3f}")
    print(f"Is surprising:  {result.is_surprising}")

    if result.baseline_size == 0:
        print("\nNo baseline yet for this speaker -- seeding with this "
              "statement instead of scoring against nothing.")
        scorer.add_to_baseline(speaker, statement)
        return

    if not result.is_surprising:
        print("\nNot surprising enough to act on. Adding to baseline and stopping.")
        scorer.add_to_baseline(speaker, statement)
        return

    # -- Surprising: pull live Kalshi data and get Gemini's read on it --
    print(f"\nSurprising statement detected. Checking Kalshi market {ticker}...")

    kalshi = KalshiClient(KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH)
    try:
        market_data = kalshi.get_market(ticker)
    except requests.HTTPError as e:
        print(f"Kalshi request failed: {e}\nResponse: {e.response.text}")
        scorer.add_to_baseline(speaker, statement)
        return

    prompt = (
        f"{speaker} just said: \"{statement}\"\n\n"
        f"Our surprise-scoring model rates this as {result.surprise_score:.2f} "
        f"(0 = matches their usual pattern, 1 = totally novel), which is "
        f"above our threshold.\n\n"
        f"Here is the current Kalshi market data for {ticker} (JSON):\n"
        f"{json.dumps(market_data, indent=2)}\n\n"
        "In a few sentences: does this statement plausibly move this "
        "market, and in which direction? Be concise and concrete."
    )
    answer = ask_gemini(prompt)
    print("\n--- Gemini's read ---")
    print(answer)

    backboard = BackboardClient(BACKBOARD_API_KEY)
    log_question = f"[{speaker}] surprise={result.surprise_score:.2f} :: {statement}"
    thread_id = backboard.log_exchange(BACKBOARD_THREAD_ID, log_question, answer)
    print(f"\nLogged to Backboard thread: {thread_id}")
    if not BACKBOARD_THREAD_ID:
        print(
            "Tip: set BACKBOARD_THREAD_ID to this value to keep appending "
            "to the same thread on future runs."
        )

    # Now that it's been scored and acted on, fold it into the baseline
    # so future statements are compared against an up-to-date history.
    scorer.add_to_baseline(speaker, statement)


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Kalshi + Gemini + Backboard pipeline")
    parser.add_argument("question", nargs="?", default=None,
                         help="Direct market-lookup question (mode A)")
    parser.add_argument("--ticker", default=MARKET_TICKER,
                         help="Kalshi market ticker to check")
    parser.add_argument("--speaker", default=None,
                         help="Speaker key for surprise scoring (mode B), e.g. kevin_warsh")
    parser.add_argument("--statement", default=None,
                         help="The statement to score/seed for --speaker")
    parser.add_argument("--seed", action="store_true",
                         help="Only add --statement to the baseline; don't score or trade")
    parser.add_argument("--search", action="store_true",
                         help="List live markets/events (optionally filtered by --series) "
                              "and exit, instead of running the pipeline")
    parser.add_argument("--series", default=None,
                         help="Series ticker to filter --search by, e.g. FED, KXWTAOPEN")
    return parser


def search_markets(series_ticker: str, limit: int):
    kalshi = KalshiClient(KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH)
    events = kalshi.list_events(series_ticker=series_ticker, limit=limit)

    event_list = events.get("events", [])
    if len(event_list) == 0:
        print("No events found. Try a different --series, or drop it to browse broadly.")
        return

    for event in event_list:
        event_ticker = event.get("event_ticker", "?")
        title = event.get("title", "(no title)")
        print(f"\nEVENT: {event_ticker} -- {title}")

        markets = event.get("markets", [])
        for market in markets:
            market_ticker = market.get("ticker", "?")
            market_title = market.get("title", "(no title)")
            status = market.get("status", "?")
            print(f"   MARKET: {market_ticker} -- {market_title} [{status}]")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.search:
        search_markets(args.series, limit=50)
        return

    if args.speaker and args.statement:
        handle_statement(args.speaker, args.statement, args.ticker, args.seed)
        return

    # Fall back to mode A: direct market lookup.
    if args.question:
        question = args.question
    else:
        question = f"Summarize the current state of Kalshi market {args.ticker}."

    kalshi = KalshiClient(KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH)
    try:
        market_data = kalshi.get_market(args.ticker)
    except requests.HTTPError as e:
        print(f"Kalshi request failed: {e}\nResponse: {e.response.text}")
        return
    print("Fetched Kalshi market data.")

    prompt = (
        f"{question}\n\n"
        f"Here is the raw Kalshi market data (JSON):\n"
        f"{json.dumps(market_data, indent=2)}\n\n"
        "Give a concise, plain-language summary of what this market is "
        "showing right now."
    )
    answer = ask_gemini(prompt)
    print("\n--- Gemini's analysis ---")
    print(answer)

    backboard = BackboardClient(BACKBOARD_API_KEY)
    thread_id = backboard.log_exchange(BACKBOARD_THREAD_ID, question, answer)
    print(f"\nLogged to Backboard thread: {thread_id}")
    if not BACKBOARD_THREAD_ID:
        print(
            "Tip: set BACKBOARD_THREAD_ID to this value to keep appending "
            "to the same thread on future runs."
        )


if __name__ == "__main__":
    main()