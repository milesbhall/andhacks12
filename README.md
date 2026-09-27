# MarketPulse

**Live site:** https://yellow-boar-756344.hostingersite.com · Built at &hacks XII, William & Mary (Sept 2026)

MarketPulse listens to Fed speakers (and the President) live, scores each passage's
policy stance **against that speaker's own history**, and when someone breaks character
(|z| ≥ 2) it trades the next FOMC decision on **Kalshi** and **Polymarket** within about
a second. "Inflation is too high" is routine for a hawk and a shock from a dove; a generic
sentiment model can't tell the difference. A personal baseline can.

## How it works

```
 audio (mic / upload / YouTube / Fed livestream / replay)
   │  ElevenLabs Scribe v2 Realtime
   ▼
 transcript ──► Gemini scores −1 dovish … +1 hawkish (with prior context + FRED macro data)
   │            z = (stance − speaker mean) / speaker sd
   ▼
 |z| ≥ 2 ? ──► pre-picked Kalshi + Polymarket markets (+ Dylan's live recommenders)
   │            risk limits: $10/order, $50/day, slippage cap, kill switch, flip guard
   ▼
 orders ──► Solana memo receipt · Tiger Data (TimescaleDB) · Backboard memory
   ▼
 publish.py ──► Hostinger page (live chart, transcript, orders, crowd + macro)
```

| Step | What happens |
|---|---|
| Listen | ElevenLabs realtime speech-to-text from the laptop mic, an uploaded recording (1×–20×), a YouTube or federalreserve.gov stream, or a saved transcript replay. |
| Context | **FRED** (fed funds, CPI, core PCE, unemployment, payrolls, 2y/10y, breakevens) goes into the scoring prompt. **Bluesky + Reddit** (via SocialCrawl) posts are scored on the same scale so each surprise is compared with what the crowd expects. |
| Score | Gemini rates each passage and labels who is talking (reporter questions are ignored). |
| Decide | z-score vs. the speaker's baseline, **adjusted with FRED** (+0.25 per point of extra core PCE inflation, −0.15 per point of extra unemployment since the baseline was recorded): Warsh (46 answers, 2 press conferences), Powell (123 answers, 4 press conferences), President Trump (5 economy speeches). |
| Trade | Markets are chosen before the speech, so no search delay. Dry run, Kalshi demo, or LIVE (typed confirmation). |
| Prove | Every surprise is hashed onto Solana devnet, stored in Tiger Data, and remembered by Backboard ("Ask the desk"). |

## The website

The Hostinger page is the whole product; no Streamlit needed. Anyone can watch; the
operator signs in to run it. A small worker on the laptop (`control_worker.py`) polls the
site over HTTPS and runs the Python pipeline locally.

- **Control room:** start/stop a session from the mic, a recording, a stream URL, or a replay; choose speaker, mode (dry / Kalshi demo / LIVE), contracts, venues, playback speed.
- **Live:** z-score chart, transcript, "Hearing:" line for the mic, order cards, the markets ready to trade, live recommenders.
- **Ask the desk:** questions answered from Backboard memory of every signal and order.
- **Analyze:** score any statement and see which markets it would move.
- **Crowd & macro:** Bluesky + Reddit sentiment and the FRED backdrop.
- **Replay:** 17 press conferences and speeches (Warsh, Powell, presidential remarks) scored answer by answer.
- **History:** every signal (Tiger Data) and order.

## Run it

```bash
pip install -r requirements.txt
# keys go in gitignored files: gemapi.txt, elevenapi.txt, kalshikey.txt + privkey.txt,
# polymarketkey.txt + polymarketsecret.txt, backboardapi.txt, tigerdb.txt, fredapi.txt,
# socialcrawlapi.txt (Reddit), hostinger_url.txt + hostinger_token.txt

python control_worker.py                         # then use the website's control room

# or from the command line
python live.py --simulate 20260916 --speed 10     # replay a press conference, dry run
python mic.py & python live.py --fast --source mic --surprises-only
python publish.py                                # push live state to the website
```

Data tools: `fed_transcripts.py` (FOMC press conference PDFs), `fed_speeches.py`
(Fed speeches + `--president` remarks from the American Presidency Project),
`stance_scorer.py --seed` (build a speaker baseline), `build_replays.py`,
`fred_client.py`, `social_sentiment.py`.

## Repo map

| File | Role |
|---|---|
| `live.py` | Live desk: chunk transcript, score, decide, trade, persist |
| `stance_scorer.py` | Gemini stance rubric, speaker baselines, z-scores |
| `speechtxt.py`, `mic.py` | ElevenLabs realtime transcription (streams, files, YouTube, mic) |
| `market_router.py`, `kalshi_trader.py`, `polymarket_client.py`, `trading_common.py` | Market search, pricing, orders, risk limits, modes |
| `kalshi_ticker2.py`, `run_kalshi_ticker2.py`, `run_polymarket.py`, `recommendation_schema.py` | Dylan's baseline-aware market recommenders |
| `fred_client.py`, `social_sentiment.py` | FRED macro backdrop; Bluesky + Reddit crowd sentiment |
| `solana_proof.py`, `tiger_store.py`, `backboard_client.py` | Receipts, time-series storage, memory / Ask the desk |
| `control_worker.py`, `publish.py`, `web/` | Hosted site: PHP control endpoints, worker, publisher, page |
| `pipeline.py`, `dashboard.py` | One-shot analysis / replays; the original local Streamlit dashboard |
| `transcripts/`, `stance_baselines.json` | Replay transcripts and speaker baselines |

## Risk controls

$10 per order, $50 per day across venues (demo orders excluded), max 2¢ slippage,
`STOP_TRADING` kill switch file, no adding to a market already traded in a session, and
an opposite-direction trade needs two surprises in a row. LIVE mode on the website
requires the operator password and typing LIVE. Real money is never the default.

## Sponsors used

Gemini · ElevenLabs · Solana · Tiger Data · Backboard · Auth0 (local dashboard) · Hostinger · Kalshi · Polymarket · FRED · Bluesky · Reddit via SocialCrawl

## Team

Om Patel · Dylan Ball · Miles Hall. Built with help from Claude (Anthropic) and Codex.

Not financial advice.
