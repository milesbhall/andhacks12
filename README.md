# MarketPulse

**Live site:** https://yellow-boar-756344.hostingersite.com · Built at &hacks XII, William & Mary (Sept 2026)

MarketPulse listens to Fed speakers (and the President) live, scores each passage's
policy stance **against that speaker's own history**, and when someone breaks character
(|z| ≥ 2) it can act on the next FOMC decision on **Kalshi** and **Polymarket**. End-to-end
latency depends on transcription, model scoring, market data, and order submission; the live
dashboard displays the measured time for each signal. "Inflation is too high" is routine for a hawk and a shock from a dove; a generic
sentiment model can't tell the difference. A personal baseline can.

## How it works

```
 audio (mic / upload / YouTube / Fed livestream / replay)
   │  ElevenLabs Scribe v2 Realtime
   ▼
 transcript ──► Gemini scores −1 dovish … +1 hawkish (with prior context + FRED macro data)
   │            z = (stance − speaker mean) / speaker sd
   ▼
 |z| ≥ 2 ? ──► Kalshi + Polymarket candidates refreshed from the speaker's recent topic
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
| Trade | Market recommenders follow recent speaker words. Dry run, Kalshi demo, or guarded LIVE execution. |
| Prove | Every surprise is hashed onto Solana devnet, stored in Tiger Data, and remembered by Backboard ("Ask the desk"). |

## The website

The Hostinger page starts with sign-in and local key setup. An approved Auth0 operator or the operator password can open the desk; only approved Auth0 operators can access private order controls. A small worker on the team laptop (`control_worker.py`) polls the
site over HTTPS and runs the Python pipeline locally.

- **Control room:** start/stop a session from the mic, a recording, a stream URL, or a replay; choose speaker, mode (dry / Kalshi demo / LIVE), contracts, venues, playback speed.
- **Live:** z-score chart, transcript, "Hearing:" line for the mic, order cards, the markets ready to trade, live recommenders.
- **Orders:** private exchange open orders, holdings, recent manual attempts, and Buy/Sell tickets. A ticket checks a fresh quote and risk limits, then requires a separate confirmation. A submission is not a confirmed fill. Sell requires verified holdings.
- **Ask the desk:** questions answered from Backboard memory of every signal and order.
- **Analyze:** score any statement and see which markets it would move. This tool always runs dry and never sends orders.
- **Crowd & macro:** Bluesky + Reddit sentiment and the FRED backdrop.
- **Replay:** 17 press conferences and speeches (Warsh, Powell, presidential remarks) scored answer by answer.
- **History:** every signal (Tiger Data) and order.

The Live page labels the desk update, last speech update, and market update separately. It shows whether audio is waiting, listening, stalled, or ended. Markets in view use the latest available speaker topic and disappear when their source is stale or a session ends. The hosted page and worker share **one team trading account**; independent user keys require separate deployments and workers.

There are two replay paths. **Control room replay** times a saved transcript through the live pipeline, refreshing market recommendations and producing dry/demo/LIVE order attempts according to the chosen mode. **Replay tab** shows previously computed, answer-by-answer stance scores for inspection and sends no orders.

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

Hosted controls and Auth0 setup are in [docs/HOSTED_CONTROLS.md](docs/HOSTED_CONTROLS.md).
Real-money mode requires all three gates: an approved Auth0 identity with LIVE access,
typing `LIVE` when selecting the mode, and starting the local worker with
`MARKETPULSE_ALLOW_LIVE=1`. The default worker rejects LIVE requests; dry run and
Kalshi demo remain available. A locally created `STOP_TRADING` file still stops order
execution. Use dry run for judging unless the team deliberately enables real money.
Manual Buy/Sell additionally requires a private exchange snapshot, a 90-second, single-use preview bound to the Auth0 session, a fresh limit and position check, and an explicit final confirmation. The worker records a request before its one send attempt and never retries an ambiguous exchange response automatically. The same $10/order and $50/day caps apply to automated and manual LIVE orders. Polymarket Sell fails closed when its account response does not verify the exact held side and quantity.

## Repo map

| File | Role |
|---|---|
| `live.py` | Live desk: chunk transcript, score, decide, trade, persist |
| `stance_scorer.py` | Gemini stance rubric, speaker baselines, z-scores |
| `speechtxt.py`, `mic.py` | ElevenLabs realtime transcription (streams, files, YouTube, mic) |
| `market_router.py`, `kalshi_trader.py`, `polymarket_client.py`, `trading_common.py` | Market search, pricing, orders, risk limits, modes |
| `manual_orders.py` | Private account snapshots, short-lived manual previews, guarded Buy/Sell execution |
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
requires an approved Auth0 identity, typing LIVE, and local worker opt-in. Real money is never the default.

## Full stack

| Layer | Technology and role |
|---|---|
| Front end and hosting | Logo-led HTML, CSS, JavaScript dashboard on Hostinger; PHP endpoints for authentication, control queue, private order jobs, and publishing |
| Identity | Auth0 Authorization Code flow for allowlisted operators; password fallback for non-LIVE controls |
| Local runtime | Python worker on the operator laptop, outbound HTTPS polling, subprocess supervision, replay and audio handling |
| Speech and scoring | ElevenLabs Scribe v2 Realtime transcription; Gemini stance scoring with FRED macro context and speaker baselines |
| Markets and execution | Kalshi and Polymarket market data/order clients, topic-aware recommendations, dry/demo/LIVE modes, private manual order flow, shared limits and a local kill switch |
| Context and history | FRED, Bluesky, Reddit via SocialCrawl, Tiger Data (TimescaleDB), Backboard memory |
| Proof | Solana devnet memo receipt when a funded wallet is configured |

## Sponsors used

Gemini · ElevenLabs · Solana · Tiger Data · Backboard · Auth0 · Hostinger · Kalshi · Polymarket · FRED · Bluesky · Reddit via SocialCrawl

## Team

Om Patel · Dylan Ball. Built with help from Claude (Anthropic) and Codex.

Not financial advice.
