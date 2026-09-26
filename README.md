# [Project Name]

**[Project Name]** transcribes speeches and posts from market-moving figures (e.g. Fed Chair Kevin Warsh, Donald Trump) in real time and scores each statement for **surprise relative to that speaker's own history**, not generic sentiment. It flags when someone sounds unusually hawkish, dovish, or off-script *for them*, maps the statement to the relevant contracts on **Kalshi, Polymarket US, and other regulated prediction markets**, and can place trades automatically within set risk limits. Every prediction is logged against the actual outcome, so the system builds a public, calibrated track record.

## Why it exists

Markets move on surprises, not on the tone of the words themselves. Research has long measured policy surprises against *market expectations* (Gürkaynak, Sack & Swanson, 2005), and Chicago Fed work shows that how closely Fed officials' messages line up with each other changes how markets react.

Existing tools score sentiment in general terms: Bloomberg and news analytics say "this is negative," and live tools like fedspeech label a line hawkish or dovish. None of them ask whether a statement is unusual **for this particular speaker**, and none connect that question to tradeable markets with a public track record.

We test the hypothesis that speaker-relative surprise predicts market moves, live and end to end, and we let the logged results show whether it works.

## How it works

1. **Ingest:** live audio (speeches, press conferences) streamed through ElevenLabs Scribe v2 Realtime, which returns word-level text and speaker labels, plus posts from speakers' official accounts.
2. **Baseline:** each speaker has a profile, stored in Backboard, built from their past transcripts and posts: usual stance, topics, and wording.
3. **Score:** Gemini rates each new statement, and the system measures how far it departs from that speaker's baseline (e.g. "baseline hawkishness 0.62 across 8 press conferences; this answer scored 0.91, a 2.3σ surprise").
4. **Map:** high-surprise statements are matched to the related contracts (rate decisions, mention markets, policy outcomes) across supported exchanges. Gemini checks each market's resolution rules to confirm a statement qualifies.
5. **Trade:** orders go out automatically when the model's price differs from the market price by more than fees, subject to risk limits.
6. **Log:** every signal, trade, and outcome is recorded to measure accuracy and profit over time, and each result updates the speaker's baseline.

## Tech stack

- **Gemini API**: statement scoring (stance, topics, market-rule matching) and plain-English trade rationales
- **Backboard.io**: persistent memory for each speaker's historical baseline and the running log of signals and outcomes
- **Kalshi API** and **Polymarket US API**: live market data, automated order execution, and settlement results
- **ElevenLabs Scribe v2 Realtime**: live speech-to-text (~150 ms) with speaker labels, so only the speaker being tracked is scored (not reporters' questions)
- **Federal Reserve caption files**: official, timed, speaker-labeled transcripts used for the replay demo and to build speaker baselines

## Repo map

**Run these**

| File | What it does |
|---|---|
| `pipeline.py` | The whole system: statement or replay -> stance surprise -> markets -> trades -> logs |
| `dashboard.py` | Streamlit demo screen (`streamlit run dashboard.py`) |

**Pieces the pipeline uses**

| File | What it does |
|---|---|
| `stance_scorer.py` | Gemini scores each answer -1 (dovish) to +1 (hawkish); surprise = z-score vs. that speaker's past answers |
| `fed_transcripts.py` | Downloads Fed press conference PDFs and splits them into Chair answers vs. reporter questions |
| `market_router.py` | Finds related markets on Kalshi and Polymarket and trades them the same way |
| `kalshi_ticker_finder.py` | Searches all ~130k live Kalshi markets for ones related to a statement |
| `kalshi_trader.py` | Kalshi prices and orders |
| `polymarket_client.py` | Polymarket US search, prices and orders |
| `trading_common.py` | Shared risk limits, kill switch, fees and trade log for both venues |
| `backboard_client.py` | Logs every signal and trade to a Backboard thread |
| `tiger_store.py` | Stores signals, prices and trades as Tiger Data time series |
| `solana_proof.py` | Writes a hash of each signal to Solana, a public timestamp that proves when we made the call |
| `speechtxt.py` | ElevenLabs speech-to-text |

**Data and config:** `transcripts/` (parsed press conferences), `stance_baselines.json` (each speaker's usual stance), `Dockerfile` and `deploy/` (DigitalOcean), `.streamlit/secrets.example.toml` (Auth0).

**Work in progress:** `kalshi_ticker2.py` and `run_kalshi_ticker2.py` (next version of the Kalshi market search).

## Quick start

```
pip install -r requirements.txt
python pipeline.py --replay 20260916 --venues polymarket     # replay the Sept 16 press conference (dry run)
python pipeline.py --statement "..."                          # score one statement
streamlit run dashboard.py
```

Keys go in gitignored files next to the code (`gemapi.txt`, `polymarketkey.txt`, `polymarketsecret.txt`, `kalshikey.txt` + `privkey.txt`, `elevenapi.txt`, `backboardapi.txt`, `tigerdb.txt`) or in environment variables. Orders are dry runs unless you pass `--live`.

## Supported venues

- **Kalshi** (live, plus demo environment for testing)
- **Polymarket US**
- More regulated venues planned

## Risk controls

- Per-trade size cap and daily loss limit
- Manual kill switch
- Log-only mode that records signals without trading
- API keys stored in environment variables and never committed

## Demo

We replay the **September 16, 2026 FOMC press conference** with Kalshi's real trade history synced to the transcript, showing each surprise signal, the trade it would have placed, and how the market moved. The same pipeline runs live for the next press conference on **October 29, 2026**.

## Roadmap

- **Crowd sentiment:** measure how X, Reddit, StockTwits, and news react to each statement, and compare the speaker's surprise with the crowd's reaction.
- **More context sources:** official press releases, interviews, and hearing testimony, to build richer speaker baselines.
- **More speakers and events:** other Fed officials, Treasury, and CEOs on earnings calls.
- **Public track record page:** live accuracy and calibration charts.

## Team

- Om Patel
- Dylan Ball

Research, data analysis, and drafting assisted by Claude (Anthropic).

Built at &hacks XII, William & Mary, September 2026.
