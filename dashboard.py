"""
dashboard.py
============
The demo screen. Streamlit app over the whole pipeline.

  Analyze : paste a statement or upload audio (ElevenLabs Scribe) -> stance
            surprise -> matching Kalshi + Polymarket markets -> trades
  Replay  : answer-by-answer stance chart for a past press conference
  History : trades, plus Tiger Data time series and Solana receipts

Login (Auth0 via Streamlit's built-in st.login):
  Anyone can look. Running the pipeline needs a login, and sending LIVE
  orders additionally needs your email in ALLOWED_TRADERS.
  Config goes in .streamlit/secrets.toml (gitignored), see
  secrets.example.toml. Without an [auth] section the app runs in local mode
  (everything unlocked, dry run only).

  pip install streamlit Authlib
  streamlit run dashboard.py
"""

import glob
import json
import os

os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash-lite")

import pandas as pd
import requests
import streamlit as st

import trading_common as tc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ALLOWED_TRADERS = {e.strip().lower() for e in os.environ.get("ALLOWED_TRADERS", "").split(",") if e.strip()}

st.set_page_config(page_title="Fed Surprise Desk", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ #
# AUTH
# ------------------------------------------------------------------ #

def auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def current_user():
    """(logged_in, email). Local mode counts as logged in, but never live."""
    if not auth_configured():
        return True, None
    return bool(st.user.is_logged_in), (st.user.get("email") if st.user.is_logged_in else None)


def can_trade_live(email) -> bool:
    return bool(email) and email.lower() in ALLOWED_TRADERS


# ------------------------------------------------------------------ #
# ELEVENLABS (file transcription)
# ------------------------------------------------------------------ #

def elevenlabs_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    path = os.path.join(SCRIPT_DIR, "elevenapi.txt")
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
    return key


def transcribe(file_bytes: bytes, filename: str) -> str:
    resp = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": elevenlabs_key()},
        data={"model_id": os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v2")},
        files={"file": (filename, file_bytes)},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("text", "")


# ------------------------------------------------------------------ #
# HELPERS
# ------------------------------------------------------------------ #

def load_trades() -> pd.DataFrame:
    if not os.path.isfile(tc.TRADE_LOG_PATH):
        return pd.DataFrame()
    rows = []
    with open(tc.TRADE_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({"time": r.get("time"), "venue": r.get("venue"), "market": r.get("market"),
                         "side": r.get("side"), "qty": r.get("qty"), "yes_limit": r.get("yes_limit"),
                         "max_cost": r.get("max_cost"), "status": tc.trade_status(r),
                         "reason": r.get("reason")})
    return pd.DataFrame(rows).iloc[::-1]


def show_record(rec: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stance", f"{rec['stance']:+.2f}", help="-1 dovish ... +1 hawkish")
    c2.metric("Speaker's usual", f"{rec['baseline_mean']:+.2f}")
    c3.metric("Surprise (z)", f"{rec['z']:+.1f}σ")
    c4.metric("Read", rec["direction"])
    st.caption(rec.get("summary", ""))
    if rec.get("solana"):
        st.markdown(f"On-chain receipt (Solana {rec['solana']['cluster']}): "
                    f"[{rec['solana']['signature'][:16]}…]({rec['solana']['explorer']})")
    if rec.get("matches"):
        st.subheader("Markets")
        st.dataframe(pd.DataFrame([{
            "venue": m["venue"], "market": m["market"], "title": m["title"], "direction": m["direction"],
            "relevance": m["relevance"], "bid": m["quote"]["best_bid"], "ask": m["quote"]["best_ask"],
            "why": m["reason"]} for m in rec["matches"]]), width="stretch", hide_index=True)
    if rec.get("trades"):
        st.subheader("Orders")
        st.dataframe(pd.DataFrame([{
            "venue": t.get("venue"), "market": t.get("market"), "side": t.get("side"),
            "qty": t.get("qty"), "limit": t.get("yes_limit"), "max $": t.get("max_cost"),
            "status": t.get("error") and f"skipped: {t['error']}" or tc.trade_status(t)}
            for t in rec["trades"]]), width="stretch", hide_index=True)


# ------------------------------------------------------------------ #
# PAGE
# ------------------------------------------------------------------ #

logged_in, email = current_user()

with st.sidebar:
    st.header("Fed Surprise Desk")
    if auth_configured():
        if logged_in:
            st.write(f"Signed in as **{email}**")
            if st.button("Log out"):
                st.logout()
        else:
            st.button("Log in", on_click=st.login, args=("auth0",))
    else:
        st.caption("Local mode (Auth0 not configured).")
    speaker = st.selectbox("Speaker", ["kevin_warsh", "jerome_powell"])
    venues = st.multiselect("Venues", ["kalshi", "polymarket"], default=["kalshi", "polymarket"])
    qty = st.number_input("Contracts per order", 1, tc.MAX_CONTRACTS_PER_ORDER, 2)
    live = False
    if can_trade_live(email):
        live = st.toggle("LIVE orders (real money)", value=False)
    st.caption(f"Limits: ${tc.MAX_DOLLARS_PER_ORDER:.0f}/order, ${tc.MAX_DOLLARS_PER_DAY:.0f}/day "
               f"across venues. Kill switch: STOP_TRADING file.")

tab_desk, tab_analyze, tab_replay, tab_history = st.tabs(["Ask the desk", "Analyze", "Replay", "History"])

with tab_desk:
    import backboard_client as bb
    st.subheader("Ask the desk")
    st.caption("Answers come from Backboard memory: every surprise signal and the orders it triggered.")
    if not bb.enabled():
        st.info("Add a Backboard key (backboardapi.txt or BACKBOARD_API_KEY), then run "
                "`python backboard_client.py --setup`.")
    else:
        left, right = st.columns([3, 2])
        with left:
            history = st.session_state.setdefault("desk_chat", [])
            for turn in history:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["text"])
                    if turn.get("note"):
                        st.caption(turn["note"])
            examples = ["When was Warsh most hawkish, and what did we buy?",
                        "Which markets have we traded after hawkish surprises?",
                        "What is Warsh's usual stance?"]
            pick = st.pills("Try", examples, key="desk_pick") if hasattr(st, "pills") else None
            question = st.chat_input("Ask about past signals and trades") or pick
            if question and question != st.session_state.get("desk_last"):
                st.session_state["desk_last"] = question
                history.append({"role": "user", "text": question})
                with st.spinner("Searching memory..."):
                    try:
                        r = bb.ask(question)
                        history.append({"role": "assistant", "text": r["answer"] or "(no answer)",
                                        "note": f"{len(r['memories'])} memories used · {r['model']}"})
                    except Exception as e:
                        history.append({"role": "assistant", "text": f"Backboard error: {e}"})
                st.rerun()
        with right:
            st.markdown("**What the desk remembers**")
            query = st.text_input("Search memories", placeholder="e.g. rate hike, Polymarket, Warsh")
            try:
                mems = bb.search_memories(query, limit=15) if query else bb.list_memories()
            except Exception as e:
                mems = []
                st.warning(f"Backboard unavailable: {e}")
            signals = [m for m in mems if (m.get("metadata") or {}).get("kind") == "signal"]
            c1, c2 = st.columns(2)
            c1.metric("Memories", len(mems))
            c2.metric("Signals", len(signals))
            for m in mems[:30]:
                meta = m.get("metadata") or {}
                tag = meta.get("direction") or meta.get("kind", "")
                with st.container(border=True):
                    st.caption(f"{tag} · {str(m.get('created_at', ''))[:16]}")
                    st.write(m.get("content", ""))
            runs = bb.runs()
            if runs:
                st.markdown("**Run logs (Backboard threads)**")
                run = st.selectbox("Run", runs, label_visibility="collapsed")
                for msg in bb.run_log(run):
                    st.text(msg.get("content", ""))

with tab_analyze:
    st.subheader("How surprising is this, for this speaker?")
    if not logged_in:
        st.info("Log in to run the analysis. Replays and history are open to everyone.")
    else:
        audio = st.file_uploader("Audio clip (optional, transcribed with ElevenLabs Scribe)",
                                 type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"])
        if audio and st.button("Transcribe"):
            with st.spinner("ElevenLabs Scribe..."):
                st.session_state["statement"] = transcribe(audio.getvalue(), audio.name)
        statement = st.text_area("Statement", key="statement", height=120,
                                 placeholder="Paste a quote from a press conference, speech or post")
        c1, c2 = st.columns(2)
        stance_only = c2.checkbox("Stance only (skip markets)")
        if c1.button("Analyze", type="primary", disabled=not statement.strip()):
            import pipeline
            import stance_scorer
            with st.spinner("Scoring stance with Gemini..."):
                r = stance_scorer.score_statement(speaker, statement)
            with st.spinner("Finding markets and pricing orders (first Kalshi search can take ~2 min)..."):
                rec = pipeline.act_on(r, speaker, venues, live, int(qty), stance_only, source="dashboard")
            pipeline.save(f"dashboard_{pd.Timestamp.utcnow().value // 10**9}", [rec])
            st.session_state["last"] = rec
        if st.session_state.get("last"):
            show_record(st.session_state["last"])

with tab_replay:
    st.subheader("Press conference replay")
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "replay_*.json")), reverse=True)
    if not files:
        st.info("No replays yet. Run: python pipeline.py --replay 20260916")
    else:
        choice = st.selectbox("Run", files, format_func=os.path.basename)
        with open(choice, encoding="utf-8") as f:
            records = json.load(f)["records"]
        df = pd.DataFrame([{"answer": i + 1, "z": r["z"], "stance": r["stance"],
                            "direction": r["direction"], "summary": r["summary"]}
                           for i, r in enumerate(records)])
        st.bar_chart(df.set_index("answer")["z"])
        st.caption("Bars are z-scores vs. the speaker's own history. |z| ≥ 2 is flagged as a surprise.")
        st.dataframe(df, width="stretch", hide_index=True)
        for i, r in enumerate(records):
            if r["direction"] != "IN LINE" and (r.get("matches") or r.get("solana")):
                with st.expander(f"Answer {i + 1}: {r['direction']} z={r['z']:+.1f}"):
                    st.write(r["statement"])
                    show_record(r)

with tab_history:
    import tiger_store
    if tiger_store.enabled():
        st.subheader("Signals (Tiger Data)")
        try:
            sig = pd.DataFrame(tiger_store.query(
                "SELECT time, speaker, direction, z, summary, solana_sig FROM signals "
                "ORDER BY time DESC LIMIT 200"))
            if not sig.empty:
                st.line_chart(sig.set_index("time")["z"])
                st.dataframe(sig, width="stretch", hide_index=True)
            ticks = pd.DataFrame(tiger_store.query(
                "SELECT time_bucket('1 minute', time) AS minute, market, avg((bid + ask) / 2) AS mid "
                "FROM ticks WHERE bid IS NOT NULL AND ask IS NOT NULL "
                "GROUP BY minute, market ORDER BY minute"))
            if not ticks.empty:
                st.subheader("Market mid prices (1-minute buckets)")
                st.line_chart(ticks.pivot(index="minute", columns="market", values="mid"))
        except Exception as e:
            st.warning(f"Tiger Data unavailable: {e}")
    st.subheader("Orders (trades.jsonl)")
    trades = load_trades()
    if trades.empty:
        st.info("No orders yet.")
    else:
        st.dataframe(trades, width="stretch", hide_index=True)
