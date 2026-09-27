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
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash-lite")

import pandas as pd
import requests
import streamlit as st

import trading_common as tc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ALLOWED_TRADERS = {e.strip().lower() for e in os.environ.get("ALLOWED_TRADERS", "").split(",") if e.strip()}

st.set_page_config(page_title="MarketPulse", page_icon="📈", layout="wide")


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
    allowed = set(ALLOWED_TRADERS)
    try:
        allowed |= {e.strip().lower() for e in st.secrets.get("allowed_traders", [])}
    except Exception:
        pass
    return bool(email) and email.lower() in allowed


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


def _clean_title(title: str, market: str) -> str:
    title = (title or "").replace("?", "").strip()
    if " -- " in title:
        event, outcome = title.split(" -- ", 1)
        title = f"{outcome.strip()} ({event.strip()})"
    return title or market


def render_trade_cards(trades: list, titles: dict = None, limit: int = 8):
    """Orders as readable cards; priced-in / skipped ones folded into one line."""
    titles = titles or {}
    placed = [t for t in trades if not (t.get("skipped") or t.get("error") or t.get("blocked"))]
    skipped = [t for t in trades if t not in placed]
    if not placed:
        st.caption("No orders placed yet.")
    for t in list(reversed(placed))[:limit]:
        side = str(t.get("side", "")).upper()
        yes_limit = t.get("yes_limit") or 0
        per = yes_limit if side == "YES" else 1 - yes_limit
        qty = t.get("qty") or 0
        status = t.get("status") or tc.trade_status(t)
        badge = {"SENT": ":green-badge[SENT: real money]", "SENT (demo)": ":blue-badge[SENT: Kalshi demo]",
                 "DRY RUN": ":gray-badge[DRY RUN]"}.get(status, f":red-badge[{status[:40]}]")
        title = _clean_title(t.get("title") or titles.get((t.get("venue"), t.get("market")), ""), t.get("market", ""))
        with st.container(border=True):
            st.markdown(f"**BUY {side}** · {title}  \n"
                        f"{str(t.get('venue', '')).capitalize()} · {qty:g} contracts at up to {per * 100:.0f}¢ "
                        f"· costs at most ${t.get('max_cost', 0):.2f} · pays ${qty:g} if right  \n"
                        f"{badge}" + (f" · triggered by {t['trigger']}" if t.get("trigger") else ""))
    if skipped:
        names = ", ".join(_clean_title(t.get("title") or titles.get((t.get("venue"), t.get("market")), ""),
                                       t.get("market", ""))[:40] for t in skipped[-4:])
        st.caption(f"Skipped {len(skipped)} (already priced in or blocked): {names}"
                   + ("…" if len(skipped) > 4 else ""))


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
        render_trade_cards(rec["trades"], {(m["venue"], m["market"]): m.get("title", "")
                                           for m in rec.get("matches", [])})


# ------------------------------------------------------------------ #
# PAGE
# ------------------------------------------------------------------ #

logged_in, email = current_user()

with st.sidebar:
    st.header("MarketPulse")
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
    mode_labels = {"dry": "Dry run", "demo": "Kalshi demo (fake money)", "live": "LIVE (real money)"}
    modes = ["dry", "demo", "live"] if can_trade_live(email) else ["dry", "demo"]
    live = st.radio("Trading mode", modes, format_func=mode_labels.get, key="trading_mode",
                    help="Demo sends Kalshi orders to demo.kalshi.co (needs kalshikey_demo.txt + "
                         "privkey_demo.txt); Polymarket has no demo, so it stays dry. "
                         "LIVE is only offered to signed-in allowlisted accounts.")
    if live == "live":
        st.warning("Real orders. Limits still apply; create a STOP_TRADING file to halt.")
    st.caption(f"Limits: ${tc.MAX_DOLLARS_PER_ORDER:.0f}/order, ${tc.MAX_DOLLARS_PER_DAY:.0f}/day "
               f"across venues. Kill switch: STOP_TRADING file.")

tab_live, tab_desk, tab_analyze, tab_replay, tab_history = st.tabs(["Live", "Ask the desk", "Analyze", "Replay", "History"])

LIVE_STATE_PATH = os.path.join(SCRIPT_DIR, "live_state.json")


@st.fragment(run_every=1.0)
def live_panel():
    if not os.path.isfile(LIVE_STATE_PATH):
        st.info("Nothing running. Choose a source above and press Start, or run "
                "`python live.py --simulate 20260916 --speed 10` for a saved demo.")
        return
    try:
        with open(LIVE_STATE_PATH, encoding="utf-8") as f:
            s = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    base = s["baseline"]
    spoken = [c for c in s["chunks"] if c.get("z") is not None]
    last = spoken[-1] if spoken else None
    alerts = s.get("alerts", [])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", s["status"].split(" (")[0].capitalize())
    c2.metric("Latest stance", f"{last['stance']:+.2f}" if last else "—",
              delta=f"z {last['z']:+.1f}" if last else None, delta_color="off")
    c3.metric("Surprises", len(alerts))
    c4.metric("Signal → order", f"{alerts[-1]['latency_ms'] / 1000:.1f}s" if alerts else "—")
    st.caption(f"{s['speaker']} · source: {s['source']} · usual stance {base['mean']:+.2f} "
               f"(spread {base['stdev']:.2f}) · { {'live': 'LIVE ORDERS', 'demo': 'Kalshi demo orders'}.get(s.get('mode') or ('live' if s.get('live_orders') else 'dry'), 'dry run') }")

    if alerts:
        a = alerts[-1]
        (st.error if a["direction"] == "HAWKISH" else st.success)(
            f"**{a['direction']} surprise** (z {a['z']:+.1f}): {a['summary']} · "
            f"{a['orders']} order(s) in {a['latency_ms'] / 1000:.1f}s")

    left, right = st.columns([3, 2])
    with left:
        if spoken:
            start = pd.Timestamp(s["chunks"][0]["time"])
            minutes = [round((pd.Timestamp(c["time"]) - start).total_seconds() / 60, 2) for c in spoken]
            chart = pd.DataFrame({"minutes into speech": minutes,
                                  "z-score": [c["z"] for c in spoken],
                                  "hawkish line (+2)": [2.0] * len(spoken),
                                  "dovish line (-2)": [-2.0] * len(spoken)}).set_index("minutes into speech")
            st.line_chart(chart, height=220, x_label="minutes into speech", y_label="z-score")
        st.markdown("**Transcript**")
        for c in reversed(s["chunks"][-12:]):
            if c["role"] != "speaker":
                st.caption(f"Question: {c['text'][:220]}")
                continue
            tag = c["direction"] if c["direction"] in ("HAWKISH", "DOVISH") else "in line"
            st.markdown(f"`{c['stance']:+.2f} · z {c['z']:+.1f} · {tag}`  {c['text'][:300]}")
    with right:
        st.markdown("**Orders**")
        watch_titles = {(m["venue"], m["market"]): m.get("title", "")
                        for ms in s["watchlist"].values() for m in ms}
        render_trade_cards(s["trades"], watch_titles, limit=6)
        with st.expander("Ready to trade (picked before the speech)"):
            for d, ms in s["watchlist"].items():
                st.markdown(f"**If {d.lower()}:**")
                for m in ms[:6]:
                    st.caption(f"BUY {m['side'].upper()} · {_clean_title(m.get('title', ''), m['market'])} "
                               f"({m['venue'].capitalize()})")


TRANSCRIPT_LIVE_PATH = os.path.join(SCRIPT_DIR, "live_transcript.json")
PARTIAL_LIVE_PATH = os.path.join(SCRIPT_DIR, "live_partial.json")
DEMO_PIDS_PATH = os.path.join(SCRIPT_DIR, "mic_demo_pids.json")
DEMO_SOURCE_PATH = os.path.join(SCRIPT_DIR, "audio_source_session.json")


def _source_session() -> dict:
    try:
        with open(DEMO_SOURCE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _log_tail(path: str) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "No log output yet."
    return next((line[:350] for line in reversed(lines) if line.strip()), "No log output yet.")


def _demo_running() -> dict:
    """PIDs of the dashboard audio processes that are still alive."""
    import subprocess
    try:
        with open(DEMO_PIDS_PATH, encoding="utf-8") as f:
            pids = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    alive = {}
    for name, pid in pids.items():
        try:
            if os.name == "nt":
                out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
                if str(pid) in out:
                    alive[name] = pid
            else:
                os.kill(pid, 0)
                alive[name] = pid
        except Exception:
            pass
    return alive


def _start_demo(mode: str, venues: list, speaker: str, qty: int, source_kind: str,
                source_value: str = "", source_label: str = "", recommenders: bool = False):
    import subprocess
    import sys
    if source_kind not in {"mic", "browser", "file", "url"}:
        raise ValueError(f"Unknown audio source: {source_kind}")
    if source_kind == "url" and urlparse(source_value).scheme not in {"http", "https"}:
        raise ValueError("Enter an http:// or https:// stream URL.")
    if source_kind == "file" and not Path(source_value).is_file():
        raise ValueError("Choose an audio file before starting.")
    with open(DEMO_SOURCE_PATH, "w", encoding="utf-8") as f:
        json.dump({"kind": source_kind,
                   "uploaded_path": source_value if source_kind == "file" else ""}, f)

    # The desk must start from an empty transcript; the source begins once its
    # watchlist is ready so a recording is scored at the pace it is played.
    with open(TRANSCRIPT_LIVE_PATH, "w", encoding="utf-8") as f:
        json.dump({"source": source_kind, "segments": []}, f)
    try:
        os.remove(LIVE_STATE_PATH)
    except FileNotFoundError:
        pass
    for stale_recommendations in ("live_recommendations.json", "live_polymarket_recommendations.json"):
        try:
            os.remove(os.path.join(SCRIPT_DIR, stale_recommendations))
        except FileNotFoundError:
            pass
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    pids = {}

    def launch(name: str, command: list[str], log_name: str):
        with open(os.path.join(SCRIPT_DIR, log_name), "w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=SCRIPT_DIR, creationflags=flags,
                                       stdout=log, stderr=subprocess.STDOUT)
        pids[name] = process.pid
        with open(DEMO_PIDS_PATH, "w", encoding="utf-8") as f:
            json.dump(pids, f)
        return process

    try:
        desk_command = [sys.executable, "live.py", "--speaker", speaker, "--fast",
                        "--mode", mode, "--qty", str(qty), "--venues", *venues,
                        "--source", "mic" if source_kind in {"mic", "browser"} else "live",
                        "--source-label", source_label or source_kind]
        if source_kind in {"mic", "browser"}:
            desk_command.append("--surprises-only")
        desk = launch("desk", desk_command, "live.log")
        if recommenders:
            for name, script in (("kalshi recommender", "run_kalshi_ticker2.py"),
                                 ("polymarket recommender", "run_polymarket.py")):
                launch(name, [sys.executable, script, "--watch", "--speaker", speaker],
                       script.replace(".py", ".log"))
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if desk.poll() is not None:
                raise RuntimeError("Live desk stopped: " + _log_tail(os.path.join(SCRIPT_DIR, "live.log")))
            try:
                with open(LIVE_STATE_PATH, encoding="utf-8") as f:
                    ready = json.load(f).get("status") == "listening"
            except (OSError, json.JSONDecodeError):
                ready = False
            if ready:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("Live desk did not become ready within 45 seconds. Check live.log.")
        if source_kind != "browser":
            if source_kind == "mic":
                launch("mic", [sys.executable, "mic.py"], "mic.log")
            else:
                source_flag = "--realtime-file" if source_kind == "file" else "--url"
                launch("audio source", [sys.executable, "speechtxt.py", source_flag, source_value],
                       "speechtxt.log")
        if all(os.path.isfile(os.path.join(SCRIPT_DIR, name))
               for name in ("hostinger_url.txt", "hostinger_token.txt")):
            launch("website updates", [sys.executable, "publish.py"], "publish.log")
    except Exception:
        _stop_demo()
        raise


def _stop_demo():
    import signal
    import subprocess
    for pid in _demo_running().values():
        try:
            if os.name == "nt":   # kill the whole process tree so the mic really stops
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    for path, empty in ((TRANSCRIPT_LIVE_PATH, {"segments": []}), (PARTIAL_LIVE_PATH, {"text": ""})):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(empty, f)
        except OSError:
            pass
    try:
        os.remove(DEMO_PIDS_PATH)
    except OSError:
        pass
    uploaded = _source_session().get("uploaded_path")
    if uploaded:
        path = Path(uploaded)
        try:
            path.unlink(missing_ok=True)
            path.parent.rmdir()
        except OSError:
            pass
    try:
        os.remove(DEMO_SOURCE_PATH)
    except OSError:
        pass


def _save_uploaded_audio(uploaded) -> str:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".wav", ".webm"}:
        raise ValueError("Unsupported audio file type.")
    folder = Path(tempfile.mkdtemp(prefix="andhacks12_audio_"))
    path = folder / ("recording" + suffix)
    path.write_bytes(uploaded.getvalue())
    return str(path)


def _append_spoken(text: str):
    try:
        with open(TRANSCRIPT_LIVE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {"source": "browser microphone", "segments": []}
    data.setdefault("segments", []).append({"speaker": "microphone", "role": "speaker", "text": text})
    with open(TRANSCRIPT_LIVE_PATH + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(TRANSCRIPT_LIVE_PATH + ".tmp", TRANSCRIPT_LIVE_PATH)


@st.fragment(run_every=0.5)
def hearing_now():
    try:
        with open(PARTIAL_LIVE_PATH, encoding="utf-8") as f:
            text = json.load(f).get("text", "")
    except (OSError, json.JSONDecodeError):
        text = ""
    if text and _demo_running().get("mic"):
        st.markdown(f"**Hearing:** _{text}_")


@st.fragment(run_every=2.0)
def audio_source_status():
    session = _source_session()
    source_kind = session.get("kind")
    process_name = "mic" if source_kind == "mic" else "audio source"
    if source_kind not in {"mic", "file", "url"} or process_name in _demo_running():
        return
    log_name = "mic.log" if source_kind == "mic" else "speechtxt.log"
    detail = _log_tail(os.path.join(SCRIPT_DIR, log_name))
    if source_kind in {"file", "url"} and detail.startswith("Transcript saved to "):
        st.success("Audio finished. The desk remains open so you can review its results. Press Stop to end the session.")
    else:
        st.error(f"Audio source stopped: {detail}")


RECOMMENDER_FILES = {"Kalshi": os.path.join(SCRIPT_DIR, "live_recommendations.json"),
                     "Polymarket": os.path.join(SCRIPT_DIR, "live_polymarket_recommendations.json")}


@st.fragment(run_every=2.0)
def recommenders_panel():
    """Latest output of the two live recommenders (recommendation_schema format)."""
    shown = False
    cols = st.columns(2)
    for col, (venue, path) in zip(cols, RECOMMENDER_FILES.items()):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        shown = True
        base = data.get("baseline") or {}
        with col:
            z = base.get("z")
            read = base.get("surprise_direction", "UNSCORED")
            st.markdown(f"**{venue}** · update {data.get('transcript_update') or '—'} · "
                        f"latest statement: **{read}**" + (f" (z {z:+.1f})" if z is not None else ""))
            recs = data.get("recommendations") or []
            items = recs or (data.get("candidates") or [])
            if not items:
                st.caption("No related markets yet.")
            for r in items[:3]:
                side = (r.get("side") or "").upper()
                q = r.get("quote") or {}
                label = f"BUY {side} · " if (recs and side) else ""
                text = (f"{label}{r.get('market_title') or r.get('ticker')}  \n"
                        f"relevance {r.get('relevance_score', 0):.2f} · bid {q.get('best_bid')} / ask {q.get('best_ask')}")
                if recs:
                    st.success(text)
                else:
                    st.caption(text)
            if items and not recs:
                st.caption("Candidates only: promoted to recommendations when the statement is a surprise.")
    if not shown:
        st.caption("Recommenders not running. Tick the box above before starting, or run "
                   "`python run_kalshi_ticker2.py --watch` and `python run_polymarket.py --watch`.")


with tab_live:
    with st.expander("Live audio: microphone, recording, or stream", expanded=True):
        running = _demo_running()
        if not running:
            if _source_session():
                st.error("The previous session stopped: " + _log_tail(os.path.join(SCRIPT_DIR, "live.log")))
            source_kind = st.radio("Audio source", ["mic", "browser", "file", "url"],
                                   format_func={"mic": "Laptop microphone", "browser": "Browser recorder",
                                                "file": "Upload audio file", "url": "Live stream URL"}.get,
                                   horizontal=True)
            uploaded = (st.file_uploader("Audio recording", type=["aac", "flac", "m4a", "mp3",
                                                                  "mp4", "ogg", "wav", "webm"])
                        if source_kind == "file" else None)
            stream_url = (st.text_input("Stream URL", placeholder="https://example.com/live.m3u8")
                          if source_kind == "url" else "")
            with_recs = st.checkbox("Also run the live Kalshi + Polymarket recommenders",
                                    help="Dylan's run_kalshi_ticker2 / run_polymarket --watch. Kalshi takes "
                                         "about 2 minutes to load its catalog.")
            disabled = not logged_in or not venues or (source_kind == "file" and uploaded is None) or (
                source_kind == "url" and not stream_url.strip())
            if st.button("Start", type="primary", disabled=disabled):
                source_value = ""
                try:
                    if os.path.isfile(DEMO_PIDS_PATH) or os.path.isfile(DEMO_SOURCE_PATH):
                        _stop_demo()
                    if source_kind == "file":
                        source_value = _save_uploaded_audio(uploaded)
                    elif source_kind == "url":
                        source_value = stream_url.strip()
                    label = {"mic": "Laptop microphone", "browser": "Browser recorder",
                             "file": "Uploaded audio",
                             "url": f"Live stream: {urlparse(source_value).hostname or ''}"}[source_kind]
                    with st.spinner("Preparing the live desk..."):
                        _start_demo(live, venues, speaker, qty, source_kind, source_value,
                                    source_label=label, recommenders=with_recs)
                    st.rerun()
                except Exception as exc:
                    if source_value and source_kind == "file":
                        path = Path(source_value)
                        path.unlink(missing_ok=True)
                        try:
                            path.parent.rmdir()
                        except OSError:
                            pass
                    st.error(f"Could not start audio session: {exc}")
            st.caption(f"Scores against {speaker}'s usual stance. Trading mode: **{live}**. "
                       "Uploaded audio plays at its natural speed; URL audio follows the live stream.")
        else:
            if st.button("Stop", type="primary"):
                _stop_demo()
                st.rerun()
            active_kind = _source_session().get("kind")
            st.caption("Running: " + ", ".join(running) + ".")
            audio_source_status()
            if active_kind == "mic":
                hearing_now()
            elif active_kind == "browser":
                clip = st.audio_input("Record a statement (desk is listening for it)")
                if clip is not None and st.session_state.get("last_clip") != clip.file_id:
                    st.session_state["last_clip"] = clip.file_id
                    with st.spinner("ElevenLabs Scribe..."):
                        heard = transcribe(clip.getvalue(), clip.name or "clip.wav")
                    if heard.strip():
                        _append_spoken(heard.strip())
                        st.success(f"Heard: {heard.strip()}")
        if not running or _source_session().get("kind") in {"mic", "browser"}:
            st.caption('Try: "Inflation is still far too high. We are prepared to raise rates again in October." '
                       'or "The labor market is weakening fast and we are ready to cut rates at the next meeting."')
    live_panel()
    with st.expander("Live market recommenders (Kalshi + Polymarket)", expanded=False):
        recommenders_panel()

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
