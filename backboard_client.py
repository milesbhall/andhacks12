"""
backboard_client.py
===================
Backboard is the desk's long-term memory.

  Assistant  "Fed Surprise Desk": one per account, created on first use.
  Memories   one per surprise signal ("Warsh HAWKISH z=+3.0 on 2026-09-16 ...,
             bought YES on ...") plus each speaker's usual stance. Searchable,
             and used automatically when you ask the desk a question.
  Threads    one per run (a replay, a live session, the dashboard). Every
             signal and its trades is appended as a message, so a run reads
             like a trading log.
  Ask        questions in plain English, answered from the stored memories
             ("When was Warsh most hawkish, and what did we buy?").

Key (gitignored): backboardapi.txt, or env BACKBOARD_API_KEY.
IDs Backboard gives us are kept in backboard_state.json (gitignored).

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  python backboard_client.py --setup                  # create the assistant, store baselines
  python backboard_client.py --memories               # list what the desk remembers
  python backboard_client.py --search "rate hike"     # search memories
  python backboard_client.py --import-results results/replay_20260916.json
  python backboard_client.py --ask "When was Warsh most hawkish and what did we trade?"
------------------------------------------------------------------------
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://app.backboard.io/api"
STATE_PATH = os.path.join(SCRIPT_DIR, "backboard_state.json")
ASSISTANT_NAME = "Fed Surprise Desk"

# Model the desk answers questions with. Backboard routes to the provider;
# if this pair isn't available on the account we fall back to Backboard's default.
ASK_PROVIDER = os.environ.get("BACKBOARD_LLM_PROVIDER", "google")
ASK_MODEL = os.environ.get("BACKBOARD_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are the memory of a trading desk that listens to Federal Reserve officials and "
    "other market-moving speakers. Each stored memory is a signal: a statement that was "
    "unusually hawkish or dovish FOR THAT SPEAKER (measured as a z-score against their own "
    "past answers), plus the prediction-market orders it triggered on Kalshi or Polymarket. "
    "Answer questions using those memories. Quote dates, z-scores, markets and prices "
    "exactly as stored. If the memories don't contain the answer, say so plainly. "
    "Keep answers short. You are not giving financial advice."
)


def _read_key() -> str:
    key = os.environ.get("BACKBOARD_API_KEY", "")
    path = os.path.join(SCRIPT_DIR, "backboardapi.txt")
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
    return key


API_KEY = _read_key()


def enabled() -> bool:
    return bool(API_KEY)


# ------------------------------------------------------------------ #
# LOW LEVEL
# ------------------------------------------------------------------ #

def _request(method: str, path: str, json_body=None, params=None, timeout=60):
    if not API_KEY:
        raise RuntimeError("No Backboard key: put it in backboardapi.txt or BACKBOARD_API_KEY.")
    for attempt in range(3):   # Backboard occasionally returns a one-off 500
        resp = requests.request(method, BASE_URL + path, params=params, json=json_body, timeout=timeout,
                                headers={"X-API-Key": API_KEY, "Content-Type": "application/json"})
        if resp.status_code < 500:
            break
        time.sleep(2 * (attempt + 1))
    if resp.status_code >= 400:
        raise RuntimeError(f"Backboard {method} {path} failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json() if resp.text else {}


def _load_state() -> dict:
    if os.path.isfile(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"assistant_id": None, "threads": {}}


def _save_state(state: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ------------------------------------------------------------------ #
# ASSISTANT + THREADS
# ------------------------------------------------------------------ #

def assistant_id() -> str:
    """Our assistant's id. Reuses one with the same name, otherwise creates it."""
    state = _load_state()
    if state.get("assistant_id"):
        return state["assistant_id"]
    existing = _request("GET", "/assistants")
    items = existing if isinstance(existing, list) else existing.get("assistants", [])
    match = next((a for a in items if a.get("name") == ASSISTANT_NAME), None)
    if match:
        aid = match["assistant_id"]
    else:
        aid = _request("POST", "/assistants", {"name": ASSISTANT_NAME, "system_prompt": SYSTEM_PROMPT})["assistant_id"]
    state["assistant_id"] = aid
    _save_state(state)
    return aid


def thread_id(run: str) -> str:
    """One Backboard thread per run name (e.g. 'replay_20260916', 'dashboard')."""
    state = _load_state()
    if run in state["threads"]:
        return state["threads"][run]
    tid = _request("POST", f"/assistants/{assistant_id()}/threads", {})["thread_id"]
    state = _load_state()
    state["threads"][run] = tid
    _save_state(state)
    return tid


def log_message(run: str, content: str, metadata: dict = None) -> dict:
    """Store a message in the run's thread without calling an LLM."""
    body = {"content": content, "thread_id": thread_id(run), "send_to_llm": "false",
            "memory": "off", "stream": False}
    if metadata:
        body["metadata"] = {k: str(v) for k, v in metadata.items() if v is not None}
    return _request("POST", "/threads/messages", body)


def run_log(run: str) -> list:
    """Messages stored for a run, oldest first."""
    state = _load_state()
    if run not in state["threads"]:
        return []
    return _request("GET", f"/threads/{state['threads'][run]}").get("messages", [])


def runs() -> list:
    return sorted(_load_state()["threads"].keys())


# ------------------------------------------------------------------ #
# MEMORIES
# ------------------------------------------------------------------ #

def add_memory(content: str, metadata: dict = None) -> dict:
    body = {"content": content}
    if metadata:
        # Backboard 500s on numeric metadata values; send everything as text.
        body["metadata"] = {k: str(v) for k, v in metadata.items() if v is not None}
    return _request("POST", f"/assistants/{assistant_id()}/memories", body)


def list_memories(page_size: int = 100) -> list:
    data = _request("GET", f"/assistants/{assistant_id()}/memories", params={"page": 1, "page_size": page_size})
    return data.get("memories", [])


def search_memories(query: str, limit: int = 10) -> list:
    data = _request("POST", f"/assistants/{assistant_id()}/memories/search", {"query": query, "limit": limit})
    return data.get("memories", [])


def remember_baselines(path: str = None):
    """Store each speaker's usual stance so the desk can explain z-scores."""
    path = path or os.path.join(SCRIPT_DIR, "stance_baselines.json")
    with open(path, encoding="utf-8") as f:
        store = json.load(f)
    for speaker, b in store.items():
        add_memory(
            f"Baseline for {speaker}: across {b['n']} past press conference answers "
            f"({', '.join(b.get('dates', []))}), usual stance is {b['mean']:+.2f} on a -1 (dovish) "
            f"to +1 (hawkish) scale, spread {b['stdev']:.2f}. An answer 2+ spreads away is a surprise.",
            {"kind": "baseline", "speaker": speaker, "mean": b["mean"], "stdev": b["stdev"]},
        )


def _trade_line(t: dict) -> str:
    if t.get("error"):
        return f"{t.get('venue')} {t.get('market')}: skipped ({t['error']})"
    status = "BLOCKED" if t.get("blocked") else ("SENT LIVE" if t.get("sent") else "dry run")
    return (f"{t.get('venue')} buy {str(t.get('side', '')).upper()} x{t.get('qty')} {t.get('market')} "
            f"at YES limit {t.get('yes_limit')} (max ${t.get('max_cost')}, {status})")


def record_signal(record: dict, run: str, when: str = None):
    """Save one surprise to Backboard: a searchable memory plus a line in the run's log."""
    when = when or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    trades = [_trade_line(t) for t in record.get("trades", [])]
    markets = [f"{m['venue']} {m['market']} ({m['direction']}, bid {m['quote']['best_bid']} / "
               f"ask {m['quote']['best_ask']})" for m in record.get("matches", [])[:5]]

    memory = (f"[{run}] {record['speaker']} sounded {record['direction']} (z={record['z']:+.1f}, stance "
              f"{record['stance']:+.2f} vs usual {record['baseline_mean']:+.2f}): {record['summary'].rstrip('.')}. "
              + (f"Orders: {'; '.join(trades)}." if trades else "No orders placed."))
    meta = {"kind": "signal", "run": run, "speaker": record["speaker"], "direction": record["direction"],
            "z": record["z"], "stance": record["stance"], "logged_at": when}
    if record.get("solana"):
        meta["solana_signature"] = record["solana"].get("signature")
    add_memory(memory, meta)

    log = [f"{when} | {record['speaker']} | {record['direction']} z={record['z']:+.1f}",
           f"Statement: {record['statement'][:600]}",
           f"Read: {record['summary']}"]
    if markets:
        log.append("Markets: " + "; ".join(markets))
    if trades:
        log.append("Orders: " + "; ".join(trades))
    if record.get("solana"):
        log.append(f"Solana receipt: {record['solana'].get('explorer')}")
    log_message(run, "\n".join(log), meta)


# ------------------------------------------------------------------ #
# ASK
# ------------------------------------------------------------------ #

def _answer_with_gemini(question: str, memories: list) -> str:
    """Write the answer from Backboard's retrieved memories using our own Gemini key."""
    import polymarket_client
    from google import genai
    notes = "\n".join(f"- {m.get('content', '')}" for m in memories) or "(no memories found)"
    client = genai.Client(api_key=polymarket_client.GEMINI_API_KEY)
    prompt = f"{SYSTEM_PROMPT}\n\nMemories:\n{notes}\n\nQuestion: {question}"
    first = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    models = list(dict.fromkeys([first, "gemini-3.1-flash-lite", "gemini-3.8-flash"]))
    last = None
    for attempt in range(2):
        for model in models:
            try:
                return client.models.generate_content(model=model, contents=prompt).text.strip()
            except Exception as e:   # 429 quota / 503 overload: try the next model
                last = e
                if "429" not in str(e) and "503" not in str(e):
                    raise
        time.sleep(20)   # per-minute limits reset quickly
    top = "\n".join(f"- {m.get('content', '')}" for m in memories[:5])
    return ("Gemini is rate-limited right now, so here are the closest saved memories instead:\n"
            + (top or "(none found)") + f"\n\n({type(last).__name__}: quota exceeded, try again in a minute)")


def ask(question: str, run: str = "questions") -> dict:
    """Plain-English question answered from the desk's memories.

    Normal path: Backboard's chat, which pulls in the memories itself. If the
    Backboard account has no LLM credits, Backboard still does the memory
    search and Gemini writes the answer.
    """
    body = {"content": question, "thread_id": thread_id(run), "memory": "Readonly",
            "memory_response_citation": True, "stream": False,
            "llm_provider": ASK_PROVIDER, "model_name": ASK_MODEL}
    resp = {}
    try:
        try:
            resp = _request("POST", "/threads/messages", body, timeout=120)
        except RuntimeError as e:
            if "model" not in str(e).lower() and "provider" not in str(e).lower():
                raise
            body.pop("llm_provider"); body.pop("model_name")   # use Backboard's default model
            resp = _request("POST", "/threads/messages", body, timeout=120)
    except RuntimeError:
        resp = {}
    answer = resp.get("content") or resp.get("message") or ""
    if answer and "out of credits" not in answer.lower() and resp.get("model_name"):
        return {"answer": answer, "memories": resp.get("retrieved_memories") or [],
                "model": f"Backboard {resp.get('model_provider')}/{resp.get('model_name')}"}

    memories = search_memories(question, limit=10)
    return {"answer": _answer_with_gemini(question, memories), "memories": memories,
            "model": f"Backboard memory search + Gemini {os.environ.get('GEMINI_MODEL', 'gemini-3.5-flash-lite')}"}


# Backwards-compatible helper (older code called backboard_client.log)
def log(question: str, answer: str):
    if not enabled():
        return None
    return log_message("general", f"{question}\n\n{answer}")


def import_results(path: str) -> int:
    """Load a saved pipeline run (results/<run>.json) into Backboard without rerunning it."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    run = data.get("run") or os.path.splitext(os.path.basename(path))[0]
    count = 0
    for record in data.get("records", []):
        if record.get("direction") in ("HAWKISH", "DOVISH"):
            record_signal(record, run=run, when=str(data.get("created", ""))[:16].replace("T", " ") + " UTC")
            count += 1
    return count


def delete_memory(memory_id: str):
    return _request("DELETE", f"/assistants/{assistant_id()}/memories/{memory_id}")


def main():
    parser = argparse.ArgumentParser(description="Backboard memory for the desk")
    parser.add_argument("--setup", action="store_true", help="Create the assistant and store speaker baselines")
    parser.add_argument("--memories", action="store_true")
    parser.add_argument("--search")
    parser.add_argument("--ask")
    parser.add_argument("--import-results", metavar="RESULTS_JSON", help="Load a saved run, e.g. results/replay_20260916.json")
    args = parser.parse_args()

    if args.setup:
        print("assistant:", assistant_id())
        remember_baselines()
        print("Stored speaker baselines. Memories:", len(list_memories()))
    elif args.import_results:
        print(f"Imported {import_results(args.import_results)} signals.")
    elif args.memories:
        for m in list_memories():
            print(f"- {m.get('content')}")
    elif args.search:
        for m in search_memories(args.search):
            print(f"{m.get('score', 0):.2f}  {m.get('content')}")
    elif args.ask:
        r = ask(args.ask)
        print(r["answer"], f"\n\n({r['model']}, {len(r['memories'])} memories used)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
