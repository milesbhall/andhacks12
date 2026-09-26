"""
backboard_client.py
===================
Logs each surprise signal and the trades it triggered to a Backboard thread,
so the system keeps a running memory of what it called and why.

Key (gitignored): backboardapi.txt, or env BACKBOARD_API_KEY.
Optional: BACKBOARD_THREAD_ID to keep writing to one thread.
"""

import os
from datetime import datetime, timezone

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKBOARD_BASE_URL = "https://app.backboard.io/api"
BACKBOARD_THREAD_ID = os.environ.get("BACKBOARD_THREAD_ID", "")


def _read_key() -> str:
    key = os.environ.get("BACKBOARD_API_KEY", "")
    path = os.path.join(SCRIPT_DIR, "backboardapi.txt")
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
    return key


BACKBOARD_API_KEY = _read_key()


class BackboardClient:
    def __init__(self, api_key: str = BACKBOARD_API_KEY):
        if not api_key:
            raise RuntimeError("Set BACKBOARD_API_KEY or create backboardapi.txt.")
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def log_exchange(self, thread_id: str, question: str, answer: str) -> str:
        content = (f"[Signal log -- {datetime.now(timezone.utc).isoformat()}]\n\n"
                   f"Signal: {question}\n\nAnalysis and trades:\n{answer}")
        payload = {"content": content, "stream": False}
        if thread_id:
            payload["thread_id"] = thread_id
        resp = requests.post(f"{BACKBOARD_BASE_URL}/threads/messages", headers=self.headers,
                             json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("thread_id", thread_id)


def log(question: str, answer: str):
    """Best effort: returns None if no key is set."""
    if not BACKBOARD_API_KEY:
        return None
    return BackboardClient().log_exchange(BACKBOARD_THREAD_ID, question, answer)
