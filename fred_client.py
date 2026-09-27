"""
fred_client.py
==============
Macro backdrop from FRED (St. Louis Fed): the numbers a Fed speaker is reacting to.
Shown on the website and given to Gemini as context when scoring a passage.

  python fred_client.py          # print the latest readings

Key: fredapi.txt (gitignored) or FRED_API_KEY.
"""

import json
import os
import time
from datetime import datetime, timezone

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "macro_cache.json")
CACHE_SECONDS = 3600
API = "https://api.stlouisfed.org/fred/series/observations"

# series id -> (label, how to show it)
SERIES = {
    "DFEDTARU": ("Fed funds target (upper)", "level"),
    "CPIAUCSL": ("CPI inflation, y/y", "yoy"),
    "PCEPILFE": ("Core PCE inflation, y/y", "yoy"),
    "UNRATE": ("Unemployment rate", "level"),
    "PAYEMS": ("Payrolls, monthly change (k)", "diff"),
    "DGS2": ("2-year Treasury yield", "level"),
    "DGS10": ("10-year Treasury yield", "level"),
    "T5YIE": ("5-year breakeven inflation", "level"),
}


def api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    path = os.path.join(SCRIPT_DIR, "fredapi.txt")
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
    return key


def _observations(series_id: str, key: str, limit: int = 14) -> list:
    r = requests.get(API, params={"series_id": series_id, "api_key": key, "file_type": "json",
                                  "sort_order": "desc", "limit": limit}, timeout=20)
    r.raise_for_status()
    return [o for o in r.json().get("observations", []) if o.get("value") not in (".", "", None)]


def fetch() -> dict:
    key = api_key()
    if not key:
        return {}
    out = []
    for sid, (label, kind) in SERIES.items():
        try:
            obs = _observations(sid, key)
        except requests.RequestException:
            continue
        if not obs:
            continue
        latest = float(obs[0]["value"])
        if kind == "yoy" and len(obs) >= 13:
            value, unit = (latest / float(obs[12]["value"]) - 1) * 100, "%"
        elif kind == "diff" and len(obs) >= 2:
            value, unit = latest - float(obs[1]["value"]), "k"
        elif kind == "level":
            value, unit = latest, "%"
        else:
            continue
        prev = None
        if kind == "level" and len(obs) >= 2:
            prev = float(obs[1]["value"])
        out.append({"id": sid, "label": label, "value": round(value, 2), "unit": unit,
                    "date": obs[0]["date"], "prev": prev})
    return {"updated_at": datetime.now(timezone.utc).isoformat(), "series": out}


def latest(max_age: int = CACHE_SECONDS) -> dict:
    """Cached for an hour so the live desk never waits on FRED."""
    try:
        if time.time() - os.path.getmtime(CACHE_PATH) < max_age:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
    except (OSError, ValueError):
        pass
    data = fetch()
    if data.get("series"):
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    return data


def context_line() -> str:
    """One line for the Gemini prompt, e.g. 'CPI inflation, y/y 2.9%; Unemployment rate 4.3%; ...'"""
    data = latest()
    return "; ".join(f"{s['label']} {s['value']:g}{s['unit']} ({s['date']})" for s in data.get("series", []))


if __name__ == "__main__":
    for s in latest(max_age=0).get("series", []):
        print(f"{s['label']:32} {s['value']:>8g}{s['unit']}   as of {s['date']}")


def value_on(series_id: str, date: str):
    """Latest observation on or before date (YYYY-MM-DD); cached in macro_cache.json."""
    cache = {}
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        pass
    hist = cache.setdefault("history", {})
    k = f"{series_id}@{date}"
    if k not in hist:
        key = api_key()
        if not key:
            return None
        r = requests.get(API, params={"series_id": series_id, "api_key": key, "file_type": "json",
                                      "sort_order": "desc", "limit": 14, "observation_end": date}, timeout=20)
        r.raise_for_status()
        obs = [o for o in r.json().get("observations", []) if o.get("value") not in (".", "")]
        if not obs:
            return None
        v = float(obs[0]["value"])
        if series_id in ("PCEPILFE", "CPIAUCSL") and len(obs) >= 13:
            v = (v / float(obs[12]["value"]) - 1) * 100
        hist[k] = v
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1)
    return hist[k]


def macro_shift(baseline_dates: list, today: str = None) -> dict:
    """How much more hawkish than usual a speaker *should* sound today, given how inflation
    and unemployment have moved since their baseline was recorded (FRED core PCE + UNRATE).
    +0.25 stance per point of extra core inflation, -0.15 per point of extra unemployment,
    capped at +/-0.3. A dove turning hawkish while inflation surges is less of a surprise."""
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dates = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in baseline_dates if len(d) >= 8 and d[:8].isdigit()]
    if not dates:
        return {"shift": 0.0}
    try:
        pce_then = sum(value_on("PCEPILFE", d) for d in dates) / len(dates)
        un_then = sum(value_on("UNRATE", d) for d in dates) / len(dates)
        pce_now, un_now = value_on("PCEPILFE", today), value_on("UNRATE", today)
    except (TypeError, requests.RequestException):
        return {"shift": 0.0}
    shift = max(-0.3, min(0.3, 0.25 * (pce_now - pce_then) - 0.15 * (un_now - un_then)))
    return {"shift": round(shift, 3), "core_pce_then": round(pce_then, 2), "core_pce_now": round(pce_now, 2),
            "unrate_then": round(un_then, 2), "unrate_now": round(un_now, 2)}
