"""
fed_speeches.py
===============
Downloads Fed speeches (federalreserve.gov) into the same format as press
conference transcripts, so they can be replayed on the live desk:

    transcripts/speech_<id>.json = {date, source, title, speaker_key, segments: [...]}

and rebuilds transcripts/catalog.json, the list of everything the website can replay.

  python fed_speeches.py                       # default speech list below
  python fed_speeches.py warsh20260828a        # any speech id from federalreserve.gov
"""

import argparse
import html
import json
import os
import re

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_DIR = os.path.join(SCRIPT_DIR, "transcripts")
SPEECH_URL = "https://www.federalreserve.gov/newsevents/speech/{id}.htm"
SPEAKER_KEYS = {"warsh": "kevin_warsh", "powell": "jerome_powell"}
DEFAULT_SPEECHES = [
    "warsh20260828a",   # Warsh, "In Our Time"
    "powell20250822a",  # Powell, Jackson Hole: Monetary Policy and the Fed's Framework Review
    "powell20250923a",  # Powell, Economic Outlook
    "powell20250416a",  # Powell, Economic Outlook (tariffs)
    "powell20251014a",  # Powell, Understanding the Fed's Balance Sheet
]
CHAIR_NAMES = {"kevin_warsh": "Chair Kevin Warsh", "jerome_powell": "Chair Jerome Powell"}


def fetch_speech(speech_id: str) -> dict:
    url = SPEECH_URL.format(id=speech_id)
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (andhacks research)"})
    r.raise_for_status()
    page = r.content.decode("utf-8", "replace")
    title = re.search(r'<h3 class="title">\s*<em>(.*?)</em>', page, re.S) or re.search(r"<title>(.*?)</title>", page, re.S)
    title = html.unescape(re.sub(r"<[^>]+>", "", title.group(1))).strip() if title else speech_id
    title = re.sub(r"\s*-\s*Federal Reserve Board\s*$", "", title)
    body = re.search(r'<div class="col-xs-12 col-sm-8 col-md-8">(.*?)<div class="footnotes"|<div id="article">(.*?)</div>\s*</div>', page, re.S)
    body = (body.group(1) or body.group(2)) if body else page
    paragraphs = []
    for p in re.findall(r"<p[^>]*>(.*?)</p>", body, re.S):
        text = html.unescape(re.sub(r"<[^>]+>", "", p))
        text = re.sub(r"\[\d+\]|\s+", " ", text).strip()
        if len(text.split()) >= 12 and not text.lower().startswith(("return to text", "last update")):
            paragraphs.append(text)
    key = next((v for k, v in SPEAKER_KEYS.items() if speech_id.startswith(k)), "kevin_warsh")
    date = re.search(r"(\d{8})", speech_id).group(1)
    return {"date": date, "source": url, "title": title, "speaker_key": key,
            "segments": [{"speaker": key.upper(), "role": "chair", "text": t} for t in paragraphs]}


def build_catalog() -> list:
    """Every replayable transcript: press conferences + speeches, newest first."""
    items = []
    for name in sorted(os.listdir(TRANSCRIPT_DIR)):
        if not name.endswith(".json") or name == "catalog.json":
            continue
        rid = name[:-5]
        try:
            with open(os.path.join(TRANSCRIPT_DIR, name), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        date = data.get("date") or rid[-8:]
        if rid.startswith("speech_"):
            speaker = data.get("speaker_key", "kevin_warsh")
            label = f"Speech: {data.get('title', rid)}"
        elif re.fullmatch(r"20\d{6}", rid):
            speaker = "kevin_warsh" if date >= "20260601" else "jerome_powell"
            label = "FOMC press conference"
        else:
            continue
        items.append({"id": rid, "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}", "speaker": speaker,
                      "label": f"{label} · {CHAIR_NAMES.get(speaker, speaker)} · {date[:4]}-{date[4:6]}-{date[6:8]}"})
    items.sort(key=lambda i: i["date"], reverse=True)
    with open(os.path.join(TRANSCRIPT_DIR, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1, ensure_ascii=False)
    return items


def main():
    parser = argparse.ArgumentParser(description="Download Fed speeches for replay")
    parser.add_argument("ids", nargs="*", default=DEFAULT_SPEECHES)
    args = parser.parse_args()
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    for sid in args.ids:
        try:
            data = fetch_speech(sid)
        except Exception as e:
            print(f"{sid}: failed ({e})")
            continue
        with open(os.path.join(TRANSCRIPT_DIR, f"speech_{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print(f"{sid}: {data['title']} · {len(data['segments'])} paragraphs")
    for item in build_catalog():
        print(" ", item["id"], "->", item["label"])


if __name__ == "__main__":
    main()
