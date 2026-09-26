"""
fed_transcripts.py
==================
Downloads official FOMC press conference transcripts (PDF) from
federalreserve.gov and splits them into who-said-what:
    transcripts/<date>.json = {date, source, segments: [{speaker, role, text}]}
role is "chair" for the Chair's answers and "other" for reporters.

stance_scorer.py calls this automatically when a transcript is missing.

  python fed_transcripts.py 20261028                  # Warsh (default)
  python fed_transcripts.py 20260318 --chair-label POWELL
"""

import argparse
import io
import json
import os
import re

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_DIR = os.path.join(SCRIPT_DIR, "transcripts")
FED_PDF_URL = "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{date}.pdf"
DEFAULT_CHAIR_LABEL = "WARSH"   # also matches the transcript's "CHARIMAN WARSH" typo


def download_pdf_text(date: str) -> str:
    from pypdf import PdfReader

    url = FED_PDF_URL.format(date=date)
    resp = requests.get(url, timeout=60, headers={"User-Agent": "andhacks-research"})
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def split_speakers(raw_text: str, chair_label: str) -> list:
    """Returns [{"speaker": "CHAIRMAN WARSH", "role": "chair"|"other", "text": ...}, ...]
    in the order they spoke.
    """
    text = raw_text
    # Drop running page headers/footers like "Page 3 of 25" and the title line.
    text = re.sub(r"(?m)^.*Press Conference\s+FINAL\s*$", "", text)
    text = re.sub(r"(?m)^\s*Page \d+ of \d+\s*$", "", text)

    # Speaker labels look like "CHAIRMAN WARSH." or "COLBY SMITH." at the start of a line.
    parts = re.split(r"(?m)^\s*([A-Z][A-Z\.\-’' ]{2,40})\.\s", text)

    segments = []
    index = 1
    while index < len(parts) - 1:
        speaker = parts[index].strip()
        body = re.sub(r"\s+", " ", parts[index + 1]).strip()
        # pypdf sometimes inserts a space before punctuation ("Warsh ," / "it ’s").
        body = re.sub(r"\s+([,\.;:\?\!’%\)])", r"\1", body)
        body = re.sub(r"(\w)’ (\w)", r"\1’\2", body)
        index = index + 2
        if not body:
            continue
        role = "chair" if chair_label in speaker else "other"
        segments.append({"speaker": speaker, "role": role, "text": body})
    return segments


def load_or_parse(date: str, chair_label: str) -> list:
    """Uses transcripts/<date>.json if it exists, otherwise downloads + parses."""
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    path = os.path.join(TRANSCRIPT_DIR, f"{date}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["segments"]

    print(f"Downloading {FED_PDF_URL.format(date=date)} ...")
    segments = split_speakers(download_pdf_text(date), chair_label)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": date, "source": FED_PDF_URL.format(date=date),
                   "segments": segments}, f, indent=1, ensure_ascii=False)
    return segments


def main():
    parser = argparse.ArgumentParser(description="Download + split FOMC press conference transcripts")
    parser.add_argument("dates", nargs="+", help="YYYYMMDD")
    parser.add_argument("--chair-label", default=DEFAULT_CHAIR_LABEL)
    args = parser.parse_args()
    for date in args.dates:
        segments = load_or_parse(date, args.chair_label)
        chair = sum(1 for s in segments if s["role"] == "chair")
        print(f"{date}: {len(segments)} segments, {chair} chair answers -> transcripts/{date}.json")


if __name__ == "__main__":
    main()
