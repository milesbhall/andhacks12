"""
seed_baselines.py
=================
Builds each speaker's baseline from their real past press conferences, so the
surprise scorer has something to compare new statements against.

What it does:
  1. Downloads official FOMC press conference transcripts (PDF) from
     federalreserve.gov
  2. Splits each one into who-said-what: the Chair's answers vs. reporters'
     questions, and saves the result to transcripts/<date>.json
     (these files are also what the replay demo uses)
  3. Breaks the Chair's answers into ~120-word chunks and adds each chunk to
     the speaker's baseline (one Gemini embedding per chunk)
  4. Calibrates: scores every baseline chunk against the baseline centroid to
     see what "normal" surprise looks like for this speaker, and suggests a
     threshold (mean + 2 standard deviations)

Warsh's press conferences as Chair: 2026-06-17, 2026-07-29, 2026-09-16.
By default we seed with June + July and keep September 16 OUT of the
baseline, because September is the replay demo. Seeding with it would be
cheating.

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
  pip install requests pypdf google-genai

  # Parse only (no Gemini key needed): download + split transcripts
  python seed_baselines.py --parse-only

  # Seed Warsh's baseline from June + July and calibrate
  python seed_baselines.py

  # Start over (wipes this speaker's existing baseline first)
  python seed_baselines.py --reset

  # Other speakers/dates, e.g. Powell's 2026 press conferences
  python seed_baselines.py --speaker jerome_powell --chair-label POWELL \
      --dates 20260318 20260429
------------------------------------------------------------------------
"""

import argparse
import io
import json
import os
import re
import statistics

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_DIR = os.path.join(SCRIPT_DIR, "transcripts")
BASELINE_STORE_PATH = os.path.join(SCRIPT_DIR, "baselines.json")
CALIBRATION_PATH = os.path.join(SCRIPT_DIR, "baseline_calibration.json")

FED_PDF_URL = "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{date}.pdf"

DEFAULT_SPEAKER = "kevin_warsh"
DEFAULT_CHAIR_LABEL = "WARSH"                 # matches "CHAIRMAN WARSH" (and the transcript's typo "CHARIMAN WARSH")
DEFAULT_DATES = ["20260617", "20260729"]      # Sept 16 is held out for the replay demo

CHUNK_WORDS = 120      # target size of each baseline chunk
MIN_CHUNK_WORDS = 15   # skip "Thank you." / "Sure." style fragments


# ------------------------------------------------------------------ #
# 1. DOWNLOAD + PARSE
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# 2. CHUNK THE CHAIR'S ANSWERS
# ------------------------------------------------------------------ #

def chunk_text(text: str) -> list:
    """Split on sentence boundaries into chunks of about CHUNK_WORDS words."""
    sentences = re.split(r"(?<=[\.\?\!])\s+", text)
    chunks = []
    current = []
    current_words = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and current_words + words > CHUNK_WORDS:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words = current_words + words
    if current:
        chunks.append(" ".join(current))

    kept = []
    for chunk in chunks:
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            kept.append(chunk)
    return kept


def chair_chunks(segments: list) -> list:
    chunks = []
    for segment in segments:
        if segment["role"] == "chair":
            chunks.extend(chunk_text(segment["text"]))
    return chunks


# ------------------------------------------------------------------ #
# 3. SEED + 4. CALIBRATE
# ------------------------------------------------------------------ #

def calibrate(scorer, speaker: str) -> dict:
    """How far is each baseline chunk from the speaker's own centroid?
    That spread defines what 'normal' looks like for this speaker.
    """
    from speaker_baseline import centroid, cosine_similarity

    embeddings = scorer.store.get_embeddings(speaker)
    center = centroid(embeddings)
    surprises = []
    for embedding in embeddings:
        surprises.append(1.0 - cosine_similarity(embedding, center))

    mean = statistics.mean(surprises)
    stdev = statistics.pstdev(surprises)
    ordered = sorted(surprises)
    result = {
        "speaker": speaker,
        "n": len(surprises),
        "mean": round(mean, 4),
        "stdev": round(stdev, 4),
        "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 4),
        "max": round(ordered[-1], 4),
        "suggested_threshold": round(mean + 2 * stdev, 4),
    }

    all_calibration = {}
    if os.path.isfile(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, encoding="utf-8") as f:
            all_calibration = json.load(f)
    all_calibration[speaker] = result
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(all_calibration, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Seed speaker baselines from Fed press conference transcripts")
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--chair-label", default=DEFAULT_CHAIR_LABEL,
                        help="Text that identifies the speaker's label in the transcript, e.g. WARSH")
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES,
                        help="Press conference dates as YYYYMMDD")
    parser.add_argument("--parse-only", action="store_true",
                        help="Download and split transcripts, but don't call Gemini")
    parser.add_argument("--reset", action="store_true",
                        help="Remove this speaker's existing baseline before seeding")
    args = parser.parse_args()

    all_chunks = []
    for date in args.dates:
        segments = load_or_parse(date, args.chair_label)
        chunks = chair_chunks(segments)
        chair_answers = sum(1 for s in segments if s["role"] == "chair")
        print(f"{date}: {len(segments)} segments, {chair_answers} chair answers -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal baseline chunks for {args.speaker}: {len(all_chunks)}")
    if args.parse_only:
        print("Parse-only mode: transcripts saved to transcripts/. No embeddings created.")
        return

    import speaker_baseline

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    key_file = os.path.join(SCRIPT_DIR, "gemapi.txt")
    if not gemini_key and os.path.isfile(key_file):
        with open(key_file, encoding="utf-8") as f:
            gemini_key = f.read().strip()

    # Make sure the store keeps every seeded chunk instead of trimming to the default.
    speaker_baseline.MAX_BASELINE_SIZE = max(speaker_baseline.MAX_BASELINE_SIZE, len(all_chunks) + 100)

    scorer = speaker_baseline.SurpriseScorer(store_path=BASELINE_STORE_PATH, api_key=gemini_key)
    if args.reset and args.speaker in scorer.store.data:
        del scorer.store.data[args.speaker]
        scorer.store._save()
        print(f"Cleared existing baseline for {args.speaker}.")

    for number, chunk in enumerate(all_chunks, start=1):
        scorer.add_to_baseline(args.speaker, chunk)
        if number % 10 == 0 or number == len(all_chunks):
            print(f"  embedded {number}/{len(all_chunks)}")

    stats = calibrate(scorer, args.speaker)
    print("\nCalibration (surprise of the speaker's OWN past answers vs. their baseline):")
    print(f"  n={stats['n']}  mean={stats['mean']}  stdev={stats['stdev']}  "
          f"p90={stats['p90']}  max={stats['max']}")
    print(f"  Suggested SURPRISE_THRESHOLD (mean + 2 sd): {stats['suggested_threshold']}")
    print(f"  Current SURPRISE_THRESHOLD in speaker_baseline.py: {speaker_baseline.SURPRISE_THRESHOLD}")
    print(f"Saved to {os.path.basename(BASELINE_STORE_PATH)} and {os.path.basename(CALIBRATION_PATH)}.")


if __name__ == "__main__":
    main()
