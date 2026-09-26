import json
import argparse
from pathlib import Path

from kalshi_ticker2 import find_relevant_tickers


INPUT_PATH = Path(__file__).with_name("20260916.json")


def load_transcript_text(path: Path) -> str:
    with path.open(encoding="utf-8") as transcript_file:
        payload = json.load(transcript_file)

    direct_text = payload.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"Expected a 'segments' list in {path}")

    text_segments = [
        segment["text"].strip()
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("text"), str)
        and segment["text"].strip()
    ]

    chair_segments = [
        segment["text"].strip()
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("role"), str)
        and segment["role"].strip().lower() == "chair"
        and isinstance(segment.get("text"), str)
        and segment["text"].strip()
    ]
    selected_segments = chair_segments or text_segments
    if not selected_segments:
        raise ValueError(f"No non-empty transcript segments found in {path}")

    return "\n\n".join(selected_segments)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank Kalshi markets against a transcript JSON file."
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        type=Path,
        default=INPUT_PATH,
        help=f"Transcript JSON (default: {INPUT_PATH.name}).",
    )
    args = parser.parse_args()

    transcript_path = args.transcript.expanduser().resolve()
    speech_text = load_transcript_text(transcript_path)
    matches = find_relevant_tickers(speech_text, top_n=3)

    print("\nRELEVANT KALSHI MARKETS")
    print(json.dumps(matches, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
