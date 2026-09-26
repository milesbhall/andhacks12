import json
from pathlib import Path

from kalshi_ticker2 import find_relevant_tickers


INPUT_PATH = Path(__file__).with_name("20260916.json")


def load_transcript_text(path: Path) -> str:
    with path.open(encoding="utf-8") as transcript_file:
        payload = json.load(transcript_file)

    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"Expected a 'segments' list in {path}")

    chair_segments = [
        segment["text"].strip()
        for segment in segments
        if isinstance(segment, dict)
        and segment.get("role", "").strip().lower() == "chair"
        and isinstance(segment.get("text"), str)
        and segment["text"].strip()
    ]

    if not chair_segments:
        raise ValueError(f"No non-empty chair speech segments found in {path}")

    return "\n\n".join(chair_segments)


def main() -> None:
    speech_text = load_transcript_text(INPUT_PATH)
    matches = find_relevant_tickers(speech_text, top_n=3)

    print("\nRELEVANT KALSHI MARKETS")
    print(json.dumps(matches, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
