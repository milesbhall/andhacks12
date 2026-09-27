"""
mic.py
======
Talk into the laptop microphone and watch the desk trade on it.

    microphone ──► ElevenLabs Scribe v2 Realtime ──► live_transcript.json ──► live.py (score + trade)
                                   └──► live_partial.json (what it's hearing right now, for the dashboard)

Writes the same live_transcript.json format as speechtxt.py, so live.py and
the dashboard's Live tab work unchanged. Start live.py first (or use the
"Start mic demo" button in the dashboard, which starts both).

  python mic.py                      # default microphone
  python mic.py --list               # show microphones
  python mic.py --device 2           # pick one

Needs an ElevenLabs key in elevenapi.txt (or ELEVENLABS_API_KEY).
Say something clearly hawkish or dovish, for example:
  "Inflation is still far too high. We are prepared to raise rates again in October."
  "The labor market is weakening quickly, and we are ready to cut rates at the next meeting."
"""

import argparse
import asyncio
import base64
import json
import os
import queue
from datetime import datetime, timezone
from urllib.parse import urlencode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_PATH = os.path.join(SCRIPT_DIR, "live_transcript.json")
PARTIAL_PATH = os.path.join(SCRIPT_DIR, "live_partial.json")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.1


def api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    path = os.path.join(SCRIPT_DIR, "elevenapi.txt")
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
    return key


def _write_json(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def save_segments(segments: list):
    _write_json(TRANSCRIPT_PATH, {
        "date": datetime.now(timezone.utc).strftime("%Y%m%d"), "source": "microphone",
        "updated_at": datetime.now(timezone.utc).isoformat(), "segments": segments,
        "text": "\n\n".join(s["text"] for s in segments),
    })


def save_partial(text: str):
    _write_json(PARTIAL_PATH, {"updated_at": datetime.now(timezone.utc).isoformat(), "text": text})


async def run(device=None):
    import sounddevice as sd
    import websockets

    key = api_key()
    if not key:
        raise SystemExit("No ElevenLabs key: put it in elevenapi.txt or ELEVENLABS_API_KEY.")

    audio = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        audio.put(bytes(indata))

    params = urlencode({"model_id": "scribe_v2_realtime", "audio_format": "pcm_16000",
                        "commit_strategy": "vad", "vad_silence_threshold_secs": 1.0,
                        "language_code": "en"})
    uri = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{params}"
    segments = []
    save_segments(segments)      # fresh transcript for this session
    save_partial("")

    async with websockets.connect(uri, additional_headers={"xi-api-key": key}, max_size=None) as ws:
        async def receive():
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("message_type")
                if kind == "session_started":
                    print("Listening... speak into the microphone (Ctrl+C to stop).")
                elif kind == "partial_transcript":
                    save_partial(msg.get("text", ""))
                elif kind in ("committed_transcript", "committed_transcript_with_timestamps"):
                    text = (msg.get("text") or "").strip()
                    if text:
                        segments.append({"speaker": "microphone", "role": "speaker", "text": text})
                        save_segments(segments)
                        save_partial("")
                        print(f"  heard: {text}")
                elif kind and ("error" in kind or kind in ("quota_exceeded", "rate_limited", "auth_error")):
                    raise RuntimeError(f"ElevenLabs: {msg}")

        receiver = asyncio.create_task(receive())
        loop = asyncio.get_running_loop()
        with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device,
                               blocksize=int(SAMPLE_RATE * CHUNK_SECONDS), callback=on_audio):
            try:
                while not receiver.done():
                    chunk = await loop.run_in_executor(None, audio.get)
                    await ws.send(json.dumps({"message_type": "input_audio_chunk",
                                              "audio_base_64": base64.b64encode(chunk).decode("ascii"),
                                              "sample_rate": SAMPLE_RATE, "commit": False}))
            finally:
                if not receiver.done():
                    receiver.cancel()
        if receiver.done() and not receiver.cancelled() and receiver.exception():
            raise receiver.exception()


def main():
    parser = argparse.ArgumentParser(description="Microphone -> ElevenLabs realtime -> live_transcript.json")
    parser.add_argument("--device", type=int, help="Input device index (see --list)")
    parser.add_argument("--list", action="store_true", help="List microphones")
    args = parser.parse_args()
    if args.list:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(i, d["name"])
        return
    try:
        asyncio.run(run(args.device))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
