import argparse
import asyncio
import base64
import json
import os
import shutil
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
import websockets


SCRIPT_DIR = Path(__file__).resolve().parent
API_KEY_PATH = SCRIPT_DIR / "elevenapi.txt"
OUTPUT_PATH = SCRIPT_DIR / "live_transcript.json"
SAMPLE_RATE = 16000
AUDIO_CHUNK_BYTES = 3200


def load_api_key() -> str:
    if API_KEY_PATH.is_file():
        return API_KEY_PATH.read_text(encoding="utf-8").strip()
    return os.environ.get("ELEVENLABS_API_KEY", "").strip()


def save_transcript(path: Path, source: str, segments: list[dict]) -> None:
    payload = {
        "date": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "source": source,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "segments": segments,
        "text": "\n\n".join(segment["text"] for segment in segments),
    }
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def transcribe_file(audio_path: Path, output_path: Path, language: str) -> None:
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("Set ELEVENLABS_API_KEY or put the key in elevenapi.txt.")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with audio_path.open("rb") as audio_file:
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
            data={
                "model_id": "scribe_v2",
                "language_code": language,
                "diarize": "true",
                "no_verbatim": "true",
            },
            files={"file": (audio_path.name, audio_file, "application/octet-stream")},
            timeout=(30, 600),
        )
    response.raise_for_status()
    result = response.json()
    text = str(result.get("text", "")).strip()
    if not text:
        raise RuntimeError("ElevenLabs returned an empty transcript.")

    words = result.get("words") or []
    segments = []
    current_speaker = None
    current_text = []
    for word in words:
        if not isinstance(word, dict):
            continue
        word_text = str(word.get("text", ""))
        speaker = word.get("speaker_id") or "speaker_unknown"
        if current_text and speaker != current_speaker:
            segments.append({
                "speaker": current_speaker,
                "role": "speaker",
                "text": "".join(current_text).strip(),
            })
            current_text = []
        current_speaker = speaker
        current_text.append(word_text)

    if current_text and "".join(current_text).strip():
        segments.append({
            "speaker": current_speaker,
            "role": "speaker",
            "text": "".join(current_text).strip(),
        })
    if not segments:
        segments = [{"speaker": "speaker_unknown", "role": "speaker", "text": text}]

    save_transcript(output_path, str(audio_path), segments)
    print(f"Transcript saved to {output_path}")
    print(text)


async def ffmpeg_audio_chunks(source: str):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required for live stream URLs. Install it with `brew install ffmpeg`."
        )

    parsed_source = urllib.parse.urlparse(source)
    direct_media_suffixes = (
        ".aac", ".m3u8", ".m4a", ".mp3", ".mp4", ".ogg", ".ts", ".wav", ".webm",
    )
    looks_like_direct_media = parsed_source.path.lower().endswith(direct_media_suffixes)
    yt_dlp = shutil.which("yt-dlp")
    if not looks_like_direct_media and yt_dlp:
        resolver = await asyncio.create_subprocess_exec(
            yt_dlp,
            "--no-warnings",
            "--format", "bestaudio/best",
            "--get-url",
            source,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        resolved_output, resolver_error = await resolver.communicate()
        if resolver.returncode == 0 and resolved_output.strip():
            source = resolved_output.decode("utf-8").splitlines()[0].strip()
        else:
            detail = resolver_error.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"yt-dlp could not resolve the stream URL: {detail}")
    elif not looks_like_direct_media and parsed_source.hostname in {
        "youtube.com", "www.youtube.com", "youtu.be", "www.youtube-nocookie.com",
    }:
        raise RuntimeError(
            "YouTube URLs require yt-dlp. Install it with `brew install yt-dlp`."
        )

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    command.extend([
        "-i", source,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ])

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        while True:
            chunk = await process.stdout.read(AUDIO_CHUNK_BYTES)
            if not chunk:
                break
            yield chunk
        return_code = await process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}.")
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()


async def microphone_audio_chunks():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Microphone mode requires sounddevice. Install it with "
            "`/usr/local/bin/python3 -m pip install sounddevice`."
        ) from exc

    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue(maxsize=20)

    def enqueue_audio(data: bytes) -> None:
        if not audio_queue.full():
            audio_queue.put_nowait(data)

    def audio_callback(indata, frames, timing, status) -> None:
        if status:
            print(f"Microphone warning: {status}", file=sys.stderr)
        loop.call_soon_threadsafe(enqueue_audio, bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=AUDIO_CHUNK_BYTES // 2,
        channels=1,
        dtype="int16",
        callback=audio_callback,
    ):
        print("Transcribing microphone input. Press Ctrl+C to stop.")
        while True:
            yield await audio_queue.get()


async def stream_realtime_audio(
    source: str,
    output_path: Path,
    language: str,
    microphone: bool,
) -> None:
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("Set ELEVENLABS_API_KEY or put the key in elevenapi.txt.")

    params = urlencode({
        "model_id": "scribe_v2_realtime",
        "audio_format": "pcm_16000",
        "sample_rate": SAMPLE_RATE,
        "commit_strategy": "vad",
        "language_code": language,
        "include_timestamps": "true",
    })
    uri = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{params}"
    transcript_segments = []
    session_started = asyncio.get_running_loop().create_future()

    async with websockets.connect(
        uri,
        additional_headers={"xi-api-key": api_key},
        ping_interval=20,
    ) as websocket:
        async def receive_transcripts() -> None:
            try:
                async for raw_message in websocket:
                    message = json.loads(raw_message)
                    message_type = message.get("message_type")
                    if message_type == "session_started":
                        if not session_started.done():
                            session_started.set_result(None)
                    elif message_type in (
                        "committed_transcript",
                        "committed_transcript_with_timestamps",
                    ):
                        text = str(message.get("text", "")).strip()
                        if text:
                            words = message.get("words") or []
                            speaker_ids = {
                                word.get("speaker_id")
                                for word in words
                                if isinstance(word, dict) and word.get("speaker_id")
                            }
                            transcript_segments.append({
                                "speaker": next(iter(speaker_ids)) if len(speaker_ids) == 1 else "speaker_unknown",
                                "role": "speaker",
                                "text": text,
                            })
                            save_transcript(output_path, source, transcript_segments)
                            print(text, flush=True)
                    elif message_type in (
                        "error", "auth_error", "quota_exceeded", "rate_limited",
                        "resource_exhausted", "invalid_request", "input_error",
                    ):
                        raise RuntimeError(message.get("error", str(message)))
            except Exception as exc:
                if not session_started.done():
                    session_started.set_exception(exc)
                raise

        receiver = asyncio.create_task(receive_transcripts())
        try:
            await asyncio.wait_for(asyncio.shield(session_started), timeout=20)
            chunks = (
                microphone_audio_chunks()
                if microphone
                else ffmpeg_audio_chunks(source)
            )
            async for audio_chunk in chunks:
                await websocket.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(audio_chunk).decode("ascii"),
                    "sample_rate": SAMPLE_RATE,
                    "commit": False,
                }))
        finally:
            if not receiver.done():
                try:
                    await websocket.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(bytes(SAMPLE_RATE)).decode("ascii"),
                        "sample_rate": SAMPLE_RATE,
                        "commit": True,
                    }))
                    await asyncio.sleep(1.0)
                except (websockets.ConnectionClosed, asyncio.CancelledError):
                    pass
                receiver.cancel()
            receiver_result = await asyncio.gather(receiver, return_exceptions=True)
            receiver_error = receiver_result[0]
            if isinstance(receiver_error, Exception):
                raise receiver_error

    if transcript_segments:
        save_transcript(output_path, source, transcript_segments)
        print(f"Transcript saved to {output_path}")
    else:
        raise RuntimeError("No committed speech was transcribed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe a Fed speech with ElevenLabs and save Kalshi-compatible JSON."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="Local audio recording to transcribe.")
    source.add_argument(
        "--url",
        help="Live stream URL (direct media/HLS, or a hosted page when yt-dlp supports it).",
    )
    source.add_argument("--mic", action="store_true", help="Transcribe audio from the selected microphone.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output transcript JSON path.")
    parser.add_argument("--language", default="en", help="Audio language code (default: en).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.file:
        transcribe_file(args.file.expanduser().resolve(), output_path, args.language)
        return

    source = "microphone" if args.mic else "live stream"
    asyncio.run(stream_realtime_audio(source, output_path, args.language, args.mic))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped transcription.")
