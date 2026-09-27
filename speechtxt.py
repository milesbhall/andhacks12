import argparse
import asyncio
import base64
import json
import os
import shutil
import urllib.parse
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

import requests
import websockets


SCRIPT_DIR = Path(__file__).resolve().parent
API_KEY_PATH = SCRIPT_DIR / "elevenapi.txt"
OUTPUT_PATH = SCRIPT_DIR / "live_transcript.json"
FED_LIVE_PAGE = "https://www.federalreserve.gov/live-broadcast.htm"
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


class BrightcoveEmbedParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.video = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "video":
            return

        attributes = dict(attrs)
        required = ("data-video-id", "data-account", "data-player", "data-embed")
        if all(attributes.get(name) for name in required):
            self.video = tuple(attributes[name] for name in required)


def resolve_brightcove_player_url(page_url: str) -> str:
    response = requests.get(page_url, timeout=30)
    response.raise_for_status()
    parser = BrightcoveEmbedParser()
    parser.feed(response.text)
    if not parser.video:
        raise RuntimeError("The Federal Reserve page does not currently expose a Brightcove video.")

    video_id, account, player, embed = parser.video
    return (
        f"https://players.brightcove.net/{account}/{player}_{embed}/index.html"
        f"?videoId={urllib.parse.quote(video_id)}"
    )


def resolve_hosted_audio(source: str) -> tuple[str, dict[str, str]]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "Hosted streams need yt-dlp to resolve audio. Run `pip install yt-dlp`."
        ) from exc

    if urllib.parse.urlparse(source).hostname == "www.federalreserve.gov":
        source = resolve_brightcove_player_url(source)

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as downloader:
        info = downloader.extract_info(source, download=False)

    formats = info.get("formats", [])
    audio_formats = [
        media_format for media_format in formats
        if media_format.get("url")
        and media_format.get("protocol", "").startswith("m3u8")
        and media_format.get("vcodec") == "none"
    ]
    if not audio_formats and info.get("url"):
        audio_formats = [info]
    if not audio_formats:
        raise RuntimeError("No HLS audio rendition was found for this stream.")

    media_format = audio_formats[0]
    return media_format["url"], media_format.get("http_headers", {})


async def ffmpeg_audio_chunks(source: str):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError(
                "Audio decoding needs FFmpeg. Install it on your system or run "
                "`pip install imageio-ffmpeg`."
            ) from exc

    parsed_source = urllib.parse.urlparse(source)
    direct_media_suffixes = (
        ".aac", ".flac", ".m3u8", ".m4a", ".mp3", ".mp4", ".ogg", ".ts", ".wav", ".webm",
    )
    looks_like_direct_media = Path(source).is_file() or parsed_source.path.lower().endswith(direct_media_suffixes)
    headers = {}
    if not looks_like_direct_media:
        source, headers = await asyncio.to_thread(resolve_hosted_audio, source)

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if headers:
        formatted_headers = "".join(
            f"{name}: {value}\r\n" for name, value in headers.items()
        )
        command.extend(["-headers", formatted_headers])
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
        buffered_audio = bytearray()
        while True:
            chunk = await process.stdout.read(AUDIO_CHUNK_BYTES)
            if not chunk:
                break
            buffered_audio.extend(chunk)
            while len(buffered_audio) >= AUDIO_CHUNK_BYTES:
                yield bytes(buffered_audio[:AUDIO_CHUNK_BYTES])
                del buffered_audio[:AUDIO_CHUNK_BYTES]
        if buffered_audio:
            yield bytes(buffered_audio)
        return_code = await process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}.")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def stream_realtime_audio(
    source: str,
    output_path: Path,
    language: str,
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
    save_transcript(output_path, source, transcript_segments)
    session_started = asyncio.get_running_loop().create_future()
    last_transcript_at = None

    async with websockets.connect(
        uri,
        additional_headers={"xi-api-key": api_key},
        ping_interval=20,
    ) as websocket:
        async def receive_transcripts() -> None:
            nonlocal last_transcript_at
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
                            last_transcript_at = asyncio.get_running_loop().time()
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
            loop = asyncio.get_running_loop()
            next_send_time = loop.time()
            async for audio_chunk in ffmpeg_audio_chunks(source):
                await websocket.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(audio_chunk).decode("ascii"),
                    "sample_rate": SAMPLE_RATE,
                    "commit": False,
                }))
                next_send_time = max(next_send_time, loop.time()) + len(audio_chunk) / (SAMPLE_RATE * 2)
                await asyncio.sleep(max(0.0, next_send_time - loop.time()))
        finally:
            if not receiver.done():
                try:
                    await websocket.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(bytes(SAMPLE_RATE)).decode("ascii"),
                        "sample_rate": SAMPLE_RATE,
                        "commit": True,
                    }))
                    # Allow Scribe to finish the final passage after the last
                    # audio bytes. A fixed one-second delay can drop the end.
                    final_audio_at = loop.time()
                    deadline = loop.time() + 10.0
                    while not receiver.done() and loop.time() < deadline:
                        quiet_since = max(final_audio_at, last_transcript_at or 0)
                        if last_transcript_at is not None and loop.time() - quiet_since >= 3.0:
                            break
                        await asyncio.sleep(0.2)
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", type=Path, help="Local audio recording to transcribe.")
    source.add_argument("--realtime-file", type=Path,
                        help="Play a local audio recording through realtime transcription at its natural speed.")
    source.add_argument(
        "--url",
        default=FED_LIVE_PAGE,
        help="Live stream URL (direct media/HLS, or a hosted page when yt-dlp supports it).",
    )
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

    if args.realtime_file:
        source = str(args.realtime_file.expanduser().resolve())
        if not Path(source).is_file():
            raise FileNotFoundError(f"Audio file not found: {source}")
    else:
        source = args.url
    asyncio.run(stream_realtime_audio(source, output_path, args.language))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped transcription.")
