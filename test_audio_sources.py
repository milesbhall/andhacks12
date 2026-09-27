"""Offline checks for the two file transcription modes and local FFmpeg input."""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import speechtxt


class AudioSourceTests(unittest.TestCase):
    def test_batch_file_cli_still_uses_one_shot_transcription(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "recording.mp3"
            audio.write_bytes(b"placeholder")
            output = Path(folder) / "batch.json"
            with patch.object(sys, "argv", ["speechtxt.py", "--file", str(audio),
                                            "--output", str(output)]), \
                 patch.object(speechtxt, "transcribe_file") as batch, \
                 patch.object(speechtxt, "stream_realtime_audio", new_callable=AsyncMock) as realtime:
                speechtxt.main()
            batch.assert_called_once_with(audio, output, "en")
            realtime.assert_not_awaited()

    def test_realtime_file_cli_uses_streaming_transcription(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "recording.mp3"
            audio.write_bytes(b"placeholder")
            output = Path(folder) / "live.json"
            with patch.object(sys, "argv", ["speechtxt.py", "--realtime-file", str(audio),
                                            "--output", str(output)]), \
                 patch.object(speechtxt, "transcribe_file") as batch, \
                 patch.object(speechtxt, "stream_realtime_audio", new_callable=AsyncMock) as realtime:
                speechtxt.main()
            realtime.assert_awaited_once_with(str(audio), output, "en")
            batch.assert_not_called()

    def test_extensionless_local_file_goes_directly_to_ffmpeg(self):
        class FakePipe:
            def __init__(self):
                self.chunks = [b"a" * speechtxt.AUDIO_CHUNK_BYTES, b""]

            async def read(self, count):
                return self.chunks.pop(0)

        class FakeProcess:
            stdout = FakePipe()
            returncode = 0

            async def wait(self):
                return 0

        async def collect(source):
            return [chunk async for chunk in speechtxt.ffmpeg_audio_chunks(source)]

        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "audio_without_extension"
            audio.write_bytes(b"placeholder")
            with patch.object(speechtxt.shutil, "which", return_value="ffmpeg"), \
                 patch.object(speechtxt.asyncio, "create_subprocess_exec",
                              new_callable=AsyncMock, return_value=FakeProcess()) as create, \
                 patch.object(speechtxt, "resolve_hosted_audio") as resolve:
                chunks = asyncio.run(collect(str(audio)))
            self.assertEqual(chunks, [b"a" * speechtxt.AUDIO_CHUNK_BYTES])
            resolve.assert_not_called()
            args = create.await_args.args
            self.assertEqual(args[args.index("-i") + 1], str(audio))


if __name__ == "__main__":
    unittest.main()
