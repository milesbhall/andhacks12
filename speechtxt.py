import asyncio
import json
import base64
import websockets
import os

# Helper function to load API key from file
def load_api_key(filename="elevenapi.txt"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return f.read().strip()
    # Fallback to environment variable if file is missing
    return os.environ.get("ELEVENLABS_API_KEY", "")

# Configuration
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # Default voice ID (Rachel)
MODEL_ID = "eleven_flash_v2_5"      # Use Flash v2.5 for sub-100ms processing
API_KEY = load_api_key("elevenapi.txt")

async def text_chunk_generator():
    """Simulates live incoming Fed transcript text chunks (e.g., from an STT or live feed)."""
    sentences = [
        "The Federal Open Market Committee ",
        "decided today to maintain ",
        "the target range for the federal funds rate ",
        "at five and a quarter to five and a half percent. "
    ]
    for text in sentences:
        yield text
        await asyncio.sleep(0.05)  # Simulate live incoming text feed

async def stream_elevenlabs_tts():
    if not API_KEY:
        raise ValueError("API key is missing. Please ensure elevenapi.txt contains a valid key.")

    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id={MODEL_ID}&output_format=mp3_44100_128"

    async with websockets.connect(uri) as ws:
        # 1. Send initial configuration message
        bos_message = {
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
            "xi_api_key": API_KEY,
        }
        await ws.send(json.dumps(bos_message))

        # 2. Start audio receiver task
        async def receive_audio():
            while True:
                try:
                    response = await ws.recv()
                    data = json.loads(response)
                    if data.get("audio"):
                        audio_chunk = base64.b64decode(data["audio"])
                        # Play or stream audio_chunk immediately to client/speaker
                        print(f"Received audio chunk: {len(audio_chunk)} bytes")
                    if data.get("isFinal"):
                        break
                except websockets.exceptions.ConnectionClosed:
                    break

        receiver = asyncio.create_task(receive_audio())

        # 3. Stream incoming text chunks immediately as they arrive
        async for chunk in text_chunk_generator():
            text_payload = {
                "text": chunk,
                "try_trigger_generation": True
            }
            await ws.send(json.dumps(text_payload))

        # 4. Send End of Stream
        eos_message = {"text": ""}
        await ws.send(json.dumps(eos_message))
        await receiver

if __name__ == "__main__":
    asyncio.run(stream_elevenlabs_tts())