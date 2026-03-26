"""
Example 07 — Voice-driven acquisition workflow.

Demonstrates:
- Push-to-talk microphone capture
- Local faster-whisper transcription
- Feeding the transcript into the Ollama MCP client
- Optional spoken reply output

Run:
    pip install "nuance-gms-mcp[ollama,voice]"
    GMS_SIMULATE=1 python examples/07_voice_acquisition.py
    GMS_SIMULATE=1 python examples/07_voice_acquisition.py --speak

Manual simulator smoke test (no microphone capture):
    GMS_SIMULATE=1 python examples/07_voice_acquisition.py \
      --transcript "Check microscope state, acquire a 256 by 256 HAADF STEM image at 5 microseconds dwell time, and report the mean intensity."
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, "src")
os.environ.setdefault("GMS_SIMULATE", "1")

from gms_mcp.client import DEFAULT_MODEL, run_agent
from gms_mcp.voice import (
    DEFAULT_MAX_RECORDING_S,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_LANGUAGE,
    DEFAULT_WHISPER_MODEL,
    LocalWhisperTranscriber,
    VoiceDependencyError,
    record_push_to_talk,
    remove_temp_audio_file,
    speak_text,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voice-driven GMS acquisition example using local Whisper"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--gms-url", default=os.environ.get("GMS_MCP_URL", ""))
    parser.add_argument("--speak", action="store_true",
                        help="Speak the agent reply after the workflow completes")
    parser.add_argument("--tts-command", default="",
                        help="Optional text-to-speech command override")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--whisper-device", default=DEFAULT_WHISPER_DEVICE)
    parser.add_argument("--whisper-language", default=DEFAULT_WHISPER_LANGUAGE)
    parser.add_argument("--voice-sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--voice-max-seconds", type=float, default=DEFAULT_MAX_RECORDING_S)
    parser.add_argument(
        "--transcript",
        default="",
        help="Use this text instead of recording audio (useful for simulator smoke tests)",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    if args.transcript:
        transcript = args.transcript
        print("\nUsing injected transcript for simulator smoke test.\n")
    else:
        print("\nPress Enter to start recording, describe the acquisition workflow, then press Enter again to stop.")
        print("Example prompt: 'Check microscope state, acquire a 512 by 512 HAADF STEM image at 10 microseconds dwell time, and report the mean intensity.'\n")

        try:
            transcriber = LocalWhisperTranscriber(
                model_name=args.whisper_model,
                device=args.whisper_device,
                language=args.whisper_language,
            )
            audio_path = record_push_to_talk(
                sample_rate=args.voice_sample_rate,
                max_duration_s=args.voice_max_seconds,
            )
            try:
                transcript = transcriber.transcribe_file(audio_path)
            finally:
                remove_temp_audio_file(audio_path)
        except VoiceDependencyError as exc:
            raise SystemExit(f"Voice dependencies are unavailable: {exc}") from exc

    print("\n─── Transcript ───")
    print(transcript)

    result = await run_agent(
        query=transcript,
        model=args.model,
        base_url=args.ollama_url,
        server_url=args.gms_url,
        verbose=True,
    )

    print("\n─── Agent Response ───")
    print(result["answer"])
    print(f"\nTools called: {[tc['tool'] for tc in result['tool_calls']]}")

    if args.speak:
        try:
            speak_text(result["answer"], command=args.tts_command)
        except Exception as exc:
            print(f"[voice] Speech output unavailable: {exc}")


if __name__ == "__main__":
    asyncio.run(main())