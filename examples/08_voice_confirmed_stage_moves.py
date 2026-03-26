"""
Example 08 — Safety-oriented voice control for stage moves and tilts.

Demonstrates:
- Push-to-talk microphone capture
- Local faster-whisper transcription
- Local confirmation gate for stage moves and tilt operations
- Feeding only confirmed commands into the Ollama MCP client
- Optional spoken reply output

Run:
    pip install "gms-mcp[ollama,voice]"
    GMS_SIMULATE=1 python examples/08_voice_confirmed_stage_moves.py
    GMS_SIMULATE=1 python examples/08_voice_confirmed_stage_moves.py --speak
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


RISK_KEYWORDS = (
    "move",
    "stage",
    "tilt",
    "alpha",
    "beta",
    "rotate",
    "x =",
    "y =",
    "z =",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voice-confirmed stage and tilt control example using local Whisper"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    parser.add_argument("--gms-url", default=os.environ.get("GMS_MCP_URL", ""))
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak the agent reply after the workflow completes",
    )
    parser.add_argument("--tts-command", default="")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--whisper-device", default=DEFAULT_WHISPER_DEVICE)
    parser.add_argument("--whisper-language", default=DEFAULT_WHISPER_LANGUAGE)
    parser.add_argument("--voice-sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--voice-max-seconds", type=float, default=DEFAULT_MAX_RECORDING_S)
    return parser.parse_args()


def _record_transcript(
    transcriber: LocalWhisperTranscriber,
    sample_rate: int,
    max_duration_s: float,
) -> str:
    audio_path = record_push_to_talk(
        sample_rate=sample_rate,
        max_duration_s=max_duration_s,
    )
    try:
        return transcriber.transcribe_file(audio_path)
    finally:
        remove_temp_audio_file(audio_path)


def _requires_confirmation(transcript: str) -> bool:
    lowered = transcript.lower()
    return any(keyword in lowered for keyword in RISK_KEYWORDS)


def _confirmed(transcript: str) -> bool:
    lowered = transcript.lower()
    return "confirm" in lowered or "yes" in lowered or "proceed" in lowered


async def main() -> None:
    args = _parse_args()

    print("\nPress Enter to start recording your microscope instruction, then press Enter again to stop.")
    print("Example prompt: 'Move the stage to x equals 100 micrometers, y equals minus 50 micrometers, then tilt alpha to minus 20 degrees.'\n")

    try:
        transcriber = LocalWhisperTranscriber(
            model_name=args.whisper_model,
            device=args.whisper_device,
            language=args.whisper_language,
        )

        transcript = _record_transcript(
            transcriber=transcriber,
            sample_rate=args.voice_sample_rate,
            max_duration_s=args.voice_max_seconds,
        )
    except VoiceDependencyError as exc:
        raise SystemExit(f"Voice dependencies are unavailable: {exc}") from exc

    print("\n─── Requested Action ───")
    print(transcript)

    if _requires_confirmation(transcript):
        print("\nThis request includes a stage move or tilt. A second confirmation is required before it will be sent to the agent.")
        print("Press Enter to record a confirmation phrase such as 'confirm move', 'yes proceed', or 'cancel'.\n")
        confirmation = _record_transcript(
            transcriber=transcriber,
            sample_rate=args.voice_sample_rate,
            max_duration_s=args.voice_max_seconds,
        )
        print("\n─── Confirmation Transcript ───")
        print(confirmation)
        if not _confirmed(confirmation):
            print("\nRequest cancelled locally. No stage or tilt command was sent to the agent.")
            return

    safe_query = (
        "Before any destructive action, restate the requested stage move or tilt, "
        "then execute it only if it remains within validated bounds. "
        f"User request: {transcript}"
    )
    result = await run_agent(
        query=safe_query,
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