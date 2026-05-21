"""Example 07 — Voice-driven acquisition through a local LLM agent.

Demonstrates how to pipe a transcribed voice command into the MCP tool
surface. The transcription path uses ``faster-whisper`` (CPU- or
Metal-accelerated, local) and the planning step uses Ollama through
``langchain-mcp-adapters``.

Both extras are optional::

    pip install "nuance-mcp[voice,ollama,gatan]"

Run:
    # Type a sentence instead of recording (smoke test, no microphone):
    python examples/07_voice_acquisition.py \\
        --transcript "Check microscope state, acquire a 256 by 256 HAADF \\
                       STEM image at 5 microseconds dwell time, and report \\
                       the mean intensity."

    # Push-to-talk:
    python examples/07_voice_acquisition.py --adapter gatan
"""

from __future__ import annotations
import argparse, asyncio, os, sys

from _common import make_server, banner


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adapter",
        default="simulator",
        choices=["simulator", "gatan", "jeol", "hitachi"],
    )
    p.add_argument("--mode", default=None)
    p.add_argument(
        "--transcript", default="", help="Skip recording; use this string instead."
    )
    p.add_argument("--ollama-model", default="qwen2.5:7b")
    p.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    p.add_argument(
        "--speak",
        action="store_true",
        help="Speak the agent reply (macOS 'say' by default).",
    )
    return p.parse_args()


async def main() -> None:
    args = _parse()
    server = make_server(args)

    if args.transcript:
        transcript = args.transcript
    else:
        try:
            from nuance_mcp.voice import (
                LocalWhisperTranscriber,
                record_push_to_talk,
                remove_temp_audio_file,
            )
        except ImportError as exc:
            sys.exit(
                f"voice extras unavailable: {exc}. pip install 'nuance-mcp[voice]'"
            )
        print("Press Enter to start recording, Enter again to stop.")
        audio = record_push_to_talk()
        transcript = LocalWhisperTranscriber().transcribe_file(audio)
        remove_temp_audio_file(audio)

    banner("Transcript")
    print(transcript)

    # Ollama + LangGraph ReAct agent
    try:
        from nuance_mcp.agent import run_agent
    except ImportError:
        sys.exit("ollama extras unavailable; pip install 'nuance-mcp[ollama]'")

    answer = await run_agent(
        server, transcript, model=args.ollama_model, base_url=args.ollama_url
    )
    banner("Agent reply")
    print(answer)

    if args.speak:
        try:
            from nuance_mcp.voice import speak_text

            speak_text(answer)
        except Exception as exc:
            print(f"[voice] speech output skipped: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
