"""
ollama_mcp_client.py
=====================
LangChain ReAct agent that bridges a locally-running Ollama LLM to the
GMS MCP server, enabling natural-language microscope control and analysis.

Architecture
------------
    Ollama (local LLM, port 11434)
        ↕  LangChain / langgraph ReAct agent
    MultiServerMCPClient (langchain-mcp-adapters)
        ↕  stdio subprocess OR HTTP
    gms_mcp.server  (FastMCP, simulation or live)
        ↕  DM Python API (inside GMS) or DMSimulator

Quick start
-----------
    # 1. Install and start Ollama
    #    https://ollama.ai → pull a tool-capable model:
    ollama pull qwen2.5:7b          # excellent tool-use (recommended)
    # or: ollama pull llama3.1:8b
    # or: ollama pull llama3.2

    # 2. Run the interactive session:
    python ollama_mcp_client.py

    # 3. Programmatic usage in scripts:
    python ollama_mcp_client.py --query "Acquire a HAADF STEM image at 512×512" --no-interactive

Supported Ollama models (tool-calling tested)
---------------------------------------------
    qwen2.5:7b     ★★★★★  Best tool-use, recommended
    qwen2.5:14b    ★★★★★  Higher accuracy, slower
    llama3.1:8b    ★★★★☆  Reliable, widely used
    llama3.2:3b    ★★★☆☆  Lightweight, limited context
    mistral-nemo   ★★★☆☆  Good but occasional mis-formatting
    command-r      ★★★☆☆  Decent, needs explicit prompting
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from gms_mcp import voice as voice_io

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.resolve()
_SERVER_SCRIPT = str(_HERE / "server.py")

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_GMS_MCP_URL = os.environ.get("GMS_MCP_URL", "")  # empty = use stdio

SYSTEM_PROMPT = """\
You are an expert electron microscopist and scientific AI assistant with full \
control over a transmission electron microscope running Gatan Microscopy Suite \
(GMS) 3.60.

You have access to the following MCP tools:
- gms_get_microscope_state   : Read all instrument parameters before acting.
- gms_acquire_tem_image      : TEM / HRTEM image acquisition.
- gms_acquire_stem           : STEM scanning (HAADF, BF, ABF).
- gms_acquire_4d_stem        : 4D-STEM / NBED dataset acquisition.
- gms_acquire_eels           : EELS spectrum acquisition.
- gms_acquire_diffraction    : Electron diffraction pattern acquisition.
- gms_get_stage_position     : Read stage coordinates.
- gms_set_stage_position     : Move stage (X, Y, Z, α, β).
- gms_set_beam_parameters    : Configure spot size, focus, beam shift/tilt, stigmators.
- gms_configure_detectors    : Insert/retract camera, set CCD temperature, enable detectors.
- gms_acquire_tilt_series    : Automated tomographic tilt series.
- gms_run_4dstem_analysis    : Virtual detector / DPC / COM analysis on a 4D dataset.

Operating protocol:
1. ALWAYS call gms_get_microscope_state first to confirm the current instrument state.
2. Interpret results scientifically — report d-spacings, lattice parameters, pixel \
   calibration in physically meaningful units.
3. Before any destructive move (large stage shift, large tilt), confirm with the user.
4. Report errors clearly with the diagnostic information from the tool response.
5. Suggest follow-up analyses based on the data (e.g. FFT, radial profile, EELS mapping).
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _build_mcp_connections(server_url: str = "") -> dict:
    """
    Return a MultiServerMCPClient connections dict.

    If server_url is provided, connect via HTTP (for Claude.ai / remote).
    Otherwise, launch the server as a stdio subprocess (for local Ollama).
    """
    if server_url:
        return {
            "gms": {
                "url": server_url.rstrip("/"),
                "transport": "http",
            }
        }
    else:
        return {
            "gms": {
                "command": sys.executable,
                "args": [_SERVER_SCRIPT, "--transport", "stdio"],
                "transport": "stdio",
                "env": {**os.environ, "GMS_SIMULATE": "1"},
            }
        }


def _import_ollama_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_ollama import ChatOllama
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:
        raise RuntimeError(
            "Ollama client support requires optional dependencies. Install with: "
            'pip install "nuance-gms-mcp[ollama]"'
        ) from exc

    return AIMessage, HumanMessage, MultiServerMCPClient, create_react_agent, ChatOllama


def _build_llm(model: str, base_url: str, temperature: float = 0.0) -> Any:
    """Create a ChatOllama instance configured for tool-calling."""
    _AIMessage, _HumanMessage, _MultiServerMCPClient, _create_react_agent, ChatOllama = (
        _import_ollama_stack()
    )
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=8192,           # 8k context — enough for multi-tool chains
        num_predict=2048,       # max tokens per response
    )


def _capture_voice_query(
    transcriber: voice_io.LocalWhisperTranscriber,
    sample_rate: int,
    max_duration_s: float,
) -> str:
    audio_path = voice_io.record_push_to_talk(
        sample_rate=sample_rate,
        max_duration_s=max_duration_s,
    )
    try:
        transcript = transcriber.transcribe_file(audio_path)
    finally:
        voice_io.remove_temp_audio_file(audio_path)
    print(f"\nYou (transcribed): {transcript}\n")
    return transcript


def _emit_agent_reply(answer: str, speak: bool = False, tts_command: str = "") -> None:
    print(f"\nAgent: {answer}\n")
    if not speak:
        return
    try:
        voice_io.speak_text(answer, command=tts_command)
    except Exception as exc:
        print(f"[voice] Speech output unavailable: {exc}")


# ---------------------------------------------------------------------------
# Core agent runner
# ---------------------------------------------------------------------------

async def run_agent(
    query: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    server_url: str = DEFAULT_GMS_MCP_URL,
    verbose: bool = False,
) -> dict:
    """
    Run a single query through the Ollama → MCP → GMS pipeline.

    Parameters
    ----------
    query      : Natural language instruction for the microscope agent.
    model      : Ollama model name (e.g. 'qwen2.5:7b').
    base_url   : Ollama server URL.
    server_url : GMS MCP HTTP URL; empty string = stdio subprocess.
    verbose    : Print intermediate tool calls to stdout.

    Returns
    -------
    dict with keys:
        "answer"      : str  — Final agent response.
        "tool_calls"  : list — Tool calls made during the session.
        "messages"    : list — Full message history.
    """
    AIMessage, HumanMessage, MultiServerMCPClient, create_react_agent, _ChatOllama = (
        _import_ollama_stack()
    )
    connections = _build_mcp_connections(server_url)
    llm = _build_llm(model, base_url)

    client = MultiServerMCPClient(connections)
    tools = await client.get_tools()

    if verbose:
        print(f"\n[MCP] Connected to {len(tools)} tools:")
        for t in tools:
            print(f"  • {t.name}")
        print()

    agent = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
    )

    messages = [HumanMessage(content=query)]
    result = await agent.ainvoke({"messages": messages})

    # Extract tool call history
    tool_calls = []
    final_answer = ""
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                })
                if verbose:
                    print(f"[TOOL] {tc['name']}({json.dumps(tc['args'], indent=2)})")
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", []):
            final_answer = str(msg.content)
            if verbose:
                print(f"\n[AGENT] {final_answer}\n")

    return {
        "answer":     final_answer,
        "tool_calls": tool_calls,
        "messages":   [
            {"role": type(m).__name__, "content": str(m.content)}
            for m in result["messages"]
        ],
    }


# ---------------------------------------------------------------------------
# Multi-turn interactive session
# ---------------------------------------------------------------------------

async def interactive_session(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    server_url: str = DEFAULT_GMS_MCP_URL,
    voice_mode: bool = False,
    speak: bool = False,
    tts_command: str = "",
    whisper_model: str = voice_io.DEFAULT_WHISPER_MODEL,
    whisper_device: str = voice_io.DEFAULT_WHISPER_DEVICE,
    whisper_language: str = voice_io.DEFAULT_WHISPER_LANGUAGE,
    voice_sample_rate: int = voice_io.DEFAULT_SAMPLE_RATE,
    voice_max_seconds: float = voice_io.DEFAULT_MAX_RECORDING_S,
) -> None:
    """
    Launch an interactive REPL for conversational microscope control.

    The session maintains conversation history so the agent remembers
    previous actions (e.g. "now tilt to +30° and acquire another image").
    """
    AIMessage, HumanMessage, MultiServerMCPClient, create_react_agent, _ChatOllama = (
        _import_ollama_stack()
    )
    connections = _build_mcp_connections(server_url)
    llm = _build_llm(model, base_url)

    client = MultiServerMCPClient(connections)
    tools = await client.get_tools()

    agent = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
    )
    transcriber = None
    if voice_mode:
        transcriber = voice_io.LocalWhisperTranscriber(
            model_name=whisper_model,
            device=whisper_device,
            language=whisper_language,
        )

    print("\n" + "=" * 64)
    print("  GMS Microscope Agent  (Ollama ×  MCP)")
    print(f"  Model   : {model}")
    print(f"  Server  : {'HTTP → ' + server_url if server_url else 'stdio subprocess'}")
    print(f"  Tools   : {len(tools)} available")
    print(f"  Input   : {'voice push-to-talk' if voice_mode else 'text'}")
    print("=" * 64)
    if voice_mode:
        print("Press Enter to start and stop recording. Say 'exit' to quit.\n")
    else:
        print("Type your instruction. 'exit' or Ctrl-C to quit.\n")

    history = []

    while True:
        try:
            if voice_mode:
                assert transcriber is not None
                user_input = _capture_voice_query(
                    transcriber=transcriber,
                    sample_rate=voice_sample_rate,
                    max_duration_s=voice_max_seconds,
                ).strip()
            else:
                user_input = input("You: ").strip()
        except voice_io.VoiceDependencyError as exc:
            print(f"[ERROR] Voice mode is unavailable: {exc}\n")
            break
        except RuntimeError as exc:
            print(f"[ERROR] Voice capture failed: {exc}\n")
            continue
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("[Session ended]")
            break

        history.append(HumanMessage(content=user_input))

        try:
            result = await agent.ainvoke({"messages": history})
        except Exception as e:
            print(f"[ERROR] Agent invocation failed: {e}\n")
            continue

        # Append agent messages to history for multi-turn continuity
        history.extend(result["messages"][len(history):])

        # Extract and print the final AI response
        for msg in reversed(result["messages"]):
            if (isinstance(msg, AIMessage) and msg.content
                    and not getattr(msg, "tool_calls", [])):
                _emit_agent_reply(str(msg.content), speak=speak, tts_command=tts_command)
                break


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GMS Ollama MCP Client — natural-language microscope control"
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Ollama model (default: {DEFAULT_MODEL})")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                   help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})")
    p.add_argument("--gms-url", default=DEFAULT_GMS_MCP_URL,
                   help="GMS MCP HTTP URL (empty = stdio subprocess)")
    p.add_argument("--query", default="",
                   help="Single query to run non-interactively")
    p.add_argument("--no-interactive", action="store_true",
                   help="Exit after processing --query (no REPL)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print tool calls and intermediate steps")
    p.add_argument("--output-json", action="store_true",
                   help="Print final result as JSON (for scripting)")
    p.add_argument("--voice", action="store_true",
                   help="Use push-to-talk microphone capture instead of typed input")
    p.add_argument("--speak", action="store_true",
                   help="Speak the agent reply using a local TTS command")
    p.add_argument("--tts-command", default="",
                   help="Override the local TTS command (default: auto-detect, e.g. 'say')")
    p.add_argument("--whisper-model", default=voice_io.DEFAULT_WHISPER_MODEL,
                   help=(
                       "Local faster-whisper model name for voice transcription "
                       f"(default: {voice_io.DEFAULT_WHISPER_MODEL})"
                   ))
    p.add_argument("--whisper-device", default=voice_io.DEFAULT_WHISPER_DEVICE,
                   help=(
                       "Device passed to faster-whisper "
                       f"(default: {voice_io.DEFAULT_WHISPER_DEVICE})"
                   ))
    p.add_argument("--whisper-language", default=voice_io.DEFAULT_WHISPER_LANGUAGE,
                   help=(
                       "Language hint for faster-whisper "
                       f"(default: {voice_io.DEFAULT_WHISPER_LANGUAGE})"
                   ))
    p.add_argument("--voice-sample-rate", type=int, default=voice_io.DEFAULT_SAMPLE_RATE,
                   help=f"Microphone sample rate in Hz (default: {voice_io.DEFAULT_SAMPLE_RATE})")
    p.add_argument("--voice-max-seconds", type=float,
                   default=voice_io.DEFAULT_MAX_RECORDING_S,
                   help=(
                       "Maximum duration for a push-to-talk utterance in seconds "
                       f"(default: {voice_io.DEFAULT_MAX_RECORDING_S})"
                   ))
    return p.parse_args(argv)


def _print_run_result(
    result: dict,
    verbose: bool = False,
    output_json: bool = False,
    speak: bool = False,
    tts_command: str = "",
) -> None:
    if output_json:
        print(json.dumps(result, indent=2))
        return

    _emit_agent_reply(str(result.get("answer", "")), speak=speak, tts_command=tts_command)
    if verbose:
        print(f"Tool calls: {len(result['tool_calls'])}")
        for tc in result["tool_calls"]:
            print(f"  → {tc['tool']}")


async def _main() -> None:
    args = _parse_args()

    query = args.query
    if args.voice and not query and args.no_interactive:
        try:
            transcriber = voice_io.LocalWhisperTranscriber(
                model_name=args.whisper_model,
                device=args.whisper_device,
                language=args.whisper_language,
            )
            query = _capture_voice_query(
                transcriber=transcriber,
                sample_rate=args.voice_sample_rate,
                max_duration_s=args.voice_max_seconds,
            )
        except voice_io.VoiceDependencyError as exc:
            raise SystemExit(f"Voice mode is unavailable: {exc}") from exc

    if query:
        result = await run_agent(
            query=query,
            model=args.model,
            base_url=args.ollama_url,
            server_url=args.gms_url,
            verbose=args.verbose,
        )
        _print_run_result(
            result,
            verbose=args.verbose,
            output_json=args.output_json,
            speak=args.speak,
            tts_command=args.tts_command,
        )

        if args.no_interactive:
            return

    await interactive_session(
        model=args.model,
        base_url=args.ollama_url,
        server_url=args.gms_url,
        voice_mode=args.voice,
        speak=args.speak,
        tts_command=args.tts_command,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_language=args.whisper_language,
        voice_sample_rate=args.voice_sample_rate,
        voice_max_seconds=args.voice_max_seconds,
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
