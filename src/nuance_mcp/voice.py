"""Local voice I/O for the agent loop.

Three responsibilities, each with a clean fallback path:

* **Push-to-talk recording** — :func:`record_push_to_talk` opens a
  microphone stream with :mod:`sounddevice`, captures audio between
  two Enter presses, and writes a temporary WAV file. Hard limit on
  recording length defends against a runaway capture.
* **Local transcription** — :class:`LocalWhisperTranscriber` wraps
  ``faster-whisper`` (a CTranslate2 reimplementation of OpenAI
  Whisper). It runs entirely on the workstation, supports CPU
  (``device="cpu"``) or Apple-Silicon Metal (``device="auto"``), and
  loads a model size that you choose at construction.
* **Spoken reply** — :func:`speak_text` plays back text through a
  facility-appropriate TTS path. On macOS the system ``say`` command
  is used; on Linux ``espeak``/``espeak-ng``; on Windows ``powershell
  Add-Type … Speak``. A custom ``command=`` override is accepted for
  air-gapped sites with their own TTS engine.

The whole module is **optional**: nothing here is imported by the
core server, the schema layer, or any built-in adapter. Importing it
without the ``[voice]`` extra raises :class:`VoiceDependencyError`
with a clear remediation hint.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_MAX_RECORDING_S = 60.0
DEFAULT_WHISPER_MODEL = os.environ.get(
    "NUANCE_MCP_WHISPER_MODEL", "base.en"
)
DEFAULT_WHISPER_DEVICE = os.environ.get(
    "NUANCE_MCP_WHISPER_DEVICE", "auto"
)   # "auto" | "cpu" | "cuda"
DEFAULT_WHISPER_LANGUAGE = os.environ.get(
    "NUANCE_MCP_WHISPER_LANGUAGE", "en"
)


# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------

class VoiceDependencyError(ImportError):
    """Raised when the ``[voice]`` extra is not installed correctly."""


def _require_recording_extras():
    missing = []
    try:
        import sounddevice          # noqa: F401
    except ImportError:
        missing.append("sounddevice")
    try:
        import numpy                # noqa: F401
    except ImportError:
        missing.append("numpy")
    if missing:
        raise VoiceDependencyError(
            "Microphone capture requires the [voice] extra. "
            "Install with `pip install 'nuance-mcp[voice]'`. "
            f"Missing: {', '.join(missing)}"
        )


def _require_whisper_extras():
    try:
        import faster_whisper       # noqa: F401
    except ImportError as exc:
        raise VoiceDependencyError(
            "Local transcription requires faster-whisper. "
            "Install with `pip install 'nuance-mcp[voice]'`."
        ) from exc


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_push_to_talk(
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration_s: float = DEFAULT_MAX_RECORDING_S,
    prompt: str = "Press Enter to start recording; Enter again to stop.",
) -> Path:
    """Capture audio from the system microphone, return path to WAV.

    Two-stage interaction:

    1. Print ``prompt`` and wait for Enter (signals "start").
    2. Open the input stream and keep capturing until either the user
       presses Enter again **or** ``max_duration_s`` elapses.
    """
    _require_recording_extras()
    import numpy as np
    import sounddevice as sd

    print(prompt, flush=True)
    try:
        input()
    except EOFError:
        raise RuntimeError("stdin closed; cannot push-to-talk")

    frames: list = []
    stop_at = time.monotonic() + max_duration_s

    def _on_audio(indata, frames_count, time_info, status):
        if status:
            print(f"  [voice] status: {status}", file=sys.stderr)
        frames.append(indata.copy())

    print("  recording … press Enter to stop", flush=True)
    with sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype="int16", callback=_on_audio):
        try:
            # Spawn a tiny background thread that reads stdin so we can
            # interrupt the recording without blocking the audio
            # callback. fall back to polling time if stdin isn't a tty.
            import threading
            stop_event = threading.Event()

            def _waiter():
                try:
                    input()
                except EOFError:
                    pass
                stop_event.set()

            t = threading.Thread(target=_waiter, daemon=True)
            t.start()
            while not stop_event.is_set() and time.monotonic() < stop_at:
                sd.sleep(50)
        except KeyboardInterrupt:
            pass

    if not frames:
        raise RuntimeError("no audio captured")

    audio = np.concatenate(frames, axis=0)
    out = Path(tempfile.mkstemp(prefix="nuance-mcp-", suffix=".wav")[1])
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return out


def remove_temp_audio_file(path: Path) -> None:
    """Best-effort cleanup of a recording WAV file."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class LocalWhisperTranscriber:
    """Lazy-loading wrapper around :mod:`faster_whisper`.

    Parameters
    ----------
    model_name
        Whisper model size: ``"tiny"``, ``"base"``, ``"base.en"``,
        ``"small"``, ``"medium"``, ``"large-v3"``, ``"large-v3-turbo"``.
        Default ``"base.en"`` is a good CPU-friendly trade-off.
    device
        ``"auto"`` (default; CUDA if available, Metal on Apple Silicon,
        else CPU), ``"cpu"``, or ``"cuda"``.
    language
        ISO language hint (e.g. ``"en"``). Pass ``None`` to let Whisper
        auto-detect.
    compute_type
        ``faster-whisper`` precision: ``"int8"`` (fast / CPU-friendly),
        ``"float16"`` (GPU), or ``"int8_float16"``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        device: str = DEFAULT_WHISPER_DEVICE,
        language: Optional[str] = DEFAULT_WHISPER_LANGUAGE,
        compute_type: str = "int8",
    ) -> None:
        _require_whisper_extras()
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.device = device
        self.language = language
        self.compute_type = compute_type
        self._model = WhisperModel(
            model_name, device=device, compute_type=compute_type,
        )

    def transcribe_file(self, wav_path) -> str:
        segments, _info = self._model.transcribe(
            str(wav_path),
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        return "".join(seg.text for seg in segments).strip()


# ---------------------------------------------------------------------------
# Speech output
# ---------------------------------------------------------------------------

def speak_text(text: str, *, command: str = "") -> None:
    """Speak ``text`` through the platform-appropriate TTS.

    The default for each OS is summarised below. Callers can override
    with ``command=``, which is interpreted as a template containing
    ``{text}``::

        speak_text("hello", command="my-tts --say {text}")
    """
    text = (text or "").strip()
    if not text:
        return

    if command:
        formatted = command.format(text=_shell_quote(text))
        subprocess.run(formatted, shell=True, check=False)
        return

    system = platform.system()
    if system == "Darwin" and shutil.which("say"):
        subprocess.run(["say", text], check=False)
        return

    if system == "Linux":
        for exe in ("espeak-ng", "espeak"):
            if shutil.which(exe):
                subprocess.run([exe, text], check=False)
                return

    if system == "Windows":
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
            f".Speak('{text.replace(chr(39), chr(39)*2)}')"
        )
        subprocess.run(["powershell", "-Command", ps], check=False)
        return

    raise RuntimeError(
        f"No TTS backend available on {system!r}. "
        "Pass command='your-tts {text}' to speak_text() to override."
    )


def _shell_quote(text: str) -> str:
    """Minimal shell quoting for the ``command=`` template."""
    if "'" not in text:
        return f"'{text}'"
    return '"' + text.replace('"', r"\"") + '"'
