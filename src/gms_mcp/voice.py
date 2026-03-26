from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from contextlib import suppress
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_MAX_RECORDING_S = 30.0
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
DEFAULT_WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
DEFAULT_WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")


class VoiceDependencyError(RuntimeError):
    """Raised when optional local voice dependencies are unavailable."""


def _import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise VoiceDependencyError(
            "Microphone capture requires 'sounddevice'. Install with: "
            'pip install "nuance-gms-mcp[voice]"'
        ) from exc
    return sd


def _import_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise VoiceDependencyError(
            "Local transcription requires 'faster-whisper'. Install with: "
            'pip install "nuance-gms-mcp[voice]"'
        ) from exc
    return WhisperModel


def _default_tts_command() -> list[str] | None:
    if sys.platform == "darwin" and shutil.which("say"):
        return ["say"]
    if shutil.which("spd-say"):
        return ["spd-say"]
    if shutil.which("espeak"):
        return ["espeak"]
    return None


def _write_wav_file(audio: np.ndarray, sample_rate: int, output_path: Path) -> Path:
    pcm = np.asarray(np.clip(audio, -1.0, 1.0) * 32767.0, dtype=np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output_path


def record_push_to_talk(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration_s: float = DEFAULT_MAX_RECORDING_S,
    prompt: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> Path:
    """Capture a push-to-talk utterance and return a temporary WAV path."""
    sd = _import_sounddevice()
    chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            output(f"[voice] audio status: {status}")
        chunks.append(np.copy(indata[:, 0]))

    def wait_for_stop() -> None:
        with suppress(EOFError):
            prompt("Press Enter to stop recording...")
        stop_event.set()

    output("[voice] Press Enter to start recording.")
    with suppress(EOFError):
        prompt("")
    output("[voice] Recording... speak now.")

    stopper = threading.Thread(target=wait_for_stop, daemon=True)
    deadline = time.time() + max_duration_s
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        stopper.start()
        while not stop_event.is_set():
            if time.time() >= deadline:
                output(
                    f"[voice] Reached {max_duration_s:.1f} s maximum duration; stopping."
                )
                stop_event.set()
                break
            sd.sleep(100)

    if not chunks:
        raise RuntimeError("No audio was captured from the microphone.")

    fd, temp_path = tempfile.mkstemp(prefix="gms_mcp_voice_", suffix=".wav")
    os.close(fd)
    audio = np.concatenate(chunks, axis=0)
    return _write_wav_file(audio, sample_rate, Path(temp_path))


class LocalWhisperTranscriber:
    """Cache a faster-whisper model for repeated local transcriptions."""

    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        device: str = DEFAULT_WHISPER_DEVICE,
        language: str = DEFAULT_WHISPER_LANGUAGE,
    ) -> None:
        whisper_model_cls = _import_whisper_model()
        self._model = whisper_model_cls(
            model_name,
            device=device,
            compute_type="default",
        )
        self.language = language

    def transcribe_file(self, audio_path: str | Path) -> str:
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self.language or None,
            vad_filter=True,
        )
        transcript = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
        if not transcript:
            raise RuntimeError("Whisper returned an empty transcript.")
        return transcript


def transcribe_audio_file(
    audio_path: str | Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_WHISPER_DEVICE,
    language: str = DEFAULT_WHISPER_LANGUAGE,
) -> str:
    transcriber = LocalWhisperTranscriber(
        model_name=model_name,
        device=device,
        language=language,
    )
    return transcriber.transcribe_file(audio_path)


def speak_text(text: str, command: str = "") -> None:
    if not text.strip():
        return
    command_parts = command.split() if command else _default_tts_command()
    if not command_parts:
        raise VoiceDependencyError(
            "No local text-to-speech command found. On macOS, 'say' is supported "
            "automatically."
        )
    subprocess.run([*command_parts, text], check=True)


def remove_temp_audio_file(audio_path: str | Path) -> None:
    path = Path(audio_path)
    if path.exists():
        path.unlink()
