"""Generic live-processing job lifecycle.

The lifecycle is owned by the *server*: when an adapter does not declare the
``LIVE_JOBS`` capability, the server falls back to a local
:class:`JobRegistry` that performs the job in-process against the simulator
or the adapter's read-only methods (radial profile, FFT, etc.). When an
adapter advertises ``LIVE_JOBS`` (typically because a host-process bridge
maintains live derived-image references in the vendor GUI), the server
forwards lifecycle calls straight to the adapter.

This keeps the (start, status, result, stop) contract identical from the
agent's perspective regardless of where state actually lives.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RESULT_READY = "result_ready"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    state: JobState = JobState.PENDING
    iterations: int = 0
    started_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    error: Optional[str] = None
    latest: Any = None              # latest derived payload
    history: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_update = time.time()

    def summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "state": self.state.value,
            "iterations": self.iterations,
            "started_at": self.started_at,
            "last_update": self.last_update,
            "error": self.error,
        }


class JobRegistry:
    """Thread-safe in-process live-job registry used when the adapter
    does not own job state itself."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stops: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # --- public api ---------------------------------------------------

    def new_job(self, job_type: str, params: dict) -> JobRecord:
        with self._lock:
            jid = f"job-{uuid.uuid4().hex[:8]}"
            rec = JobRecord(job_id=jid, job_type=job_type, params=params)
            self._jobs[jid] = rec
            self._stops[jid] = threading.Event()
            return rec

    def start_thread(self, rec: JobRecord, target, *args, **kwargs) -> None:
        rec.state = JobState.RUNNING
        rec.touch()
        t = threading.Thread(
            target=target, args=(rec, self._stops[rec.job_id]) + tuple(args),
            kwargs=kwargs, daemon=True, name=f"livejob-{rec.job_id}",
        )
        self._threads[rec.job_id] = t
        t.start()

    def status(self, job_id: str) -> dict:
        rec = self._get(job_id)
        return rec.summary()

    def result(self, job_id: str) -> Any:
        rec = self._get(job_id)
        return rec.latest

    def stop(self, job_id: str) -> dict:
        rec = self._get(job_id)
        stop = self._stops.get(job_id)
        if stop is not None:
            stop.set()
        rec.state = JobState.STOPPED
        rec.touch()
        return rec.summary()

    def all(self) -> list[dict]:
        with self._lock:
            return [r.summary() for r in self._jobs.values()]

    # --- internals ----------------------------------------------------

    def _get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job_id: {job_id}")
            return self._jobs[job_id]
