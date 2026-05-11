"""Bridge plugin that runs inside the GMS Python environment.

This file replaces ``gms_mcp.dm_plugin`` from the v0.1 layout. It runs as a
daemon thread within the live ``DigitalMicrograph`` process and serves the
JSON contract documented in ``docs/spec/nuance-mcp-bridge-1.0.md``. The
contract is now versioned (``v: "nuance-mcp-bridge/1.0"``) so other
adapters (JEOL, Hitachi) can reuse it if their vendor APIs are also
host-process bound.

Default binding is loopback (``tcp://127.0.0.1:5555``). Override via
``GMS_MCP_ZMQ_BIND`` when, and only when, the bridge needs to be reachable
from another machine; in that case the operator is responsible for the
firewall, allow-list, and reverse-proxy guards.
"""
from __future__ import annotations

import base64
import json
import os
import threading
from typing import Any

try:
    import DigitalMicrograph as DM
except ImportError as exc:                                       # pragma: no cover
    raise RuntimeError(
        "nuance_mcp.adapters.gatan.bridge must run inside the GMS "
        "Python environment. `import DigitalMicrograph` failed."
    ) from exc

try:
    import zmq
except ImportError as exc:                                       # pragma: no cover
    raise RuntimeError(
        "pyzmq is required inside the GMS environment. "
        "Install it with `pip install pyzmq --break-system-packages`."
    ) from exc

import numpy as np

_DEFAULT_BIND = "tcp://127.0.0.1:5555"
ZMQ_BIND = os.environ.get("GMS_MCP_ZMQ_BIND", _DEFAULT_BIND)

_thread: threading.Thread | None = None
_stop = threading.Event()


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, dict): return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_jsonable(v) for v in obj]
    return obj


def _dispatch(method: str, params: dict) -> dict:
    """Translate a generic method name into a DM API call.

    Method names mirror :class:`MicroscopeAdapter` exactly.
    """
    # The full dispatcher table lives in the v0.1 dm_plugin; the skeleton
    # below shows the pattern for two representative methods so the structure
    # is clear. The actual migration copies the existing dispatcher and
    # renames functions to the generic names.
    if method == "hello":
        return {"server": "nuance-mcp-bridge", "version": "1.0",
                "vendor": "Gatan", "model": "GMS 3.60"}

    if method == "get_microscope_state":
        ms = DM.GetTagGroup
        # ... (real implementation copied from v0.1.dm_plugin) ...
        return {"high_tension_kV": None, "mode": "TEM"}

    if method == "acquire_tem":
        cam = DM.GetActiveCamera()
        prm = DM.CM_CreateAcquisitionParameters_FullCCD(
            params["processing"], params["exposure_s"], params["binning"],
            params["binning"],
        )
        img = DM.CM_AcquireImage(cam, prm)
        img.ShowImage()
        arr = img.GetNumArray()
        return {
            "name": img.GetName(),
            "shape": list(arr.shape),
            "data_dtype": str(arr.dtype),
            "data_b64": base64.b64encode(arr.tobytes()).decode(),
            "calibration": {"scale": img.GetDimensionCalibration(0, 0)[1],
                            "unit": "nm"},
            "metadata": {"exposure_s": params["exposure_s"]},
        }

    raise KeyError(f"unknown method: {method}")


def _serve() -> None:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(ZMQ_BIND)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    while not _stop.is_set():
        try:
            DM.DoEvents()
        except Exception:
            pass
        socks = dict(poller.poll(100))
        if sock in socks:
            try:
                msg = json.loads(sock.recv_string())
                method = msg.get("method")
                params = msg.get("params", {})
                result = _dispatch(method, params)
                sock.send_string(json.dumps({"status": "ok",
                                              "result": _to_jsonable(result)}))
            except Exception as exc:                              # pragma: no cover
                sock.send_string(json.dumps({"status": "error",
                                              "error": str(exc)}))
    sock.close(0)


def start_bridge() -> None:
    """Start the bridge in a daemon thread inside the GMS Python interpreter."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_serve, daemon=True,
                               name="nuance-mcp-bridge")
    _thread.start()
    DM.OkDialog(f"nuance-mcp bridge listening on {ZMQ_BIND}")


def stop_bridge() -> None:
    _stop.set()
