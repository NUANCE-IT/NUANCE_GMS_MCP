"""Adapter discovery.

Adapters self-register through Python entry points under the
``nuance_mcp.adapters`` group::

    [project.entry-points."nuance_mcp.adapters"]
    gatan      = "nuance_mcp.adapters.gatan:GatanGMSAdapter"
    jeol       = "nuance_mcp.adapters.jeol:JEOLAdapter"
    hitachi    = "nuance_mcp.adapters.hitachi:HitachiAdapter"
    simulator  = "nuance_mcp.core.simulator:SimulatorAdapter"

The CLI uses :func:`load_adapter` to resolve a backend by name. Third-party
adapters can be installed without touching this package.
"""

from __future__ import annotations

import importlib
from typing import Type

from ..core.adapter import MicroscopeAdapter

# Built-in adapters (importable regardless of installed extras; vendor-only
# imports happen lazily inside each module).
_BUILTIN = {
    "simulator": "nuance_mcp.core.simulator:SimulatorAdapter",
    "gatan":     "nuance_mcp.adapters.gatan:GatanGMSAdapter",
    "jeol":      "nuance_mcp.adapters.jeol:JEOLAdapter",
    "hitachi":   "nuance_mcp.adapters.hitachi:HitachiAdapter",
}


def load_adapter(name: str) -> Type[MicroscopeAdapter]:
    """Resolve an adapter name to its class. Lazy import on demand."""

    try:
        # Entry points first (third-party adapters take precedence)
        from importlib.metadata import entry_points
        eps = entry_points(group="nuance_mcp.adapters")
        for ep in eps:
            if ep.name == name:
                return ep.load()
    except Exception:
        pass

    if name not in _BUILTIN:
        raise ValueError(
            f"unknown adapter {name!r}; built-in choices: "
            + ", ".join(_BUILTIN)
        )
    module_path, _, class_name = _BUILTIN[name].partition(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def available_adapters() -> list[str]:
    return list(_BUILTIN.keys())
