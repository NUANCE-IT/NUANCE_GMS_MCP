"""
gms-mcp — Model Context Protocol server for Gatan Microscopy Suite 3.60.

Authors: Roberto dos Reis, Vinayak P. Dravid
         NUANCE Center, Northwestern University
License: MIT
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nuance-gms-mcp")
except PackageNotFoundError:
    __version__ = "0.1.2-dev"

__all__ = ["__version__"]
