"""Shared pytest fixtures."""

import os
import sys

# Ensure the in-repo src/ is importable for development without `pip install`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
