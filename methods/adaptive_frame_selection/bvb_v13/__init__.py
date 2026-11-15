"""BVB V12: current seven-task integration, with isolated bundled dependencies."""
from pathlib import Path as _Path
import sys as _sys

__version__ = "13.0.0"
_RUNTIME = _Path(__file__).resolve().parent / "_runtime"
# Resolve V12's pinned adapters before the workspace's historical packages.
if str(_RUNTIME) not in _sys.path:
    _sys.path.insert(0, str(_RUNTIME))
