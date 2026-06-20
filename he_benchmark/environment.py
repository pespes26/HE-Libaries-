"""Capture environment information for reproducibility (saved into the run config)."""
from __future__ import annotations

import multiprocessing
import platform
import sys


def capture() -> dict:
    """Return a dict of interpreter, OS, CPU, and key library versions."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
    }
    for mod in ("tenseal", "numpy", "pandas", "cryptography", "psutil", "matplotlib"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = "not installed"
    try:
        info["cpu_count"] = multiprocessing.cpu_count()
    except Exception:
        info["cpu_count"] = None
    return info
