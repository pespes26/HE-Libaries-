"""Capture environment information for reproducibility (saved into the run config)."""
from __future__ import annotations

import multiprocessing
import platform
import subprocess
import sys


def _cpu_model() -> str:
    """Best-effort exact CPU model name.

    `platform.processor()` returns only the architecture ("x86_64") on Linux, which
    does not identify the machine a benchmark ran on; read the real model name instead.
    """
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def capture() -> dict:
    """Return a dict of interpreter, OS, CPU (model + cores + RAM), and library versions."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_model": _cpu_model(),
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
    try:
        import psutil
        info["memory_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        info["memory_total_gb"] = None
    return info
