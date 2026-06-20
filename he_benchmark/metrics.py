"""Measurement helpers: wall-clock timing, peak-memory sampling, repeat-runner.

Memory note: TenSEAL ciphertexts are allocated in C++ and are invisible to
`tracemalloc`, which tracks only Python-level allocations and would massively
under-report. We therefore sample the whole-process Resident Set Size (RSS) via
psutil to capture true peak memory during an operation.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import psutil

_PROCESS = psutil.Process()


def _rss_mb() -> float:
    return _PROCESS.memory_info().rss / (1024 * 1024)


class MemorySampler:
    """Context manager that records peak process RSS (MB) while a block runs.

    Usage:
        with MemorySampler() as s:
            ... work ...
        peak, delta = s.peak_mb, s.delta_mb

    Caveat: this is a whole-process RSS delta. The baseline captured on entry already
    includes residue from prior operations (the allocator rarely returns freed pages to
    the OS even after gc.collect()), so per-operation deltas are noisy and not strictly
    comparable across operations. Treat `delta_mb` as indicative only; the authoritative
    size/memory signal in this project is serialized ciphertext size (he_ckks).
    """

    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mb = 0.0
        self.peak_mb = 0.0

    def _run(self) -> None:
        peak = self.baseline_mb
        while not self._stop.is_set():
            peak = max(peak, _rss_mb())
            time.sleep(self.interval)
        peak = max(peak, _rss_mb())  # final sample after stop
        self.peak_mb = peak

    def __enter__(self) -> "MemorySampler":
        self.baseline_mb = _rss_mb()
        self.peak_mb = self.baseline_mb
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def delta_mb(self) -> float:
        """Peak RSS minus the baseline captured on entry (never negative)."""
        return max(0.0, self.peak_mb - self.baseline_mb)


@dataclass
class TimingResult:
    """Aggregated wall-clock timing across repeated runs (warm-up excluded)."""
    mean: float
    std: float
    runs: list = field(default_factory=list)


def repeat(fn: Callable[[], object], n: int, warmup: int = 1):
    """Run `fn` warmup+n times; time each of the last n with perf_counter.

    Returns (TimingResult, last_return_value). The return value lets callers reuse
    the actual object produced by the final run (e.g. a ciphertext) for size and
    correctness checks.
    """
    last = None
    for _ in range(max(0, warmup)):
        last = fn()
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return TimingResult(mean=float(arr.mean()), std=float(arr.std(ddof=0)), runs=times), last
