"""Tests for the measurement layer itself (no TenSEAL needed).

The benchmark's numbers are only as good as `repeat`/`MemorySampler`, so their
contracts are pinned here: warm-up discarding, batch sizing for sub-millisecond
calls, return-value passthrough, and sane std/peak semantics.
"""
from __future__ import annotations

import time

from he_benchmark import metrics


def test_repeat_discards_warmup_and_runs_n_samples():
    """A slow (>= min_sample_s) fn keeps k=1: calls = warmup + probe + n."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        time.sleep(0.002)  # 2 ms >= the 1 ms batching threshold -> no batching
        return calls["n"]

    t, last = metrics.repeat(fn, n=5, warmup=1)
    assert t.calls_per_sample == 1
    assert calls["n"] == 1 + 1 + 5  # warmup + probe + n samples
    assert last == calls["n"]  # the value produced by the final run is returned
    assert len(t.runs) == 5
    assert t.mean >= 0.002


def test_repeat_batches_submillisecond_calls():
    """A microsecond-scale fn must be looped k>1 times per sample (noise floor)."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return calls["n"]

    t, _ = metrics.repeat(fn, n=3, warmup=1)
    assert t.calls_per_sample > 1
    assert calls["n"] == 1 + 1 + 3 * t.calls_per_sample
    assert t.mean < 1e-3  # per-call mean, not per-batch total


def test_repeat_single_sample_has_zero_std():
    t, _ = metrics.repeat(lambda: time.sleep(0.002), n=1, warmup=0)
    assert t.std == 0.0  # ddof=1 is undefined for one sample; pinned to 0


def test_memory_sampler_peak_and_delta_are_sane():
    with metrics.MemorySampler(interval=0.001) as s:
        block = bytearray(8 * 1024 * 1024)  # touch 8 MB so RSS has something to see
        block[::4096] = b"x" * len(block[::4096])
    assert s.peak_mb >= s.baseline_mb
    assert s.delta_mb >= 0.0
    del block
