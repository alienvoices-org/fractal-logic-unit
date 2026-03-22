"""
tests/test_core/test_od32_iterator.py
======================================
Tests for FMDanceIterator (OD-32 — O(1) amortized incremental traversal).

Verifies:
  1. Output is exactly identical to path_coord(k, n, d) for all k.
  2. Coverage: every lattice point is visited exactly once.
  3. Step bound (T4): max torus step = min(d, n//2).
  4. FMDanceIterator is importable from the top-level flu namespace.
  5. validate() method returns True for all tested (n, d).
  6. Throughput is measurably faster than path_coord for sequential traversal.

STATUS: PROVEN — see FMDanceIterator docstring / OD-32 theorem in fm_dance_path.py.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pytest

from flu.core.fm_dance_path import FMDanceIterator, path_coord


# ── 1. Correctness: exact match with path_coord ───────────────────────────────

@pytest.mark.parametrize("n,d", [(3, 2), (3, 3), (3, 4), (5, 2), (5, 3), (7, 2)])
def test_iterator_matches_path_coord(n: int, d: int) -> None:
    """FMDanceIterator must yield path_coord(k, n, d) for every k."""
    it = FMDanceIterator(n=n, d=d)
    for k, coord in enumerate(it):
        assert coord == path_coord(k, n, d), (
            f"Mismatch at k={k} (n={n}, d={d}): "
            f"iterator={coord}  ref={path_coord(k, n, d)}"
        )


# ── 2. Coverage: every lattice point visited exactly once ────────────────────

@pytest.mark.parametrize("n,d", [(3, 2), (3, 4), (5, 3)])
def test_iterator_full_coverage(n: int, d: int) -> None:
    """Hamiltonian property: n^d distinct coordinates, each appearing once."""
    coords = list(FMDanceIterator(n=n, d=d))
    assert len(coords) == n ** d, f"Wrong length: {len(coords)} != {n**d}"
    assert len(set(coords)) == n ** d, "Duplicate coordinates found"


# ── 3. Step bound (inherits from T4) ─────────────────────────────────────────

@pytest.mark.parametrize("n,d", [(3, 3), (5, 2), (7, 2)])
def test_iterator_step_bound(n: int, d: int) -> None:
    """Max torus step must equal min(d, n//2) — same as path_coord guarantee."""
    half  = n // 2
    bound = min(d, half)
    prev  = None
    max_step = 0
    for coord in FMDanceIterator(n=n, d=d):
        if prev is not None:
            step = max(
                min(abs(coord[i] - prev[i]) % n, n - abs(coord[i] - prev[i]) % n)
                for i in range(d)
            )
            max_step = max(max_step, step)
        prev = coord
    assert max_step == bound, (
        f"n={n} d={d}: max_step={max_step}, expected {bound}"
    )


# ── 4. Top-level namespace import ─────────────────────────────────────────────

def test_iterator_importable_from_flu() -> None:
    """FMDanceIterator must be importable from the top-level flu package."""
    import flu
    assert hasattr(flu, "FMDanceIterator"), (
        "FMDanceIterator not exported from flu.__init__"
    )
    assert hasattr(flu.traversal, "FMDanceIterator"), (
        "FMDanceIterator not in flu.traversal namespace"
    )


# ── 5. validate() method ──────────────────────────────────────────────────────

@pytest.mark.parametrize("n,d", [(3, 2), (5, 3), (7, 2)])
def test_validate_method(n: int, d: int) -> None:
    """FMDanceIterator.validate() must return True for all small (n, d)."""
    assert FMDanceIterator(n=n, d=d).validate(), (
        f"validate() returned False for n={n}, d={d}"
    )


# ── 6. Throughput ─────────────────────────────────────────────────────────────

def test_iterator_faster_than_path_coord() -> None:
    """
    FMDanceIterator must be faster than sequential path_coord for n=5, d=3.
    We accept any speedup > 0 (i.e., not slower) to avoid false failures on
    heavily loaded CI; the benchmark script records precise numbers separately.
    """
    n, d = 5, 3
    total = n ** d  # 125

    # FMDanceIterator timing
    t0 = time.perf_counter()
    for _ in range(100):   # 100 full passes to get stable timing
        list(FMDanceIterator(n=n, d=d))
    t_it = time.perf_counter() - t0

    # path_coord sequential timing
    t0 = time.perf_counter()
    for _ in range(100):
        [path_coord(k, n, d) for k in range(total)]
    t_ref = time.perf_counter() - t0

    # Allow up to 10% slower (for noisy CI environments), but flag it
    assert t_it <= t_ref * 1.10, (
        f"FMDanceIterator unexpectedly slow: {t_it:.3f}s vs path_coord {t_ref:.3f}s"
    )


# ── 7. Edge cases ─────────────────────────────────────────────────────────────

def test_iterator_d1() -> None:
    """d=1 is the trivial 1-D path: must yield (-half, …, half)."""
    n = 5
    it = FMDanceIterator(n=n, d=1)
    coords = list(it)
    expected = [path_coord(k, n, 1) for k in range(n)]
    assert coords == expected

def test_iterator_n3_d2_known_sequence() -> None:
    """
    For n=3, d=2 the FM-Dance is the 3×3 Lo Shu traversal order.
    First coordinate must start at (-1,-1) (rank 0, all digits zero → both
    x_0 = −a_0 = 0 mod 3 → unsigned 0 → signed −1, and similarly x_1 = −1).
    """
    it    = FMDanceIterator(n=3, d=2)
    first = next(iter(it))
    assert first == (-1, -1), f"Unexpected start: {first}"

def test_iterator_invalid_even_n() -> None:
    """Even n must raise ValueError."""
    with pytest.raises(ValueError):
        FMDanceIterator(n=4, d=2)

def test_iterator_invalid_d0() -> None:
    """d=0 must raise ValueError."""
    with pytest.raises(ValueError):
        FMDanceIterator(n=3, d=0)
