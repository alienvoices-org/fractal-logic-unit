"""
tests/test_core/test_hypothesis_deterministic.py
==================================================
Deterministic equivalents of the hypothesis property tests.
Runs exhaustively over the (n, d) grid, giving full coverage
when hypothesis is not installed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
from flu.core.fm_dance import index_to_coords, coords_to_index
from flu.theory.theory_latin import (
    verify_constant_line_sum, verify_holographic_repair, line_sum_constant
)
from flu.core.n_ary import nary_step_bound

# Safe grid matching hypothesis strategies (n^d <= 500)
SAFE_ND = [(n, d) for n in [3, 5, 7, 9, 11] for d in range(2, 5) if n**d <= 500]

def _value_hyperprism(n, d):
    half = n // 2
    arr  = np.zeros([n]*d, dtype=float)
    for idx in np.ndindex(*[n]*d):
        arr[idx] = (sum(idx) % n) - half
    return arr


def test_bijection_round_trip():
    """T1: k → coords → k is identity for all k in [0, n^d)."""
    for n, d in SAFE_ND:
        for k in range(n**d):
            c  = index_to_coords(k, n, d)
            k2 = coords_to_index(c, n, d)
            assert k2 == k, f"T1 n={n},d={d}: k={k} → {c} → {k2}"


def test_latin_property_arbitrary_n_d():
    """T3: The value hyperprism is Latin on every axis."""
    for n, d in SAFE_ND:
        arr = _value_hyperprism(n, d)
        for axis in range(d):
            for fixed in np.ndindex(*([n]*(d-1))):
                idx = [slice(None)] * d
                for fi, fv in enumerate(fixed):
                    ta = fi if fi < axis else fi + 1
                    idx[ta] = fv
                sl = arr[tuple(idx)].flatten()
                assert len(set(sl.tolist())) == n, (
                    f"T3 Latin n={n},d={d},axis={axis},fixed={fixed}: not a permutation"
                )


def test_line_sum_zero():
    """L1: All axis line sums of the signed value hyperprism are zero."""
    for n, d in SAFE_ND:
        arr = _value_hyperprism(n, d)
        r   = verify_constant_line_sum(arr, n, signed=True)
        assert r["line_sum_ok"], f"L1 n={n},d={d}: {r}"


def test_holographic_repair_arbitrary_coord():
    """L2: Single-point recovery from line sums works for all (n,d)."""
    for n, d in SAFE_ND:
        arr = _value_hyperprism(n, d)
        r   = verify_holographic_repair(arr, n, signed=True)
        assert r["repair_ok"], f"L2 n={n},d={d}: {r}"


def test_hamiltonian_coverage():
    """T2: FM-Dance visits every lattice point exactly once."""
    for n, d in SAFE_ND:
        total   = n**d
        visited = set()
        for k in range(total):
            c = index_to_coords(k, n, d)
            assert c not in visited, f"T2 n={n},d={d},k={k}: duplicate {c}"
            visited.add(c)
        assert len(visited) == total


def test_step_bound_theorem_t4():
    """T4: Every step satisfies ||Δ||_∞ ≤ ⌊n/2⌋."""
    for n, d in SAFE_ND:
        bound = nary_step_bound(n, d)
        for k in range(n**d - 1):
            c0 = index_to_coords(k, n, d)
            c1 = index_to_coords(k+1, n, d)
            step = max(min(abs(int(a)-int(b)), n-abs(int(a)-int(b))) for a, b in zip(c0, c1))
            assert step <= bound, f"T4 n={n},d={d},k={k}: step={step} > {bound}"


def test_nary_bijection_n_ary1():
    """N-ARY-1: The n-ary FM-Dance is bijective for all odd n."""
    for n, d in SAFE_ND:
        if n % 2 == 0: continue
        total = n**d
        seen  = set()
        for k in range(total):
            c = index_to_coords(k, n, d)
            assert c not in seen, f"N-ARY-1 n={n},d={d},k={k}: duplicate"
            seen.add(c)
        assert len(seen) == total
