"""
tests/test_core/test_fmd_net.py
================================
Tests for FMD-NET: FractalNet is a (0,D,D)-net at full n^D-point blocks.

Theorem (FMD-NET, PROVEN):
  FractalNet(n,d).generate(n^D) produces exactly one point per cell of the
  n-ary grid {0/n, 1/n, ..., (n-1)/n}^D.  Proof is a direct corollary of T1.

Note on floating point:
  Values are of the form k/n (integer over n).  Due to IEEE 754, floor(k/n * n)
  can give k-1 for some k (e.g. 5/7 * 7 = 4.999...).  Tests use rounded integer
  arithmetic (round(pt * n)) rather than floor to avoid spurious failures.

STATUS: PROVEN — direct corollary of T1 (n-ary Coordinate Bijection).
"""
from __future__ import annotations

import numpy as np
import pytest

from flu.core.fractal_net import FractalNet
from flu.core.fm_dance import index_to_coords


# ── Core theorem verification ─────────────────────────────────────────────────

@pytest.mark.parametrize("n,d", [(3, 2), (3, 3), (3, 4), (5, 2), (5, 3), (7, 2), (7, 3)])
def test_fmd_net_one_point_per_cell(n: int, d: int) -> None:
    """
    FMD-NET: every grid cell contains exactly one point.

    Grid cell (i_0,...,i_{d-1}) = [i_j/n, (i_j+1)/n) for each axis j.
    The first n^D points of FractalNet map bijectively onto these cells.
    """
    N = n ** d
    net = FractalNet(n=n, d=d)
    pts = net.generate(N)  # shape (N, d), values in [0, 1)

    # Convert to integer cell indices using rounding (handles IEEE 754)
    cells = np.round(pts * n).astype(int)

    # Each cell index must be in {0, ..., n-1}
    assert cells.min() >= 0, f"n={n} d={d}: negative cell index"
    assert cells.max() <= n - 1, f"n={n} d={d}: cell index >= n"

    # All cells must be distinct → exactly one point per cell
    cell_set = set(map(tuple, cells))
    assert len(cell_set) == N, (
        f"n={n} d={d}: {len(cell_set)} distinct cells, expected {N}"
    )


def test_fmd_net_values_in_unit_interval() -> None:
    """Points must lie in [0, 1)^d."""
    for n, d in [(3, 4), (5, 3), (7, 2)]:
        net = FractalNet(n=n, d=d)
        pts = net.generate(n ** d)
        assert (pts >= 0).all(), f"Negative value for n={n},d={d}"
        assert (pts < 1).all(),  f"Value >= 1 for n={n},d={d}"


def test_fmd_net_matches_bijection() -> None:
    """
    The cells produced must equal index_to_coords(k, n, d) / n
    (after rounding), confirming the corollary from T1.
    """
    n, d = 3, 3
    net = FractalNet(n=n, d=d)
    half = net.half
    pts = net.generate(n ** d)
    for k in range(n ** d):
        coords = index_to_coords(k, n, d)  # signed: in {-half,...,half}
        expected_unsigned = tuple(c + half for c in coords)  # in {0,...,n-1}
        actual_cell = tuple(int(round(pts[k, j] * n)) for j in range(d))
        assert actual_cell == expected_unsigned, (
            f"k={k}: actual_cell={actual_cell}, expected={expected_unsigned}"
        )


def test_fmd_net_deep_blocks() -> None:
    """
    (0,D,D)-net holds at each individual n^D block, not just the first.

    For k*n^D points (k full blocks), each block independently satisfies
    the (0,D,D)-net property.
    """
    n, d = 3, 2
    N = n ** d   # 9
    k_blocks = 4  # 36 total points
    net = FractalNet(n=n, d=d)
    pts = net.generate(N * k_blocks)

    for block in range(k_blocks):
        block_pts = pts[block * N:(block + 1) * N]
        cells = np.round(block_pts * n).astype(int)
        # Subtract the block offset to get within-block cells
        # The cells themselves may overlap between blocks in general QMC,
        # but within each block they must be distinct.
        cell_set = set(map(tuple, cells % n))
        assert len(cell_set) == N, (
            f"Block {block}: {len(cell_set)} distinct cells, expected {N}"
        )


# ── Relation to the registry ──────────────────────────────────────────────────

def test_fmd_net_in_theorem_registry() -> None:
    """FMD-NET must be registered as PROVEN in the theorem registry."""
    from flu.theory.theorem_registry import REGISTRY
    assert "FMD-NET" in REGISTRY, "FMD-NET not registered"
    th = REGISTRY["FMD-NET"]
    assert th.status == "PROVEN", f"Expected PROVEN, got {th.status}"
