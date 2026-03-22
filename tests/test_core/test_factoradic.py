"""
Tests for flu.core.factoradic — Lehmer-code unranking + FM-Dance bridge.

ITER-3A: factoradic_to_fm_coords / fm_coords_to_factoradic
"""

import math
import pytest
import numpy as np

from flu.core.factoradic import (
    factoradic_unrank,
    factoradic_rank,
    arrow_generator,
    factoradic_to_fm_coords,
    fm_coords_to_factoradic,
    ArrowStep,
)


# ── factoradic_unrank (existing) ──────────────────────────────────────────────

class TestFactoradicUnrank:
    def test_pivot_at_centre(self):
        arrow = factoradic_unrank(0, 5, signed=True, pivot=0)
        assert arrow[2] == 0
        assert sorted(arrow.tolist()) == [-2, -1, 0, 1, 2]

    def test_distinct_k_distinct_arrow(self):
        a0 = factoradic_unrank(0, 5, pivot=0)
        a1 = factoradic_unrank(1, 5, pivot=0)
        assert not np.array_equal(a0, a1)

    def test_all_arrows_unique(self):
        n = 4
        arrows = [tuple(factoradic_unrank(k, n, pivot=0)) for k in range(math.factorial(n - 1))]
        assert len(set(arrows)) == math.factorial(n - 1)

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError):
            factoradic_unrank(math.factorial(3), 4, pivot=0)

    def test_invalid_pivot_raises(self):
        with pytest.raises(ValueError):
            factoradic_unrank(0, 3, signed=True, pivot=99)


# ── factoradic_rank ───────────────────────────────────────────────────────────

class TestFactoradicRank:
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_round_trip_no_pivot(self, n):
        for k in range(math.factorial(n)):
            arrow  = factoradic_unrank(k, n, signed=True, pivot=None)
            k_back = factoradic_rank(arrow, n, signed=True, pivot=None)
            assert k_back == k

    @pytest.mark.parametrize("n,pivot", [(3, 0), (4, 0), (5, -1), (5, 2)])
    def test_round_trip_with_pivot(self, n, pivot):
        for k in range(math.factorial(n - 1)):
            arrow  = factoradic_unrank(k, n, signed=True, pivot=pivot)
            k_back = factoradic_rank(arrow, n, signed=True, pivot=pivot)
            assert k_back == k

    def test_pivot_mismatch_raises(self):
        arrow = factoradic_unrank(0, 3, pivot=0)
        with pytest.raises(ValueError):
            factoradic_rank(arrow, 3, pivot=1)  # wrong pivot


# ── ArrowStep ─────────────────────────────────────────────────────────────────

class TestArrowStep:
    def test_is_named_tuple(self):
        arrow = np.array([-1, 0, 1])
        step  = ArrowStep(arrow=arrow, fm_coords=(-1, 0, 1, 0))
        assert step.fm_coords == (-1, 0, 1, 0)
        assert np.array_equal(step.arrow, arrow)


# ── factoradic_to_fm_coords ───────────────────────────────────────────────────

class TestFactoradicToFmCoords:
    @pytest.mark.parametrize("n,d,pdim,pval", [
        (3, 4, 0,  0),
        (3, 4, 1, -1),
        (3, 2, 0,  1),
        (5, 3, 2,  2),
    ])
    def test_pivot_dim_constraint(self, n, d, pdim, pval):
        total = (n ** (d - 1)) * math.factorial(n - 1)
        for k in range(total):
            step = factoradic_to_fm_coords(k, n, d, pdim, pval)
            assert step.fm_coords[pdim] == pval

    @pytest.mark.parametrize("n,d,pdim,pval", [
        (3, 4, 0,  0),
        (3, 3, 1,  1),
    ])
    def test_arrow_centre_constraint(self, n, d, pdim, pval):
        total = (n ** (d - 1)) * math.factorial(n - 1)
        for k in range(total):
            step = factoradic_to_fm_coords(k, n, d, pdim, pval)
            assert step.arrow[n // 2] == pval

    @pytest.mark.parametrize("n,d", [(3, 4), (3, 2), (5, 3)])
    def test_count_combinatorial_identity(self, n, d):
        """Total unique pairs = n^(d-1) * (n-1)!"""
        pval     = 0
        expected = (n ** (d - 1)) * math.factorial(n - 1)
        seen = set()
        for k in range(expected):
            step = factoradic_to_fm_coords(k, n, d, 0, pval)
            key  = (step.fm_coords, tuple(step.arrow.tolist()))
            assert key not in seen
            seen.add(key)
        assert len(seen) == expected

    @pytest.mark.parametrize("n,d,pdim,pval", [
        (3, 4, 0,  0),
        (3, 4, 1, -1),
        (3, 3, 2,  1),
    ])
    def test_round_trip(self, n, d, pdim, pval):
        """k → ArrowStep → k must be identity."""
        total = (n ** (d - 1)) * math.factorial(n - 1)
        for k in range(total):
            step   = factoradic_to_fm_coords(k, n, d, pdim, pval)
            k_back = fm_coords_to_factoradic(
                step.arrow, step.fm_coords, n, d, pdim, pval
            )
            assert k_back == k, f"Round-trip failed at k={k}"

    def test_k_out_of_range_raises(self):
        total = (3 ** 3) * math.factorial(2)
        with pytest.raises(ValueError):
            factoradic_to_fm_coords(total, 3, 4, 0, 0)

    def test_invalid_pivot_val_raises(self):
        with pytest.raises(ValueError):
            factoradic_to_fm_coords(0, 3, 4, 0, 99)

    def test_invalid_pivot_dim_raises(self):
        with pytest.raises(ValueError):
            factoradic_to_fm_coords(0, 3, 4, 10, 0)

    def test_all_fm_coords_in_digit_set(self):
        """Every coordinate in every fm_coords is in {-1, 0, 1} for n=3."""
        n, d, pdim, pval = 3, 4, 0, 0
        total = (n ** (d - 1)) * math.factorial(n - 1)
        for k in range(total):
            step = factoradic_to_fm_coords(k, n, d, pdim, pval)
            for c in step.fm_coords:
                assert c in {-1, 0, 1}, f"Coord {c} out of digit set"
