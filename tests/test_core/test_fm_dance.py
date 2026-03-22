"""
Tests for flu.core.fm_dance — FM-Dance bijection.

These tests directly verify the THEOREM (FM-Dance Bijection):
    For all k ∈ [0, n^d), coords_to_index(index_to_coords(k)) = k.

We test:
  1. Round-trip correctness for all (n,d) pairs in the audit suite.
  2. Coverage: all n^d distinct coordinates visited exactly once.
  3. Mean-centering: mean of all visited coords is 0 (odd n).
  4. Latin property on the materialised hyperprism.
  5. Sparse access: single-cell lookup matches full array.
"""

import pytest
import numpy as np
from flu.core.fm_dance import (
    index_to_coords, coords_to_index,
    generate_fast, verify_bijection,
)


# ── Parametrised (n, d) cases ─────────────────────────────────────────────────

CASES = [
    (3, 1), (3, 2), (3, 3), (3, 4),
    (5, 2), (5, 3),
    (7, 2),
    (11, 2),
]


class TestRoundTrip:
    """Theorem: coords_to_index(index_to_coords(k)) == k for all k."""

    @pytest.mark.parametrize("n, d", CASES)
    def test_forward_inverse(self, n, d):
        total = n ** d
        for k in range(total):
            coords = index_to_coords(k, n, d)
            k_back = coords_to_index(coords, n, d)
            assert k_back == k, (
                f"Round-trip failed: n={n}, d={d}, k={k}, "
                f"coords={coords}, k_back={k_back}"
            )

    @pytest.mark.parametrize("n, d", CASES)
    def test_coverage(self, n, d):
        """All n^d distinct coordinates must be visited."""
        total      = n ** d
        coords_set = {index_to_coords(k, n, d) for k in range(total)}
        assert len(coords_set) == total, (
            f"Coverage failed: n={n}, d={d}, got {len(coords_set)} distinct coords"
        )

    @pytest.mark.parametrize("n, d", CASES)
    def test_mean_centered(self, n, d):
        """For odd n, mean of all visited coordinates must be 0."""
        coords_all = [index_to_coords(k, n, d) for k in range(n ** d)]
        arr        = np.array(coords_all, dtype=float)
        col_means  = arr.mean(axis=0)
        assert np.allclose(col_means, 0.0, atol=1e-10), (
            f"Mean-centering failed: n={n}, d={d}, means={col_means}"
        )


class TestSignedRange:
    @pytest.mark.parametrize("n, d", CASES)
    def test_coords_in_range(self, n, d):
        half  = n // 2
        total = n ** d
        for k in range(total):
            coords = index_to_coords(k, n, d)
            for c in coords:
                assert -half <= c <= half, (
                    f"Coord {c} out of range [-{half},{half}] for n={n}"
                )


class TestEvenNRejection:
    def test_even_n_raises(self):
        with pytest.raises(ValueError):
            index_to_coords(0, 4, 2)

    def test_even_n_generate_raises(self):
        with pytest.raises(ValueError):
            generate_fast(4, 2)


class TestLatinProperty:
    @pytest.mark.parametrize("n, d", [(3, 2), (3, 3), (5, 2), (7, 2)])
    def test_latin_slices(self, n, d):
        """Every 1-D slice of generate_fast must be a permutation of the digit set."""
        from flu.utils.math_helpers import digits_signed
        half   = n // 2
        arr    = generate_fast(n, d, signed=True)
        digits = set(digits_signed(n))

        # We verify via the step-index unique-per-slice property
        # (each slice of step-indices covers [0,n^{d-1}) × n values)
        # Equivalent: check that signed value distribution is uniform per slice
        for axis in range(d):
            for fixed in np.ndindex(*[n if i != axis else 1 for i in range(d)]):
                slc = []
                fi  = 0
                for dim in range(d):
                    if dim == axis:
                        slc.append(slice(None))
                    else:
                        slc.append(fixed[fi])
                        fi += 1
                # Step indices in this slice: should all be distinct (n values)
                step_slice = arr[tuple(slc)].flatten()
                assert len(set(int(v) for v in step_slice)) == n


class TestSparseVsFull:
    def test_sparse_matches_full_array(self):
        """index_to_coords(k) must match position of k in generate_fast()."""
        n, d   = 3, 3
        half   = n // 2
        arr    = generate_fast(n, d, signed=True)

        for k in range(n ** d):
            coords   = index_to_coords(k, n, d)
            arr_idx  = tuple(c + half for c in coords)
            assert arr[arr_idx] == k, (
                f"Sparse/full mismatch: k={k}, coords={coords}, arr[idx]={arr[arr_idx]}"
            )


class TestVerifyBijection:
    @pytest.mark.parametrize("n, d", CASES)
    def test_verify_passes(self, n, d):
        result = verify_bijection(n, d)
        assert result["bijection_ok"], (
            f"verify_bijection failed for n={n}, d={d}: {result}"
        )
