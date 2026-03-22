"""
Tests for flu.core.even_n — even-n Latin hyperprism generation.

Regression suite for the even-n Latin property (macro_step modulo fix).

BUG (archived): `a = macro_val` (global step index) → Latin property DESTROYED.
FIX: `a = macro_val % m`                     → Latin property RESTORED.

We verify this explicitly for all documented failure cases.
"""

import pytest
import numpy as np
from flu.core.even_n import generate, verify, decompose_n


class TestDecomposeN:
    @pytest.mark.parametrize("n, expected_k, expected_m", [
        (2,   1, 1),
        (4,   2, 1),
        (6,   1, 3),
        (8,   3, 1),
        (10,  1, 5),
        (12,  2, 3),
        (14,  1, 7),
        (16,  4, 1),
    ])
    def test_decomposition(self, n, expected_k, expected_m):
        k, m = decompose_n(n)
        assert k == expected_k, f"n={n}: expected k={expected_k}, got {k}"
        assert m == expected_m, f"n={n}: expected m={expected_m}, got {m}"
        assert 2**k * m == n


class TestLatinPropertyRegression:
    """
    CRITICAL: Latin property regression tests.

    If the bug is re-introduced (using raw macro_step instead of
    macro_step % m), these tests must fail.  They are the guard.
    """

    @pytest.mark.parametrize("n, d", [
        (6,  2),   # original reported failure case
        (6,  3),
        (10, 2),
        (10, 3),
        (14, 2),
    ])
    def test_latin_property(self, n, d):
        result = verify(n, d)
        assert result["latin_ok"], (
            f"Latin property FAILED for n={n}, d={d}.  "
            f"Violations: {result['violations'][:3]}.  "
            "Check for macro_step modulo regression."
        )

    @pytest.mark.parametrize("n, d", [
        (4,  2), (4, 3),   # pure power-of-2 (m=1 path)
        (8,  2),
    ])
    def test_latin_pure_power_of_2(self, n, d):
        result = verify(n, d)
        assert result["latin_ok"], (
            f"Latin property FAILED for pure-power-of-2 n={n}, d={d}"
        )

    @pytest.mark.parametrize("n, d", [
        (6, 2), (6, 3), (10, 2), (4, 2), (8, 2),
    ])
    def test_coverage(self, n, d):
        result = verify(n, d)
        assert result["coverage_ok"], (
            f"Coverage FAILED for n={n}, d={d}"
        )


class TestValueRange:
    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_unsigned_range(self, n):
        hp = generate(n, 2, signed=False)
        assert int(hp.min()) == 0
        assert int(hp.max()) == n - 1

    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_signed_range(self, n):
        hp   = generate(n, 2, signed=True)
        half = n // 2
        assert int(hp.min()) == -half
        assert int(hp.max()) == half - 1 or int(hp.max()) == half


class TestShape:
    @pytest.mark.parametrize("n, d", [(4, 2), (6, 3), (8, 2), (10, 2)])
    def test_shape(self, n, d):
        hp = generate(n, d)
        assert hp.shape == tuple([n] * d)


class TestOddNRejection:
    def test_odd_n_raises(self):
        with pytest.raises(ValueError):
            generate(3, 2)
        with pytest.raises(ValueError):
            generate(5, 3)


# ── SparseEvenManifold tests ───────────────────────────────────────────────────

class TestSparseEvenManifoldParity:
    """
    CRITICAL: SparseEvenManifold must be a point-for-point oracle equivalent
    of flu.core.even_n.generate().  Any divergence is a correctness bug.

    Tests both the pure-power-of-2 path (m=1, XOR only) and the mixed
    Kronecker path (m > 1, XOR micro + sum-mod macro).
    """

    @pytest.mark.parametrize("n, d", [
        (4,  2),   # pure 2^2, m=1
        (4,  3),
        (8,  2),   # pure 2^3, m=1
        (6,  2),   # 2^1 * 3, m=3
        (6,  3),
        (10, 2),   # 2^1 * 5, m=5
        (12, 2),   # 2^2 * 3, m=3
        (14, 2),   # 2^1 * 7, m=7
    ])
    def test_matches_dense_generate_signed(self, n, d):
        from flu.container.sparse import SparseEvenManifold
        dense  = generate(n, d, signed=True)
        sparse = SparseEvenManifold(n=n, d=d, signed=True)
        half   = n // 2
        for idx in np.ndindex(*([n] * d)):
            coord     = tuple(i - half for i in idx)
            dense_val = int(dense[idx])
            sparse_val = int(sparse[coord])
            assert dense_val == sparse_val, (
                f"n={n}, d={d}: mismatch at idx={idx} coord={coord}: "
                f"dense={dense_val}, sparse={sparse_val}"
            )

    @pytest.mark.parametrize("n, d", [
        (4, 2), (6, 2), (8, 2), (10, 2),
    ])
    def test_matches_dense_generate_unsigned(self, n, d):
        from flu.container.sparse import SparseEvenManifold
        dense  = generate(n, d, signed=False)
        sparse = SparseEvenManifold(n=n, d=d, signed=False)
        half   = n // 2
        for idx in np.ndindex(*([n] * d)):
            coord      = tuple(i - half for i in idx)
            dense_val  = int(dense[idx])
            sparse_val = int(sparse[coord])
            assert dense_val == sparse_val, (
                f"n={n}, d={d} unsigned: mismatch at coord={coord}: "
                f"dense={dense_val}, sparse={sparse_val}"
            )

    @pytest.mark.parametrize("n, d", [
        (4, 2), (6, 2), (10, 2),
    ])
    def test_batch_matches_single(self, n, d):
        """Vectorised batch evaluation must equal single-coord evaluation."""
        from flu.container.sparse import SparseEvenManifold
        sparse = SparseEvenManifold(n=n, d=d, signed=True)
        half   = n // 2
        # Build all signed coords as an (n^d, d) array
        all_idx    = list(np.ndindex(*([n] * d)))
        coords_arr = np.array([[i - half for i in idx] for idx in all_idx], dtype=int)
        batch_vals  = sparse[coords_arr]
        single_vals = np.array([sparse[tuple(c)] for c in coords_arr])
        np.testing.assert_array_equal(
            batch_vals, single_vals,
            err_msg=f"Batch vs single mismatch for n={n}, d={d}"
        )


class TestSparseEvenManifoldLatin:
    """Materialised SparseEvenManifold must be a valid Latin hyperprism."""

    @pytest.mark.parametrize("n, d", [
        (4, 2), (6, 2), (8, 2), (10, 2), (4, 3), (6, 3),
    ])
    def test_latin_property(self, n, d):
        from flu.container.sparse import SparseEvenManifold
        from flu.utils.verification import check_latin
        sparse = SparseEvenManifold(n=n, d=d, signed=False)
        half   = n // 2
        arr    = np.zeros([n] * d, dtype=int)
        for idx in np.ndindex(*([n] * d)):
            arr[idx] = int(sparse[tuple(i - half for i in idx)])
        result = check_latin(arr, n, signed=False)
        assert result["latin_ok"], (
            f"SparseEvenManifold Latin property FAILED for n={n}, d={d}. "
            f"Violations: {result['violations'][:3]}"
        )


class TestSparseEvenManifoldRange:
    @pytest.mark.parametrize("n", [4, 6, 8, 10, 12])
    def test_signed_range(self, n):
        from flu.container.sparse import SparseEvenManifold
        sparse = SparseEvenManifold(n=n, d=2, signed=True)
        half   = n // 2
        vals   = [sparse[tuple(i - half for i in idx)]
                  for idx in np.ndindex(n, n)]
        assert min(vals) == -half
        assert max(vals) == half - 1

    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_unsigned_range(self, n):
        from flu.container.sparse import SparseEvenManifold
        sparse = SparseEvenManifold(n=n, d=2, signed=False)
        half   = n // 2
        vals   = [sparse[tuple(i - half for i in idx)]
                  for idx in np.ndindex(n, n)]
        assert min(vals) == 0
        assert max(vals) == n - 1

    def test_odd_n_rejected(self):
        from flu.container.sparse import SparseEvenManifold
        with pytest.raises(ValueError):
            SparseEvenManifold(n=5, d=2)

    def test_shape_attribute(self):
        from flu.container.sparse import SparseEvenManifold
        m = SparseEvenManifold(n=6, d=3)
        assert m.shape == (6, 6, 6)
        assert m.n == 6
        assert m.d == 3


# ── manifold() factory routing tests ─────────────────────────────────────────

class TestManifoldFactory:
    """
    flu.manifold() routing: dense/sparse × odd/even should all work and
    return the correct type with correct mathematical properties.
    """

    # ── Dense path ────────────────────────────────────────────────────────────

    def test_dense_odd_returns_ndarray(self):
        import flu
        M = flu.manifold(5, 3)
        assert isinstance(M, np.ndarray)
        assert M.shape == (5, 5, 5)

    def test_dense_even_returns_ndarray(self):
        import flu
        M = flu.manifold(6, 2)
        assert isinstance(M, np.ndarray)
        assert M.shape == (6, 6)

    def test_dense_signed_false(self):
        import flu
        M = flu.manifold(6, 2, signed=False)
        assert int(M.min()) == 0
        assert int(M.max()) == 5

    # ── Sparse odd path ───────────────────────────────────────────────────────

    def test_sparse_odd_returns_sparse_communion(self):
        import flu
        from flu.container.sparse import SparseCommunionManifold
        M = flu.manifold(3, 4, sparse=True)
        assert isinstance(M, SparseCommunionManifold)
        assert M.n == 3
        assert M.d == 4

    def test_sparse_odd_rejects_even_n_internally(self):
        """SparseCommunionManifold must not be returned for even n."""
        import flu
        from flu.container.sparse import SparseCommunionManifold
        M = flu.manifold(4, 2, sparse=True)
        assert not isinstance(M, SparseCommunionManifold)

    # ── Sparse even path ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("n", [4, 6, 8, 10, 12, 16])
    def test_sparse_even_returns_sparse_even_manifold(self, n):
        import flu
        from flu.container.sparse import SparseEvenManifold
        M = flu.manifold(n, 3, sparse=True)
        assert isinstance(M, SparseEvenManifold), (
            f"manifold(n={n}, sparse=True) should return SparseEvenManifold"
        )
        assert M.n == n
        assert M.d == 3

    @pytest.mark.parametrize("n, d", [
        (4, 2), (6, 2), (8, 2), (10, 2),
    ])
    def test_sparse_even_factory_matches_dense(self, n, d):
        """Factory sparse oracle must agree with dense generate() on every cell."""
        import flu
        dense  = generate(n, d, signed=True)
        sparse = flu.manifold(n, d, sparse=True, signed=True)
        half   = n // 2
        for idx in np.ndindex(*([n] * d)):
            coord = tuple(i - half for i in idx)
            assert int(dense[idx]) == int(sparse[coord]), (
                f"Factory mismatch at n={n}, d={d}, coord={coord}"
            )

    def test_sparse_even_signed_false(self):
        import flu
        M    = flu.manifold(6, 2, sparse=True, signed=False)
        half = 3
        vals = [M[tuple(i - half for i in idx)] for idx in np.ndindex(6, 6)]
        assert min(vals) == 0
        assert max(vals) == 5

    def test_sparse_even_large_d_no_materialisation(self):
        """
        Large-d even manifold should construct in O(1) without allocating n^d memory.
        Just checks that construction and a single lookup succeed.
        """
        import flu
        M   = flu.manifold(4, 64, sparse=True)
        val = M[tuple([0] * 64)]
        assert isinstance(val, (int, float, np.integer, np.floating))
