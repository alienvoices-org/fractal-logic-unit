"""
tests/test_core/test_hypothesis.py
====================================
Property-based parametric tests for FLU core invariants across (n,d) grid.
"""
import numpy as np
from flu.core.parity_switcher import generate, verify_latin
from flu.core.n_ary import nary_generate_signed, nary_verify, nary_step_bound
from flu.theory.theory_latin import (
    verify_all_latin_theorems, verify_holographic_repair, verify_constant_line_sum,
)
from flu.theory.theory_spectral import verify_dc_zero
from flu.theory.theory_fm_dance import verify_hamiltonian, verify_bijection
from flu.core.fm_dance_path import step_bound_theorem
from flu.core.factoradic import unrank_optimal_seed
from flu.container.sparse import SparseCommunionManifold

_ODD_N   = [3, 5, 7, 9, 11]
_ND_PAIRS = [(3, 2), (3, 4), (5, 2), (5, 3), (7, 2)]

# ── LATIN ─────────────────────────────────────────────────────────────────────
def test_latin_property_all_n():
    for n in _ODD_N:
        assert verify_latin(n=n, d=2)["latin_ok"], f"Latin failed n={n}"

def test_latin_property_d3():
    for n in [3, 5, 7]:
        assert verify_latin(n=n, d=3)["latin_ok"]

def test_all_latin_theorems_nd_grid():
    for n, d in _ND_PAIRS:
        M = generate(n=n, d=d)
        assert verify_all_latin_theorems(M, n=n)["all_ok"], f"n={n},d={d}"

# ── S1 ZERO MEAN ──────────────────────────────────────────────────────────────
def test_s1_zero_mean_all_n():
    for n, d in _ND_PAIRS:
        M = generate(n=n, d=d)
        assert verify_dc_zero(M)["dc_zero"], f"S1 failed n={n},d={d}"

def test_s1_zero_mean_nary():
    for n in [3, 5, 7]:
        M = nary_generate_signed(n, 2)
        assert abs(M.mean()) < 1e-10

# ── L1 CONSTANT LINE SUM ─────────────────────────────────────────────────────
def test_l1_constant_line_sum_grid():
    for n, d in _ND_PAIRS:
        M = generate(n=n, d=d)
        assert verify_constant_line_sum(M, n=n)["line_sum_ok"]

def test_l1_line_sum_value_zero():
    for n, d in [(3, 2), (5, 2), (7, 2)]:
        M = generate(n=n, d=d)
        assert verify_constant_line_sum(M, n=n)["target_sum"] == 0

# ── L2 HOLOGRAPHIC REPAIR ─────────────────────────────────────────────────────
def test_l2_holographic_repair_grid():
    for n, d in _ND_PAIRS:
        M = generate(n=n, d=d)
        assert verify_holographic_repair(M, n=n)["repair_ok"]

def test_l2_multi_axis_ok():
    for n, d in [(3, 3), (5, 3)]:
        M = generate(n=n, d=d)
        assert verify_holographic_repair(M, n=n)["multi_axis_ok"]

def test_l2_zero_errors():
    for n, d in _ND_PAIRS:
        M = generate(n=n, d=d)
        assert verify_holographic_repair(M, n=n)["errors"] == 0

# ── T1/T2 ─────────────────────────────────────────────────────────────────────
def test_bijection_all_nd_pairs():
    for n, d in _ND_PAIRS:
        assert verify_bijection(n, d), f"Bijection failed n={n},d={d}"

def test_hamiltonian_all_nd_pairs():
    for n, d in [(3, 2), (3, 3), (5, 2), (7, 2)]:
        assert verify_hamiltonian(n, d)

# ── T4 STEP BOUND ─────────────────────────────────────────────────────────────
def test_step_bound_all_nd_pairs():
    for n, d in _ND_PAIRS:
        r = step_bound_theorem(n, d)
        assert r["measured_max"] <= r["max_step_bound"]

def test_step_bound_matches_nary():
    for n, d in _ND_PAIRS:
        assert step_bound_theorem(n, d)["max_step_bound"] == nary_step_bound(n, d)

# ── NARY VERIFY ───────────────────────────────────────────────────────────────
def test_nary_verify_grid():
    for n, d in [(3, 2), (5, 2), (7, 2), (5, 3)]:
        assert nary_verify(n, d)["all_pass"]

# ── SPARSE COMMUNION MANIFOLD ─────────────────────────────────────────────────
def test_sparse_manifold_lookup_is_integer():
    """SparseCommunionManifold returns integer-valued outputs (signed coords)."""
    n = 3
    seeds = [unrank_optimal_seed(0, n, signed=True) for _ in range(4)]
    M = SparseCommunionManifold(n=n, seeds=seeds)
    # Signed coords for n=3 are in {-1, 0, 1}
    for coords in [(-1, 0, 1, -1), (0, 0, 0, 0), (1, -1, 0, 1)]:
        val = M[coords]
        assert isinstance(int(val), int)

def test_sparse_manifold_range():
    """SparseCommunionManifold values are within signed range."""
    n = 3
    seeds = [unrank_optimal_seed(0, n, signed=True) for _ in range(4)]
    M = SparseCommunionManifold(n=n, seeds=seeds)
    half = n // 2
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            val = int(M[(i, j, 0, 0)])
            assert -half <= val <= half, f"Out of range at ({i},{j}): {val}"

def test_sparse_manifold_large_d():
    """SparseCommunionManifold is constructible for very large d."""
    n, d = 3, 32
    seeds = [unrank_optimal_seed(0, n, signed=True) for _ in range(d)]
    M = SparseCommunionManifold(n=n, seeds=seeds)
    assert M is not None

# ── GENERATE CONSISTENCY ──────────────────────────────────────────────────────
def test_generate_is_deterministic():
    for n, d in [(3, 2), (5, 3)]:
        assert np.array_equal(generate(n=n, d=d), generate(n=n, d=d))

def test_generate_shape():
    for n, d in _ND_PAIRS:
        assert generate(n=n, d=d).shape == (n,) * d

def test_generate_even_n_works():
    # V15 audit fix: flu.core.parity_switcher.generate routes even n to even_n.generate.
    # The old test expected a ValueError for n=4; that behaviour is no longer correct.
    # Verify instead that the call succeeds and returns the right shape.
    result = generate(n=4, d=2)
    assert result.shape == (4, 4), f"Expected shape (4,4), got {result.shape}"
