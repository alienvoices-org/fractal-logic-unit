"""
tests/test_core/test_parity_switcher.py
=======================================
Verification of flu.core.parity_switcher.

Checks the unified factory dispatch logic:
  - Odd n → FM-Dance (Kinetic branch)
  - Even n → even_kronecker (EVEN-1 branch)
  - Metadata introspection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import numpy as np
from flu.core.parity_switcher import generate, generate_metadata, verify_latin

# ── 1. Dispatch Logic ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,d,expected_branch", [
    (3, 2, "fm_dance"),
    (5, 3, "fm_dance"),
    (4, 2, "even_kronecker"),
    (6, 2, "even_kronecker"),
])
def test_dispatch_logic(n, d, expected_branch):
    """Verify dispatch logic via metadata (T-PS-04)."""
    meta = generate_metadata(n, d)
    assert meta["branch"] == expected_branch, \
        f"PS dispatch failed: n={n}, expected {expected_branch}, got {meta['branch']}"

def test_dispatch_parity_flag():
    """Verify metadata parity flag."""
    assert generate_metadata(3, 2)["parity"] == "odd"
    assert generate_metadata(4, 2)["parity"] == "even"


# ── 2. Functional Correctness ────────────────────────────────────────────────

@pytest.mark.parametrize("n,d", [(3, 2), (4, 2), (5, 3), (6, 2)])
def test_generate_latin_property(n, d):
    """Verify Latin property holds for both branches (T3/PROVEN)."""
    M = generate(n, d, signed=False)
    # verify_latin(arr, n, signed=False) verifies Latin-square slices
    r = verify_latin(M, n, signed=False)
    assert r["latin_ok"], f"Latin property FAILED for n={n}, d={d}"

def test_generate_shape():
    """Verify shape is always (n,)*d."""
    for n, d in [(3, 3), (4, 3)]:
        M = generate(n, d)
        assert M.shape == (n,) * d


# ── 3. Rigor/Constraint Tests ─────────────────────────────────────────────────

def test_generate_raises_n_lt_2():
    with pytest.raises(ValueError):
        generate(1, 2)

def test_generate_raises_d_lt_1():
    with pytest.raises(ValueError):
        generate(3, 0)

def test_metadata_hamiltonian_flag():
    """Hamiltonian property is only PROVEN for odd n (T2)."""
    assert generate_metadata(3, 2)["hamiltonian"] is True
    assert generate_metadata(4, 2)["hamiltonian"] is False

def test_metadata_step_bound():
    """Step bound exists only for odd n (T4)."""
    assert generate_metadata(3, 3)["step_bound"] == min(3, 3 // 2)
    assert generate_metadata(4, 3)["step_bound"] is None
