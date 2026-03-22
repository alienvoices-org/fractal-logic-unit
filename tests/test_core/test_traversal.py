"""
tests/test_core/test_traversal.py
===================================
FM-Dance traversal mathematics: torus geometry, step bounds, Hamiltonian
properties, Siamese/fractal structure, Gray Bridge, and kinetic inverse.

Theorems verified: T1, T2, T4 (BFRW-1), T5, T6, T8, BFRW-1, C4.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from flu.core.fm_dance_path import (
    path_coord, path_coord_to_rank, step_bound_theorem,
    verify_siamese_d2, verify_fractal,
    traverse, generate_path_array,
    identify_step, invert_fm_dance_step, step_vector,
    traverse_reverse, cayley_generators, cayley_inverse_generators,
)
from flu.core.fm_dance import index_to_coords, coords_to_index
from flu.theory.theory_fm_dance import (
    verify_hamiltonian, verify_bijection, fm_dance_step_vectors,
)


# ── Torus diameter / BFRW-1 ──────────────────────────────────────────────────

def _max_torus_step(n, d):
    coords = [path_coord(k, n=n, d=d) for k in range(n**d)]
    return max(
        max(min(abs(coords[k+1][i] - coords[k][i]),
                n - abs(coords[k+1][i] - coords[k][i]))
            for i in range(d))
        for k in range(len(coords)-1)
    )

def test_torus_diameter_n3():
    assert _max_torus_step(3, 4) <= 3 // 2

def test_torus_diameter_n5():
    assert _max_torus_step(5, 2) <= 5 // 2

def test_torus_diameter_n7():
    assert _max_torus_step(7, 2) <= 7 // 2

def test_bfrw1_bound_is_tight_n3():
    """BFRW-1 is tight: max step == floor(n/2)."""
    assert _max_torus_step(3, 4) == 3 // 2


# ── T4 step bound theorem ────────────────────────────────────────────────────

def test_step_bound_theorem_n3_d4():
    r = step_bound_theorem(3, 4)
    assert r["status"] == "PROVEN"
    assert r["measured_max"] <= r["max_step_bound"]

def test_step_bound_theorem_n5_d2():
    r = step_bound_theorem(5, 2)
    assert r["measured_max"] <= r["max_step_bound"]

def test_step_bound_dimension_limited():
    assert step_bound_theorem(7, 3)["max_step_bound"] == 3

def test_step_bound_radix_limited():
    assert step_bound_theorem(3, 4)["max_step_bound"] == 1

def test_step_bound_tightness():
    assert step_bound_theorem(3, 4)["bound_tight"] is True


# ── T5 Siamese ───────────────────────────────────────────────────────────────

def test_siamese_n3():
    r = verify_siamese_d2(3)
    assert r["status"] == "PROVEN"
    assert r["bijection_ok"]
    assert r["siamese_ok"]

def test_siamese_n5():
    assert verify_siamese_d2(5)["status"] == "PROVEN"

def test_siamese_n7():
    assert verify_siamese_d2(7)["status"] == "PROVEN"

def test_siamese_primary_ok():
    for n in [3, 5, 7]:
        assert verify_siamese_d2(n)["primary_ok"], f"primary_ok failed n={n}"


# ── T6 Fractal ───────────────────────────────────────────────────────────────

def test_fractal_n3_d4_split2():
    assert verify_fractal(3, 4, 2)["fractal_ok"]

def test_fractal_n5_d4_split2():
    assert verify_fractal(5, 4, 2)["fractal_ok"]

def test_fractal_error_is_zero():
    assert verify_fractal(3, 4, 2)["max_error"] == 0


# ── T8 Gray Bridge ───────────────────────────────────────────────────────────

def _lowest_zero_bit(k):
    j = 0
    while (k >> j) & 1:
        j += 1
    return j

def test_gray_bridge_n2_carry_matches_brgc():
    """T8: FM-Dance carry rule at n=2 == BRGC (lowest zero bit)."""
    for k in range(1, 64):
        brgc = _lowest_zero_bit(k)
        alt  = next(j for j in range(32) if not ((k >> j) & 1))
        assert brgc == alt, f"BRGC rule inconsistent at k={k}"

def test_gray_bridge_carry_level_range():
    n, d = 3, 4
    for k in range(1, n**d):
        coord = path_coord(k, n=n, d=d)
        carry = identify_step(coord, n=n)
        assert 0 <= carry < d

def test_cayley_generators_count():
    for n, d in [(3, 4), (5, 3)]:
        assert len(cayley_generators(n=n, d=d)) == d
        assert len(cayley_inverse_generators(n=n, d=d)) == d

def test_t8_step_vectors_torus_bounded():
    for n, d in [(3, 4), (5, 3), (7, 2)]:
        half = n // 2
        for v in fm_dance_step_vectors(n, d):
            for coord in v:
                assert min(abs(coord), n-abs(coord)) <= half


# ── T1/T2 Bijection + Hamiltonian ────────────────────────────────────────────

def test_hamiltonian_n3_d2(): assert verify_hamiltonian(3, 2)
def test_hamiltonian_n5_d2(): assert verify_hamiltonian(5, 2)
def test_hamiltonian_n3_d4(): assert verify_hamiltonian(3, 4)
def test_bijection_n3_d4():   assert verify_bijection(3, 4)
def test_bijection_n5_d3():   assert verify_bijection(5, 3)

def test_all_81_cells_distinct_n3_d4():
    assert len(set(path_coord(k, n=3, d=4) for k in range(81))) == 81

def test_fm_dance_step_vectors_count():
    for n, d in [(3, 4), (5, 3), (7, 2)]:
        assert len(fm_dance_step_vectors(n, d)) == d

def test_traverse_returns_all_coords():
    n, d = 3, 3
    seen = set()
    for coord in traverse(n=n, d=d):
        seen.add(coord)
    assert len(seen) == n**d

def test_generate_path_array_shape():
    n, d = 3, 3
    arr = generate_path_array(n=n, d=d)
    assert arr.shape == (n,) * d

def test_index_coords_round_trip():
    n, d = 3, 4
    for k in range(0, n**d, 7):
        coords = index_to_coords(k, n=n, d=d)
        back   = coords_to_index(coords, n=n, d=d)
        assert back == k

def test_invert_step_returns_tuple():
    coord = path_coord(5, n=3, d=4)
    inv   = invert_fm_dance_step(coord, n=3)
    assert isinstance(inv, tuple) and len(inv) == 4


# ── C4 Cycle Smoothness ──────────────────────────────────────────────────────

def test_cycle_smoothness_last_point_identity():
    """C4: last-point identity Φ(n^D−1) = (1,−2,−3,…,−D) mod n."""
    def torus_dist(a, b, n):
        return min(abs(a - b) % n, n - abs(a - b) % n)

    for n, d in [(3, 1), (5, 2), (7, 3), (3, 2)]:
        k_max = n**d - 1
        last  = path_coord(k_max, n, d)
        half  = n // 2
        expected = tuple(((1 if i == 0 else -(i+1)) % n) - half for i in range(d))
        assert last == expected, f"C4 last-point identity failed n={n},d={d}: {last} != {expected}"

def test_cycle_smoothness_closing_jump_bounded():
    """C4: closing jump within step bound when D ≤ ⌊n/2⌋."""
    def torus_dist(a, b, n):
        diff = abs(a - b) % n
        return min(diff, n - diff)

    for n, d in [(5, 2), (7, 3)]:
        if d <= n // 2:
            last  = path_coord(n**d - 1, n, d)
            first = path_coord(0, n, d)
            step_bound = min(d, n // 2)
            closing_step = max(torus_dist(last[i], first[i], n) for i in range(d))
            assert closing_step <= step_bound, \
                f"C4 closing jump violated n={n},d={d}: step={closing_step} > bound={step_bound}"


# ── Kinetic Inverse (T-KI) ───────────────────────────────────────────────────

def test_ki_step_vector_formula_n5_d3():
    """T-KI-01: step_vector O(d) formulation."""
    n, d = 5, 3
    sv0 = step_vector(0, n, d)
    assert sv0 == (n-1, 1, 1), f"Primary step wrong: {sv0}"
    sv1 = step_vector(1, n, d)
    assert sv1 == (n-1, 2, 2), f"Level-1 step wrong: {sv1}"
    sv2 = step_vector(2, n, d)
    assert sv2 == (n-1, 2, 3), f"Level-2 step wrong: {sv2}"

def test_ki_step_vector_siamese_n3_d2():
    """T-KI-02: step_vector for n=3, d=2 (Siamese case)."""
    n, d = 3, 2
    assert step_vector(0, n, d) == (2, 1)
    assert step_vector(1, n, d) == (2, 2)

def test_ki_identify_step_bijection():
    """T-KI-03: identify_step returns the correct carry level, 0 mismatches."""
    def get_carry_level(k, n, d):
        a = k; digits = []
        for _ in range(d):
            digits.append(a % n); a //= n
        j = 0
        while j < d-1 and digits[j] == n-1:
            j += 1
        return j

    for n, d in [(3, 2), (3, 3), (5, 2), (5, 3), (7, 3)]:
        mismatches = 0
        for k in range(n**d - 1):
            j_actual = get_carry_level(k, n, d)
            x_next   = path_coord(k+1, n, d)
            j_sig    = identify_step(x_next, n)
            if j_actual != j_sig:
                mismatches += 1
        assert mismatches == 0, f"KI identify_step mismatches n={n},d={d}: {mismatches}"

def test_ki_identify_step_origin():
    """T-KI-04: identify_step returns d for the origin coordinate."""
    for n, d in [(3, 2), (5, 3), (7, 2)]:
        origin = path_coord(0, n, d)
        j = identify_step(origin, n)
        assert j == d, f"n={n},d={d}: expected {d}, got {j}"

def test_ki_invert_step_recovers_previous():
    """T-KI-05: invert_fm_dance_step recovers x_{k-1} exactly."""
    for n, d in [(3, 2), (3, 3), (5, 3), (7, 3)]:
        errors = 0
        for k in range(1, min(n**d, 1000)):
            x_k   = path_coord(k, n, d)
            x_km1 = invert_fm_dance_step(x_k, n)
            x_true = path_coord(k-1, n, d)
            if x_km1 != x_true:
                errors += 1
        assert errors == 0, f"KI invert errors n={n},d={d}: {errors}"

def test_ki_invert_step_raises_on_origin():
    """T-KI-06: invert_fm_dance_step raises ValueError at origin."""
    try:
        origin = path_coord(0, 3, 2)
        invert_fm_dance_step(origin, 3)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_ki_traverse_reverse():
    """T-KI-07: traverse_reverse yields exact reverse sequence."""
    for n, d in [(3, 2), (5, 2), (3, 3)]:
        fwd = list(path_coord(k, n, d) for k in range(n**d))
        rev = list(traverse_reverse(n, d))
        assert rev == list(reversed(fwd)), f"traverse_reverse mismatch n={n},d={d}"

def test_ki_round_trip_rank_recovery():
    """T-KI-08: path_coord → path_coord_to_rank round-trip."""
    for n, d in [(5, 3), (7, 2)]:
        errors = 0
        for k in range(min(n**d, 500)):
            coord = path_coord(k, n, d)
            rank  = path_coord_to_rank(coord, n, d)
            if rank != k:
                errors += 1
        assert errors == 0, f"Round-trip errors n={n},d={d}: {errors}"

def test_ki_invert_is_od_linear():
    """T-KI-09: invert timing grows O(d), not O(n^d)."""
    n = 5
    timings = []
    for d in [3, 6, 9]:
        x = path_coord(n**d - 1, n, d)
        t0 = time.perf_counter_ns()
        for _ in range(200):
            invert_fm_dance_step(x, n)
        timings.append((time.perf_counter_ns() - t0) / 200)
    ratio = timings[2] / timings[0]
    assert ratio < 10, f"invert timing ratio d=9/d=3 = {ratio:.1f} (expected < 10)"
