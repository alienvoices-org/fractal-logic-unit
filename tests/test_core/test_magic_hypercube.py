"""
tests/test_core/test_magic_hypercube.py
========================================
Regression tests for the FM-Dance magic hypercube construction (Theorem MH)
and the Trump/Boyer perfect magic cube constant (Theorem MH-COMPARE).

Covers:
  - generate_magic / magic_coord  (flu.core.fm_dance)
  - FM_DANCE_5_NP / TRUMP_BOYER_5_NP  (flu.constants)
  - Three-way object distinction: generate_fast / path_coord / generate_magic
  - LHS per-slice digit balance
  - Planar diagonal coverage
  - Layer antisymmetry of FM-Dance bones representation

All tests are deterministic and parameter-free (no fixtures required).
"""
import numpy as np
import pytest

from flu.core.fm_dance import generate_fast, generate_magic, magic_coord
from flu.core.fm_dance_path import generate_path_array
from flu.constants import (
    FM_DANCE_5_NP,
    TRUMP_BOYER_5_NP,
    MAGIC_SUM_5,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def magic_sum(n: int, d: int) -> int:
    return n * (n**d + 1) // 2


def check_all_axis_magic(cube: np.ndarray, M: int) -> bool:
    return all(np.all(cube.sum(axis=a) == M) for a in range(cube.ndim))


def check_completeness(cube: np.ndarray) -> bool:
    n_d = cube.size
    return sorted(cube.flatten().tolist()) == list(range(1, n_d + 1))


def per_slice_digit_balanced(cube: np.ndarray) -> bool:
    """True iff every axis-slice is digit-balanced for all 3 digit positions."""
    n = cube.shape[0]
    c = cube - 1
    for dp in range(cube.ndim):
        dg = (c // (n ** dp)) % n
        for ax in range(cube.ndim):
            for i in range(n):
                sl = np.take(dg, i, axis=ax).flatten()
                if np.bincount(sl, minlength=n).tolist() != [n] * n:
                    return False
    return True


def spectral_block_per_line(cube: np.ndarray) -> bool:
    """True iff every axis-aligned line contains one value from each spectral block."""
    n = cube.shape[0]; d = cube.ndim
    blk = (cube - 1) // (n ** (d - 1))
    for ax in range(d):
        # iterate over all (d-1)-dimensional index combos
        for indices in np.ndindex(*[n if i != ax else 1 for i in range(d)]):
            slc = [indices[i] if i != ax else slice(None) for i in range(d)]
            line = blk[tuple(slc)].flatten()
            if np.bincount(line, minlength=n).tolist() != [1] * n:
                return False
    return True


def count_magic_planar_diags(cube: np.ndarray, M: int) -> int:
    n = cube.shape[0]; count = 0
    for ax in range(cube.ndim):
        for i in range(n):
            layer = np.take(cube, i, axis=ax)
            d1 = int(sum(layer[k, k]     for k in range(n)))
            d2 = int(sum(layer[k, n-1-k] for k in range(n)))
            if d1 == M: count += 1
            if d2 == M: count += 1
    return count


def space_diagonals(cube: np.ndarray) -> list:
    """Return all 2^(d-1) main space diagonal sums."""
    from itertools import product as iprod
    n = cube.shape[0]; d = cube.ndim
    sums = []
    for signs in iprod([1, -1], repeat=d):
        s = int(sum(cube[tuple((i if sg == 1 else n-1-i) for sg in signs)]
                    for i in range(n)))
        sums.append(s)
    return sums


# ─────────────────────────────────────────────────────────────────────────────
# Tests: magic_coord and generate_magic
# ─────────────────────────────────────────────────────────────────────────────

class TestMagicCoord:
    """Unit tests for the magic_coord closed-form formula."""

    def test_centre_value_at_geometric_centre_n5(self):
        """Value 63 (mean of 1..125) must sit at position [2,2,2]."""
        cube = generate_magic(5, 3)
        assert cube[2, 2, 2] == 63

    def test_centre_value_at_geometric_centre_n3(self):
        """Value 14 (mean of 1..27) must sit at position [1,1,1]."""
        cube = generate_magic(3, 3)
        assert cube[1, 1, 1] == 14

    @pytest.mark.parametrize("n,d", [
        (3, 2), (5, 2), (3, 3), (5, 3), (7, 3), (3, 4),
    ])
    def test_bijection(self, n, d):
        cube = generate_magic(n, d)
        assert check_completeness(cube), f"Not a bijection for n={n}, d={d}"

    @pytest.mark.parametrize("n,d", [
        (3, 2), (5, 2), (3, 3), (5, 3), (7, 3), (3, 4),
    ])
    def test_all_axis_sums_equal_magic_constant(self, n, d):
        cube = generate_magic(n, d)
        M = magic_sum(n, d)
        assert check_all_axis_magic(cube, M), f"Axis sums fail for n={n}, d={d}, M={M}"

    @pytest.mark.parametrize("n,d", [(3, 3), (5, 3), (3, 4)])
    def test_space_diagonals(self, n, d):
        cube = generate_magic(n, d)
        M = magic_sum(n, d)
        sds = space_diagonals(cube)
        assert all(s == M for s in sds), f"Space diagonals {sds} ≠ {M} for n={n},d={d}"

    @pytest.mark.parametrize("n,d", [(3, 2), (5, 2), (3, 3), (5, 3)])
    def test_spectral_block_per_line(self, n, d):
        cube = generate_magic(n, d)
        assert spectral_block_per_line(cube), f"Spectral block property fails n={n},d={d}"

    def test_d2_is_magic_square_n3(self):
        """d=2 reduces to the classical Lo Shu magic square (up to rotation)."""
        sq = generate_magic(3, 2)
        M = magic_sum(3, 2)   # 15
        assert check_completeness(sq)
        assert check_all_axis_magic(sq, M)
        assert sum(sq[i, i]   for i in range(3)) == M   # main diagonal
        assert sum(sq[i, 2-i] for i in range(3)) == M   # anti diagonal

    def test_d2_is_magic_square_n5(self):
        sq = generate_magic(5, 2)
        M = magic_sum(5, 2)   # 65
        assert check_all_axis_magic(sq, M)
        assert sum(sq[i, i]   for i in range(5)) == M
        assert sum(sq[i, 4-i] for i in range(5)) == M


# ─────────────────────────────────────────────────────────────────────────────
# Tests: FM_DANCE_5_NP constant
# ─────────────────────────────────────────────────────────────────────────────

class TestFMDance5Constant:
    """Tests for the FM_DANCE_5_NP constant in flu.constants."""

    def test_equals_generate_magic(self):
        assert np.all(FM_DANCE_5_NP == generate_magic(5, 3)), \
            "FM_DANCE_5_NP must equal generate_magic(5,3)"

    def test_completeness(self):
        assert check_completeness(FM_DANCE_5_NP)

    def test_all_axis_sums(self):
        assert check_all_axis_magic(FM_DANCE_5_NP, MAGIC_SUM_5)

    def test_space_diagonals(self):
        M = MAGIC_SUM_5
        assert all(s == M for s in space_diagonals(FM_DANCE_5_NP))

    def test_magic_sum_constant(self):
        assert MAGIC_SUM_5 == 315 == 5 * (5**3 + 1) // 2

    def test_centre_cell(self):
        assert FM_DANCE_5_NP[2, 2, 2] == 63

    def test_per_slice_digit_balance(self):
        """Key LHS property: every digit position is balanced in every slice."""
        assert per_slice_digit_balanced(FM_DANCE_5_NP), \
            "FM_DANCE_5_NP must satisfy per-slice 5-ary digit balance"

    def test_layer_antisymmetry_bones(self):
        """Bones[k,y,x] == -Bones[4-k,4-y,4-x] for all k,y,x (Theorem MH)."""
        bones = FM_DANCE_5_NP - 63
        for k in range(2):   # pairs (0,4) and (1,3)
            assert np.all(bones[k] == -bones[4 - k, ::-1, ::-1]), \
                f"Layer antisymmetry fails for pair ({k},{4-k})"
        assert np.all(bones[2] == -bones[2, ::-1, ::-1]), \
            "Centre layer must be self-antisymmetric in bones"

    def test_spectral_block_per_line(self):
        """FM-Dance satisfies spectral block per-line (TB does not)."""
        assert spectral_block_per_line(FM_DANCE_5_NP)

    def test_point_symmetry(self):
        """FM-Dance: for every v, 126-v is at the geometrically opposite cell."""
        FM = FM_DANCE_5_NP; n = 5
        for z in range(n):
            for y in range(n):
                for x in range(n):
                    assert int(FM[z,y,x]) + int(FM[n-1-z,n-1-y,n-1-x]) == 126

    def test_planar_diagonal_count(self):
        """FM-Dance achieves 18/30 planar diagonals (not perfect)."""
        cnt = count_magic_planar_diags(FM_DANCE_5_NP, MAGIC_SUM_5)
        assert cnt == 18, f"Expected 18 planar diagonals = 315, got {cnt}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: TRUMP_BOYER_5_NP constant
# ─────────────────────────────────────────────────────────────────────────────

class TestTrumpBoyer5Constant:
    """Tests for the TRUMP_BOYER_5_NP constant in flu.constants."""

    def test_completeness(self):
        assert check_completeness(TRUMP_BOYER_5_NP)

    def test_all_axis_sums(self):
        assert check_all_axis_magic(TRUMP_BOYER_5_NP, MAGIC_SUM_5)

    def test_space_diagonals(self):
        M = MAGIC_SUM_5
        assert all(s == M for s in space_diagonals(TRUMP_BOYER_5_NP))

    def test_centre_cell(self):
        assert TRUMP_BOYER_5_NP[2, 2, 2] == 63

    def test_perfect_planar_diagonals(self):
        """Trump/Boyer is a PERFECT magic cube: all 30 planar diagonals = 315."""
        cnt = count_magic_planar_diags(TRUMP_BOYER_5_NP, MAGIC_SUM_5)
        assert cnt == 30, f"Trump/Boyer must have 30/30 planar diagonals, got {cnt}"

    def test_not_per_slice_digit_balanced(self):
        """Trump/Boyer is NOT a Latin hypercube: per-slice digit balance fails
        (only 2 of 45 slice×digit pairs are accidentally balanced)."""
        assert not per_slice_digit_balanced(TRUMP_BOYER_5_NP), \
            "Trump/Boyer must NOT satisfy per-slice digit balance (LHS incompatible)"

    def test_spectral_block_per_line_fm_only(self):
        """Spectral-block-per-line holds for FM-Dance but NOT Trump/Boyer.
        TB axis-0 has only 5/25 magic lines; FM has 25/25 on every axis."""
        assert spectral_block_per_line(FM_DANCE_5_NP), \
            "FM_DANCE_5_NP must satisfy spectral block per-line"
        assert not spectral_block_per_line(TRUMP_BOYER_5_NP), \
            "TRUMP_BOYER_5_NP must NOT satisfy spectral block per-line"

    def test_fm_point_symmetry(self):
        """FM-Dance: for every v, antipodal value 126-v is at the opposite cell."""
        FM = FM_DANCE_5_NP; n = 5
        for z in range(n):
            for y in range(n):
                for x in range(n):
                    assert int(FM[z,y,x]) + int(FM[n-1-z,n-1-y,n-1-x]) == 126

    def test_tb_no_point_symmetry(self):
        """Trump/Boyer (exhaustive-search) does NOT have full point symmetry."""
        TB = TRUMP_BOYER_5_NP; n = 5
        violations = [(z,y,x) for z in range(n) for y in range(n) for x in range(n)
                      if int(TB[z,y,x]) + int(TB[n-1-z,n-1-y,n-1-x]) != 126]
        assert len(violations) > 0, "TB must not have full point symmetry"

    def test_global_digit_balance(self):
        """Global digit balance: 25 of each digit value over all 125 cells."""
        c = TRUMP_BOYER_5_NP - 1
        for dp in range(3):
            dg = (c // (5 ** dp)) % 5
            counts = np.bincount(dg.flatten(), minlength=5).tolist()
            assert counts == [25] * 5, f"Global digit-{dp} balance fails: {counts}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Three-way object distinction
# ─────────────────────────────────────────────────────────────────────────────

class TestThreeWayDistinction:
    """
    Verify that generate_fast, path_coord, and generate_magic are three
    distinct bijections with different magic properties (Theorem MH, fm_dance.py).
    """

    def test_generate_fast_is_identity_map(self):
        """generate_fast stores rank k at its own digit-address: trivial bijection."""
        GF = generate_fast(n=5, d=3, signed=False)
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    assert GF[x, y, z] == x + 5*y + 25*z

    def test_generate_fast_not_magic(self):
        """Axis-0 sums of generate_fast are not all equal."""
        GF = generate_fast(n=5, d=3, signed=False) + 1
        axis0_sums = np.unique(GF.sum(axis=0)).tolist()
        assert len(axis0_sums) > 1, "generate_fast must NOT be magic on axis-0"

    def test_path_coord_not_magic_on_all_axes(self):
        """T-matrix path is not magic (axis-0 sums are stratified by block)."""
        PA = generate_path_array(n=5, d=3) + 1
        axis0_sums = np.unique(PA.sum(axis=0)).tolist()
        assert len(axis0_sums) > 1, "path_coord cube must NOT be fully magic"

    def test_all_three_are_bijections(self):
        """All three objects contain each value 1..125 exactly once."""
        GF = generate_fast(n=5, d=3, signed=False) + 1
        PA = generate_path_array(n=5, d=3) + 1
        GM = generate_magic(5, 3)
        for cube, name in [(GF,"generate_fast"),(PA,"path_coord"),(GM,"generate_magic")]:
            assert check_completeness(cube), f"{name} is not a bijection"

    def test_all_three_are_distinct(self):
        """The three cubes must be genuinely different arrays."""
        GF = generate_fast(n=5, d=3, signed=False) + 1
        PA = generate_path_array(n=5, d=3) + 1
        GM = generate_magic(5, 3)
        assert not np.all(GF == PA), "generate_fast and path_coord must differ"
        assert not np.all(GF == GM), "generate_fast and generate_magic must differ"
        assert not np.all(PA == GM), "path_coord and generate_magic must differ"

    def test_only_generate_magic_is_fully_magic(self):
        """Only generate_magic has all axis sums = M = 315."""
        M = MAGIC_SUM_5
        GF = generate_fast(n=5, d=3, signed=False) + 1
        PA = generate_path_array(n=5, d=3) + 1
        GM = generate_magic(5, 3)
        assert     check_all_axis_magic(GM, M), "generate_magic must be fully magic"
        assert not check_all_axis_magic(GF, M), "generate_fast must NOT be fully magic"
        assert not check_all_axis_magic(PA, M), "path_coord must NOT be fully magic"

    def test_fm_and_tb_different_from_each_other(self):
        """The FM-Dance and Trump/Boyer cubes are distinct."""
        assert not np.all(FM_DANCE_5_NP == TRUMP_BOYER_5_NP)
