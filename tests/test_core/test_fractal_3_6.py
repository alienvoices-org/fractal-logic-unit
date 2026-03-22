"""
Tests for flu.core.fractal_3_6 — FractalHyperCell_3_6.

ITER-3B: 3⁶ = 3⁴ (macro FLUHyperCell) × 3² (Lo Shu micro grid).
"""

import pytest
import numpy as np

from flu.core.hypercell   import FLUHyperCell
from flu.core.fractal_3_6 import FractalHyperCell_3_6, CellPair, MicroCell
from flu.core.lo_shu      import CellStrata


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def macro():
    return FLUHyperCell()

@pytest.fixture(scope="module")
def frac(macro):
    return FractalHyperCell_3_6(macro)


# ── MicroCell ─────────────────────────────────────────────────────────────────

class TestMicroCell:
    def test_norm0_equals_value_minus_1(self):
        mc = MicroCell(0, 0, 5)
        assert mc.norm0 == 4

    def test_coords_2d_in_digit_set(self):
        for value in range(1, 10):
            mc = MicroCell(0, 0, value)
            assert all(c in (-1, 0, 1) for c in mc.coords_2d)

    def test_9_unique_coords(self):
        coords = {MicroCell(0, 0, v).coords_2d for v in range(1, 10)}
        assert len(coords) == 9


# ── FractalHyperCell_3_6 construction ────────────────────────────────────────

class TestConstruction:
    def test_total_cells(self, frac):
        assert len(frac) == 729

    def test_reverse_index_size(self, frac):
        assert len(frac._reverse) == 729

    def test_bad_micro_shape_raises(self, macro):
        with pytest.raises(ValueError):
            FractalHyperCell_3_6(macro, micro_lo_shu=np.ones((4, 4), dtype=int))

    def test_custom_micro_lo_shu(self, macro):
        alt = np.array([[2, 7, 6], [9, 5, 1], [4, 3, 8]], dtype=np.int64)
        frac_alt = FractalHyperCell_3_6(macro, micro_lo_shu=alt)
        assert len(frac_alt._reverse) == 729


# ── 729 unique 6D addresses ───────────────────────────────────────────────────

class TestAddresses:
    def test_unique_count(self, frac):
        seen = set()
        for mr in range(9):
            for mc in range(9):
                for ur in range(3):
                    for uc in range(3):
                        coords = frac.sparse_address_6d(mr, mc, ur, uc)
                        seen.add(coords)
        assert len(seen) == 729

    def test_all_coords_in_digit_set(self, frac):
        for coords in frac._reverse:
            assert all(c in (-1, 0, 1) for c in coords), \
                f"Out-of-range coord in {coords}"

    def test_coords_length_is_6(self, frac):
        coords = frac.sparse_address_6d(0, 0, 0, 0)
        assert len(coords) == 6

    def test_macro_prefix_matches_sparse_address(self, frac, macro):
        for mr in range(9):
            for mc in range(9):
                macro_4d = macro.sparse_address(mr, mc)
                for ur in range(3):
                    for uc in range(3):
                        coords_6d = frac.sparse_address_6d(mr, mc, ur, uc)
                        assert coords_6d[:4] == macro_4d

    def test_out_of_bounds_macro_raises(self, frac):
        with pytest.raises(ValueError):
            frac.sparse_address_6d(9, 0, 0, 0)

    def test_out_of_bounds_micro_raises(self, frac):
        with pytest.raises(ValueError):
            frac.sparse_address_6d(0, 0, 3, 0)


# ── Round-trip ────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_address_to_position_to_address(self, frac):
        """sparse_address_6d → _reverse lookup → sparse_address_6d == identity."""
        for mr in range(9):
            for mc in range(9):
                for ur in range(3):
                    for uc in range(3):
                        coords      = frac.sparse_address_6d(mr, mc, ur, uc)
                        pos_back    = frac._reverse[coords]
                        coords_back = frac.sparse_address_6d(*pos_back)
                        assert coords_back == coords

    def test_invalid_6d_raises(self, frac):
        with pytest.raises(ValueError):
            frac.cell_at_6d((0, 0, 0, 0, 0, 0))  # only valid if in index


# ── cell_at_6d ────────────────────────────────────────────────────────────────

class TestCellAt6d:
    def test_returns_cell_pair(self, frac):
        coords = frac.sparse_address_6d(4, 4, 1, 1)
        pair   = frac.cell_at_6d(coords)
        assert isinstance(pair, CellPair)

    def test_macro_is_cell_strata(self, frac):
        coords = frac.sparse_address_6d(0, 0, 0, 0)
        pair   = frac.cell_at_6d(coords)
        assert isinstance(pair.macro, CellStrata)

    def test_micro_is_micro_cell(self, frac):
        coords = frac.sparse_address_6d(0, 0, 0, 0)
        pair   = frac.cell_at_6d(coords)
        assert isinstance(pair.micro, MicroCell)

    def test_micro_coords_match_suffix(self, frac):
        """The last 2 coords of the 6D address match micro.coords_2d."""
        for mr in range(9):
            for mc in range(9):
                for ur in range(3):
                    for uc in range(3):
                        coords = frac.sparse_address_6d(mr, mc, ur, uc)
                        pair   = frac.cell_at_6d(coords)
                        assert coords[4:] == pair.micro.coords_2d


# ── verify() ─────────────────────────────────────────────────────────────────

class TestVerify:
    def test_seam_verified(self, frac):
        result = frac.verify(silent=True)
        assert result["seam_verified"] is True

    def test_total_cells_729(self, frac):
        result = frac.verify(silent=True)
        assert result["total_cells"] == 729

    def test_unique_addresses_729(self, frac):
        result = frac.verify(silent=True)
        assert result["unique_addresses"] == 729

    def test_no_range_errors(self, frac):
        result = frac.verify(silent=True)
        assert result["range_errors"] == 0

    def test_no_round_trip_errors(self, frac):
        result = frac.verify(silent=True)
        assert result["round_trip_errors"] == 0


# ── embed_as_3_6 on FLUHyperCell ──────────────────────────────────────────

class TestEmbedAs3_6:
    def test_returns_fractal_type(self):
        frac = FLUHyperCell().embed_as_3_6()
        assert isinstance(frac, FractalHyperCell_3_6)

    def test_length_729(self):
        frac = FLUHyperCell().embed_as_3_6()
        assert len(frac) == 729

    def test_custom_micro_passthrough(self):
        alt  = np.array([[2, 7, 6], [9, 5, 1], [4, 3, 8]], dtype=np.int64)
        frac = FLUHyperCell().embed_as_3_6(micro_lo_shu=alt)
        assert len(frac._reverse) == 729

    def test_verify_seam_after_embed(self):
        frac   = FLUHyperCell().embed_as_3_6()
        result = frac.verify(silent=True)
        assert result["seam_verified"]


# ── Top-level package access ──────────────────────────────────────────────────

class TestPackageAccess:
    def test_symbols_importable_from_flu(self):
        import flu
        assert hasattr(flu, "FractalHyperCell_3_6")
        assert hasattr(flu, "CellPair")
        assert hasattr(flu, "MicroCell")

    def test_smoke_via_flu_namespace(self):
        import flu
        frac = flu.FractalHyperCell_3_6(flu.FLUHyperCell())
        assert len(frac) == 729
