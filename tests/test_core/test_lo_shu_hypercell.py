"""
tests/test_core/test_lo_shu_hypercell.py
=========================================
Tests for previously untested public methods in:
  - flu.core.lo_shu    : address_of(), from_d1_d2(), set_perspective()
  - flu.core.hypercell : address_of(), gnosis(), set_omega(),
                         set_perspective(), sparse_step_index(),
                         zoom_in(), zoom_out()
"""

import pytest
import numpy as np

from flu.core.lo_shu    import LoShuHyperCell, Perspective, CellStrata
from flu.core.hypercell import FLUHyperCell


# ── CellStrata.from_d1_d2 ────────────────────────────────────────────────────

class TestCellStrataFromD1D2:

    @pytest.mark.parametrize("d1,d2", [(1,1), (5,5), (9,9), (1,9)])
    def test_norm0_formula(self, d1, d2):
        """norm0 = (d1-1)*9 + (d2-1)."""
        cell = CellStrata.from_d1_d2(d1, d2)
        assert cell.norm0 == (d1 - 1) * 9 + (d2 - 1)

    def test_norm1_is_norm0_plus_1(self):
        cell = CellStrata.from_d1_d2(3, 7)
        assert cell.norm1 == cell.norm0 + 1

    def test_bt_length_is_4(self):
        cell = CellStrata.from_d1_d2(5, 5)
        assert len(cell.bt) == 4

    def test_bt_values_in_digit_set(self):
        for d1 in range(1, 10):
            for d2 in range(1, 10):
                cell = CellStrata.from_d1_d2(d1, d2)
                assert all(b in (-1, 0, 1) for b in cell.bt), \
                    f"bt={cell.bt} at ({d1},{d2})"

    def test_unity_positive(self):
        for d1 in range(1, 10):
            cell = CellStrata.from_d1_d2(d1, 1)
            assert cell.unity > 0


# ── LoShuHyperCell.address_of ─────────────────────────────────────────────────

class TestLoShuAddressOf:

    @pytest.fixture
    def hc(self):
        return LoShuHyperCell()

    def test_all_81_values_found(self, hc):
        """address_of(norm1) returns a valid position for every norm1 ∈ [1,81]."""
        for norm1 in range(1, 82):
            pos = hc.address_of(norm1)
            assert pos is not None, f"norm1={norm1} not found"
            r, c = pos
            assert 0 <= r < 9 and 0 <= c < 9

    def test_round_trip(self, hc):
        """address_of(cell(r,c).norm1) == (r, c)."""
        for r in range(9):
            for c in range(9):
                norm1 = hc.cell(r, c).norm1
                pos   = hc.address_of(norm1)
                assert pos == (r, c), f"Round-trip failed at ({r},{c})"

    def test_missing_value_returns_none(self, hc):
        assert hc.address_of(0)  is None   # out of range
        assert hc.address_of(82) is None


# ── LoShuHyperCell.set_perspective ────────────────────────────────────────────

class TestLoShuSetPerspective:

    def test_perspective_changes_grid(self):
        hc   = LoShuHyperCell()
        orig = hc.norm1().copy()
        p    = Perspective(phase_idx=1, shift_r=0, shift_c=0)
        hc.set_perspective(p)
        new  = hc.norm1()
        assert not np.array_equal(orig, new), "Perspective change had no effect"

    def test_identity_perspective_stable(self):
        hc1 = LoShuHyperCell()
        hc2 = LoShuHyperCell()
        hc2.set_perspective(Perspective())   # identity
        assert np.array_equal(hc1.norm1(), hc2.norm1())

    def test_returns_self(self):
        hc  = LoShuHyperCell()
        ret = hc.set_perspective(Perspective())
        assert ret is hc


# ── FLUHyperCell untested methods ──────────────────────────────────────────

class TestHyperCellUntested:

    @pytest.fixture
    def hc(self):
        return FLUHyperCell()

    # ── address_of ────────────────────────────────────────────────────────

    def test_address_of_all_cells(self, hc):
        """address_of(norm1) finds every cell via the inner LoShuHyperCell."""
        for r in range(9):
            for c in range(9):
                norm1 = hc.cell(r, c).norm1
                pos   = hc.address_of(norm1)
                assert pos == (r, c)

    def test_address_of_missing_returns_none(self, hc):
        assert hc.address_of(0)  is None
        assert hc.address_of(99) is None

    # ── gnosis ────────────────────────────────────────────────────────────

    def test_gnosis_shape(self, hc):
        g = hc.gnosis()
        assert g.shape == (9, 9)

    def test_gnosis_equals_unity_times_omega(self, hc):
        """gnosis = unity × Ω."""
        np.testing.assert_allclose(hc.gnosis(), hc.unity() * hc.contract.omega)

    def test_gnosis_rescales_with_omega(self, hc):
        hc.set_omega(2.0)
        np.testing.assert_allclose(hc.gnosis(), hc.unity() * 2.0)

    # ── set_omega ────────────────────────────────────────────────────────

    def test_set_omega_updates_contract(self, hc):
        hc.set_omega(3.14)
        assert hc.contract.omega == pytest.approx(3.14)

    def test_set_omega_frozen_ignored(self, hc):
        hc.contract.freeze()
        old_omega = hc.contract.omega
        hc.set_omega(99.0)       # should silently skip contract mutation
        assert hc.contract.omega == old_omega

    def test_set_omega_returns_self(self, hc):
        assert hc.set_omega(1.5) is hc

    # ── set_perspective ──────────────────────────────────────────────────

    def test_set_perspective_changes_grid(self, hc):
        orig = hc.norm1().copy()
        p    = Perspective(phase_idx=2, shift_r=1, shift_c=0)
        hc.set_perspective(p)
        assert not np.array_equal(orig, hc.norm1())

    def test_set_perspective_syncs_contract_phi(self, hc):
        p = Perspective(phase_idx=3, shift_r=0, shift_c=2)
        hc.set_perspective(p)
        assert hc.contract.phi["phase_idx"] == 3
        assert hc.contract.phi["shift_c"]   == 2

    def test_set_perspective_frozen_skips_contract(self, hc):
        hc.contract.freeze()
        old_phi = dict(hc.contract.phi)
        hc.set_perspective(Perspective(phase_idx=5, shift_r=0, shift_c=0))
        assert hc.contract.phi == old_phi   # unchanged

    def test_set_perspective_returns_self(self, hc):
        assert hc.set_perspective(Perspective()) is hc

    # ── sparse_step_index ────────────────────────────────────────────────

    def test_sparse_step_index_range(self, hc):
        """Every cell's sparse_step_index is in [0, 80]."""
        for r in range(9):
            for c in range(9):
                idx = hc.sparse_step_index(r, c)
                assert 0 <= idx <= 80, f"({r},{c}): step_index={idx}"

    def test_sparse_step_index_equals_norm0(self, hc):
        for r in range(9):
            for c in range(9):
                assert hc.sparse_step_index(r, c) == hc.cell(r, c).norm0

    def test_sparse_step_indices_all_unique(self, hc):
        indices = {hc.sparse_step_index(r, c) for r in range(9) for c in range(9)}
        assert len(indices) == 81

    # ── zoom_in / zoom_out ────────────────────────────────────────────────

    def test_zoom_in_increments_tau(self, hc):
        tau_before = hc.contract.tau
        hc.zoom_in()
        assert hc.contract.tau == tau_before + 1

    def test_zoom_out_decrements_tau(self, hc):
        hc.zoom_in()
        tau_after_in = hc.contract.tau
        hc.zoom_out()
        assert hc.contract.tau == tau_after_in - 1

    def test_zoom_in_returns_self(self, hc):
        assert hc.zoom_in() is hc

    def test_zoom_out_returns_self(self, hc):
        assert hc.zoom_out() is hc

    def test_zoom_sequence(self, hc):
        """zoom_in three times then zoom_out three times returns to tau=0."""
        for _ in range(3):
            hc.zoom_in()
        for _ in range(3):
            hc.zoom_out()
        assert hc.contract.tau == 0
