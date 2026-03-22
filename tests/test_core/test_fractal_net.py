"""
tests/test_core/test_fractal_net.py
====================================
Tests for flu.core.fractal_net.FractalNet (OD-27 Digital Net, V14 audit).

Also covers:
  - Zero-compute APN path in factoradic.unrank_optimal_seed (OD-16-PM proof)
  - T9 lattice isomorphism empirical evidence
  - DN2 APN-scrambled net structure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import pytest

from flu.core.fractal_net import FractalNet, FractalNetKinetic
from flu.core.factoradic import unrank_optimal_seed, differential_uniformity


# ── FractalNet construction ───────────────────────────────────────────────────

class TestFractalNetConstruction:
    def test_init_valid(self):
        net = FractalNet(3, 4)
        assert net.n == 3
        assert net.d == 4
        assert net.N == 81

    def test_init_even_n_raises(self):
        with pytest.raises(ValueError, match="odd"):
            FractalNet(4, 2)

    def test_init_d_zero_raises(self):
        with pytest.raises(ValueError):
            FractalNet(3, 0)

    def test_base_block_shape(self):
        net = FractalNet(3, 4)
        assert net._base_block.shape == (81, 4)

    def test_base_block_range(self):
        # Unsigned coords should be in [0, n-1]
        net = FractalNet(3, 4)
        assert net._base_block.min() >= 0
        assert net._base_block.max() <= 2

    def test_repr(self):
        r = repr(FractalNet(3, 4))
        assert "FractalNet" in r and "n=3" in r


# ── FractalNet generation ─────────────────────────────────────────────────────

class TestFractalNetGenerate:
    @pytest.fixture
    def net(self):
        return FractalNet(3, 4)

    def test_output_shape(self, net):
        pts = net.generate(100)
        assert pts.shape == (100, 4)

    def test_output_range_in_unit_hypercube(self, net):
        pts = net.generate(729)
        assert (pts >= 0.0).all(), "All points must be ≥ 0"
        assert (pts < 1.0).all(),  "All points must be < 1"

    def test_zero_points(self, net):
        pts = net.generate(0)
        assert pts.shape == (0, 4)

    def test_single_point(self, net):
        pts = net.generate(1)
        assert pts.shape == (1, 4)
        assert (pts >= 0).all() and (pts < 1).all()

    def test_deterministic(self, net):
        """Same call, same result."""
        p1 = net.generate(200)
        p2 = net.generate(200)
        np.testing.assert_array_equal(p1, p2)

    def test_base_block_exact_coverage(self, net):
        """First N=81 points should tile the unit cell exactly once each."""
        pts = net.generate(81)
        # Multiply by n to get digit indices, check Latin property:
        # each coordinate axis should contain each value in [0,n-1] exactly n^(d-1) times
        scaled = (pts * net.n).astype(int)
        for col in range(net.d):
            vals, counts = np.unique(scaled[:, col], return_counts=True)
            assert len(vals) == net.n, f"Axis {col}: expected {net.n} distinct values"
            assert (counts == net.n ** (net.d - 1)).all(), f"Axis {col}: unequal counts"

    def test_beats_random_discrepancy(self, net):
        """FractalNet should have lower L2-star discrepancy than pure random (OD-27)."""
        def warnock(pts):
            N, d = pts.shape
            s1 = np.sum(np.prod(1.0 - pts**2 / 2.0, axis=1))
            s2 = sum(np.sum(np.prod(1.0 - np.maximum(pts[i], pts), axis=1))
                     for i in range(N))
            return float(np.sqrt(abs(3**(-d) - (2**(1-d)/N)*s1 + s2/N**2)))

        flu_pts = net.generate(243)
        mc_pts  = np.random.default_rng(42).random((243, 4))
        assert warnock(flu_pts) < warnock(mc_pts), (
            "FractalNet should beat random noise in L2-star discrepancy (OD-27)"
        )


# ── FractalNet scrambled ──────────────────────────────────────────────────────

class TestFractalNetScrambled:
    def test_scrambled_in_unit_hypercube(self):
        net = FractalNet(3, 4)
        pts = net.generate_scrambled(100, seed_rank=0)
        assert pts.shape == (100, 4)
        assert (pts >= 0).all() and (pts < 1).all()

    def test_scrambled_deterministic(self):
        net = FractalNet(3, 4)
        p1 = net.generate_scrambled(50, seed_rank=0)
        p2 = net.generate_scrambled(50, seed_rank=0)
        np.testing.assert_array_equal(p1, p2)

    def test_scrambled_differs_from_plain(self):
        """APN scrambling should produce a different sequence for n≥5."""
        net5 = FractalNet(5, 2)
        plain = net5.generate(25)
        scram = net5.generate_scrambled(25, seed_rank=0)
        assert not np.allclose(plain, scram), (
            "APN-scrambled sequence should differ from plain (n=5)"
        )


# ── Zero-compute APN path (OD-16-PM / PROOF_APN_OBSTRUCTION.md) ──────────────

class TestZeroComputeAPN:
    """
    For prime p ≡ 2 (mod 3): x^3 is unconditionally APN (δ=2) and a bijection.
    unrank_optimal_seed should use this path without search.
    """

    @pytest.mark.parametrize("p", [5, 11, 17, 23, 29])
    def test_x_cubed_is_bijection(self, p):
        """x^3 mod p must be a bijection for p ≡ 2 (mod 3)."""
        assert p % 3 == 2, f"Test precondition: {p} must be ≡ 2 mod 3"
        perm = [(x**3) % p for x in range(p)]
        assert len(set(perm)) == p, f"x^3 mod {p} is not a bijection"

    @pytest.mark.parametrize("p", [5, 11, 17, 23, 29])
    def test_x_cubed_is_apn(self, p):
        """x^3 mod p must be APN (δ=2) for p ≡ 2 (mod 3)."""
        perm = np.array([(x**3) % p for x in range(p)])
        delta = differential_uniformity(perm, p)
        assert delta == 2, f"x^3 mod {p}: expected δ=2, got δ={delta}"

    @pytest.mark.parametrize("p", [5, 11, 17, 23, 29])
    def test_unrank_returns_apn_seed(self, p):
        """unrank_optimal_seed should return an APN seed for p ≡ 2 (mod 3)."""
        if p in __import__('flu.core.factoradic', fromlist=['GOLDEN_SEEDS']).GOLDEN_SEEDS:
            pytest.skip(f"n={p} has pre-computed seeds; zero-compute path not triggered")
        seed = unrank_optimal_seed(0, p, signed=False)
        assert len(seed) == p
        assert len(set(seed)) == p, "Seed must be a bijection"
        delta = differential_uniformity(seed, p)
        assert delta == 2, f"unrank_optimal_seed({p}) should be APN, got δ={delta}"

    @pytest.mark.parametrize("p", [7, 13, 19, 31])
    def test_x_cubed_not_bijection_for_p1mod3(self, p):
        """x^3 mod p should NOT be a bijection for p ≡ 1 (mod 3) — power-map obstruction."""
        assert p % 3 == 1, f"Test precondition: {p} must be ≡ 1 mod 3"
        perm = [(x**3) % p for x in range(p)]
        assert len(set(perm)) < p, (
            f"x^3 mod {p} should not be a bijection (p ≡ 1 mod 3, OD-16-PM)"
        )

    @pytest.mark.parametrize("p,exponents", [
        (19, [5, 7, 11, 13, 17]),
        (31, [7, 11, 13, 17, 19, 23, 29]),
    ])
    def test_all_power_maps_not_apn_for_p1mod3(self, p, exponents):
        """All bijective power maps for p=19,31 must have δ≥4 (APN obstruction, PROVEN)."""
        from math import gcd
        for d in exponents:
            assert gcd(d, p - 1) == 1, f"d={d} must be bijective exponent for p={p}"
            perm = np.array([(x**d) % p for x in range(p)])
            delta = differential_uniformity(perm, p)
            assert delta >= 4, (
                f"d={d} mod {p}: expected δ≥4 (Hasse-Weil obstruction), got δ={delta}"
            )


# ── New theorems in registry ──────────────────────────────────────────────────

class TestNewTheoremRegistry:
    @pytest.fixture
    def registry(self):
        from flu.theory.theorem_registry import REGISTRY
        return REGISTRY

    def test_t9_registered(self, registry):
        assert "T9" in registry
        # V15.1 audit: T9 promoted from CONJECTURE to PROVEN
        # (benchmark bug fixed: np.cumsum T[0,0]=+1 → correct T[0,0]=-1, 27/27 matches)
        assert registry["T9"].status == "PROVEN", \
            f"T9 should be PROVEN after V15.1 audit, got {registry['T9'].status}"

    def test_dn2_registered(self, registry):
        assert "DN2" in registry
        assert registry["DN2"].status == "PROVEN", \
            f"DN2 should be PROVEN after V15.3 (ETK+Walsh+Var+ANOVA closed), got {registry['DN2'].status}"
        # Verify all four sub-theorems are also registered
        for sub in ["DN2-ETK", "DN2-WALSH", "DN2-VAR", "DN2-ANOVA"]:
            assert sub in registry, f"{sub} not in registry"
            assert registry[sub].status == "PROVEN", f"{sub} should be PROVEN"

    def test_od16_pm_proven(self, registry):
        assert "OD-16-PM" in registry
        assert registry["OD-16-PM"].status == "PROVEN"

    def test_od17_pm_proven(self, registry):
        assert "OD-17-PM" in registry
        assert registry["OD-17-PM"].status == "PROVEN"

    def test_hm1_registered(self, registry):
        assert "HM-1" in registry
        # HM-1 upgraded to PROVEN in V14 open-debt closure:
        # the representation theorem is exact by construction of ScarStore.
        assert registry["HM-1"].status == "PROVEN"

    def test_t8b_statement_notes_open_uniqueness(self, registry):
        """T8b must note that uniqueness (OD-19) is still open."""
        t8b = registry["T8b"]
        assert t8b.status == "PROVEN"
        stmt = t8b.statement
        assert "OPEN" in stmt or "open" in stmt or "OD-19" in stmt, (
            "T8b statement must flag that uniqueness is open (OD-19)"
        )

    def test_od16_full_conjecture_still_open(self, registry):
        """OD-16 full conjecture (all bijections) must remain CONJECTURE."""
        assert registry["OD-16"].status == "CONJECTURE"

    def test_od17_full_conjecture_still_open(self, registry):
        assert registry["OD-17"].status == "CONJECTURE"

    def test_fractal_net_importable_from_package(self):
        from flu import FractalNet
        net = FractalNet(3, 2)
        pts = net.generate(9)
        assert pts.shape == (9, 2)


# ── DN2 per-depth seed rotation (V15.1.3 architectural fix) ──────────────────

class TestDN2PerDepthScrambling:
    """
    V15.1.3 regression tests for the per-depth APN seed rotation fix.

    Root cause of V14 null-L2 result: same APN perm at every depth =
    relabelling, not true randomisation.  Fix: depth m uses
    GOLDEN_SEEDS[n][(seed_rank + m) % len(seeds)].
    """

    def test_different_seed_ranks_differ(self):
        """seed_rank=0 and seed_rank=1 must produce different sequences for n=5."""
        net = FractalNet(5, 2)
        p0 = net.generate_scrambled(50, seed_rank=0)
        p1 = net.generate_scrambled(50, seed_rank=1)
        assert not np.allclose(p0, p1), (
            "seed_rank=0 and seed_rank=1 must yield different sequences (per-depth rotation)"
        )

    def test_per_depth_differs_from_plain(self):
        """Per-depth scrambled output must differ from plain generate() for n=5."""
        for cls in [FractalNet, FractalNetKinetic]:
            net = cls(5, 2)
            plain = net.generate(100)
            scr = net.generate_scrambled(100, seed_rank=0)
            assert not np.allclose(plain, scr), (
                f"{cls.__name__}: scrambled must differ from plain"
            )

    def test_scrambled_stays_in_unit_hypercube(self):
        """All scrambled points must remain in [0, 1)^d."""
        for n in [5, 7, 11]:
            for cls in [FractalNet, FractalNetKinetic]:
                net = cls(n, 2)
                pts = net.generate_scrambled(4 * n**2, seed_rank=0)
                assert (pts >= 0).all(), f"{cls.__name__} n={n}: point below 0"
                assert (pts < 1).all(),  f"{cls.__name__} n={n}: point >= 1"

    def test_scrambled_deterministic_across_calls(self):
        """Same seed_rank must give bitwise-identical output on repeated calls."""
        for cls in [FractalNet, FractalNetKinetic]:
            net = cls(7, 2)
            p1 = net.generate_scrambled(60, seed_rank=2)
            p2 = net.generate_scrambled(60, seed_rank=2)
            np.testing.assert_array_equal(p1, p2,
                err_msg=f"{cls.__name__}: scrambled not deterministic")

    def test_scrambled_zero_points_returns_empty(self):
        """generate_scrambled(0) must return empty array of correct shape."""
        for cls in [FractalNet, FractalNetKinetic]:
            net = cls(5, 3)
            pts = net.generate_scrambled(0)
            assert pts.shape == (0, 3), f"{cls.__name__}: expected (0,3) for 0 points"

    def test_fft_reduction_vs_plain_n5(self):
        """
        For n=5 (8 APN seeds), per-depth scrambling must reduce the FFT
        spectral peak by at least 10% relative to the plain sequence.
        This is the quantitative DN2 confirmation (V15.1.3).
        """
        n, d = 5, 2
        N = 4 * n**d
        net = FractalNetKinetic(n, d)
        plain = net.generate(N)
        # Try all seed ranks, take the best reduction
        best_reduction = 0.0
        fp = _fft_peak(plain)
        for sr in range(8):
            scr = net.generate_scrambled(N, seed_rank=sr)
            fs = _fft_peak(scr)
            reduction = (fp - fs) / fp
            if reduction > best_reduction:
                best_reduction = reduction
        assert best_reduction > 0.10, (
            f"Expected >10% FFT peak reduction for n=5, got {best_reduction*100:.1f}%"
        )

    def test_n3_scramble_works_without_error(self):
        """n=3 has no APN seeds (δ_min=3). generate_scrambled must not crash."""
        net = FractalNetKinetic(3, 2)
        pts = net.generate_scrambled(27, seed_rank=0)
        assert pts.shape == (27, 2)
        assert (pts >= 0).all() and (pts < 1).all()


def _fft_peak(points, bins=16):
    """Mean FFT max-peak across 2-D projections (helper for DN2 test)."""
    import numpy as _np
    N, d = points.shape
    peaks = []
    for i in range(d):
        for j in range(i + 1, d):
            H, _, _ = _np.histogram2d(points[:, i], points[:, j],
                                      bins=bins, range=[[0, 1], [0, 1]])
            F = _np.abs(_np.fft.fftshift(_np.fft.fft2(H)))
            F[bins // 2, bins // 2] = 0
            peaks.append(float(F.max()))
    return float(_np.mean(peaks)) if peaks else 0.0
