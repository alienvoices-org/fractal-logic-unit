"""
tests/test_core/test_apn_seeds.py
====================================
APN seed arithmetic, golden seeds, spectral flatness (S2) and spectral
dispersion bound (S2-Prime).

Theorems verified: S2, S2-Prime, OD-16-PM / OD-17-PM, GEN-1.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import unittest


# ── PN Flatness / S2 ─────────────────────────────────────────────────────────

def test_s2_pn_flatness():
    """S2 (PROVEN): communion with PN seeds has spectral mixed-variance ≈ 0."""
    from flu.theory.theory_spectral import compute_spectral_profile
    n = 3
    half = n // 2
    # [0,2,1] = x ↦ 2x mod 3, a PN permutation for n=3
    pi_s = np.array([0, 2, 1]) - half   # signed: [-1, 1, 0]
    M = np.array([[pi_s[i] + pi_s[j] for j in range(n)] for i in range(n)])
    profile = compute_spectral_profile(M, n)
    assert profile["mixed_variance"] < 0.01, \
        f"S2: mixed_variance={profile['mixed_variance']} expected < 0.01"
    assert abs(profile["dc_magnitude"]) < 1e-6, "S2: DC should be zero"


# ── APN seed hub ─────────────────────────────────────────────────────────────

def test_apn_identity_has_high_delta():
    """Identity permutation has δ = n−1 (worst case)."""
    from flu.core.factoradic import differential_uniformity
    for n in [3, 5]:
        identity = np.arange(n)
        delta = differential_uniformity(identity, n)
        assert delta >= 2, f"identity delta n={n}: {delta}"

def test_apn_nonlinearity_equals_differential_uniformity():
    """nonlinearity_score == differential_uniformity for same permutation."""
    from flu.core.factoradic import differential_uniformity, nonlinearity_score
    pi3 = np.array([0, 2, 1])
    assert nonlinearity_score(pi3, 3) == differential_uniformity(pi3, 3)

def test_apn_identity_not_pn():
    """Identity permutation is NOT PN for n >= 5."""
    from flu.core.factoradic import is_pn_permutation
    assert not is_pn_permutation(np.arange(5), 5)

def test_apn_golden_seeds_exist():
    """GOLDEN_SEEDS has entries for standard n values."""
    from flu.core.factoradic import GOLDEN_SEEDS
    for n in [3, 5, 7]:
        assert n in GOLDEN_SEEDS, f"GOLDEN_SEEDS missing n={n}"

def test_apn_unrank_optimal_seed_shape():
    """unrank_optimal_seed(0, n) returns a valid permutation array."""
    from flu.core.factoradic import unrank_optimal_seed
    seed = unrank_optimal_seed(0, 3, signed=False)
    assert len(seed) == 3
    assert set(int(x) for x in seed) == {0, 1, 2}

def test_apn_unrank_optimal_seed_signed():
    """unrank_optimal_seed signed=True returns length-n array."""
    from flu.core.factoradic import unrank_optimal_seed
    seed_s = unrank_optimal_seed(0, 3, signed=True)
    assert len(seed_s) == 3


# ── apn_search_vectorized ────────────────────────────────────────────────────

class TestAPNSearchVectorized(unittest.TestCase):

    def test_function_importable(self):
        from flu.core.factoradic import apn_search_vectorized
        self.assertTrue(callable(apn_search_vectorized))

    def test_small_n_finds_apn(self):
        """For n=7 (APN known), short search should find δ=2."""
        from flu.core.factoradic import apn_search_vectorized
        result = apn_search_vectorized(n=7, trials=200_000, rng_seed=42)
        self.assertEqual(result["best_delta"], 2,
            f"Expected δ=2 for n=7, got {result['best_delta']}")
        self.assertEqual(result["status"], "FOUND")
        self.assertGreater(len(result["ranks"]), 0)

    def test_n19_best_delta_is_3(self):
        """500K trials for n=19 — must not find δ=2 (OD-16 conjecture)."""
        from flu.core.factoradic import apn_search_vectorized
        result = apn_search_vectorized(n=19, trials=500_000, rng_seed=99)
        self.assertGreaterEqual(result["best_delta"], 3,
            f"Unexpectedly found δ<3 for n=19: {result}")
        self.assertIn(result["status"], ("BEST_DELTA_ONLY", "SEARCHED_NO_RESULT"))

    def test_returns_valid_structure(self):
        from flu.core.factoradic import apn_search_vectorized
        result = apn_search_vectorized(n=5, trials=10_000, rng_seed=0)
        for key in ("n", "best_delta", "ranks", "elapsed_sec", "status"):
            self.assertIn(key, result)
        self.assertIsInstance(result["ranks"], list)

    def test_found_ranks_round_trip(self):
        """Every returned rank decodes to a permutation with the claimed δ."""
        from flu.core.factoradic import (
            apn_search_vectorized, factoradic_unrank, differential_uniformity,
        )
        result = apn_search_vectorized(n=5, trials=50_000, rng_seed=7)
        for rank in result["ranks"]:
            pi    = factoradic_unrank(rank, 5, signed=False)
            delta = differential_uniformity(pi, 5)
            self.assertEqual(delta, result["best_delta"],
                f"Rank {rank} decoded to δ={delta}, expected {result['best_delta']}")


# ── Golden Seeds n=19 / n=31 ─────────────────────────────────────────────────

class TestGoldenSeedsN19N31(unittest.TestCase):

    def _verify_seeds(self, n, expected_delta):
        from flu.core.factoradic import GOLDEN_SEEDS, factoradic_unrank, differential_uniformity
        self.assertIn(n, GOLDEN_SEEDS)
        seeds = GOLDEN_SEEDS[n]
        self.assertGreaterEqual(len(seeds), 8, f"Expected ≥8 seeds for n={n}")
        for rank in seeds:
            pi    = factoradic_unrank(rank, n, signed=False)
            delta = differential_uniformity(pi, n)
            self.assertEqual(delta, expected_delta,
                f"n={n} rank={rank}: expected δ={expected_delta}, got {delta}")

    def test_n19_seeds_all_delta3(self):   self._verify_seeds(19, 3)
    def test_n31_seeds_all_delta3(self):   self._verify_seeds(31, 3)

    def test_n19_seeds_are_bijections(self):
        from flu.core.factoradic import GOLDEN_SEEDS, factoradic_unrank
        for rank in GOLDEN_SEEDS[19]:
            pi = factoradic_unrank(rank, 19, signed=False)
            self.assertEqual(sorted(pi.tolist()), list(range(19)))

    def test_n31_seeds_are_bijections(self):
        from flu.core.factoradic import GOLDEN_SEEDS, factoradic_unrank
        for rank in GOLDEN_SEEDS[31]:
            pi = factoradic_unrank(rank, 31, signed=False)
            self.assertEqual(sorted(pi.tolist()), list(range(31)))

    def test_n19_ranks_are_unique(self):
        from flu.core.factoradic import GOLDEN_SEEDS
        ranks = GOLDEN_SEEDS[19]
        self.assertEqual(len(ranks), len(set(ranks)))

    def test_n31_ranks_are_unique(self):
        from flu.core.factoradic import GOLDEN_SEEDS
        ranks = GOLDEN_SEEDS[31]
        self.assertEqual(len(ranks), len(set(ranks)))


# ── S2-Prime Spectral Dispersion Bound ───────────────────────────────────────

def test_s2prime_pn_bound_lt_apn_bound():
    """S2-Prime: PN bound (δ=1) < APN bound (δ=2)."""
    from flu.theory.theory_spectral import spectral_dispersion_bound
    b1 = spectral_dispersion_bound(delta_max=1, n=5, d=2)
    b2 = spectral_dispersion_bound(delta_max=2, n=5, d=2)
    assert b1 < b2

def test_s2prime_bound_positive():
    from flu.theory.theory_spectral import spectral_dispersion_bound
    assert spectral_dispersion_bound(delta_max=2, n=5, d=2) > 0

def test_s2prime_class_bound_positive():
    from flu.theory.theory_spectral import SpectralDispersionBound
    sdb = SpectralDispersionBound(n=5, d=2, delta_max=2)
    assert sdb.bound > 0

def test_s2prime_communion_satisfies_bound():
    """Communion array built from APN seeds satisfies the S2-Prime bound."""
    from flu.theory.theory_spectral import SpectralDispersionBound
    from flu.core.factoradic import factoradic_unrank
    n, d = 5, 2
    sdb = SpectralDispersionBound(n=n, d=d, delta_max=2)
    seeds = [factoradic_unrank(k, n, signed=True) for k in range(d)]
    M = np.array([[seeds[0][i] + seeds[1][j] for j in range(n)] for i in range(n)])
    result = sdb.verify(M)
    assert result["satisfied"]
    assert result["theorem"] == "S2-Prime — Bounded Spectral Dispersion"


# ── Benchmarks smoke test ────────────────────────────────────────────────────

def test_benchmark_addressing():
    from flu.utils.benchmarks import addressing_benchmark
    addr = addressing_benchmark(n=3, d_values=[2, 4, 8], n_reps=20)
    assert "complexity_ok" in addr

def test_benchmark_traversal():
    from flu.utils.benchmarks import traversal_benchmark
    trav = traversal_benchmark(n=3, d=3, n_steps=50)
    assert "amortised_ok" in trav

def test_benchmark_spectral():
    from flu.utils.benchmarks import spectral_variance_bench
    spec = spectral_variance_bench(n=5, d_values=[2])
    assert "all_within_s2prime_bound" in spec

def test_benchmark_full_report():
    from flu.utils.benchmarks import full_benchmark_report
    report = full_benchmark_report(n=3, verbose=False)
    assert all(k in report for k in ["addressing", "traversal", "spectral", "avalanche", "summary"])
