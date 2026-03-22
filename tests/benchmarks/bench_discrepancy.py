"""
tests/benchmarks/bench_discrepancy.py
======================================
L2-Star Discrepancy Benchmark for FractalNet vs. Competitors (OD-27, V15.1.4).

Measures how uniformly the sequence fills [0,1)^d using the Warnock
L2-star discrepancy formula.  Compares:
  - FLU FractalNet (van der Corput ordering)
  - FLU FractalNetKinetic (FM-Dance ordering, T9 PROVEN)
  - FLU FractalNet APN-scrambled (DN2 PARTIAL)
  - Halton sequence (base-prime)
  - SciPy Sobol (when available; warning: base-2, degrades at non-power-of-2 N)
  - NumPy pseudo-random Monte Carlo

STATUS: Empirical validation of OD-27 — confirming low-discrepancy property.
REFERENCES: FMD-NET (PROVEN), OD-33 (PROVEN), T9 (PROVEN), DN2 (PARTIAL).
"""

import time
import warnings
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from flu.core.fractal_net import FractalNet, FractalNetKinetic

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def warnock_l2_discrepancy(points: np.ndarray) -> float:
    """
    Warnock's analytical formula for L2-star discrepancy.  O(N²) time.
    Lower is better; random sequence ≈ 0.188 for N=600, d=4.
    """
    N, d = points.shape
    sum1 = np.sum(np.prod(1.0 - (points ** 2) / 2.0, axis=1))
    sum2 = 0.0
    for i in range(N):
        max_vals = np.maximum(points[i], points)
        sum2 += np.sum(np.prod(1.0 - max_vals, axis=1))
    term1 = 3.0 ** (-d)
    term2 = (2.0 ** (1 - d) / N) * sum1
    term3 = (1.0 / (N ** 2)) * sum2
    return float(np.sqrt(abs(term1 - term2 + term3)))


def halton_points(N: int, d: int) -> np.ndarray:
    """Pure-Python Halton sequence using the first d primes (base-prime)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:d]
    pts = np.zeros((N, d))
    for j, p in enumerate(primes):
        for i in range(N):
            r = 0.0; f = 1.0 / p; k = i + 1
            while k > 0:
                r += (k % p) * f
                k //= p; f /= p
            pts[i, j] = r
    return pts


def run(n: int = 3, d: int = 4, num_points: int = 600, verbose: bool = True) -> dict:
    """Run the discrepancy comparison across all sequences and return results dict."""

    if verbose:
        print(f"Digital Net Discrepancy Benchmark (OD-27, V15.1.4)")
        print(f"Parameters: n={n}, d={d}, points={num_points}\n")

    rng = np.random.default_rng(42)

    # 1. FLU FractalNet (plain — van der Corput ordering)
    t0 = time.perf_counter()
    net = FractalNet(n, d)
    flu_pts = net.generate(num_points)
    flu_gen_time = time.perf_counter() - t0
    flu_disc = warnock_l2_discrepancy(flu_pts)

    # 2. FLU FractalNetKinetic (FM-Dance ordering, T9 PROVEN)
    net_k = FractalNetKinetic(n, d)
    flu_k_pts = net_k.generate(num_points)
    flu_k_disc = warnock_l2_discrepancy(flu_k_pts)

    # 3. FLU FractalNet (APN-scrambled per-depth, DN2 architecture V15.1.4)
    flu_scr_pts  = net.generate_scrambled(num_points, seed_rank=0)
    flu_scr_disc = warnock_l2_discrepancy(flu_scr_pts)

    # 4. Halton sequence (pure-Python, base-prime)
    hlt_pts  = halton_points(num_points, d)
    hlt_disc = warnock_l2_discrepancy(hlt_pts)

    # 5. Standard Monte Carlo (Pseudo-Random)
    mc_pts   = rng.random((num_points, d))
    mc_disc  = warnock_l2_discrepancy(mc_pts)

    # 6. Sobol (if SciPy installed)
    sobol_disc = None
    if HAS_SCIPY:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler    = qmc.Sobol(d=d, scramble=False, seed=42)
            sobol_pts  = sampler.random(num_points)
        sobol_disc = warnock_l2_discrepancy(sobol_pts)

    if verbose:
        print(f"1. Monte Carlo (Random)       L2* : {mc_disc:.6f}")
        print(f"2. Halton (base-prime)        L2* : {hlt_disc:.6f}")
        if HAS_SCIPY:
            print(f"3. SciPy Sobol (base-2)†     L2* : {sobol_disc:.6f}  †degrades at non-2^k N")
        print(f"4. FLU FractalNet            L2* : {flu_disc:.6f}  "
              f"(gen: {flu_gen_time:.4f}s)")
        print(f"5. FLU FractalNetKinetic     L2* : {flu_k_disc:.6f}  (T9 PROVEN)")
        print(f"6. FLU FractalNet (APN)      L2* : {flu_scr_disc:.6f}  (DN2 PARTIAL per-depth)")
        print()
        print("Analysis:")
        if flu_disc < mc_disc:
            pct = (1 - flu_disc / mc_disc) * 100
            print(f"  ✓ FLU FractalNet beats random by {pct:.1f}% (OD-27 confirmed).")
        else:
            print(f"  ✗ FLU FractalNet is worse than random noise.")
        if flu_disc < hlt_disc:
            print(f"  ✓ FLU FractalNet beats Halton ({flu_disc:.4f} < {hlt_disc:.4f}).")
        if HAS_SCIPY and sobol_disc is not None:
            if flu_disc <= sobol_disc:
                print(f"  🏆 FLU matches or beats Sobol ({flu_disc:.4f} ≤ {sobol_disc:.4f}).")
            else:
                pct_sob = (1 - flu_disc / sobol_disc) * 100
                print(f"  - Sobol: {sobol_disc:.4f} vs FLU: {flu_disc:.4f} "
                      f"(note: Sobol degrades at non-2^k N).")
        if flu_k_disc < flu_disc:
            print(f"  ✓ FractalNetKinetic has better L2 ({flu_k_disc:.4f} < {flu_disc:.4f}).")
        elif flu_k_disc > flu_disc:
            print(f"  - FractalNet better than FractalNetKinetic at this N "
                  f"({flu_disc:.4f} < {flu_k_disc:.4f}); they converge at N=n^(2d).")

    return {
        "n"              : n,
        "d"              : d,
        "num_points"     : num_points,
        "flu_disc"       : flu_disc,
        "flu_k_disc"     : flu_k_disc,
        "flu_scr_disc"   : flu_scr_disc,
        "halton_disc"    : hlt_disc,
        "mc_disc"        : mc_disc,
        "sobol_disc"     : sobol_disc,
        "beats_random"   : flu_disc < mc_disc,
        "beats_halton"   : flu_disc < hlt_disc,
        "beats_sobol"    : sobol_disc is None or flu_disc <= sobol_disc,
        "scramble_helps" : flu_scr_disc < flu_disc,
    }


def run_sweep(n: int = 3, d: int = 4) -> None:
    """
    Run discrepancy comparison at N = n^k for k=1..2d, printing a full table.
    This reproduces Table 1 from docs/BENCHMARK_FRACTALNET.md with Halton added.
    """
    import warnings
    rng = np.random.default_rng(42)
    net_c = FractalNet(n, d)
    net_k = FractalNetKinetic(n, d)

    print(f"\nL2-star discrepancy sweep (n={n}, d={d})")
    print(f"{'N':>6} | {'FractalNet':>10} | {'Kinetic':>10} | {'APN':>10} | "
          f"{'Halton':>10} | {'Sobol†':>10} | {'MC':>10} | C-imp% | K-imp%")
    print("-" * 90)

    for exp in range(1, 2 * d + 1):
        N = n ** exp
        flu_c = warnock_l2_discrepancy(net_c.generate(N))
        flu_k = warnock_l2_discrepancy(net_k.generate(N))
        flu_s = warnock_l2_discrepancy(net_c.generate_scrambled(N, 0))
        mc = warnock_l2_discrepancy(rng.random((N, d)))
        hlt = warnock_l2_discrepancy(halton_points(N, d))
        sob_str = "  n/a   "
        if HAS_SCIPY:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sob = warnock_l2_discrepancy(
                    qmc.Sobol(d=d, scramble=False, seed=42).random(N))
            sob_str = f"{sob:10.6f}"
        c_imp = (1 - flu_c / mc) * 100
        k_imp = (1 - flu_k / mc) * 100
        print(f"{N:>6} | {flu_c:10.6f} | {flu_k:10.6f} | {flu_s:10.6f} | "
              f"{hlt:10.6f} | {sob_str} | {mc:10.6f} | {c_imp:5.1f}% | {k_imp:5.1f}%")
    print("\n†Sobol: SciPy implementation uses base-2; balance degrades at non-power-of-2 N.")


if __name__ == "__main__":
    # Standard single-point run
    run(n=3, d=4, num_points=600, verbose=True)
    # Full sweep for comparison table
    run_sweep(n=3, d=4)
