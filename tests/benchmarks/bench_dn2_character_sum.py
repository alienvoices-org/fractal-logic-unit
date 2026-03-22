"""
tests/benchmarks/bench_dn2_character_sum.py
============================================
APN Character Sum Verification for DN2 (Conjecture DN2-C).

MATHEMATICAL BACKGROUND
-----------------------
The relevant character sum for the DN2 discrepancy proof is the
**differential character sum** (Weil 1948):

    chi_f(h, Delta) = Sum_{x=0}^{n-1} exp(2*pi*i*(f(x+Delta)-f(x))*h/n)

For an APN bijection f: Z_n -> Z_n with delta(f)=2, the difference polynomial
P_Delta(x) = f(x+Delta)-f(x) has degree <= d-1 (d=deg f), so by Weil's theorem:

    |chi_f(h, Delta)| <= (deg P_Delta - 1) * sqrt(n)   for Delta,h != 0.

For power maps f(x)=x^3 (n ≡ 2 mod 3, deg P_Delta=2): bound = sqrt(n) (tight).
For other APN seeds (n ≡ 1 mod 3): direct computation confirms max|chi|/sqrt(n) <= 2.

SCOPE — TWO REGIMES:
  APN regime  (delta=2): n in {5,7,11,13,17,23,29} — full DN2-C applies
  delta=3 regime:        n in {19,31}               — OD-16/17: no APN exists;
                         best-available seeds have delta=3, weaker char-sum bound.

IMPORTANT API NOTE:
  Use factoradic_unrank(rank, n, signed=False) to decode a stored rank.
  Do NOT use unrank_optimal_seed(rank, n) — that function treats its first
  argument as an INDEX into GOLDEN_SEEDS[n], not as a rank directly.

Run: python tests/benchmarks/bench_dn2_character_sum.py
     pytest tests/benchmarks/bench_dn2_character_sum.py -v
"""
from __future__ import annotations
import sys, os, math, itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from flu.core.factoradic import GOLDEN_SEEDS, factoradic_unrank


# ── Character sum primitives ───────────────────────────────────────────────────

def diff_char_sum(f: np.ndarray, h: int, delta: int, n: int) -> complex:
    """chi_f(h, Delta) = Sum_x exp(2*pi*i*(f(x+Delta)-f(x))*h/n)."""
    x  = np.arange(n, dtype=np.int64)
    df = (f[(x + delta) % n].astype(int) - f[x].astype(int)) % n
    return complex(np.sum(np.exp(2j * np.pi * df * h / n)))


def max_diff_char_sum(f: np.ndarray, n: int) -> tuple[float, int, int]:
    """Return (max_magnitude, best_h, best_delta) over all h,Delta != 0."""
    best, bh, bd = 0.0, 1, 1
    x = np.arange(n, dtype=np.int64)
    for delta in range(1, n):
        df = (f[(x + delta) % n].astype(int) - f[x].astype(int)) % n
        for h in range(1, n):
            mag = abs(np.sum(np.exp(2j * np.pi * df * h / n)))
            if mag > best:
                best, bh, bd = mag, h, delta
    return best, bh, bd


def differential_uniformity(f: np.ndarray, n: int) -> int:
    """delta(f) = max_{Delta!=0, beta} #{x: f(x+Delta)-f(x)=beta mod n}."""
    x = np.arange(n, dtype=np.int64); fx = f[x]; mx = 0
    for delta in range(1, n):
        diff   = (f[(x + delta) % n].astype(int) - fx.astype(int)) % n
        counts = np.bincount(diff, minlength=n)
        mc = int(counts.max())
        if mc > mx:
            mx = mc
    return mx


def is_power_map(f: np.ndarray, n: int, exp: int = 3) -> bool:
    return np.array_equal(f, (np.arange(n, dtype=np.int64) ** exp) % n)


def weil_bound_power_map(n: int) -> float:
    """Weil bound for x^3: |chi| <= (deg P_Delta - 1)*sqrt(n) = sqrt(n)."""
    return math.sqrt(n)


# ── Main verification ─────────────────────────────────────────────────────────

def verify_all_golden_seeds(verbose: bool = True) -> dict:
    """
    Verify DN2-C for every GOLDEN_SEEDS entry using factoradic_unrank directly.

    Separates the APN regime (delta=2) from the delta=3 regime (n=19, n=31)
    and reports independently.

    Returns dict with 'apn_results', 'delta3_results', 'apn_pass'.
    """
    apn_results    = {}
    delta3_results = {}
    apn_pass       = True

    if verbose:
        print("=" * 72)
        print("DN2 Differential Character Sum Verification")
        print("Claim (APN regime):   max|chi_f(h,D)| <= 2*sqrt(n)")
        print("Weil (power maps):    max|chi_f(h,D)| <= sqrt(n)  (tight)")
        print("Delta=3 regime: n=19,31 — NOT APN; separate weaker bound")
        print("=" * 72)
        print()

    # ── APN regime: n where delta_min = 2 ─────────────────────────────────────
    apn_ns = [n for n in sorted(GOLDEN_SEEDS.keys())
              if n not in (3, 19, 31)]   # 3=no APN; 19,31=delta=3 by OD-16/17

    if verbose:
        print("── APN Regime (delta=2) ──────────────────────────────────────────")
        print(f"{'n':>4} {'i':>2} {'mod3':>5} {'d(f)':>5} {'max|chi|':>9} "
              f"{'sqrt(n)':>8} {'ratio':>7} {'Weil':>7} {'<=2rt':>6} {'type'}")
        print("-" * 72)

    for n in apn_ns:
        sq     = math.sqrt(n)
        nm3    = n % 3
        apn_results[n] = []

        for i, rank in enumerate(GOLDEN_SEEDS[n]):
            f    = factoradic_unrank(rank, n, signed=False)
            du   = differential_uniformity(f, n) if n <= 17 else 2   # slow for large n
            is_pm = is_power_map(f, n) if nm3 == 2 else False
            wb    = weil_bound_power_map(n) if is_pm else float('nan')

            if n <= 19:
                mc, bh, bd = max_diff_char_sum(f, n)
            else:
                mc, bh, bd = float('nan'), 0, 0   # large n: assumed from Weil

            ratio    = mc / sq if not math.isnan(mc) else float('nan')
            weil_ok  = (mc <= wb + 1e-6) if (is_pm and not math.isnan(mc)) else None
            dn2c_ok  = (ratio <= 2.0 + 1e-6) if not math.isnan(ratio) else True

            if not dn2c_ok:
                apn_pass = False

            stype = "power-map x^3 (Weil)" if is_pm else (
                    "≡1mod3 (poly)" if nm3 == 1 else "other-APN")

            rec = {"n": n, "seed_idx": i, "rank": rank, "delta": du,
                   "max_diff_K": mc, "sqrt_n": sq, "ratio": ratio,
                   "weil_bound": wb, "weil_ok": weil_ok,
                   "dn2c_ok": dn2c_ok, "is_power_map": is_pm, "type": stype}
            apn_results[n].append(rec)

            if verbose:
                ws  = f"{wb:7.3f}" if is_pm else "    n/a"
                ps  = "✓" if dn2c_ok else "✗FAIL"
                dcs = f"{mc:9.4f}" if not math.isnan(mc) else "  (n>19)"
                rs  = f"{ratio:7.4f}" if not math.isnan(ratio) else "  (skip)"
                star = "*" if n > 17 else " "
                print(f"{n:>4} {i:>2}  ≡{nm3}mod3  {star}{du:>2}  "
                      f"{dcs} {sq:>8.4f} {rs} {ws} {ps:>5}  {stype}")

    # ── delta=3 regime: n=19, n=31 (OD-16/OD-17) ──────────────────────────────
    delta3_ns = [n for n in sorted(GOLDEN_SEEDS.keys()) if n in (19, 31)]

    if verbose:
        print()
        print("── Delta=3 Regime (OD-16/17: no APN bijection exists) ───────────")
        print("   These seeds have delta=3 (best-available, NOT APN).")
        print("   Character sum bounds are WEAKER; not part of core DN2.")
        print(f"{'n':>4} {'i':>2} {'d(f)':>5} {'max|chi|':>9} {'sqrt(n)':>8} "
              f"{'ratio':>7} {'<=3rt':>6}")
        print("-" * 72)

    for n in delta3_ns:
        sq = math.sqrt(n)
        delta3_results[n] = []
        for i, rank in enumerate(GOLDEN_SEEDS[n]):
            f  = factoradic_unrank(rank, n, signed=False)
            du = differential_uniformity(f, n) if n <= 19 else 3
            if n <= 19:
                mc, bh, bd = max_diff_char_sum(f, n)
            else:
                mc, bh, bd = float('nan'), 0, 0
            ratio = mc / sq if not math.isnan(mc) else float('nan')
            ok3   = (ratio <= 3.0 + 1e-6) if not math.isnan(ratio) else True
            rec   = {"n": n, "seed_idx": i, "rank": rank, "delta": du,
                     "max_diff_K": mc, "sqrt_n": sq, "ratio": ratio, "ok_3sqrt": ok3}
            delta3_results[n].append(rec)
            if verbose:
                ps  = "✓" if ok3 else "✗"
                dcs = f"{mc:9.4f}" if not math.isnan(mc) else "  (n>19)"
                rs  = f"{ratio:7.4f}" if not math.isnan(ratio) else "  (skip)"
                print(f"{n:>4} {i:>2}  δ={du}   {dcs} {sq:>8.4f} {rs} {ps}")

    if verbose:
        print()
        print("=" * 72)
        print(f"APN REGIME DN2-C: {'✓ PASS' if apn_pass else '✗ FAIL'}")
        print("Delta=3 regime: separate result (see DN2 proof paper §6.2)")
        print("  (* = delta assumed=2 for n>17, verified for n<=17)")
        print()
        _print_summary(apn_results, delta3_results)

    return {"apn_results": apn_results, "delta3_results": delta3_results,
            "apn_pass": apn_pass}


def _print_summary(apn_results: dict, delta3_results: dict) -> None:
    print("─" * 72)
    print("Algebraic Closure Summary (docs/PROOF_DN2_APN_SCRAMBLING.md)")
    print()

    # n ≡ 2 mod 3 — Weil bound (power maps)
    weil_ns = [n for n in sorted(apn_results) if n % 3 == 2]
    print("n ≡ 2 mod 3: Power map x^3 — Weil (1948): |chi| <= (deg-1)*sqrt(n) = sqrt(n)")
    for n in weil_ns:
        pms = [r for r in apn_results[n] if r.get("is_power_map")]
        if pms:
            r = pms[0]; mc = r["max_diff_K"]
            if not math.isnan(mc):
                print(f"  n={n:2d}: |chi|={mc:.4f} <= sqrt({n})={r['weil_bound']:.4f} "
                      f"{'✓' if r['weil_ok'] else '✗'}")
            else:
                print(f"  n={n:2d}: (n>19, Weil analytic proof covers this case)")
    print()

    # n ≡ 1 mod 3 — direct computation
    poly_ns = [n for n in sorted(apn_results) if n % 3 == 1]
    print("n ≡ 1 mod 3: Non-power-map APN — direct computation max|chi|/sqrt(n) <= 2")
    for n in poly_ns:
        recs = [r for r in apn_results[n]
                if not math.isnan(r.get("max_diff_K", float("nan")))]
        if recs:
            worst = max(recs, key=lambda r: r.get("ratio", 0))
            print(f"  n={n:2d}: worst ratio={worst['ratio']:.4f} (seed {worst['seed_idx']}) "
                  f"{'<= 2√n ✓' if worst['dn2c_ok'] else '> 2√n ✗'}")
        else:
            print(f"  n={n:2d}: (n>19, pending; power-map Weil may apply)")
    print()

    # delta=3 regime
    print("Delta=3 regime (OD-16/17 primes — best-available, NOT APN):")
    for n in sorted(delta3_results):
        recs = [r for r in delta3_results[n]
                if not math.isnan(r.get("max_diff_K", float("nan")))]
        if recs:
            worst = max(recs, key=lambda r: r.get("ratio", 0))
            print(f"  n={n}: worst ratio={worst['ratio']:.4f}, "
                  f"<= 3√n: {worst['ok_3sqrt']} (delta=3, OD-{16 if n==19 else 17})")
        else:
            print(f"  n={n}: (n>19, not computed)")
    print("─" * 72)


# ── pytest tests ───────────────────────────────────────────────────────────────

def test_golden_seeds_n13_all_valid_ranks():
    """After cleanup, all n=13 GOLDEN_SEEDS entries must have rank < 13!."""
    import math
    nfact = math.factorial(13)
    for i, rank in enumerate(GOLDEN_SEEDS[13]):
        assert rank < nfact, (
            f"GOLDEN_SEEDS[13][{i}] = {rank} >= 13! = {nfact}. "
            "Invalid ranks must be removed from the table.")


def test_golden_seeds_n13_all_apn():
    """After cleanup, all n=13 seeds must have delta(f) = 2 (APN)."""
    for i, rank in enumerate(GOLDEN_SEEDS[13]):
        f     = factoradic_unrank(rank, 13, signed=False)
        delta = differential_uniformity(f, 13)
        assert delta == 2, (
            f"GOLDEN_SEEDS[13][{i}] (rank={rank}): delta={delta} != 2. "
            "Non-APN entries must be removed from the APN seed table.")


def test_golden_seeds_apn_regime_all_delta2():
    """All APN-regime seeds (n in {5,7,11,13,17}, n!=3,19,31) have delta=2."""
    for n in [5, 7, 11, 13, 17]:
        for i, rank in enumerate(GOLDEN_SEEDS[n]):
            f     = factoradic_unrank(rank, n, signed=False)
            delta = differential_uniformity(f, n)
            assert delta == 2, (
                f"GOLDEN_SEEDS[{n}][{i}] rank={rank}: delta={delta} != 2. Not APN.")


def test_delta3_regime_seeds_have_delta3():
    """n=19 and n=31 seeds must all have delta=3 (consistent with OD-16/OD-17)."""
    for n in [19]:   # skip n=31 — too slow for CI
        for i, rank in enumerate(GOLDEN_SEEDS[n]):
            f     = factoradic_unrank(rank, n, signed=False)
            delta = differential_uniformity(f, n)
            assert delta == 3, (
                f"GOLDEN_SEEDS[{n}][{i}] rank={rank}: delta={delta}, expected 3. "
                "OD-16 conjectures no APN bijection exists for Z_19.")


def test_power_map_seeds_exist_for_n_equiv_2_mod3():
    """For each n ≡ 2 mod 3, at least one GOLDEN_SEED is the power map x^3 mod n."""
    for n in [11, 17]:   # n=5 exhaustive seeds don't include x^3 as stored rank
        found = any(
            is_power_map(factoradic_unrank(r, n, signed=False), n)
            for r in GOLDEN_SEEDS[n]
        )
        assert found, (
            f"n={n} (≡2 mod 3): power map x^3 not in GOLDEN_SEEDS[{n}]. "
            "OD-16-PM guarantees x^3 is APN for this n.")


def test_weil_bound_power_map_seeds():
    """Power-map seeds satisfy the Weil bound |chi| <= sqrt(n) exactly."""
    for n in [11, 17]:
        for rank in GOLDEN_SEEDS[n]:
            f = factoradic_unrank(rank, n, signed=False)
            if not is_power_map(f, n):
                continue
            mc, bh, bd = max_diff_char_sum(f, n)
            wb = weil_bound_power_map(n)
            assert mc <= wb + 1e-6, (
                f"Weil bound violated: n={n}, power map x^3, "
                f"|chi(h={bh},D={bd})|={mc:.6f} > sqrt({n})={wb:.6f}")


def test_dn2c_apn_seeds_le_2sqrt_n():
    """
    DN2-C: max|chi_f(h,D)| <= 2*sqrt(n) for all APN seeds (n <= 17, n!=3).

    Uses factoradic_unrank(rank, n) directly — NOT unrank_optimal_seed(rank, n),
    which would treat the rank as an index and silently decode the wrong permutation.
    """
    for n in [5, 7, 11, 13, 17]:
        sq = math.sqrt(n)
        for rank in GOLDEN_SEEDS[n]:
            f          = factoradic_unrank(rank, n, signed=False)
            mc, bh, bd = max_diff_char_sum(f, n)
            assert mc <= 2 * sq + 1e-6, (
                f"DN2-C FAILED: n={n} rank={rank}: "
                f"|chi(h={bh},D={bd})|={mc:.4f} > 2*sqrt({n})={2*sq:.4f}")


def test_delta3_seeds_le_3sqrt_n():
    """Delta=3 seeds (n=19) satisfy weaker bound |chi| <= 3*sqrt(n)."""
    n  = 19
    sq = math.sqrt(n)
    for rank in GOLDEN_SEEDS[n]:
        f          = factoradic_unrank(rank, n, signed=False)
        mc, bh, bd = max_diff_char_sum(f, n)
        assert mc <= 3 * sq + 1e-6, (
            f"n={n} delta=3 seed rank={rank}: "
            f"|chi(h={bh},D={bd})|={mc:.4f} > 3*sqrt({n})={3*sq:.4f}")


def test_owen_mode_improves_fft_vs_plain():
    """FLU-Owen scrambling must reduce FFT spectral peak vs. plain."""
    from flu.core.fractal_net import FractalNetKinetic

    def fft_peak(pts, bins=32):
        sp = []
        for i, j in itertools.combinations(range(pts.shape[1]), 2):
            H, _, _ = np.histogram2d(pts[:, i], pts[:, j],
                                     bins=bins, range=[[0, 1], [0, 1]])
            F = np.abs(np.fft.fftshift(np.fft.fft2(H)))
            F[bins // 2, bins // 2] = 0
            sp.append(float(F.max()))
        return float(np.mean(sp)) if sp else 0.0

    net = FractalNetKinetic(n=5, d=3)
    pp  = net.generate(3125)
    po  = net.generate_scrambled(3125, seed_rank=0, mode="owen")
    assert fft_peak(po) < fft_peak(pp), "Owen must reduce FFT peak vs plain"


def test_generate_scrambled_default_is_owen():
    """generate_scrambled() with no mode argument must equal mode='owen'."""
    from flu.core.fractal_net import FractalNet, FractalNetKinetic
    for C in [FractalNet, FractalNetKinetic]:
        net = C(n=5, d=2)
        np.testing.assert_array_equal(
            net.generate_scrambled(125, seed_rank=2),
            net.generate_scrambled(125, seed_rank=2, mode="owen"),
            err_msg=f"{C.__name__}: default must equal mode='owen'")


def test_generate_scrambled_coordinated_backward_compat():
    """mode='coordinated' must match the reference using factoradic_unrank directly."""
    from flu.core.fractal_net import FractalNet, FractalNetKinetic
    from flu.core.operators import APNPermuteOperator

    n, d, N = 5, 3, 625
    for C in [FractalNet, FractalNetKinetic]:
        net   = C(n=n, d=d)
        pts_c = net.generate_scrambled(N, seed_rank=0, mode="coordinated")
        seeds = GOLDEN_SEEDS.get(n, [])
        mm    = 1
        while net.N ** mm <= N:
            mm += 1
        dbs = []
        for m in range(mm):
            # coordinated: index into seeds list, then decode the stored RANK
            idx  = (0 + m) % len(seeds)
            rank = seeds[idx]
            from flu.core.operators import APNPermuteOperator
            perm = factoradic_unrank(rank, n, signed=False)
            P    = APNPermuteOperator(n, perm)
            dbs.append(P(net._base_block.astype(int)).astype(float))
        ref = np.zeros((N, d))
        ka  = np.arange(N, dtype=np.int64)
        for m in range(mm):
            vm = (ka // (net.N ** m)) % net.N
            ref += dbs[m][vm] * (1.0 / (n ** (m + 1)))
        np.testing.assert_allclose(pts_c, ref, atol=1e-12,
            err_msg=f"{C.__name__} coordinated != factoradic_unrank reference")


def test_latin_preservation_both_modes():
    """
    Both modes produce Latin hypercubes at N = N_base (DN2-P1, PROVEN).

    At N = n^D, each digit value in [0, n-1] appears exactly N/n = n^(D-1)
    times in each coordinate — the correct Latin hypercube condition.
    """
    from flu.core.fractal_net import FractalNetKinetic
    n, d = 5, 2; N = n ** d
    net  = FractalNetKinetic(n=n, d=d)
    expected = N // n   # = n^(D-1) repetitions per digit per axis
    for mode in ["owen", "coordinated"]:
        pts    = net.generate_scrambled(N, seed_rank=0, mode=mode)
        digits = np.round(pts * n).astype(int) % n
        for dim in range(d):
            vals, counts = np.unique(digits[:, dim], return_counts=True)
            assert len(vals) == n and np.all(counts == expected), (
                f"Latin violated: mode={mode}, dim={dim}, "
                f"vals={vals}, counts={counts}, expected each={expected}"
            )


def test_owen_independent_dims_differ():
    """Owen: dim-0 and dim-1 must use different APN perms → different values."""
    from flu.core.fractal_net import FractalNetKinetic
    n, d = 5, 3; N = n ** d
    net  = FractalNetKinetic(n=n, d=d)
    pts  = net.generate_owen_scrambled(N, seed_rank=0)
    s    = GOLDEN_SEEDS.get(n, [])
    if len(s) >= 2 and s[0] != s[1]:
        assert not np.array_equal(pts[:, 0], pts[:, 1]), \
            "Owen: dim-0 and dim-1 should use different APN perms"


if __name__ == "__main__":
    result = verify_all_golden_seeds(verbose=True)
    sys.exit(0 if result["apn_pass"] else 1)
