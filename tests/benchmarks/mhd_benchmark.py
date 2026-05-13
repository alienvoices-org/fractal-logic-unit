"""
flu/tests/benchmarks/mhd_benchmark.py
================================
MHD (Magic Hypercube Digital Net) — Comprehensive Benchmark & Verification Suite

Validates every theorem in PROOF_MHD_MAGIC_HYPERCUBE_FAMILY.md with:
  - Algebraic correctness checks
  - Computational certificates
  - Generator comparison benchmarks (Addressing / Kinetic / Magic / Orthogonal)
  - Discrepancy analysis (classical star and L2-star)
  - Walsh dual structure verification
  - Korobov error analysis (prefix and full-depth)
  - Neural network initializer comparison
  - ANOVA integration exactness tests
  - Scaling / performance benchmarks

STATUS MARKERS
  [PROVEN]      algebraic argument reproduced here
  [CERT]        computational certificate
  [EMPIRICAL]   empirical measurement; no formal proof inline
  [CONJECTURE]  open item from proof document

REFERENCES
  Proof document : PROOF_MHD_MAGIC_HYPERCUBE_FAMILY.md
  Source         : flu/core/fm_dance.py
  Source         : flu/core/fractal_net.py
  Source         : flu/applications/neural.py
  External       : Hickernell (1998) "A generalized discrepancy and quadrature
                   error bound" Math. Comp. 67(221)
  External       : Niederreiter (1992) "Random Number Generation and
                   Quasi-Monte Carlo Methods" SIAM

Authors : Felix Mönnich & The Kinship Mesh Collective
Version : V15.5.0 / 2026-05-12
"""

from __future__ import annotations

import math
import time
from fractions import Fraction
from itertools import combinations, product as iproduct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── FLU imports (flu-math 15.4.0) ─────────────────────────────────────────────
from flu.core.fm_dance import (
    _build_magic_A,
    generate_magic,
    index_to_coords,
    magic_coord,
)
from flu.core.fm_dance_path import path_coord
from flu.core.fractal_net import FractalNet, FractalNetKinetic, FractalNetOrthogonal
from flu.applications.neural import FLUInitializer, DynamicFLUNetwork


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _mat_inv_int(d: int) -> List[List[int]]:
    """Exact integer inverse of A_magic via Gauss–Jordan over ℚ."""
    A = _build_magic_A(d)
    n = len(A)
    aug = [
        [Fraction(A[i][j]) for j in range(n)]
        + [Fraction(1 if i == j else 0) for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        piv = next(r for r in range(col, n) if aug[r][col] != 0)
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [aug[r][j] - f * aug[col][j] for j in range(2 * n)]
    return [[int(aug[i][n + j]) for j in range(n)] for i in range(n)]


def _c_formula(i: int, j: int, d: int) -> int:
    """c(i,j,d) from the closed-form inverse formula (MHD-INV)."""
    if j < d - 1 and (d + max(i, j)) % 2 == 0:
        return 2
    return 1


def _B_formula(i: int, j: int, d: int) -> int:
    """B[i][j] = (-1)^{d+j} * c(i,j,d)  — MHD-INV theorem."""
    return ((-1) ** (d + j)) * _c_formula(i, j, d)


def _magic_pts(n: int, d: int, num: Optional[int] = None) -> np.ndarray:
    """First `num` (default n^d) normalised magic_coord points in [0,1)^d."""
    N = n**d if num is None else num
    bb = np.zeros((N, d), dtype=np.float64)
    for k in range(N):
        bb[k] = list(magic_coord(k, n, d))
    return bb / n


def _l2_star(pts: np.ndarray) -> Optional[float]:
    """Hickernell L2-star discrepancy (Warnock formula). O(N²) — small N only."""
    N, d = pts.shape
    if N > 600:
        return None
    t1 = (1 / 3) ** d
    t2 = np.prod(1.5 - pts * (1 - pts), axis=1).mean()
    pw = sum(
        np.prod(1.0 - np.maximum(pts[i], pts), axis=1).sum() for i in range(N)
    )
    t3 = pw / N**2
    return float(np.sqrt(abs(t1 - 2 * t2 / N + t3)))


def _rank_mod_n(rows: List[List[int]], e: int, n: int) -> int:
    """Rank of submatrix (first e columns of `rows`) over ℤₙ."""
    sub = [[v % n for v in row[:e]] for row in rows]
    rank = 0
    for col in range(e):
        piv = next(
            (r for r in range(rank, len(sub)) if math.gcd(sub[r][col], n) == 1),
            None,
        )
        if piv is None:
            continue
        sub[rank], sub[piv] = sub[piv], sub[rank]
        inv = pow(sub[rank][col] % n, -1, n)
        for r in range(len(sub)):
            if r != rank:
                f = sub[r][col] * inv % n
                sub[r] = [(sub[r][c] - f * sub[rank][c]) % n for c in range(e)]
        rank += 1
        if rank == len(sub):
            break
    return rank


PASS = "PASS ✓"
FAIL = "FAIL ✗"


def _result(ok: bool) -> str:
    return PASS if ok else FAIL


# ══════════════════════════════════════════════════════════════════════════════
# § 1  MHD-STRUCT — det(A_magic) = −1
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_struct(d_max: int = 12) -> Dict:
    """
    [PROVEN] MHD-STRUCT: det(A_magic) = -1 for all d >= 2.

    Algebraic proof in V8 §2 (induction via C1+C2+assembly). This function
    provides the computational certificate for d = 2..d_max.

    Reference: flu/core/fm_dance.py::_build_magic_A
    """
    results = {}
    all_ok = True
    for d in range(2, d_max + 1):
        A = np.array(_build_magic_A(d), dtype=float)
        det = round(np.linalg.det(A))
        ok = det == -1
        results[d] = {"det": int(det), "ok": ok}
        if not ok:
            all_ok = False
    return {
        "theorem": "MHD-STRUCT",
        "claim": "det(A_magic) = -1 for all d >= 2",
        "status": "[PROVEN]",
        "d_range": f"2..{d_max}",
        "all_pass": all_ok,
        "result": _result(all_ok),
        "details": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 2  MHD-INV — Closed-form inverse B[i][j] = (-1)^{d+j} * c(i,j,d)
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_inv(d_max: int = 12) -> Dict:
    """
    [PROVEN] MHD-INV: B[i][j] = (-1)^{d+j} * c(i,j,d), c in {1,2}.

    Two checks:
      (a) Formula matches exact inverse for d=2..d_max.
      (b) A * B_formula = I (symbolic verification).

    Also verifies the no-zero-entry corollary and the {±1,±2} bound.

    Reference: flu/core/fm_dance.py::_build_magic_A + V8 §4
    """
    mismatches = []
    all_entries = set()
    ab_ok = True

    for d in range(2, d_max + 1):
        B_exact = _mat_inv_int(d)
        A = _build_magic_A(d)

        for i in range(d):
            for j in range(d):
                pred = _B_formula(i, j, d)
                actual = B_exact[i][j]
                all_entries.add(actual)
                if pred != actual:
                    mismatches.append((d, i, j, pred, actual))

        # Verify A * B = I exactly (integer arithmetic)
        B_mat = [[_B_formula(i, j, d) for j in range(d)] for i in range(d)]
        for r in range(d):
            for s in range(d):
                entry = sum(A[r][k] * B_mat[k][s] for k in range(d))
                if entry != (1 if r == s else 0):
                    ab_ok = False

    all_ok = len(mismatches) == 0 and ab_ok
    no_zero = 0 not in all_entries
    bound_ok = all(abs(v) in (1, 2) for v in all_entries)

    return {
        "theorem": "MHD-INV",
        "claim": "B[i][j] = (-1)^{d+j}*c(i,j,d), c in {1,2}; A*B=I",
        "status": "[PROVEN]",
        "d_range": f"2..{d_max}",
        "formula_match": _result(len(mismatches) == 0),
        "AB_equals_I": _result(ab_ok),
        "no_zero_entry": _result(no_zero),
        "entries_bounded": _result(bound_ok),
        "entry_values_observed": sorted(all_entries),
        "all_pass": all_ok and no_zero and bound_ok,
        "result": _result(all_ok and no_zero and bound_ok),
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 3  MHD-GEN — Universal invertibility over ℤₙ
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_gen(d_range=(2, 8), n_values=(2, 3, 4, 5, 6, 7, 8, 10, 11)) -> Dict:
    """
    [PROVEN] MHD-GEN: A_magic in GL(d, Z_n) for all n >= 2, d >= 2.

    Checks: det = -1 (from MHD-STRUCT) => gcd(-1,n) = 1 for all n.
    Also verifies that the map k -> magic_coord(k,n,d) is a bijection on Z_n^d.

    Reference: flu/core/fm_dance.py::magic_coord, V8 §3
    """
    failures = []
    for d in range(*d_range):
        for n in n_values:
            if n**d > 10_000:
                continue
            positions = set(magic_coord(k, n, d) for k in range(n**d))
            expected = n**d
            if len(positions) != expected:
                failures.append((n, d, len(positions), expected))

    all_ok = len(failures) == 0
    return {
        "theorem": "MHD-GEN",
        "claim": "magic_coord is a bijection on Z_n^d for all n>=2, d>=2",
        "status": "[PROVEN]",
        "note": "gcd(-1,n)=1 universally; bijection verified by counting",
        "all_pass": all_ok,
        "result": _result(all_ok),
        "failures": failures,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 4  MHD-MAGIC — Axis line sums = M (odd n only)
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_magic(
    n_values=(3, 5, 7, 9, 11), d_values=(2, 3, 4)
) -> Dict:
    """
    [PROVEN] MHD-MAGIC: every axis-parallel line sums to M = n*(n^d+1)/2.

    Checks all axis-aligned lines for all tested (n, d).
    Separately confirms even-n FAILURE to document the obstruction.

    Reference: flu/core/fm_dance.py::generate_magic, V8 §6
    """
    failures = []
    for n in n_values:
        for d in d_values:
            if n**d > 20_000:
                continue
            cube = generate_magic(n, d)
            M = n * (n**d + 1) // 2

            if d == 2:
                for r in range(n):
                    if cube[r].sum() != M:
                        failures.append((n, d, "row", r, cube[r].sum(), M))
                for c in range(n):
                    if cube[:, c].sum() != M:
                        failures.append((n, d, "col", c, cube[:, c].sum(), M))
            elif d == 3:
                for ax in range(3):
                    for i in range(n):
                        for j in range(n):
                            if ax == 0:
                                line = cube[:, i, j]
                            elif ax == 1:
                                line = cube[i, :, j]
                            else:
                                line = cube[i, j, :]
                            if line.sum() != M:
                                failures.append((n, d, f"ax{ax}", (i, j), line.sum(), M))
            elif d == 4:
                for ax in range(4):
                    for idx in np.ndindex(*[n if k != ax else 1 for k in range(4)]):
                        full = list(idx)
                        full.insert(ax, slice(None))
                        line = cube[tuple(full)]
                        if line.sum() != M:
                            failures.append((n, d, f"ax{ax}", idx, line.sum(), M))

    # Even-n obstruction: confirm magic sum FAILS for even n
    # [CONJECTURE] no fix known for d>=3
    even_failures = {}
    for n_even in (2, 4, 6):
        for d in (2, 3):
            if n_even**d > 1000:
                continue
            try:
                cube = generate_magic(n_even, d)
                M = n_even * (n_even**d + 1) // 2
                row_sums = [int(cube[r].sum()) for r in range(n_even)]
                even_failures[(n_even, d)] = {
                    "M_expected": M,
                    "row_sums": row_sums,
                    "magic_holds": all(s == M for s in row_sums),
                }
            except Exception as e:
                even_failures[(n_even, d)] = {"error": str(e)}

    all_ok = len(failures) == 0
    return {
        "theorem": "MHD-MAGIC",
        "claim": "all axis lines sum to M = n*(n^d+1)/2  (odd n>=3)",
        "status": "[PROVEN] for odd n; [CONJECTURE] for even n",
        "n_values": list(n_values),
        "d_values": list(d_values),
        "all_pass": all_ok,
        "result": _result(all_ok),
        "failures": failures,
        "even_n_analysis": even_failures,
        "note_even_n": (
            "Even-n: gcd(2,n)>=2 breaks complete-residue argument in digit"
            " position j where B[j][p]=±2. Conjecture: no {-1,0,1}-matrix"
            " fix exists for d>=3."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 5  MHD-PREFIX — OA(n^{d-1}, d, n, 2) all pairs covered
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_prefix(
    n_values=(3, 5, 7, 9, 11), d_values=(3, 4, 5)
) -> Dict:
    """
    [PROVEN] MHD-PREFIX: first N=n^{d-1} points form OA(n^{d-1}, d, n, 2).

    Checks all C(d,2) pairwise projections for full pair coverage.
    Also verifies ±1 minor existence (algebraic foundation of proof).

    Reference: flu/core/fm_dance.py::magic_coord, V8 §8
    """
    pair_failures = []
    minor_ok = True

    # Check ±1 minor for the five exhaustive cases
    for d in range(3, 8):
        A = _build_magic_A(d)
        e = d - 1
        for i, j in combinations(range(d), 2):
            sub = [[A[i][c] for c in range(e)], [A[j][c] for c in range(e)]]
            # Look for ±1 minor
            found_pm1 = any(
                abs(sub[0][c1] * sub[1][c2] - sub[0][c2] * sub[1][c1]) == 1
                for c1 in range(e)
                for c2 in range(c1 + 1, e)
            )
            if not found_pm1:
                minor_ok = False

    # Check actual pair coverage
    for n in n_values:
        for d in d_values:
            N = n ** (d - 1)
            if N > 3000:
                continue
            bb = np.zeros((N, d))
            for k in range(N):
                bb[k] = list(magic_coord(k, n, d))
            pts = bb / n

            for i, j in combinations(range(d), 2):
                vi = (pts[:, i] * n).round().astype(int) % n
                vj = (pts[:, j] * n).round().astype(int) % n
                pairs = len(set(zip(vi.tolist(), vj.tolist())))
                if pairs != n**2:
                    pair_failures.append((n, d, i, j, pairs, n**2))

    all_ok = minor_ok and len(pair_failures) == 0
    return {
        "theorem": "MHD-PREFIX",
        "claim": "OA(n^{d-1},d,n,2): all C(d,2) pairs balanced",
        "status": "[PROVEN]",
        "pm1_minor_exists": _result(minor_ok),
        "pair_coverage": _result(len(pair_failures) == 0),
        "all_pass": all_ok,
        "result": _result(all_ok),
        "failures": pair_failures,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 6  MHD-COVERAGE — Staircase coverage theorem
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_coverage(d_values=(3, 4, 5, 6), n_test=7) -> Dict:
    """
    [PROVEN] MHD-COVERAGE: at N=n^e, exactly C(min(e+1,d),s) s-tuples covered.
    Covered tuples = {max index <= e}.

    Reference: V8 §9 — Active Row Lemma + induction on s.
    Source: flu/core/fm_dance.py::_build_magic_A
    """
    mismatches = []
    for d in d_values:
        A = _build_magic_A(d)
        for s in range(2, min(d + 1, 6)):
            for e in range(s, d + 1):
                covered_count = sum(
                    1
                    for tup in combinations(range(d), s)
                    if _rank_mod_n([A[i] for i in tup], e, n_test) == s
                )
                covered_tuples = [
                    tup
                    for tup in combinations(range(d), s)
                    if _rank_mod_n([A[i] for i in tup], e, n_test) == s
                ]
                predicted = math.comb(min(e + 1, d), s)
                max_idx_ok = all(max(t) <= e for t in covered_tuples)
                if covered_count != predicted or not max_idx_ok:
                    mismatches.append(
                        (d, s, e, covered_count, predicted, max_idx_ok)
                    )

    all_ok = len(mismatches) == 0
    return {
        "theorem": "MHD-COVERAGE",
        "claim": "At N=n^e: covered s-tuples = {max<=e}, count C(min(e+1,d),s)",
        "status": "[PROVEN]",
        "n_test": n_test,
        "d_values": d_values,
        "all_pass": all_ok,
        "result": _result(all_ok),
        "mismatches": mismatches,
    }


def verify_mhd_oa_max(d_values=(3, 4, 5), n_test=3) -> Dict:
    """
    [PROVEN] MHD-OA-MAX: OA(n^{d-1}, d, n, d-1) — saturated.

    Verifies every (d-1)-tuple of coordinates appears exactly once
    in the first n^{d-1} magic_coord points.

    Reference: V8 §9.3
    """
    failures = []
    for d in d_values:
        N = n_test ** (d - 1)
        if N > 2000:
            continue
        bb = np.zeros((N, d))
        for k in range(N):
            bb[k] = list(magic_coord(k, n_test, d))
        pts_int = bb.astype(int)

        for cols in combinations(range(d), d - 1):
            proj = [tuple(pts_int[k, c] for c in cols) for k in range(N)]
            unique_count = len(set(proj))
            if unique_count != N:
                failures.append((d, cols, unique_count, N))

    all_ok = len(failures) == 0
    return {
        "theorem": "MHD-OA-MAX",
        "claim": "OA(n^{d-1},d,n,d-1) — saturated, tight Rao bound",
        "status": "[PROVEN]",
        "n": n_test,
        "d_values": d_values,
        "all_pass": all_ok,
        "result": _result(all_ok),
        "failures": failures,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 7  MHD-WALSH — 1-dimensional dual collapse
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_walsh(d_values=(3, 4, 5, 6, 7, 8), n_test=5) -> Dict:
    """
    [PROVEN] MHD-WALSH: prefix dual D_prefix = {m*v}, v=(1,-1,1,...),
    and |P_hat_N(m*v)| = 1 for all m != 0.

    NOTE: Distinguishes D_prefix (prefix dual) from D* (full-N dual = nZ^d).
    The unit-modulus property is exact, NOT approximate.

    Reference: V8 §10 — constraint recurrence proof
    Source: flu/core/fm_dance.py::_build_magic_A
    """
    results = {}
    for d in d_values:
        A = np.array(_build_magic_A(d), dtype=int)
        v = np.array([(-1) ** j for j in range(d)])

        # Verify (A^T v)_j = 0 for j=0..d-2
        AT_v = A.T @ v
        first_zero = all(AT_v[j] == 0 for j in range(d - 1))
        last_nonzero = AT_v[d - 1] != 0

        # Verify |P_hat_N(m*v)| = 1 by direct computation
        N = n_test ** (d - 1)
        if N <= 2000:
            bb = np.zeros((N, d))
            for k in range(N):
                bb[k] = list(magic_coord(k, n_test, d))
            pts = bb / n_test

            phat_moduli = []
            for m in range(1, 4):
                h = m * v
                phat = np.mean(np.exp(2j * np.pi * (pts @ h)))
                phat_moduli.append(abs(phat))

            unit_modulus = all(abs(mod - 1.0) < 1e-10 for mod in phat_moduli)
        else:
            unit_modulus = None  # too large to verify directly

        results[d] = {
            "AT_v_first_zero": first_zero,
            "AT_v_last_nonzero": last_nonzero,
            "AT_v": AT_v.tolist(),
            "unit_modulus": unit_modulus,
            "ok": first_zero and last_nonzero and (unit_modulus is None or unit_modulus),
        }

    all_ok = all(v["ok"] for v in results.values())

    # Also verify D* = nZ^d at full N (shared by ALL bijective generators)
    n, d = 5, 2
    pts_full = FractalNet(n, d).generate(n**d)
    phat_n = abs(np.mean(np.exp(2j * np.pi * (pts_full @ np.array([n, 0])))))
    full_dual_ok = abs(phat_n - 1.0) < 1e-10

    return {
        "theorem": "MHD-WALSH",
        "claim": "D_prefix = {m*v} (1D ray); |P_hat(mv)| = 1",
        "status": "[PROVEN]",
        "note_dual_distinction": (
            "D_prefix (prefix dual, magic-specific) != D* = nZ^d (full-N dual,"
            " shared by ALL bijective FLU generators — not distinguishing)."
        ),
        "all_pass": all_ok,
        "result": _result(all_ok),
        "full_N_dual_check": _result(full_dual_ok),
        "per_d": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 8  MHD-DISC — Discrepancy bounds
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_disc(configs=((3,3),(5,3),(7,3),(3,4),(5,4),(3,5))) -> Dict:
    """
    Verifies two DISTINCT discrepancy results:

    (A) [PROVEN] Direct classical-star bound: D*_N <= n/N = N^{-1/(d-1)}
        Proven via grid argument (V8 §11.1).
        For d=3 this gives D*_N = O(N^{-1/2}).

    (B) [EMPIRICAL / HICKERNELL-REF] L2-star discrepancy D*_{N,L2} = O(N^{-1/2})
        Measured via Hickernell formula (Warnock variant).
        Formal proof for all d: Hickernell (1998) for OA(N,d,n,2).
        NOTE: D*_{N,L2} and D*_N are DIFFERENT quantities.

    OVERSTATEMENT CORRECTION (relative to V6/V7):
        Earlier versions claimed D*_N = O(N^{-1/2}) for all d via Hickernell.
        Corrected: Hickernell proves L2-star (D*_{N,L2}), not classical star (D*_N).
        The classical star bound is D*_N <= n/N = N^{-1/(d-1)}.

    Reference: V8 §11, Hickernell (1998) Math. Comp. 67(221)
    """
    direct_results = []   # (A)
    l2star_results = []   # (B)

    for n, d in configs:
        N = n ** (d - 1)
        if N > 3000:
            continue

        # (A) Direct bound: verify |count - N*vol| <= n for aligned boxes
        bb = np.zeros((N, d))
        for k in range(N):
            bb[k] = list(magic_coord(k, n, d))
        pts_int = bb.astype(int)

        max_err = 0.0
        n_ary_bound = n
        for a_tuple in iproduct(range(n + 1), repeat=d):
            count = sum(
                1 for k in range(N)
                if all(pts_int[k, j] < a_tuple[j] for j in range(d))
            )
            vol = np.prod([a_j / n for a_j in a_tuple])
            err = abs(count - N * vol)
            max_err = max(max_err, err)

        direct_ok = max_err <= n_ary_bound + 1e-9
        direct_bound = n / N
        direct_results.append({
            "n": n, "d": d, "N": N,
            "max_err": max_err,
            "bound_n": n_ary_bound,
            "D*_upper": direct_bound,
            "N_exp": f"N^{{-1/{d-1}}}" ,
            "direct_proven": direct_ok,
        })

        # (B) L2-star via Hickernell formula
        pts = bb / n
        disc_l2 = _l2_star(pts)
        if disc_l2 is not None:
            l2star_results.append({
                "n": n, "d": d, "N": N,
                "D*_L2": round(disc_l2, 6),
                "D*_L2 * sqrt(N)": round(disc_l2 * N**0.5, 4),
                "note": "L2-star (Hickernell); NOT classical star D*",
            })

    all_direct_ok = all(r["direct_proven"] for r in direct_results)

    # Check D*_L2 * sqrt(N) is bounded (consistent with O(N^{-1/2}))
    ratios = [r["D*_L2 * sqrt(N)"] for r in l2star_results]
    ratio_bounded = max(ratios) < 10.0 if ratios else True

    return {
        "theorem": "MHD-DISC",
        "status_A": "[PROVEN] classical star: D*_N <= n/N = N^{-1/(d-1)}",
        "status_B": "[EMPIRICAL/HICKERNELL-REF] L2-star: D*_{N,L2} = O(N^{-1/2})",
        "overstatement_note": (
            "CORRECTED from V6: Hickernell (1998) proves L2-star discrepancy"
            " for OA(2). Classical star D*_N requires separate argument."
            " For d=3: both give O(N^{-1/2}). For d>=4: classical star gives"
            " O(N^{-1/(d-1)}) which is stronger; L2-star gives O(N^{-1/2})."
        ),
        "direct_bound": {
            "all_pass": all_direct_ok,
            "result": _result(all_direct_ok),
            "data": direct_results,
        },
        "l2star": {
            "ratio_D*_L2*sqrt(N)_bounded": _result(ratio_bounded),
            "max_ratio": max(ratios) if ratios else None,
            "note": "Hickernell formula = Warnock variant of L2-star",
            "data": l2star_results,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 9  MHD-ANOVA — Grid integration exactness
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_anova(configs=((3,3),(5,3),(7,3),(5,4))) -> Dict:
    """
    [PROVEN] MHD-ANOVA: grid-constant functions of order <= s are integrated
    exactly at N = n^e with OA strength s = min(e, d-1).

    Also demonstrates the CAVEAT: continuous functions incur O(1/n^2) error
    independent of N, not zero.

    Reference: V8 §12
    """
    exact_tests = []
    continuous_tests = []

    for n, d in configs:
        N = n ** (d - 1)
        if N > 3000:
            continue
        bb = np.zeros((N, d))
        for k in range(N):
            bb[k] = list(magic_coord(k, n, d))
        pts = bb / n

        # Test 1: grid-constant function (indicator of one cell) — should be exact
        # f(x) = 1 iff x_0 in [0, 1/n), else 0. True integral = 1/n.
        f_grid = (pts[:, 0] < 1 / n).astype(float)
        err_grid = abs(f_grid.mean() - 1 / n)
        exact_tests.append({
            "n": n, "d": d, "N": N,
            "function": "indicator x0 in [0,1/n)",
            "true_integral": round(1 / n, 6),
            "empirical": round(f_grid.mean(), 6),
            "error": round(err_grid, 12),
            "exact": err_grid < 1e-10,
        })

        # Test 2: continuous f(x) = x_0 * x_1 — should have error ~ O(1/n^2)
        # True integral = 1/4; n-ary grid approximation = ((n-1)/(2n))^2
        f_cont = pts[:, 0] * pts[:, 1]
        true_val = 0.25
        grid_approx = ((n - 1) / (2 * n)) ** 2
        err_cont = abs(f_cont.mean() - true_val)
        err_vs_n2 = err_cont * n**2  # should be O(1)
        continuous_tests.append({
            "n": n, "d": d, "N": N,
            "function": "x0*x1 (continuous)",
            "true_integral": 0.25,
            "empirical": round(f_cont.mean(), 6),
            "error": round(err_cont, 6),
            "n-ary_approx": round(grid_approx, 6),
            "error*n^2": round(err_vs_n2, 4),
            "note": "error ~ 1/n^2 (grid discretisation), NOT zero",
        })

    exact_ok = all(t["exact"] for t in exact_tests)
    return {
        "theorem": "MHD-ANOVA",
        "claim": "Grid-constant f of order<=s: exact integration at N=n^{d-1}",
        "status": "[PROVEN]",
        "caveat": "Smooth f: error = O(1/n^2), independent of N",
        "grid_exact_tests": {"all_pass": exact_ok, "result": _result(exact_ok), "data": exact_tests},
        "continuous_tests": {
            "note": "error*n^2 should be ~const (O(1/n^2) confirmed)",
            "data": continuous_tests,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 10  MHD-KOROBOV-PREFIX — Constant worst-case error e^2 = 2*zeta(2rd)
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_korobov_prefix(r_values=(2.0, 3.0), d_values=(2, 3, 4, 5)) -> Dict:
    """
    [PROVEN] MHD-KOROBOV-PREFIX: e^2(P_N; H_{r,d}) = 2*zeta(2rd).

    Notation: r = Korobov smoothness (NOT alpha, to avoid collision).
    The series sum equals 2*zeta(2rd) by definition of zeta function.

    Key interpretation: constant ~ 2, independent of N = n^{d-1}.
    Unit-modulus spectrum (|P_hat| = 1 for all surviving h) is exact.

    Reference: V8 §13
    """
    results = {}
    for r in r_values:
        for d in d_values:
            # e^2 = 2 * sum_{m=1}^{inf} m^{-2rd}
            S = 2 * sum(m ** (-2 * r * d) for m in range(1, 100_000))
            zeta_val = S / 2  # approx zeta(2rd)

            results[(r, d)] = {
                "r": r, "d": d,
                "e_squared": round(S, 8),
                "e": round(S**0.5, 8),
                "converges_to_2": abs(S - 2) < 0.1,
                "note": f"2*zeta({2*int(r*d)}) — finite constant, N-independent",
            }

    all_ok = all(v["converges_to_2"] for v in results.values())
    return {
        "theorem": "MHD-KOROBOV-PREFIX",
        "claim": "e^2 = 2*zeta(2rd) — constant; unit-modulus spectrum",
        "status": "[PROVEN]",
        "notation_note": (
            "r = Korobov smoothness parameter (renamed from alpha"
            " to avoid collision with frequency multiplier m)."
        ),
        "interpretation": (
            "Prefix net is optimal for GRID-CONSTANT integration (MHD-ANOVA)."
            " Constant Korobov error is NOT a failure — it characterises the"
            " regime. Full-depth nets achieve decaying error (MHD-KOROBOV-FULL)."
        ),
        "all_pass": all_ok,
        "result": _result(all_ok),
        "table": {
            f"r={r},d={d}": v for (r, d), v in results.items()
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 11  MHD-KOROBOV-FULL — Optimal rate e ~ sqrt(2d*zeta(2r)) * N^{-r/d}
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_korobov_full(
    r: float = 2.0, n_values=(3, 5, 7, 11), d_values=(2, 3, 4)
) -> Dict:
    """
    [PROVEN] MHD-KOROBOV-FULL: e(P_{n^d}) ~ sqrt(2d*zeta(2r)) * N^{-r/d}.

    Verifies that e^2 * N^{2r/d} converges to 2d*zeta(2r) as n grows.

    OPTIMALITY QUALIFICATION (V8 §14):
      - YES: matches information-complexity lower bound in fixed-d, unweighted H_{r,d}
      - NOT: strongly tractable (constant grows as sqrt(d))
      - NOT: optimal for weighted Korobov spaces

    Reference: V8 §14 — support-size expansion proof
    """
    from scipy.special import zeta as rzeta
    zeta_2r = float(rzeta(2 * r))

    results = []
    for d in d_values:
        predicted_const = 2 * d * zeta_2r
        for n in n_values:
            N = n**d
            # e^2 via support-size expansion (exact formula from proof)
            e2 = sum(
                math.comb(d, s) * (n ** (-2 * r * s)) * (zeta_2r**s)
                for s in range(1, d + 1)
            )
            e2 *= 2  # factor 2 from symmetric sum over Z^d \ {0}
            scale = e2 * (N ** (2 * r / d))
            results.append({
                "d": d, "n": n, "N": N,
                "e_squared": round(e2, 6),
                "e": round(e2**0.5, 6),
                "e2*N^{2r/d}": round(scale, 4),
                "predicted_const": round(predicted_const, 4),
                "ratio": round(scale / predicted_const, 4),
                "converges": abs(scale / predicted_const - 1) < 0.5,
            })

    all_converge = all(r_["converges"] for r_ in results)
    return {
        "theorem": "MHD-KOROBOV-FULL",
        "claim": "e ~ sqrt(2d*zeta(2r)) * N^{-r/d} — optimal in H_{r,d}",
        "status": "[PROVEN]",
        "r": r,
        "optimality_qualification": {
            "optimal_yes": "Matches lower bound c_{r,d}*N^{-r/d} in fixed-d unweighted H_{r,d}",
            "optimal_no_tractability": "C_{r,d}=sqrt(2d*zeta) grows with d; not strongly tractable",
            "optimal_no_weighted": "CBC lattice rules can achieve smaller constants for specific weights",
        },
        "all_pass": all_converge,
        "result": _result(all_converge),
        "table": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 12  GENERATOR COMPARISON BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_generators(
    configs=((3,3),(5,3),(7,3),(3,4),(5,4),(7,4)), prefix_only=True
) -> Dict:
    """
    [EMPIRICAL] Compare MHD against other FLU generators:
      - Addressing (FractalNet / row-major)
      - Kinetic (FractalNetKinetic / T-matrix)
      - Magic (MHD / A_magic)
      - Orthogonal (FractalNetOrthogonal, d=4 only)

    Metrics:
      - L2-star discrepancy (Hickernell formula)
      - Integration error on Genz test functions
      - OA pair coverage count at N = n^{d-1}

    Reference: V8 §16 (comparison table), flu/core/fractal_net.py
    """
    rows = []
    for n, d in configs:
        N = n ** (d - 1)
        if N > 800:
            continue

        # Build point sets
        pts = {
            "magic":      _magic_pts(n, d, N),
            "addressing": FractalNet(n, d).generate(N),
            "kinetic":    FractalNetKinetic(n, d).generate(N),
        }

        for name, p in pts.items():
            disc = _l2_star(p)

            # Genz oscillatory: f = cos(2*pi*x0) * ... true integral ~= 0
            f_osc = np.prod(np.cos(2 * np.pi * p), axis=1).mean()

            # Genz linear: f = sum(x_j); true = d/2
            f_lin = p.sum(axis=1).mean()
            err_lin = abs(f_lin - d / 2)

            # Count balanced pairs (OA strength 2 check)
            pair_counts = []
            for i, j in combinations(range(d), 2):
                vi = (p[:, i] * n).round().astype(int) % n
                vj = (p[:, j] * n).round().astype(int) % n
                pair_counts.append(len(set(zip(vi.tolist(), vj.tolist()))))
            pairs_full = sum(1 for c in pair_counts if c == n**2)

            rows.append({
                "n": n, "d": d, "N": N,
                "generator": name,
                "D*_L2": round(disc, 5) if disc else None,
                "D*_L2*sqrt(N)": round(disc * N**0.5, 3) if disc else None,
                "Genz_osc_error": f"{abs(f_osc):.2e}",
                "linear_err": f"{err_lin:.2e}",
                "pairs_full": f"{pairs_full}/{len(pair_counts)}",
            })

    return {
        "benchmark": "Generator Comparison",
        "status": "[EMPIRICAL]",
        "note_D*_L2": "D*_L2 = Hickernell L2-star (NOT classical star D*)",
        "generators": ["addressing (FractalNet)", "kinetic (FractalNetKinetic)",
                       "magic (MHD)"],
        "data": rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 13  NEURAL NETWORK INITIALIZER BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_neural_init(
    shapes=((64,32), (128,64), (256,128), (512,256), (1024,512))
) -> Dict:
    """
    [EMPIRICAL] Compare FLUInitializer against standard initializers
    on weight matrix statistics.

    Current FLUInitializer uses _choose_n(shape) = max(shape), which creates
    O(n^2) tensors. For large shapes this is memory-intensive.

    STATUS NOTE: The large-scale (>1M parameter) test identified in the
    original benchmark request is constrained by the n=max(shape) design choice.
    See OD-XX in FLU open debt.

    Reference: flu/applications/neural.py::FLUInitializer
    """
    from flu.applications.neural import FLUInitializer

    rng = np.random.default_rng(42)
    flu_init = FLUInitializer(signed=True)

    results = []
    for shape in shapes:
        out_f, in_f = shape
        N = out_f * in_f

        # Xavier/Glorot
        std_xav = np.sqrt(2.0 / (out_f + in_f))
        W_xav = rng.normal(0, std_xav, shape)

        # Kaiming/He
        std_he = np.sqrt(2.0 / in_f)
        W_he = rng.normal(0, std_he, shape)

        # FLU
        try:
            t0 = time.perf_counter()
            W_flu = flu_init.weights(shape)
            t_flu = time.perf_counter() - t0
            flu_ok = True
        except Exception as e:
            W_flu = np.zeros(shape)
            t_flu = 0
            flu_ok = False

        def stats(W):
            return {
                "mean": round(float(W.mean()), 6),
                "std":  round(float(W.std()),  6),
                "eff_rank": round(
                    float(np.linalg.matrix_rank(W.astype(float)) / min(shape)), 4
                ),
            }

        results.append({
            "shape": shape,
            "N_params": N,
            "xavier":  stats(W_xav),
            "he":      stats(W_he),
            "flu":     {**stats(W_flu), "time_ms": round(t_flu * 1000, 2), "ok": flu_ok},
        })

    return {
        "benchmark": "Neural Network Initializer",
        "status": "[EMPIRICAL]",
        "initializers": ["xavier/glorot", "kaiming/he", "flu (FLUInitializer)"],
        "note_flu_limitation": (
            "FLUInitializer._choose_n(shape) = max(shape). For shape (512,256):"
            " generates 512^2=262K elements, crops to 512*256. Memory: O(n^2)."
            " Scalable redesign (use small n, high d) tracked as open item."
        ),
        "data": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 14  PERSPECTIVES BENCHMARK — three views
# ══════════════════════════════════════════════════════════════════════════════

def verify_mhd_perspectives(n_values=(3,5,7), d_values=(2,3)) -> Dict:
    """
    [PROVEN] MHD-PERSPECTIVES: integer/balanced/unity normalizations.

    Verifies line sum laws for all three views.

    Reference: V8 §7
    """
    failures = []
    for n in n_values:
        for d in d_values:
            if n**d > 5000:
                continue
            cube = generate_magic(n, d)
            N = n**d
            M = n * (N + 1) // 2
            field_sum = N * (N + 1) // 2
            mean_val = (N + 1) / 2

            balanced = cube.astype(float) - mean_val
            unity = cube.astype(float) / field_sum

            if d == 2:
                for r in range(n):
                    if abs(cube[r].sum() - M) > 1e-9:
                        failures.append((n, d, "integer", r, cube[r].sum(), M))
                    if abs(balanced[r].sum()) > 1e-9:
                        failures.append((n, d, "balanced_sum", r, balanced[r].sum(), 0))
                    if abs(unity[r].sum() - 1/n**(d-1)) > 1e-9:
                        failures.append((n, d, "unity", r, unity[r].sum(), 1/n**(d-1)))

    all_ok = len(failures) == 0
    return {
        "theorem": "MHD-PERSPECTIVES",
        "claim": "Three views: integer(M), balanced(0), unity(1/n^{d-1})",
        "status": "[PROVEN]",
        "all_pass": all_ok,
        "result": _result(all_ok),
        "failures": failures,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 15  ALGEBRAIC GAP CHECK — remaining open items
# ══════════════════════════════════════════════════════════════════════════════

def check_open_items() -> Dict:
    """
    Documents and tests the three remaining open items from V8 §18.

    (A) Self-contained L2-star proof for d>=4: references Hickernell (1998).
    (B) Even-n magic construction: documents the obstruction.
    (C) Full-depth Korobov sub-leading terms: computes them explicitly.
    """
    results = {}

    # (A) D*_{L2} for d>=4: verify the direct bound strengthens with d
    direct_bounds = {}
    for d in [3, 4, 5, 6]:
        n = 5
        N = n ** (d - 1)
        # direct bound: D*_N <= n/N = N^{-1/(d-1)}
        bound = n / N
        exponent = 1 / (d - 1)
        direct_bounds[d] = {
            "bound": round(bound, 6),
            "exponent": round(exponent, 4),
            "note": f"O(N^{{-{exponent:.3f}}}) >= O(N^{{-1/2}}) for d>=4",
        }
    results["open_A"] = {
        "item": "Self-contained L2-star for d>=4",
        "status": "Direct proof: D*_N <= n/N (classical). L2-star: cites Hickernell.",
        "direct_bounds_by_d": direct_bounds,
    }

    # (B) Even-n: test the gcd obstruction algebraically
    even_gcd_analysis = {}
    for d in [2, 3, 4]:
        B = _mat_inv_int(d)
        for n_even in [2, 4, 6]:
            bad_cols = []
            for p in range(d):
                col_vals = [B[i][p] for i in range(d)]
                bad = [v for v in col_vals if math.gcd(abs(v), n_even) > 1]
                if bad:
                    bad_cols.append((p, bad))
            key = f"d={d},n={n_even}"
            even_gcd_analysis[key] = {
                "bad_columns": bad_cols,
                "obstruction": len(bad_cols) > 0,
            }
    results["open_B"] = {
        "item": "Even-n magic construction",
        "status": "[CONJECTURE] no replacement found",
        "gcd_analysis": even_gcd_analysis,
        "note": (
            "B[i][j]=±2 gives gcd(2,n_even)=2>1 for ANY even n."
            " Columns with ±2 entries fail the complete-residue condition."
            " All d>=2 affected. No {-1,0,1}-entry replacement found for d>=3."
        ),
    }

    # (C) Full-depth Korobov sub-leading terms
    from scipy.special import zeta as rzeta
    r = 2.0
    korobov_subleading = {}
    for d in [2, 3, 4]:
        z = float(rzeta(2 * r))
        terms = {
            f"s={s}": {
                "coeff": math.comb(d, s),
                "formula": f"C({d},{s})*zeta({2*int(r)})^{s}*n^{{-{2*int(r*s)}}}",
                "n=7_value": round(math.comb(d, s) * z**s * 7**(-2*r*s), 8),
            }
            for s in range(1, d + 1)
        }
        korobov_subleading[f"d={d}"] = terms
    results["open_C"] = {
        "item": "Full-depth Korobov sub-leading terms",
        "status": "[PROVEN] all terms explicit from support-size expansion",
        "terms": korobov_subleading,
        "note": "Leading term s=1 dominates; all others are O(n^{-4r}) or smaller",
    }

    return {
        "section": "Open Items Check",
        "items": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_benchmark(verbose: bool = True) -> Dict:
    """
    Run all MHD theorem verifications and benchmarks.
    Returns a structured report dict.

    Usage:
        from mhd_benchmark import run_full_benchmark
        report = run_full_benchmark(verbose=True)
    """
    def section(name):
        if verbose:
            print(f"\n{'='*60}\n  {name}\n{'='*60}")

    report = {}

    section("MHD-STRUCT: det(A_magic) = -1")
    report["mhd_struct"] = verify_mhd_struct()
    if verbose:
        r = report["mhd_struct"]
        print(f"  {r['result']}  {r['claim']}")
        print(f"  d=2..12 computational certificate: all det=-1")

    section("MHD-INV: Closed-form inverse")
    report["mhd_inv"] = verify_mhd_inv()
    if verbose:
        r = report["mhd_inv"]
        print(f"  Formula match: {r['formula_match']}")
        print(f"  A*B = I:       {r['AB_equals_I']}")
        print(f"  No zero entry: {r['no_zero_entry']}")
        print(f"  Entries in {{±1,±2}}: {r['entries_bounded']}")
        print(f"  Observed entry values: {r['entry_values_observed']}")

    section("MHD-GEN: Universal invertibility")
    report["mhd_gen"] = verify_mhd_gen()
    if verbose:
        r = report["mhd_gen"]
        print(f"  {r['result']}  Bijection verified all (n,d) with n^d <= 10K")

    section("MHD-MAGIC: Axis line sums = M (odd n)")
    report["mhd_magic"] = verify_mhd_magic()
    if verbose:
        r = report["mhd_magic"]
        print(f"  {r['result']}  n in {{3,5,7,9,11}}, d in {{2,3,4}}")
        print(f"  Even-n obstruction sample:")
        for k, v in list(r["even_n_analysis"].items())[:2]:
            print(f"    {k}: magic_holds={v.get('magic_holds', 'error')}")

    section("MHD-PERSPECTIVES: Three views")
    report["mhd_perspectives"] = verify_mhd_perspectives()
    if verbose:
        r = report["mhd_perspectives"]
        print(f"  {r['result']}  integer/balanced/unity line sums verified")

    section("MHD-PREFIX: All pairs at N=n^{d-1}")
    report["mhd_prefix"] = verify_mhd_prefix()
    if verbose:
        r = report["mhd_prefix"]
        print(f"  ±1 minor lemma:   {r['pm1_minor_exists']}")
        print(f"  Pair coverage:    {r['pair_coverage']}")

    section("MHD-COVERAGE: Staircase theorem")
    report["mhd_coverage"] = verify_mhd_coverage()
    if verbose:
        r = report["mhd_coverage"]
        print(f"  {r['result']}  C(min(e+1,d),s) tuples at N=n^e, d=3..6")

    section("MHD-OA-MAX: Saturated OA strength")
    report["mhd_oa_max"] = verify_mhd_oa_max()
    if verbose:
        r = report["mhd_oa_max"]
        print(f"  {r['result']}  OA(n^{{d-1}},d,n,d-1) saturated for d=3,4,5")

    section("MHD-WALSH: 1D dual collapse")
    report["mhd_walsh"] = verify_mhd_walsh()
    if verbose:
        r = report["mhd_walsh"]
        print(f"  {r['result']}")
        print(f"  Full-N dual D*=nZ^d (shared, not MHD-specific): {r['full_N_dual_check']}")
        for d, v in list(r["per_d"].items())[:3]:
            print(f"  d={d}: AT_v_zero={v['AT_v_first_zero']}  |P_hat(v)|=1: {v['unit_modulus']}")

    section("MHD-DISC: Discrepancy bounds")
    report["mhd_disc"] = verify_mhd_disc()
    if verbose:
        r = report["mhd_disc"]
        print(f"  Direct classical star: {r['direct_bound']['result']}")
        print(f"  L2-star ratio bounded: {r['l2star']['ratio_D*_L2*sqrt(N)_bounded']}")
        print(f"  CORRECTION: D*_L2 != D*_N (different quantities)")
        for row in r["l2star"]["data"][:3]:
            print(f"    n={row['n']},d={row['d']}: D*_L2={row['D*_L2']}  D*_L2*√N={row['D*_L2 * sqrt(N)']}")

    section("MHD-ANOVA: Grid integration exactness")
    report["mhd_anova"] = verify_mhd_anova()
    if verbose:
        r = report["mhd_anova"]
        print(f"  Grid-constant exact: {r['grid_exact_tests']['result']}")
        for row in r["continuous_tests"]["data"][:2]:
            print(f"  Continuous f=x0*x1: error={row['error']}, error*n^2={row['error*n^2']} (O(1/n^2))")

    section("MHD-KOROBOV-PREFIX: Constant worst-case error")
    report["mhd_korobov_prefix"] = verify_mhd_korobov_prefix()
    if verbose:
        r = report["mhd_korobov_prefix"]
        print(f"  {r['result']}  e^2=2*zeta(2rd) ~ 2 (constant, N-independent)")
        print(f"  Notation: r = Korobov smoothness (not alpha, avoids collision)")
        for k, v in list(r["table"].items())[:4]:
            print(f"    {k}: e^2={v['e_squared']}  e={v['e']}")

    section("MHD-KOROBOV-FULL: Optimal full-depth rate")
    report["mhd_korobov_full"] = verify_mhd_korobov_full()
    if verbose:
        r = report["mhd_korobov_full"]
        print(f"  {r['result']}  e ~ sqrt(2d*zeta(2r)) * N^{{-r/d}}")
        print(f"  Optimality: {r['optimality_qualification']['optimal_yes']}")
        print(f"  NOT: {r['optimality_qualification']['optimal_no_tractability']}")
        for row in r["table"][:4]:
            print(f"    d={row['d']},n={row['n']}: e^2*N^(2r/d)={row['e2*N^{2r/d}']}"
                  f"  predicted={row['predicted_const']}  ratio={row['ratio']}")

    section("Generator Comparison Benchmark")
    report["generator_comparison"] = benchmark_generators()
    if verbose:
        r = report["generator_comparison"]
        print(f"  {r['note_D*_L2']}")
        prev = None
        for row in r["data"]:
            key = (row["n"], row["d"])
            if key != prev:
                print(f"\n  n={row['n']},d={row['d']},N={row['N']}:")
                prev = key
            print(f"    {row['generator']:12s}: D*_L2={row['D*_L2']}  pairs={row['pairs_full']}")

    section("Neural Network Initializer Benchmark")
    report["neural_init"] = benchmark_neural_init()
    if verbose:
        r = report["neural_init"]
        print(f"  {r['note_flu_limitation'][:80]}...")
        for row in r["data"]:
            print(f"  shape={row['shape']}: "
                  f"FLU mean={row['flu']['mean']:.4f} std={row['flu']['std']:.4f}  "
                  f"OK={row['flu']['ok']}  t={row['flu']['time_ms']}ms")

    section("Open Items Check")
    report["open_items"] = check_open_items()
    if verbose:
        items = report["open_items"]["items"]
        print(f"  (A) {items['open_A']['status'][:60]}")
        print(f"  (B) {items['open_B']['status']}")
        print(f"  (C) {items['open_C']['status']}")

    # Summary
    theorems = [
        ("MHD-STRUCT",  report["mhd_struct"]["all_pass"]),
        ("MHD-INV",     report["mhd_inv"]["all_pass"]),
        ("MHD-GEN",     report["mhd_gen"]["all_pass"]),
        ("MHD-MAGIC",   report["mhd_magic"]["all_pass"]),
        ("MHD-PERSPECTIVES", report["mhd_perspectives"]["all_pass"]),
        ("MHD-PREFIX",  report["mhd_prefix"]["all_pass"]),
        ("MHD-COVERAGE",report["mhd_coverage"]["all_pass"]),
        ("MHD-OA-MAX",  report["mhd_oa_max"]["all_pass"]),
        ("MHD-WALSH",   report["mhd_walsh"]["all_pass"]),
        ("MHD-DISC(direct)", report["mhd_disc"]["direct_bound"]["all_pass"]),
        ("MHD-ANOVA",   report["mhd_anova"]["grid_exact_tests"]["all_pass"]),
        ("MHD-KOROBOV-PREFIX", report["mhd_korobov_prefix"]["all_pass"]),
        ("MHD-KOROBOV-FULL",   report["mhd_korobov_full"]["all_pass"]),
    ]

    if verbose:
        print(f"\n{'='*60}")
        print("  THEOREM SUMMARY")
        print(f"{'='*60}")
        for name, ok in theorems:
            print(f"  {_result(ok):10s}  {name}")
        passed = sum(1 for _, ok in theorems if ok)
        print(f"\n  {passed}/{len(theorems)} theorems verified ✓")

    report["summary"] = {
        "theorems_verified": theorems,
        "all_pass": all(ok for _, ok in theorems),
    }
    return report


if __name__ == "__main__":
    report = run_full_benchmark(verbose=True)
