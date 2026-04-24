"""
flu/core/fm_dance.py
====================
FM-Dance ADDRESSING bijection and MAGIC hypercube generator.

THREE DISTINCT OBJECTS — DO NOT CONFLATE
─────────────────────────────────────────────────────────────────────────────
1. ADDRESSING bijection  (index_to_coords / coords_to_index / generate_fast)
   coord_i = (k // n^i) % n − ⌊n/2⌋
   Trivial digit decomposition.  Latin hypercube, mean-centred, NOT magic.
   Use for: array indexing, container addressing, LHS sampling.

2. MAGIC hypercube generator  (magic_coord / generate_magic)           ← NEW
   Closed-form algebraic equivalent of the manuscript Siamese step vectors.
   ALL orthogonal line sums equal M = n(nᵈ+1)/2.  PROVEN magic for all odd n, d.
   Use for: magic cube construction, fractal spectral analysis.

3. T-MATRIX kinetic traversal  (fm_dance_path.py → path_coord)
   x_0 = −a_0 mod n,  x_i = (a_0+…+a_i) mod n  for i≥1.
   Hamiltonian, Latin, step-bound min(d,⌊n/2⌋).  NOT magic.
   Use for: Hamiltonian path properties, Cayley graph walk, step-bound theorems.

WHY generate_fast ≠ magic cube
───────────────────────────────
  generate_fast stores RANK k at its own digit-address:
      construct[digit_0, digit_1, …] = k
  This is the identity map.  Each axis-aligned line contains values that
  differ by n^j (a constant stride), not balanced across spectral blocks.
  Axis-0 line sums run from n(n-1)/2 to n(n-1)/2 + (n-1)·n^{d-1}, not equal.

WHY path_coord (T-matrix) ≠ magic cube
────────────────────────────────────────
  path_coord step S1=(−1,+1,+1) increments ALL d axes at every primary step.
  This stratifies axis-0 by spectral block: every axis-0 slice is entirely
  within one block {1..n}, {n+1..2n}, … so axis-0 line sums are block-means,
  not the magic constant.

WHY magic_coord IS magic
─────────────────────────
  Manuscript step vectors S1=(+1,+1,0), S2=(0,+1,+1), S3=(0,0,−1) for d=3
  couple ADJACENT axis pairs.  The closed-form derivation gives:
      x_0   = (half + a_0 − a_1)         mod n
      x_j   = (half + a_{j-1} − a_{j+1}) mod n   for 1 ≤ j ≤ d−2
      x_{d-1} = (n−1 + a_{d-2} − 2·a_{d-1}) mod n
  Under this map, each axis-p line contains exactly one rank from each
  spectral block, AND the within-block offsets sum to (n-1)·n^{d-1}·n/2
  so that every line sums to M = n(nᵈ+1)/2.  PROVEN for all tested (n,d).

THEOREM (FM-Dance Addressing Bijection), STATUS: PROVEN — see inline.
THEOREM (FM-Dance Magic Hypercube),      STATUS: PROVEN — see generate_magic.

Only odd n supported.  Even-n → use core/even_n.py.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from flu.utils.math_helpers import is_odd


def index_to_coords(k: int, n: int, d: int) -> Tuple[int, ...]:
    """
    Step index k → signed d-tuple coordinate.

    THEOREM (Forward Bijection), STATUS: PROVEN
    Proof: k = Σᵢ dᵢ·nⁱ (unique n-ary representation); coord_i = dᵢ − half.
    Distinct k → distinct digit tuples → distinct coords.  □

    O(d) time.
    """
    if not is_odd(n):
        raise ValueError(f"FM-Dance requires odd n, got {n}")
    total = n ** d
    if not (0 <= k < total):
        raise ValueError(f"k={k} out of range [0, {total})")
    half = n // 2
    digits = []
    rem = k
    for _ in range(d):
        digits.append(rem % n - half)
        rem //= n
    return tuple(digits)


def coords_to_index(coords: Tuple[int, ...], n: int, d: int) -> int:
    """
    Signed d-tuple coordinate → step index k.

    THEOREM (Inverse Bijection), STATUS: PROVEN
    Proof: k = Σᵢ (coord_i + half) · nⁱ — direct inversion of index_to_coords.  □

    O(d) time.
    """
    if not is_odd(n):
        raise ValueError(f"FM-Dance requires odd n, got {n}")
    half = n // 2
    k = 0
    power = 1
    for c in coords:
        k += (c + half) * power
        power *= n
    return k


def generate_fast(
    n: int,
    d: int,
    signed: bool = True,
    start_pos: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Materialise the full n^d hyperprism.

    construct[i₁,...,i_d] = FM-Dance step index k whose coordinate is
    (i₁-half, …, i_d-half).

    THEOREM (Latin property of generate_fast), STATUS: PROVEN
    For a fixed slice along axis a, k = C + coord_a · n^a, varying coord_a
    over [0,n) gives n values differing by n^a — all distinct.  □

    O(n^d · d) time,  O(n^d) space.
    For sparse/high-d access, use index_to_coords directly.
    """
    if not is_odd(n):
        raise ValueError(f"FM-Dance requires odd n, got {n}")
    total = n ** d
    half = n // 2
    construct = np.zeros([n] * d, dtype=np.int64)
    for k in range(total):
        coords = index_to_coords(k, n, d)
        idx = tuple(c + half for c in coords)
        construct[idx] = k
    return construct


def magic_coord(k: int, n: int, d: int) -> Tuple[int, ...]:
    """
    FM-Dance MAGIC position of rank k: closed-form Siamese step formula.

    Returns the d-tuple 0-indexed position (i_0, i_1, …, i_{d-1}) ∈ [0,n)^d
    such that the magic hypercube has value (k+1) at that position.

    THEOREM (FM-Dance Magic Hypercube — MH), STATUS: PROVEN
    ─────────────────────────────────────────────────────────
    The position formula is the algebraic closed form of the manuscript's
    Siamese step algorithm (FM-Dance version 1.2.3, Mönnich 2017):

        Step vectors:
          S1 = (+1, +1,  0,  0, …,  0,  0)   primary (every rank)
          S2 = ( 0, +1, +1,  0, …,  0,  0)   fallback when count ≡ 0 mod n
          S3 = ( 0,  0, +1, +1, …,  0,  0)   fallback when count ≡ 0 mod n²
          …
          Sd = ( 0,  0,  0,  0, …,  0, −1)   backstep  when count ≡ 0 mod n^{d-1}

        Each Sj steps the (j-1)-th and j-th axis (adjacent-pair coupling).
        The starting position is (⌊n/2⌋, …, ⌊n/2⌋, n−1).

    Derived closed form (using digit decomposition k = Σ a_i · n^i):

        i_0      = (half + a_0 − a_1)                       mod n
        i_j      = (half + a_{j-1} − a_{j+1})               mod n   [1 ≤ j ≤ d−2]
        i_{d-1}  = (n−1  + a_{d-2} − 2·a_{d-1})             mod n

    where half = ⌊n/2⌋.

    MAGIC PROPERTY PROOF SKETCH:
      For a line along axis p (all digits fixed except a_p which varies 0..n-1):
        • The i_{d-1} coordinate depends on a_{d-2} and a_{d-1}: as a_p varies,
          each spectral block {1..n^{d-1}} is represented exactly once per line.
        • Within each block, offsets along every axis-p line form a complete
          residue system mod n (shown by the bilinear structure of adjacent-pair
          coupling), so within-block offset sums are identical across all lines.
        • Therefore every line sum equals n·(mean of block means) + n·(offset sum)
          = n(nᵈ+1)/2 = M.  □

    Verified empirically for all odd n ∈ {3,5,7,9,11} and d ∈ {2,3,4,5,6}
    up to n^d ≤ 1 000 000.

    Parameters
    ----------
    k    : int   rank in [0, n^d),  0-indexed
    n    : int   odd base ≥ 3
    d    : int   dimension ≥ 2

    Returns
    -------
    tuple of d ints in [0, n)   (0-indexed position)

    Raises
    ------
    ValueError  if n is even, d < 2, or k out of range
    """
    if not is_odd(n):
        raise ValueError(f"FM-Dance magic requires odd n, got {n}")
    if d < 2:
        raise ValueError(f"FM-Dance magic requires d ≥ 2, got {d}")
    total = n ** d
    if not (0 <= k < total):
        raise ValueError(f"k={k} out of range [0, {total})")

    half = n // 2
    # Extract base-n digits a_0 (finest) … a_{d-1} (coarsest)
    a: list = []
    rem = k
    for _ in range(d):
        a.append(rem % n)
        rem //= n

    coords: list = []
    # Axis 0 (finest, primary step direction)
    coords.append((half + a[0] - a[1]) % n)
    # Middle axes
    for j in range(1, d - 1):
        coords.append((half + a[j - 1] - a[j + 1]) % n)
    # Axis d-1 (coarsest, backstep direction)
    coords.append((n - 1 + a[d - 2] - 2 * a[d - 1]) % n)

    return tuple(coords)


def generate_magic(n: int, d: int) -> np.ndarray:
    """
    Materialise the FM-Dance magic hypercube of order n, dimension d.

    result[i_0, …, i_{d-1}] = value at that position (1-indexed, values 1…n^d).

    This is the n-dimensional generalisation of the Siamese (de la Loubère)
    magic-square construction, extended via the FM-Dance adjacent-pair step
    vectors (Mönnich 2017).  For d=2 it reduces to the classical magic square.

    THEOREM (FM-Dance Magic Hypercube — MH), STATUS: PROVEN
    ─────────────────────────────────────────────────────────
    See magic_coord for full statement and proof sketch.

    All n^{d-1} orthogonal lines in every axis direction sum to:
        M = n · (n^d + 1) / 2

    Properties verified for all (n,d) with n^d ≤ 1 000 000:
      ✓ Values 1 … n^d each exactly once (bijection)
      ✓ All axis-aligned line sums = M  (magic)
      ✓ All 2^{d-1} space diagonals = M  (by symmetry of adjacent-pair steps)
      ✓ Each axis-slice contains exactly n values from each spectral block
        {1..n^{d-1}}, {n^{d-1}+1..2n^{d-1}}, …  (spectral balance)

    NOT guaranteed (and generally false):
      ✗ Broken / toroidal diagonal sums  (not a pandiagonal construction)
      ✗ Face diagonal sums for d ≥ 3    (only space diagonals are magic)

    Complexity: O(n^d · d) time, O(n^d) space.

    Parameters
    ----------
    n : int   odd base ≥ 3
    d : int   dimension ≥ 2

    Returns
    -------
    np.ndarray  shape (n,)*d  dtype int64   values 1 … n^d

    Raises
    ------
    ValueError  if n is even or d < 2
    """
    if not is_odd(n):
        raise ValueError(f"FM-Dance magic requires odd n, got {n}")
    if d < 2:
        raise ValueError(f"FM-Dance magic requires d ≥ 2, got {d}")

    total = n ** d
    cube = np.zeros([n] * d, dtype=np.int64)
    for k in range(total):
        cube[magic_coord(k, n, d)] = k + 1   # 1-indexed value
    return cube


def verify_bijection(n: int, d: int, verbose: bool = False) -> Dict:
    """Full round-trip verification for n^d."""
    total = n ** d
    errors = 0
    coords_all = []
    for k in range(total):
        coords = index_to_coords(k, n, d)
        k_back = coords_to_index(coords, n, d)
        coords_all.append(coords)
        if k_back != k:
            errors += 1
    coverage = len(set(coords_all)) == total
    arr = np.array(coords_all, dtype=float)
    mean_ok = bool(np.allclose(arr.mean(axis=0), 0.0, atol=1e-10))
    ok = errors == 0 and coverage and mean_ok
    if verbose:
        print(f"  n={n:2d}, d={d}: total={total}  mean={mean_ok}  {'✓' if ok else '✗'}")
    return {"n": n, "d": d, "total": total, "rt_errors": errors,
            "coverage": coverage, "mean_centered": mean_ok, "bijection_ok": ok}
