"""
flu/constants.py
================
Package-wide constants with mathematical annotations.
"""
import numpy as np

# ── Lo Shu magic square ────────────────────────────────────────────────────────
# The unique (up to symmetry) 3×3 magic square.
# Row/column/diagonal sums all equal FIELD_SUM = 15.
#
# Theorem (Lo Shu uniqueness): There is exactly one normal 3×3 magic square
# up to the 8 isometries of the square (rotations + reflections).
# Proof: centre must be 5 (only value whose pairs sum to 10); corners must
# be even; edges must be odd.  Exhaustive case analysis closes the proof.
LO_SHU = [
    [2, 7, 6],
    [9, 5, 1],
    [4, 3, 8],
]

# Magic constant for order-3: n(n²+1)/2 = 3·10/2 = 15
FIELD_SUM: int = 15

# Balanced-ternary digit set for n=3, signed
BT_DIGITS = (-1, 0, 1)

# Field sum formula for general n^d: Σ_{k=1}^{n^d} k = n^d(n^d+1)/2
# For n=3, d=4:  81 · 82 / 2 = 3321
FIELD_SUM_3_4: int = 3321

# Unity normalisation denominator for n=3, d=4
UNITY_DENOM: int = FIELD_SUM_3_4

# Step vector base for FM-Dance (n=3, d=4)
# S_k = (0,…,0, n^{k-1}, n^{k-1}, 0,…,0) pattern — computed in fm_dance.py
# We expose only the order constants here.
FM_N_DEFAULT: int = 3
FM_D_DEFAULT: int = 4

# ── FM-Dance T-Matrix (The Discrete Integral) ─────────────────────────────────
# The prefix-sum matrix T ∈ GL(d, Z_n). det(T) = -1.
# Pre-computed for the default d=4. For higher dimensions, use get_T_matrix(d).
def get_T_matrix(d: int) -> np.ndarray:
    """Return the d×d lower-triangular prefix-sum matrix (T9 PROVEN)."""
    T = np.tril(np.ones((d, d), dtype=int))
    T[0, 0] = -1
    return T

# ── Symmetry Group Identity (The Hyperoctahedral Basis) ───────────────────────
# The identity action (RotationHub = I). Used as the starting state
# for all recursive fractal embeddings (C5 PROVEN).
def get_identity_omega(d: int) -> np.ndarray:
    """Return the D-dimensional identity rotation matrix (H_D group identity)."""
    return np.eye(d, dtype=int)


# ── Trump/Boyer Perfect Magic Cube, Order 5 ───────────────────────────────────
#
# SOURCE & CREDIT:
#   Walter Trump and Christian Boyer, 2003-11-13.
#   Computed on the computer of Daniel Trump.
#   Reference: http://www.trump.de/magic-squares/magic-cubes/cubes-5.htm
#   Used here with explicit permission from Walter Trump.
#
# DESCRIPTION:
#   The smallest known "perfect magic cube" of order 5.
#   A *perfect* magic cube satisfies ALL of the following:
#     (1) All rows, columns, and pillars sum to M = 315 = n(n^3+1)/2.
#     (2) All 4 space diagonals sum to M.
#     (3) All 15 orthogonal face diagonals (both diagonals of every
#         axis-aligned cross-section plane) sum to M — 30 magic
#         planar diagonals total, confirmed.
#   Integers used: 1 through 125 (= 5^3), each exactly once.
#   Magic constant: M = 5 * (125 + 1) / 2 = 315.
#
# ARRAY LAYOUT:
#   Shape is (5, 5, 5) with axis order (z, y, x), i.e. TRUMP_BOYER_5[z][y][x].
#   z=0 is the first layer (z=1 in the original 1-indexed source notation).
#
# INCOMPATIBILITY NOTE (LHS / Latin-cube structure):
#   This cube does NOT decompose into mutually orthogonal Latin squares when
#   the 5-ary digit columns are examined axis-by-axis. The digit residues
#   (v-1) mod 5, floor((v-1)/5) mod 5, floor((v-1)/25) mod 5 are balanced
#   *globally* (exactly 25 of each value over all 125 cells) but NOT within
#   individual layers. This is the precise sense in which the cube is
#   incompatible with the FM-Dance LHS structure, which requires per-slice
#   digit balance along every axis-aligned plane. See cube_comparison.py.
#
# HISTORICAL NOTE:
#   Perfect magic cubes of order 5 are extraordinarily rare. This example,
#   found 2003-11-13, is among the first confirmed perfect magic cubes of
#   order 5 and resolved a long-standing open problem in combinatorial
#   design theory.
TRUMP_BOYER_5: list = [
    # z=0 (source z=1)
    [[ 25, 16,  80, 104,  90],
     [115, 98,   4,   1,  97],
     [ 42, 111,  85,   2,  75],
     [ 66,  72,  27, 102,  48],
     [ 67,  18, 119, 106,   5]],
    # z=1 (source z=2)
    [[ 91,  77,  71,   6,  70],
     [ 52,  64, 117,  69,  13],
     [ 30, 118,  21, 123,  23],
     [ 26,  39,  92,  44, 114],
     [116,  17,  14,  73,  95]],
    # z=2 (source z=3)  — the central layer (z=3 in 1-indexed)
    [[ 47,  61,  45,  76,  86],
     [107,  43,  38,  33,  94],
     [ 89,  68,  63,  58,  37],
     [ 32,  93,  88,  83,  19],
     [ 40,  50,  81,  65,  79]],
    # z=3 (source z=4)
    [[ 31,  53, 112, 109,  10],
     [ 12,  82,  34,  87, 100],
     [103,   3, 105,   8,  96],
     [113,  57,   9,  62,  74],
     [ 56, 120,  55,  49,  35]],
    # z=4 (source z=5)
    [[121, 108,   7,  20,  59],
     [ 29,  28, 122, 125,  11],
     [ 51,  15,  41, 124,  84],
     [ 78,  54,  99,  24,  60],
     [ 36, 110,  46,  22, 101]],
]

# Convenience numpy array (dtype int64, shape (5,5,5), axes z,y,x)
TRUMP_BOYER_5_NP: np.ndarray = np.array(TRUMP_BOYER_5, dtype=np.int64)

# Magic constant for order-5 cube: n(n^3+1)/2 = 5*126/2 = 315
MAGIC_SUM_5: int = 315


# ── FM-Dance Magic Cube, Order 5, Dimension 3 ─────────────────────────────────
#
# The FM-Dance magic cube for n=5, d=3 generated by the Siamese adjacent-pair
# step algorithm (Mönnich 2017, "Symmetrische Tanzschritte für magische Universen").
#
# Step vectors (manuscript notation, axes X/Y/Z, 1-indexed):
#   A  = (X=3, Y=3, Z=5)   starting position
#   S1 = (+1, +1,  0)      primary step (every rank)
#   S2 = ( 0, +1, +1)      fallback when count ≡ 0 mod 5
#   S3 = ( 0,  0, −1)      backstep   when count ≡ 0 mod 25
#
# Closed-form formula (digits a0=k%5, a1=(k//5)%5, a2=k//25; half=2):
#   ix = (2 + a0 − a1) mod 5   ← axis-0 (X, finest)
#   iy = (2 + a0 − a2) mod 5   ← axis-1 (Y, middle)
#   iz = (4 + a1 − 2·a2) mod 5 ← axis-2 (Z, coarsest / backstep)
#
# PROPERTIES (all verified, see tools/cube_comparison_order5.py):
#   ✓ Values 1..125 each exactly once
#   ✓ ALL axis-aligned line sums = 315 = 5·(125+1)/2   ← CORRECTED vs. prior version
#   ✓ All 4 space diagonals = 315
#   ✓ Each axis-slice contains exactly 5 values from each spectral block
#     {1-25}, {26-50}, {51-75}, {76-100}, {101-125}
#   ✓ Global 5-ary digit balance: each residue 0..4 appears 25× per digit pos
#   ✓ Per-slice digit balance along all axis directions (LHS property)
#
# AXIS ORDER: FM_DANCE_5_NP[axis-0, axis-1, axis-2] where
#   axis-0 = finest digit direction (S1/S2 coupled, manuscript X-axis)
#   axis-1 = middle digit direction (manuscript Y-axis)
#   axis-2 = coarsest digit direction (S3 backstep, manuscript Z-axis)
#
#   The formula coordinates from magic_coord(k,5,3) map directly:
#     coords[0] → axis-0,  coords[1] → axis-1,  coords[2] → axis-2
#
#   For display and comparison with TRUMP_BOYER_5_NP (stored as [z,y,x]):
#   both cubes are displayed with axis-0 fixed per layer. TB's axis-0 = z
#   (source notation); FM's axis-0 = manuscript X (finest/primary step axis).
#   The geometric centre (mean = 63) is at FM[2,2,2] = TB[2,2,2] = 63 for both.
#   The previous FM_DANCE_5_NP used the ADDRESSING bijection:
#     cube[z,y,x] = 1 + x + 5y + 25z   (trivial digit → position identity)
#   That is a valid Latin hypercube but NOT a magic cube. Axis-0 sums ranged
#   from 15 to 615. All code comparing FM_DANCE_5_NP against Trump/Boyer for
#   magic properties has been updated accordingly.
def _build_fm_dance_5() -> np.ndarray:
    """Build the FM-Dance n=5, d=3 MAGIC cube via generate_magic (1-indexed)."""
    from flu.core.fm_dance import generate_magic
    return generate_magic(5, 3)


FM_DANCE_5_NP: np.ndarray = _build_fm_dance_5()
