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
