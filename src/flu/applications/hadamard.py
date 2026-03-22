"""
src/flu/applications/hadamard.py
=================================
Walsh-Hadamard Matrix Generation via Parametrised Communion Algebra.

STATUS: PROVEN (Theorem HAD-1, algebraic_and_computational)

MATHEMATICAL BASIS
------------------
To generate the k-th row of a Sylvester-Hadamard matrix of order N = 2^D,
we parameterise the Communion seeds with the binary bits of k:

    Seeds:   π_a(x) = k_a ∧ x   (bitwise AND of the a-th bit of k with x)
    Fold:    C_k(x) = ⊕_a (k_a ∧ x_a)  = k · x  (mod 2)   [XOR-fold]
    Row:     H_k(x) = (−1)^{C_k(x)}                        [bipolar map]

The resulting matrix H satisfies H @ H.T = N · I exactly, as H is the
character table of the elementary abelian 2-group Z_2^D:
    <H_k, H_{k'}> = Σ_x (−1)^{(k ⊕ k') · x} = 0  for k ≠ k'  □

IMPORTANT DISTINCTION FROM NAIVE FOLD-XOR
------------------------------------------
An earlier attempt used the *identity* permutation [0,1] for ALL axes
(static seeds) and folded via XOR. That maps to (−1)^{Σ_a x_a mod 2},
which is parity — NOT the required dot-product structure. That matrix
has row dot-products of −2, not 0.

The correction: seeds are *parametrised* by k (different seed per row).

COMPUTATIONAL VERIFICATION
---------------------------
Verified H @ H.T == N·I exactly for d ∈ {2, 3, 4, 5, 6}:
  d=2  →  N=4    matrix 4×4,   passes
  d=3  →  N=8    matrix 8×8,   passes
  d=4  →  N=16   matrix 16×16, passes
  d=5  →  N=32   matrix 32×32, passes
  d=6  →  N=64   matrix 64×64, passes   (audit benchmark)

SCOPE NOTE
----------
Proves the 2^D (Sylvester) subfamily. Generalising to arbitrary 4k
orders is an open research direction tied to the even_n (Sum-Mod) branch.

V15 — audit integration sprint.
"""

from __future__ import annotations

import numpy as np


class HadamardGenerator:
    """
    Generates Sylvester-Hadamard matrices of order N = 2^D via
    parametrised XOR-Communion (bit-masked identity seeds).

    The generator produces exact integer matrices (entries ±1) satisfying
    the orthogonality condition H @ H.T == N * I.

    Methods
    -------
    generate(d)
        Generate the full N×N Hadamard matrix for a given depth d.
    generate_row(k, d)
        Generate a single row k of the N×N matrix in O(N) time.
    verify(d)
        Verify H @ H.T == N·I and return True/False.

    Examples
    --------
    >>> gen = HadamardGenerator()
    >>> H = gen.generate(d=3)
    >>> H.shape
    (8, 8)
    >>> import numpy as np
    >>> np.array_equal(H @ H.T, 8 * np.eye(8, dtype=int))
    True
    """

    def generate_row(self, k: int, d: int) -> np.ndarray:
        """
        Generate row k of the Sylvester-Hadamard matrix of order 2^D.

        Parametrisation:
            For each axis a in [0, d):
                k_a = (k >> a) & 1          # a-th bit of k
                seed_a = [0, k_a]           # π_a(x) = k_a ∧ x
            Fold via XOR:
                C_k(x) = ⊕_{a=0}^{d-1} seed_a[x_a]   =  k · x (mod 2)
            Bipolar map:
                H_k(x) = (−1)^{C_k(x)}

        Parameters
        ----------
        k : int
            Row index in [0, 2^d).
        d : int
            Depth / number of dimensions.

        Returns
        -------
        ndarray of shape (2^d,) with entries in {+1, −1}.
        """
        N = 2 ** d
        # Extract bits of k: k_bits[a] = (k >> a) & 1
        k_bits = np.array([(k >> a) & 1 for a in range(d)], dtype=np.int8)

        # Build the row by iterating over all column indices x in [0, N)
        # x_bits[x, a] = (x >> a) & 1
        x_range = np.arange(N, dtype=np.int32)
        # shape (N, d)
        x_bits = ((x_range[:, None] >> np.arange(d)[None, :]) & 1).astype(np.int8)

        # C_k(x) = k · x (mod 2) = XOR-fold of (k_a AND x_a) for each a
        # = sum of (k_bits[a] * x_bits[x, a]) mod 2
        dot_mod2 = (x_bits @ k_bits) & 1   # shape (N,)

        # Bipolar map: 0 -> +1,  1 -> -1
        return (1 - 2 * dot_mod2.astype(np.int32))

    def generate(self, d: int) -> np.ndarray:
        """
        Generate the full Sylvester-Hadamard matrix of order N = 2^D.

        Complexity: O(N²) total (N rows × O(N) per row), where N = 2^d.

        Parameters
        ----------
        d : int
            Depth (d ≥ 1). The resulting matrix has shape (2^d, 2^d).

        Returns
        -------
        ndarray of shape (2^d, 2^d) with integer entries in {+1, −1}.
        """
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        N = 2 ** d
        H = np.zeros((N, N), dtype=np.int32)
        for k in range(N):
            H[k, :] = self.generate_row(k, d)
        return H

    def verify(self, d: int) -> bool:
        """
        Verify the orthogonality condition H @ H.T == N * I for depth d.

        Returns True iff the generated matrix is a valid Hadamard matrix.
        """
        N = 2 ** d
        H = self.generate(d)
        expected = N * np.eye(N, dtype=np.int32)
        return bool(np.array_equal(H @ H.T, expected))
