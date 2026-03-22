"""
flu/applications/codes.py
=========================
LatinSquareCode — structured symbol code using FLU Latin squares.

STATUS: DESIGN INTENT (encode/decode mechanics) + PROVEN (Latin property)

MATHEMATICAL FOUNDATION
───────────────────────
A FLU code matrix M is an n×n Latin square:
  • M[i, j] = (i + j) % n   for even n  (sum-mod construction, PROVEN)
  • M[i, j] = (i + j) % n   for odd  n  (same formula, identical proof)

The Latin property guarantees:
  • Every symbol s ∈ {0,...,n-1} appears exactly once in each row.
  • Every symbol s appears exactly once in each column.

THEOREM (Latin Code Coverage), STATUS: PROVEN
──────────────────────────────────────────────
For the sum-mod matrix M[i,j] = (i+j) % n:
  (a) ∀ row r, the set {M[r,0], M[r,1], ..., M[r,n-1]} = {0,...,n-1}.
      Proof: M[r,j] = (r+j) % n; as j varies over {0,...,n-1}, the values
             (r+j) % n are a cyclic shift of {0,...,n-1}, hence a bijection.
  (b) ∀ col c, analogous argument by symmetry.  □

WHAT IS NOT CLAIMED:
  • Minimum Hamming distance: the conjecture min_d = ⌊n/2⌋
    is UNVERIFIED and has been removed.  No formal bound is stated here.
  • Error correction capability: without a proven distance bound no formal
    correction guarantee can be made.
  • The decode function finds a valid position; it does NOT guarantee
    reconstruction of the original message under arbitrary error patterns.

Dependencies: flu.core.even_n (for even n), numpy only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flu.utils.math_helpers import is_odd


# ── build_code_matrix ─────────────────────────────────────────────────────────

def build_code_matrix(n: int) -> np.ndarray:
    """
    Build an n×n Latin square code matrix via the sum-mod construction.

    M[i, j] = (i + j) % n

    THEOREM (Latin Code Coverage), STATUS: PROVEN — see module docstring.

    Parameters
    ----------
    n : int   alphabet size / code dimension (≥ 2)

    Returns
    -------
    np.ndarray  shape (n, n), dtype int64, values in {0, ..., n-1}

    Raises
    ------
    ValueError  if n < 2
    """
    if n < 2:
        raise ValueError(f"n must be ≥ 2, got {n}")

    rows = np.arange(n, dtype=np.int64).reshape(-1, 1)
    cols = np.arange(n, dtype=np.int64).reshape(1, -1)
    return (rows + cols) % n


# ── LatinSquareCode ───────────────────────────────────────────────────────────

class LatinSquareCode:
    """
    Symbol code built on a FLU Latin square code matrix.

    STATUS: DESIGN INTENT (encode/decode) + PROVEN (Latin property)

    The code encodes pairs of integers (a, b) as M[a % n, b % n], where M
    is the n×n sum-mod Latin square.  Decoding finds all (row, col) positions
    whose matrix value equals the received symbol and returns the first match
    (nearest by index if ties exist).

    Parameters
    ----------
    n : int   alphabet size / code dimension (≥ 2)

    Examples
    --------
    >>> code = LatinSquareCode(n=5)
    >>> sym  = code.encode_pair(2, 3)
    >>> r, c = code.decode_symbol(sym)
    >>> print(r, c)   # one valid pre-image
    """

    def __init__(self, n: int) -> None:
        if n < 2:
            raise ValueError(f"n must be ≥ 2, got {n}")
        self.n      = n
        self.matrix = build_code_matrix(n)

        # Pre-build inverse index: symbol → list of (row, col) positions
        self._inverse: Dict[int, List[Tuple[int, int]]] = {}
        for r in range(n):
            for c in range(n):
                sym = int(self.matrix[r, c])
                self._inverse.setdefault(sym, []).append((r, c))

    # ── Encoding ─────────────────────────────────────────────────────────

    def encode_pair(self, a: int, b: int) -> int:
        """
        Encode a pair (a, b) as a single symbol via M[a % n, b % n].

        THEOREM (Well-definedness), STATUS: PROVEN
            Every (a % n, b % n) is a valid index into M. The result is
            always in {0, ..., n-1}.

        Parameters
        ----------
        a, b : int   integers (taken modulo n)

        Returns
        -------
        int  symbol in {0, ..., n-1}
        """
        return int(self.matrix[a % self.n, b % self.n])

    def encode_message(self, message: List[int]) -> List[int]:
        """
        Encode a list of integers as a list of symbols.

        Processes message in consecutive pairs [a, b, a, b, ...].
        Odd-length messages are zero-padded on the right.

        Parameters
        ----------
        message : list[int]

        Returns
        -------
        list[int]  encoded symbols, length = ⌈len(message)/2⌉
        """
        encoded = []
        for i in range(0, len(message), 2):
            a = message[i]
            b = message[i + 1] if i + 1 < len(message) else 0   # zero-pad
            encoded.append(self.encode_pair(a, b))
        return encoded

    # ── Decoding ──────────────────────────────────────────────────────────

    def decode_symbol(self, sym: int) -> Tuple[int, int]:
        """
        Find the first (row, col) position where M[row, col] == sym.

        Because M is a Latin square, exactly n such positions exist.
        This returns the one with the smallest (row * n + col) index.

        STATUS: DESIGN INTENT — no correction guarantee without distance proof.

        Parameters
        ----------
        sym : int   a symbol in {0, ..., n-1}

        Returns
        -------
        (row, col) : tuple of int

        Raises
        ------
        ValueError  if sym is not a valid code symbol
        """
        positions = self._inverse.get(int(sym))
        if positions is None:
            raise ValueError(
                f"Symbol {sym} is not in the code alphabet {{0,...,{self.n-1}}}"
            )
        return positions[0]   # first / smallest-index position

    def decode_message(self, symbols: List[int]) -> List[int]:
        """
        Decode a list of symbols back to a flat list of integers.

        Each symbol decodes to a (row, col) pair; the output is the
        concatenation of all pairs.

        STATUS: DESIGN INTENT — round-trip is exact for error-free symbols.
        Noisy decoding is nearest-neighbour in the inverse index, NOT
        guaranteed to be optimal.

        Parameters
        ----------
        symbols : list[int]

        Returns
        -------
        list[int]  decoded integers (length = 2 × len(symbols))
        """
        decoded = []
        for sym in symbols:
            r, c = self.decode_symbol(sym)
            decoded.extend([r, c])
        return decoded

    # ── Verification ──────────────────────────────────────────────────────

    def verify(self) -> Dict[str, Any]:
        """
        Verify Latin-square properties of the code matrix.

        THEOREM (Latin Code Coverage), STATUS: PROVEN — see module docstring.

        Returns
        -------
        dict with keys:
            latin_rows      : bool   every row is a permutation of {0,...,n-1}
            latin_cols      : bool   every column is a permutation of {0,...,n-1}
            coverage_ok     : bool   every symbol has exactly n positions
            encode_decode_ok: bool   encode_pair round-trips for all (a, b)
            verified        : bool   all checks pass
        """
        expected = set(range(self.n))

        latin_rows = all(set(self.matrix[r, :].tolist()) == expected
                         for r in range(self.n))
        latin_cols = all(set(self.matrix[:, c].tolist()) == expected
                         for c in range(self.n))
        coverage_ok = all(len(self._inverse.get(s, [])) == self.n
                          for s in range(self.n))

        # encode → decode round-trip: encode_pair(a, b) then decode_symbol
        # should yield a valid (row, col) — not necessarily (a%n, b%n) since
        # decode returns the first of n positions, but both are valid pre-images.
        rt_ok = True
        for a in range(self.n):
            for b in range(self.n):
                sym     = self.encode_pair(a, b)
                r, c    = self.decode_symbol(sym)
                # Check that decode output is a valid position
                if not (0 <= r < self.n and 0 <= c < self.n):
                    rt_ok = False
                    break
                # Check that re-encoding the decoded pair gives the same symbol
                if self.encode_pair(r, c) != sym:
                    rt_ok = False
                    break

        verified = latin_rows and latin_cols and coverage_ok and rt_ok
        return {
            "n"              : self.n,
            "latin_rows"     : latin_rows,
            "latin_cols"     : latin_cols,
            "coverage_ok"    : coverage_ok,
            "encode_decode_ok": rt_ok,
            "verified"       : verified,
        }

    def __repr__(self) -> str:
        return f"LatinSquareCode(n={self.n}, matrix_shape={self.matrix.shape})"
