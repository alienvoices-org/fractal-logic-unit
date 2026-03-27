"""
flu/core/fractal_3_6.py
=======================
FractalHyperCell_3_6 — recursive 3⁶ embedding.

TENSOR PRODUCT STRUCTURE, STATUS: PROVEN
─────────────────────────────────────────
Each of the 81 macro cells (3⁴ torus) contains a full Lo Shu 3×3 micro
grid (3² torus).  Total 729 = 3⁶ cells, each with a unique 6D sparse
manifold address in {-1, 0, 1}⁶.

TWO MACRO GENERATORS
────────────────────
generator="sudoku"  (default, V15.3.1+)
    Macro layer: LoShuSudokuHyperCell — the Graeco-Latin 3⁴ hypercell.
    4D address:  bt(d1) + bt(d2)  — digit-wise BT of the two Sudoku layers.
    Property:    OA(81,4,3,4) — all 81 four-tuples in {-1,0,1}⁴ appear
                 exactly once (DN1-OA PROVEN V15.3+).  Better QMC.

generator="product"  (legacy, pre-V15.3.1)
    Macro layer: FLUHyperCell — the old product Lo Shu construction.
    4D address:  index_to_coords(norm0, 3, 4) — FM-Dance step index.
    Property:    OA(81,4,3,2) — pairwise balance only.

Both generators produce 729 unique 6D addresses in {-1,0,1}⁶.  The
"sudoku" generator carries the proven OA(81,4,3,4) macro structure.

Backward compatibility
──────────────────────
Passing an explicit `macro: FLUHyperCell` retains the old product
behaviour unconditionally.

Address layout:  coords_6d = macro_4d + micro_2d
    coords_6d[0:4]  ←  macro.sparse_address(macro_row, macro_col)
    coords_6d[4:6]  ←  index_to_coords(lo_shu[micro_row, micro_col]−1, 3, 2)

Dependencies: flu.core.lo_shu, flu.core.lo_shu_sudoku, flu.core.fm_dance,
              flu.core.hypercell.  No new external deps beyond numpy.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np

from flu.core.lo_shu        import _LO_SHU_NP, CellStrata
from flu.core.lo_shu_sudoku import LoShuSudokuHyperCell
from flu.core.fm_dance      import index_to_coords, coords_to_index
from flu.core.hypercell     import FLUHyperCell


# ── SudokuMacroAdapter ────────────────────────────────────────────────────────

class SudokuMacroAdapter:
    """
    Thin adapter that gives LoShuSudokuHyperCell the same interface as
    FLUHyperCell for use as the macro layer in FractalHyperCell_3_6.

    Addressing
    ----------
    sparse_address(r, c) → bt(d1) + bt(d2)
        = (t1_d1, t0_d1, t1_d2, t0_d2) ∈ {-1,0,1}⁴

    This is the Graeco-Latin BT addressing proven in DN1-OA to give
    OA(81,4,3,4): all 81 four-tuples appear exactly once.

    cell(r, c) → CellStrata
        Converts the LoShuSudoku cell dict to a CellStrata via
        CellStrata.from_d1_d2(d1, d2) for backward compat with CellPair.
    """

    def __init__(self, hypercell: Optional[LoShuSudokuHyperCell] = None) -> None:
        self._hc = hypercell if hypercell is not None else LoShuSudokuHyperCell()

    def sparse_address(self, row: int, col: int) -> Tuple[int, ...]:
        """
        Return the 4D Graeco-Latin BT address for cell (row, col).

        Returns bt(d1) + bt(d2) = (t1_d1, t0_d1, t1_d2, t0_d2) ∈ {-1,0,1}⁴.

        THEOREM (DN1-OA): all 81 resulting 4-tuples are distinct — the
        81 addresses cover {-1,0,1}⁴ exactly once.  This follows from:
          (a) All (d1,d2) pairs are unique (Graeco-Latin, DN1-GL).
          (b) bt: {1..9} → {-1,0,1}² is a bijection (BT digit map).
          (c) bt(d1)⊕bt(d2): {(d1,d2)} → {-1,0,1}⁴ is therefore bijective.
        """
        c = self._hc.cell(row, col)
        return c["bt_d1"] + c["bt_d2"]   # 4-tuple in {-1,0,1}⁴

    def cell(self, row: int, col: int) -> CellStrata:
        """
        Return a CellStrata-compatible object for (row, col).

        Converts the Sudoku cell dict to CellStrata via from_d1_d2
        so CellPair.macro typing is unchanged.
        """
        c = self._hc.cell(row, col)
        return CellStrata.from_d1_d2(c["d1"], c["d2"])

    @property
    def hypercell(self) -> LoShuSudokuHyperCell:
        """The underlying LoShuSudokuHyperCell."""
        return self._hc

    def __repr__(self) -> str:
        return "SudokuMacroAdapter(generator=sudoku, OA=81×4×3×4)"


# ── Micro cell result type ────────────────────────────────────────────────────

class MicroCell:
    """
    Minimal descriptor for one cell in the 3×3 Lo Shu micro grid.

    Attributes
    ----------
    row, col   : int   0-based position in the 3×3 micro grid
    value      : int   canonical Lo Shu value ∈ {1, …, 9}
    norm0      : int   zero-based index = value − 1  ∈ [0, 8]
    coords_2d  : (int, int)   FM-Dance 2D sparse address ∈ {-1,0,1}²
    """
    __slots__ = ("row", "col", "value", "norm0", "coords_2d")

    def __init__(self, row: int, col: int, value: int) -> None:
        self.row       = row
        self.col       = col
        self.value     = value
        self.norm0     = value - 1
        self.coords_2d: Tuple[int, int] = index_to_coords(self.norm0, 3, 2)

    def __repr__(self) -> str:
        return (
            f"MicroCell(row={self.row}, col={self.col}, "
            f"value={self.value}, coords_2d={self.coords_2d})"
        )


class CellPair:
    """
    Paired result of FractalHyperCell_3_6.cell_at_6d().

    Attributes
    ----------
    macro : CellStrata   the 3⁴ macro cell
    micro : MicroCell    the 3² micro cell
    """
    __slots__ = ("macro", "micro")

    def __init__(self, macro: CellStrata, micro: MicroCell) -> None:
        self.macro = macro
        self.micro = micro

    def __repr__(self) -> str:
        return f"CellPair(macro={self.macro}, micro={self.micro})"


# ── FractalHyperCell_3_6 ──────────────────────────────────────────────────────

class FractalHyperCell_3_6:
    """
    3⁶ Recursive Fractal Embedding: macro (3⁴) × Lo Shu micro (3²).

    Each of the 81 macro cells contains a full 3×3 Lo Shu micro grid.
    Every one of the 729 resulting (macro, micro) pairs receives a unique
    6D sparse address in {-1,0,1}⁶.

    Macro generators
    ----------------
    generator="sudoku"  (default)
        Uses LoShuSudokuHyperCell (Graeco-Latin).  4D macro address =
        bt(d1) + bt(d2).  Gives OA(81,4,3,4) — DN1-OA PROVEN.

    generator="product"  (legacy)
        Uses FLUHyperCell (FM-Dance product construction).  4D macro
        address = index_to_coords(norm0, 3, 4).  Pre-V15.3.1 behaviour.

    Parameters
    ----------
    macro        : FLUHyperCell | None
                   If provided, used directly (backward compat, overrides
                   `generator`).  If None, built from `generator`.
    micro_lo_shu : np.ndarray | None
                   3×3 seed for the micro grid.  Defaults to canonical
                   Lo Shu if None.
    generator    : "sudoku" | "product"
                   Which macro generator to use when `macro` is None.
                   Default "sudoku" (V15.3.1+).

    STATUS: PROVEN — tensor product of two proven bijections.
    """

    TOTAL_CELLS   = 3 ** 6   # 729
    MACRO_N       = 9        # macro grid side length
    MICRO_N       = 3        # micro grid side length

    def __init__(
        self,
        macro        : Optional[FLUHyperCell] = None,
        micro_lo_shu : Optional[np.ndarray]   = None,
        *,
        generator    : Literal["sudoku", "product"] = "sudoku",
    ) -> None:
        # ── Macro layer ───────────────────────────────────────────────
        if macro is not None:
            # Explicit macro passed: use directly (full backward compat)
            self.macro        = macro
            self._generator   = "product"   # explicit FLUHyperCell = product
            self._adapter: Optional[SudokuMacroAdapter] = None
        elif generator == "sudoku":
            adapter           = SudokuMacroAdapter()
            self._adapter     = adapter
            self.macro        = None         # no FLUHyperCell in sudoku mode
            self._generator   = "sudoku"
        else:
            # generator="product", no explicit macro → build default FLUHyperCell
            self.macro        = FLUHyperCell()
            self._adapter     = None
            self._generator   = "product"

        # Unified accessor: _macro_layer has .sparse_address(r,c) and .cell(r,c)
        self._macro_layer: Any = self._adapter if self._adapter else self.macro

        # ── Micro layer ───────────────────────────────────────────────
        self._micro_grid   = (
            micro_lo_shu if micro_lo_shu is not None else _LO_SHU_NP
        ).copy().astype(np.int64)

        if self._micro_grid.shape != (3, 3):
            raise ValueError(
                f"micro_lo_shu must be shape (3,3), got {self._micro_grid.shape}"
            )

        # Pre-build MicroCell objects (9, one per 3×3 position)
        self._micro_cells: Dict[Tuple[int, int], MicroCell] = {
            (r, c): MicroCell(r, c, int(self._micro_grid[r, c]))
            for r in range(3) for c in range(3)
        }

        # Build O(1) reverse index: 6D coords → (macro_row, macro_col, micro_row, micro_col)
        self._reverse: Dict[Tuple[int, ...], Tuple[int, int, int, int]] = {}
        self._build_index()

    # ── Index construction ────────────────────────────────────────────────

    def _build_index(self) -> None:
        """
        Populate self._reverse for all 729 (macro, micro) pairs.

        STATUS: PROVEN — follows from the bijection proof in the module docstring.
        Raises RuntimeError if any duplicate 6D address is detected.
        """
        for macro_r in range(self.MACRO_N):
            for macro_c in range(self.MACRO_N):
                macro_4d = self._macro_layer.sparse_address(macro_r, macro_c)
                for micro_r in range(self.MICRO_N):
                    for micro_c in range(self.MICRO_N):
                        micro_2d  = self._micro_cells[(micro_r, micro_c)].coords_2d
                        coords_6d = macro_4d + micro_2d        # tuple concat

                        if coords_6d in self._reverse:
                            raise RuntimeError(
                                f"Duplicate 6D address {coords_6d} at "
                                f"macro=({macro_r},{macro_c}) "
                                f"micro=({micro_r},{micro_c})"
                            )
                        self._reverse[coords_6d] = (macro_r, macro_c, micro_r, micro_c)

    # ── Public address interface ──────────────────────────────────────────

    def sparse_address_6d(
        self,
        macro_row: int,
        macro_col: int,
        micro_row: int,
        micro_col: int,
    ) -> Tuple[int, ...]:
        """
        Grid positions → 6D FM-Dance sparse address.

        Parameters
        ----------
        macro_row, macro_col : int   position in 9×9 macro grid  [0, 8]
        micro_row, micro_col : int   position in 3×3 micro grid  [0, 2]

        Returns
        -------
        6-tuple ∈ {-1, 0, 1}⁶

        Complexity: O(4 + 2) = O(6)
        STATUS: PROVEN — direct composition of two proven bijections.
        """
        if not (0 <= macro_row < self.MACRO_N and 0 <= macro_col < self.MACRO_N):
            raise ValueError(
                f"macro position ({macro_row},{macro_col}) out of [0,{self.MACRO_N})"
            )
        if not (0 <= micro_row < self.MICRO_N and 0 <= micro_col < self.MICRO_N):
            raise ValueError(
                f"micro position ({micro_row},{micro_col}) out of [0,{self.MICRO_N})"
            )
        macro_4d = self._macro_layer.sparse_address(macro_row, macro_col)
        micro_2d = self._micro_cells[(micro_row, micro_col)].coords_2d
        return macro_4d + micro_2d

    def cell_at_6d(self, coords_6d: Tuple[int, ...]) -> CellPair:
        """
        6D FM-Dance address → CellPair  (inverse lookup, O(1)).

        Parameters
        ----------
        coords_6d : 6-tuple ∈ {-1, 0, 1}⁶

        Returns
        -------
        CellPair  with .macro (CellStrata) and .micro (MicroCell)

        Raises
        ------
        ValueError  if coords_6d is not a valid 6D address

        Complexity: O(1) — cached reverse index.
        STATUS: PROVEN — inverse of a proven bijection is a bijection.
        """
        pos = self._reverse.get(coords_6d)
        if pos is None:
            raise ValueError(f"6D address {coords_6d} not in index")
        macro_r, macro_c, micro_r, micro_c = pos
        return CellPair(
            macro=self._macro_layer.cell(macro_r, macro_c),
            micro=self._micro_cells[(micro_r, micro_c)],
        )

    # ── Factory classmethods ──────────────────────────────────────────────

    @classmethod
    def make_sudoku(
        cls,
        micro_lo_shu: Optional[np.ndarray] = None,
    ) -> "FractalHyperCell_3_6":
        """
        Create a 3⁶ cell using the Graeco-Latin Sudoku macro generator.

        This is the new default (V15.3.1+).  The macro 4D addresses form
        OA(81,4,3,4) — all 81 four-tuples in {-1,0,1}⁴ appear exactly once.

        Parameters
        ----------
        micro_lo_shu : np.ndarray | None   optional 3×3 micro seed

        Returns
        -------
        FractalHyperCell_3_6  with generator="sudoku"
        """
        return cls(macro=None, micro_lo_shu=micro_lo_shu, generator="sudoku")

    @classmethod
    def make_product(
        cls,
        micro_lo_shu: Optional[np.ndarray] = None,
    ) -> "FractalHyperCell_3_6":
        """
        Create a 3⁶ cell using the legacy FM-Dance product macro generator.

        This preserves the pre-V15.3.1 behaviour exactly.

        Parameters
        ----------
        micro_lo_shu : np.ndarray | None   optional 3×3 micro seed

        Returns
        -------
        FractalHyperCell_3_6  with generator="product"
        """
        return cls(macro=None, micro_lo_shu=micro_lo_shu, generator="product")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def generator(self) -> str:
        """Which macro generator is active: 'sudoku' or 'product'."""
        return self._generator

    @property
    def sudoku_hypercell(self) -> Optional[LoShuSudokuHyperCell]:
        """
        The underlying LoShuSudokuHyperCell when generator='sudoku',
        else None.
        """
        if self._adapter is not None:
            return self._adapter.hypercell
        return None

    # ── Verification (ITER-3B seam check) ─────────────────────────────────

    def verify(self, silent: bool = True) -> Dict[str, Any]:
        """
        Verify the 3⁶ seam: all 729 cells have unique, valid 6D addresses
        and satisfy the round-trip property.

        For generator="sudoku", additionally verifies OA(81,4,3,4) of the
        macro 4D addresses.

        THEOREM (FractalHyperCell_3_6 seam), STATUS: PROVEN
        ─────────────────────────────────────────────────────
        For every (macro_row, macro_col, micro_row, micro_col):
          (a) coords_6d ∈ {-1,0,1}⁶
          (b) All 729 coords_6d are distinct
          (c) cell_at_6d(coords_6d) round-trips back to the same positions

        Returns
        -------
        dict with keys:
            generator          : str   "sudoku" or "product"
            total_cells        : int   should be 729
            unique_addresses   : int   should be 729
            range_errors       : int   should be 0
            round_trip_errors  : int   should be 0
            macro_oa_strength  : int   4 for sudoku, 2 for product
            seam_verified      : bool
        """
        all_coords        = {}
        range_errors      = 0
        round_trip_errors = 0

        for macro_r in range(self.MACRO_N):
            for macro_c in range(self.MACRO_N):
                for micro_r in range(self.MICRO_N):
                    for micro_c in range(self.MICRO_N):
                        coords = self.sparse_address_6d(
                            macro_r, macro_c, micro_r, micro_c
                        )
                        if not all(c in (-1, 0, 1) for c in coords):
                            range_errors += 1
                        all_coords[coords] = (macro_r, macro_c, micro_r, micro_c)
                        pos_back    = self._reverse[coords]
                        coords_back = self.sparse_address_6d(*pos_back)
                        if coords_back != coords:
                            round_trip_errors += 1

        unique_count  = len(all_coords)
        total_count   = self.MACRO_N * self.MACRO_N * self.MICRO_N * self.MICRO_N

        # OA strength of macro 4D addresses
        macro_oa = self._measure_macro_oa_strength()

        seam_verified = (
            unique_count      == self.TOTAL_CELLS
            and range_errors  == 0
            and round_trip_errors == 0
        )

        result = {
            "generator"        : self._generator,
            "total_cells"      : total_count,
            "unique_addresses" : unique_count,
            "range_errors"     : range_errors,
            "round_trip_errors": round_trip_errors,
            "macro_oa_strength": macro_oa,
            "seam_verified"    : seam_verified,
        }

        if not silent:
            status = "✓ 3⁶ SEAM VERIFIED" if seam_verified else "✗ 3⁶ SEAM FAILED"
            print(
                f"FractalHyperCell_3_6 [{self._generator}]: {status} "
                f"({unique_count}/729 unique, "
                f"macro OA strength={macro_oa}, "
                f"{range_errors} range errors, "
                f"{round_trip_errors} round-trip errors)"
            )

        return result

    def _measure_macro_oa_strength(self) -> int:
        """
        Measure the OA strength of the 81 macro 4D addresses.

        Returns the largest s such that all s-tuples of dimensions are
        uniformly covered (each appearing exactly 81/3^s times).
        """
        import itertools
        macro_addrs = [
            self._macro_layer.sparse_address(r, c)
            for r in range(self.MACRO_N)
            for c in range(self.MACRO_N)
        ]
        for s in range(1, 5):
            expected = 81 // (3 ** s)
            ok = True
            for dims in itertools.combinations(range(4), s):
                projection = [tuple(a[d] for d in dims) for a in macro_addrs]
                counts = {}
                for p in projection:
                    counts[p] = counts.get(p, 0) + 1
                if len(counts) != 3**s or any(v != expected for v in counts.values()):
                    ok = False
                    break
            if not ok:
                return s - 1
        return 4

    # ── Convenience ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.TOTAL_CELLS

    def __repr__(self) -> str:
        return (
            f"FractalHyperCell_3_6("
            f"generator={self._generator!r}, "
            f"cells=729, "
            f"index_size={len(self._reverse)})"
        )
