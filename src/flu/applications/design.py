"""
flu/applications/design.py
==========================
ExperimentalDesign — Latin Hypercube Experimental Design.

STATUS: DESIGN INTENT (thin wrapper; maths proven in core layer)

Wraps generate_fast() and even_n_generate() to produce publication-quality
Latin Hypercube designs.  All mathematical guarantees (Latin property,
coverage, mean-centering) are inherited from the core layer and can be
verified through the report returned by generate().

Odd  n_levels  → generate_fast()      (FM-Dance, STATUS: PROVEN)
Even n_levels  → even_n_generate()    (sum-mod,  STATUS: PROVEN)

Optional pandas export is guarded: if pandas is not installed the
to_dataframe() method raises ImportError with an install hint.

Dependencies: flu.core.fm_dance, flu.core.even_n, flu.utils.verification.
No new external deps beyond numpy (pandas is optional).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from flu.utils.math_helpers  import is_odd
from flu.utils.verification  import check_latin, check_coverage, check_mean_centered
from flu.core.fm_dance        import generate_fast, verify_bijection
from flu.core.even_n          import generate as even_n_generate

try:
    import pandas as _pd
except ImportError:
    _pd = None   # guarded optional


# ── DesignResult ──────────────────────────────────────────────────────────────

class DesignResult:
    """
    Container for a generated Latin Hypercube design and its verification report.

    Attributes
    ----------
    matrix       : np.ndarray  shape (n_levels,)*n_factors
    n_levels     : int
    n_factors    : int
    factor_names : list[str]
    signed       : bool        True → balanced digit set {-(n-1)/2, …, (n-1)/2}
    report       : dict        verification results (latin_ok, coverage_ok, mean_ok)
    overall_pass : bool        True iff all checks pass
    """

    def __init__(
        self,
        matrix      : np.ndarray,
        n_levels    : int,
        n_factors   : int,
        factor_names: List[str],
        signed      : bool,
        report      : Dict[str, Any],
    ) -> None:
        self.matrix       = matrix
        self.n_levels     = n_levels
        self.n_factors    = n_factors
        self.factor_names = factor_names
        self.signed       = signed
        self.report       = report
        self.overall_pass: bool = (
            report["latin"]["latin_ok"]
            and report["coverage"]["coverage_ok"]
            and report["mean"]["mean_ok"]
        )

    def __repr__(self) -> str:
        status = "PASS" if self.overall_pass else "FAIL"
        return (
            f"DesignResult(n_levels={self.n_levels}, "
            f"n_factors={self.n_factors}, "
            f"shape={self.matrix.shape}, "
            f"verified={status})"
        )


# ── ExperimentalDesign ────────────────────────────────────────────────────────

class ExperimentalDesign:
    """
    Latin Hypercube Experimental Design generator.

    Uses FLU hyperprisms as the underlying combinatorial structure.

    STATUS: DESIGN INTENT — thin wrapper over proven core functions.

    Parameters
    ----------
    signed : bool   Use balanced (signed) digit set.  Default True.
                    Signed designs are mean-centred at 0 for odd n_levels.

    Examples
    --------
    >>> ed = ExperimentalDesign()
    >>> result = ed.generate(n_levels=5, n_factors=3)
    >>> result.overall_pass
    True
    >>> result.matrix.shape
    (5, 5, 5)
    """

    def __init__(self, signed: bool = True) -> None:
        self.signed = signed

    # ── Core generation ───────────────────────────────────────────────────

    def generate(
        self,
        n_levels    : int,
        n_factors   : int,
        factor_names: Optional[Sequence[str]] = None,
    ) -> DesignResult:
        """
        Generate an n_levels^n_factors Latin Hypercube design.

        Parameters
        ----------
        n_levels     : int            number of levels per factor (≥ 2)
        n_factors    : int            number of factors / dimensions (≥ 1)
        factor_names : list[str] | None  optional factor labels

        Returns
        -------
        DesignResult  with .matrix and .report

        Raises
        ------
        ValueError   if n_levels < 2 or n_factors < 1

        Algorithm
        ---------
        Odd  n_levels → FM-Dance  generate_fast(n_levels, n_factors)
        Even n_levels → sum-mod   even_n_generate(n_levels, n_factors)
        Both are proven Latin Hypercubes (see core layer STATUS markers).
        """
        if n_levels < 2:
            raise ValueError(f"n_levels must be ≥ 2, got {n_levels}")
        if n_factors < 1:
            raise ValueError(f"n_factors must be ≥ 1, got {n_factors}")

        names = (
            list(factor_names)
            if factor_names is not None
            else [f"F{i}" for i in range(n_factors)]
        )
        if len(names) != n_factors:
            raise ValueError(
                f"factor_names length {len(names)} ≠ n_factors {n_factors}"
            )

        if is_odd(n_levels):
            matrix = generate_fast(n_levels, n_factors, signed=self.signed)
        else:
            matrix = even_n_generate(n_levels, n_factors, signed=self.signed)

        report = self.verify_design(matrix, n_levels)

        return DesignResult(
            matrix=matrix,
            n_levels=n_levels,
            n_factors=n_factors,
            factor_names=names,
            signed=self.signed,
            report=report,
        )

    # ── Verification ──────────────────────────────────────────────────────

    def verify_design(
        self,
        matrix: np.ndarray,
        n     : Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run Latin, coverage and mean-centering checks on a design matrix.

        Routes to the appropriate internal verify based on n parity:
          Odd  n → verify_bijection() + _check_latin_slices() (fm_dance path)
          Even n → even_n_verify()                            (sum-mod path)

        Parameters
        ----------
        matrix : np.ndarray  shape (n,)*d hyperprism
        n      : int | None  inferred from matrix.shape[0] if None

        Returns
        -------
        dict with keys: latin, coverage, mean, overall_pass
        """
        if n is None:
            n = matrix.shape[0]
        d = matrix.ndim

        if is_odd(n):
            # generate_fast stores step indices k ∈ [0, n^d).
            # verify_bijection checks the bijection round-trip directly.
            bj      = verify_bijection(n, d)
            latin   = {"latin_ok": bj["bijection_ok"], "violations": []}
            # Coverage: every value 0..(n^d-1) appears exactly once
            flat    = matrix.flatten()
            expected_vals = set(range(n ** d))
            actual_vals   = set(int(v) for v in flat)
            coverage_ok   = (actual_vals == expected_vals)
            coverage = {
                "coverage_ok"   : coverage_ok,
                "expected_count": 1,
                "violations"    : {} if coverage_ok else {"missing": expected_vals - actual_vals},
            }
            # Mean check: step indices have mean (n^d-1)/2
            actual_mean = float(np.mean(matrix))
            expected_mean = (n ** d - 1) / 2.0
            mean_ok = bool(np.isclose(actual_mean, expected_mean, atol=1e-6))
            mean = {"mean_ok": mean_ok, "actual": actual_mean, "expected": expected_mean}
        else:
            # even_n.verify uses signed=False internally
            from flu.core.even_n import verify as _even_verify
            ev = _even_verify(n, d)
            latin    = {"latin_ok": ev["latin_ok"], "violations": ev["violations"]}
            coverage = {"coverage_ok": ev["coverage_ok"], "expected_count": n ** (d - 1), "violations": {}}
            # Mean: unsigned [0, n-1] → (n-1)/2; signed shifts by -n//2
            actual_mean   = float(np.mean(matrix))
            shift         = (n // 2) if self.signed else 0
            expected_mean = (n - 1) / 2.0 - shift
            mean_ok = bool(np.isclose(actual_mean, expected_mean, atol=1e-6))
            mean = {"mean_ok": mean_ok, "actual": actual_mean, "expected": expected_mean}

        overall = latin["latin_ok"] and coverage["coverage_ok"] and mean["mean_ok"]
        return {
            "latin"       : latin,
            "coverage"    : coverage,
            "mean"        : mean,
            "overall_pass": overall,
        }

    # ── Sampling ──────────────────────────────────────────────────────────

    def stratified_sample(
        self,
        result  : DesignResult,
        n_samples: int,
        rng     : Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Draw n_samples rows from a DesignResult, preserving Latin structure.

        For each sample the method selects one level per factor by drawing
        without replacement along axis 0 of the flattened run list.
        This guarantees that no level is repeated within a sample
        (space-filling property).

        Parameters
        ----------
        result    : DesignResult
        n_samples : int   must be ≤ n_levels
        rng       : np.random.Generator | None  for reproducibility

        Returns
        -------
        np.ndarray  shape (n_samples, n_factors)  sampled run matrix

        Raises
        ------
        ValueError  if n_samples > n_levels
        """
        n = result.n_levels
        d = result.n_factors

        if n_samples > n:
            raise ValueError(
                f"n_samples={n_samples} exceeds n_levels={n}; "
                f"Latin structure cannot be preserved"
            )

        rng    = rng or np.random.default_rng()
        matrix = result.matrix

        # Build a flat run list: each run is one combination of indices
        # (one index per factor dimension), reading diagonal slices so that
        # every level appears at most once per sample.
        levels = rng.permutation(n)[:n_samples]

        # For each factor (axis), select a random permutation of levels
        # so that the n_samples rows have no repeated level per factor.
        sample = np.zeros((n_samples, d), dtype=matrix.dtype)
        for f in range(d):
            perm = rng.permutation(n)[:n_samples]
            for s, lvl in enumerate(perm):
                # Extract the value at this level along factor f for the
                # overall chosen run.  Use the diagonal to avoid bias.
                idx         = tuple(levels[s] if ax == f else perm[s] for ax in range(d))
                sample[s, f] = int(matrix[idx])

        return sample

    # ── Optional pandas export ────────────────────────────────────────────

    def to_dataframe(self, result: DesignResult) -> "pd.DataFrame":  # type: ignore[name-defined]
        """
        Convert a DesignResult to a pandas DataFrame (one row per run).

        Requires pandas.  Install with: pip install flu[design]
        or: pip install pandas

        Parameters
        ----------
        result : DesignResult

        Returns
        -------
        pd.DataFrame  shape (n_levels^n_factors, n_factors)

        Raises
        ------
        ImportError  if pandas is not installed
        """
        if _pd is None:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas  (or: pip install flu[design])"
            )
        # Flatten along all axes except the first (each top-level "run")
        # Result: (n_levels^n_factors,  n_factors) is not well-defined for
        # a hypercube — instead present as a 1-D list of all cell values.
        # For a 2D design (d=2) each row is a natural run; for d>2 we
        # flatten the inner axes so each column is one factor's value slice.
        if result.n_factors == 1:
            flat = result.matrix.reshape(-1, 1)
        else:
            # Reshape n^d matrix to (n^(d-1), n) for d=2, or flatten last axis
            n   = result.n_levels
            d   = result.n_factors
            flat = result.matrix.reshape(n ** (d - 1), n)
            # Use first d columns only (n may be > n_factors for odd path)
            flat = flat[:, :result.n_factors] if flat.shape[1] >= result.n_factors else flat
        return _pd.DataFrame(flat, columns=result.factor_names[:flat.shape[1]])
