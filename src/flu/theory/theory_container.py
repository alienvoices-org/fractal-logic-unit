"""
flu/theory/theory_container.py
================================
Permutation Lattice Algebra — Unified Algebraic Framework.

Single Responsibility: state the algebraic structure that unifies all
three FLU generators (FM-Dance, Factoradic, Lo Shu) and the Communion
operator.

KEY INSIGHT (from V11 Audit)
───────────────────────────────────────
  "FLU constructs permutation-labeled lattices.  The Communion operator
   is the tensor product of these permutation lattices."

UNIFIED DEFINITION
──────────────────
  H(n, d) = (Z_n^d, Π)
  where Π = (π_1,...,π_d),  π_i ∈ S_n  (permutations assigned to axes).

  Each generator specialises H(n,d) differently:

  Generator    Role          Formal map
  ──────────── ─────────────── ─────────────────────────────────────────
  FM-Dance     τ : [0,n^d) → Z_n^d   (traversal coordinate system)
  Factoradic   φ : ℕ → S_n            (permutation generator / unranker)
  Lo Shu       λ : Z_n^2 → Z_n^2     (orthogonal Latin embedding, n=3)

  The Communion operator C₁ ⊗_φ C₂ is the *tensor product* of permutation
  lattices.  Associativity of ⊗_φ requires φ to be associative (critical
  constraint, proven in PFNT-5 / theory.py).

THEOREMS
────────
  PC-1. Generator Roles          STATUS: PROVEN (by construction)
  PC-2. Communion as Tensor      STATUS: PROVEN (conditional on φ-associativity)
  PC-3. Latin Preservation       STATUS: PROVEN (consequence of PFNT-3)
  PC-4. Communion Associativity  STATUS: PROVEN (conditional on φ-associativity)

CONJECTURES
───────────
  PC-C1. Full Invariant Preservation  STATUS: CONJECTURE (subset proven; see C3)

No package-internal imports (pure-math leaf module).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Core algebraic definition ─────────────────────────────────────────────────

@dataclass(frozen=True)
class PermutationLattice:
    """
    H(n, d) — a permutation-labeled toroidal lattice.

    Attributes
    ----------
    n     : int  base (values per axis)
    d     : int  number of dimensions
    label : str  which generator produced this lattice

    The lattice has n^d cells.  Each axis a is 'colored' by a permutation
    π_a ∈ S_n which determines the digit-to-label assignment along that axis.
    """
    n    : int
    d    : int
    label: str = "abstract"

    def size(self) -> int:
        return self.n ** self.d

    def tensor_product(
        self,
        other: "PermutationLattice",
        phi_name: str = "add",
    ) -> "PermutationLattice":
        """
        Communion operator: H₁ ⊗_φ H₂.

        Requires self.n == other.n (same base).
        Result has dimension d₁ + d₂.

        THEOREM PC-2, STATUS: PROVEN (conditional on φ-associativity).
        """
        if self.n != other.n:
            raise ValueError(
                f"Communion requires equal base.  Got n={self.n} and n={other.n}."
            )
        return PermutationLattice(
            n     = self.n,
            d     = self.d + other.d,
            label = f"communion({self.label},{other.label},phi={phi_name})",
        )


# ── The three generators ──────────────────────────────────────────────────────

"""
THEOREM PC-1 (Generator Roles), STATUS: PROVEN (by construction)
─────────────────────────────────────────────────────────────────
Statement:
    The three FLU generators each specialise H(n,d) in a distinct way:

    1. FM-Dance (τ):
        τ : [0, n^d) → Z_n^d
        Traversal coordinate system — the bijection that assigns a
        unique signed d-tuple to each rank k.  Proven bijection (T1).
        Formally: τ = the kinetic prefix-sum traversal in fm_dance_path.py.

    2. Factoradic (φ):
        φ : ℕ → S_n  (via Lehmer code)
        Permutation generator — the bijection from rank k to the k-th
        permutation of S_n.  Proven bijection (PFNT-4).
        Formally: φ = factoradic_unrank in core/factoradic.py.

    3. Lo Shu (λ):
        λ : Z_3^2 → Z_3^2  (n=3 specialisation)
        Orthogonal Latin embedding — the classical 3×3 magic square,
        embedded in 9×9 Graeco-Latin structure.  72 perspectives.
        Formally: λ = LoShuHyperCell in core/lo_shu.py.

    Together: FM-Dance provides the *address space*, Factoradic provides
    the *permutation content*, Lo Shu provides the *canonical seed*.
"""

GENERATOR_ROLES: Dict[str, Dict[str, str]] = {
    "fm_dance": {
        "formal_map" : "tau : [0, n^d) -> Z_n^d",
        "role"       : "Traversal coordinate system",
        "module"     : "flu.core.fm_dance_path",
        "theorem"    : "T1 (Bijection), T2 (Hamiltonian), T3 (Latin)",
        "complexity" : "O(d) per call",
    },
    "factoradic": {
        "formal_map" : "phi : N -> S_n  (Lehmer code bijection)",
        "role"       : "Permutation generator / unranker",
        "module"     : "flu.core.factoradic",
        "theorem"    : "PFNT-4 (Kinetic Completeness)",
        "complexity" : "O(n) per unrank, O(n^2) per rank",
    },
    "lo_shu": {
        "formal_map" : "lambda : Z_3^2 -> Z_3^2",
        "role"       : "Orthogonal Latin embedding (canonical seed n=3)",
        "module"     : "flu.core.lo_shu",
        "theorem"    : "72-phase Graeco-Latin (in lo_shu.py)",
        "complexity" : "O(1) per cell lookup",
    },
}


# ── Communion operator formal theory ──────────────────────────────────────────

"""
THEOREM PC-2 (Communion as Tensor Product), STATUS: PROVEN (conditional)
──────────────────────────────────────────────────────────────────────────
Statement:
    The Communion operator C₁ ⊗_φ C₂ is the tensor product of two
    permutation lattices H(n, d₁) and H(n, d₂), producing H(n, d₁+d₂).

    Algebraic modes:
        outer     : C[i₁…i_{d₁}, j₁…j_{d₂}] = φ(C₁[i₁…], C₂[j₁…])
        direct    : combines along new leading dimension (requires d₁=d₂)
        kronecker : np.kron(flat(C₁), flat(C₂)), reshaped

    CRITICAL CONSTRAINT: φ must be associative.
    Proof of associativity of ⊗_φ given associativity of φ:
        (C₁ ⊗_φ C₂) ⊗_φ C₃ = φ(φ(c₁,c₂), c₃)
        C₁ ⊗_φ (C₂ ⊗_φ C₃) = φ(c₁, φ(c₂,c₃))
        These are equal iff φ is associative.  □

    Implemented in: flu.container.communion.CommunionEngine.
"""

"""
THEOREM PC-3 (Latin Preservation under Communion), STATUS: PROVEN
───────────────────────────────────────────────────────────────────
Statement:
    If C₁ and C₂ are Latin hyperprisms and φ = add (mod n or signed),
    then C₁ ⊗_{add} C₂ is a Latin hyperprism of dimension d₁+d₂.

Proof:
    Fix all indices in C₁ ⊗ C₂ except one axis a.
    Case a in dims of C₁: the free index i_a sweeps C₁[i_a],
        which is a permutation of D_set; adding C₂[j₁,...] shifts
        all values by the same constant ⟹ bijection preserved.
    Case a in dims of C₂: symmetric argument.
    In both cases the result is a permutation of (D_set + constant)
    = D_set (for mod n) or D_set shifted (for signed, still n distinct).  □

    This is a direct extension of PFNT-3 (theory.py) to the product dimension.
"""

"""
THEOREM PC-4 (Communion Associativity), STATUS: PROVEN (conditional)
──────────────────────────────────────────────────────────────────────
Statement:
    ⊗_φ is associative iff φ is associative.

Proof:
    PC-2 argument above.  □

    IMPLEMENTATION NOTE:
    flu.container.communion.CommunionEngine validates φ-associativity
    probabilistically on construction.  Non-associative φ raises ValueError.
    Do not bypass this check.
"""

"""
CONJECTURE PC-C1 (Full Invariant Preservation), STATUS: CONJECTURE
────────────────────────────────────────────────────────────────────
Claim:
    ⊗_φ preserves ALL FLU invariants (Latin, mean-centering, step bound,
    spectral mixed-flatness) for arbitrary associative φ beyond addition.

Partial results:
  Latin:           PROVEN for φ = add (PC-3 above).
  Mean-centering:  PROVEN for φ = add (mean 0 preserved by addition).
  Associativity:   PROVEN conditional on φ (PC-4).
  Step bound:      OPEN — bound after Communion may differ.
  Spectral:        Spectral mixed-flatness PROVEN for add (see theory_spectral.py);
                   unclear for general φ.

What is needed:
  Characterise which invariants persist exactly vs. up-to-isomorphism
  for general associative φ (XOR, multiply, etc.).
"""


# ── Permutation Lattice Algebra summary ───────────────────────────────────────

PERMUTATION_LATTICE_ALGEBRA: Dict[str, Any] = {
    "formal_definition" : "H(n,d) = (Z_n^d, Pi) where Pi = (pi_1,...,pi_d), pi_i in S_n",
    "generator_roles"   : GENERATOR_ROLES,
    "communion_insight" : (
        "The Communion operator C1 tensor_phi C2 is mathematically the tensor "
        "product of permutation lattices (outer, direct, or Kronecker product). "
        "Associativity strictly requires phi to be associative."
    ),
    "proven_theorems"   : ["PC-1", "PC-2", "PC-3", "PC-4"],
    "open_conjectures"  : ["PC-C1"],
    "critical_constraint": (
        "COMMUNION ASSOCIATIVITY REQUIRES PHI ASSOCIATIVITY. "
        "Non-associative phi (e.g. subtraction) makes ⊗_phi non-associative. "
        "The CommunionEngine constructor enforces this via probabilistic check."
    ),
}


def permutation_lattice_summary() -> str:
    """Return a human-readable summary of the permutation lattice algebra."""
    lines = [
        "FLU V14 — Permutation Lattice Algebra",
        "=" * 50,
        "",
        f"Formal definition: {PERMUTATION_LATTICE_ALGEBRA['formal_definition']}",
        "",
        "Three generators:",
    ]
    for name, info in GENERATOR_ROLES.items():
        lines.append(f"  {name:12s}: {info['formal_map']}")
        lines.append(f"              Role: {info['role']}")
        lines.append(f"              Theorem: {info['theorem']}")
        lines.append("")
    lines.append(f"Communion: {PERMUTATION_LATTICE_ALGEBRA['communion_insight'][:70]}…")
    lines.append("")
    lines.append(f"CRITICAL: {PERMUTATION_LATTICE_ALGEBRA['critical_constraint'][:80]}…")
    return "\n".join(lines)
