"""
flu/applications/quantum.py
===========================
TensorNetworkSimulator — FLU-structured tensor network state simulator.

╔══════════════════════════════════════════════════════════╗
║  STATUS: SIMULATION ONLY                                 ║
║  This is a mathematical demonstration, NOT a quantum     ║
║  computing system.  No claim of real quantum advantage   ║
║  or hardware execution is made or implied.               ║
╚══════════════════════════════════════════════════════════╝

Models n-qubit tensor-network states as FLU hyperprisms, using
CommunionEngine(mode='kronecker') to build composite states.

The mathematical structure (Latin Hypercube, balanced digit set) gives
well-behaved state vectors for simulation purposes.  This is not a
claim about quantum speedup or real quantum fidelity.

EPISTEMIC STATUS: CONJECTURE — no formal complexity proof exists.

Dependencies: flu.core.fm_dance, flu.core.even_n,
              flu.container.communion. numpy only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from flu.utils.math_helpers  import is_odd
from flu.core.fm_dance        import generate_fast
from flu.core.even_n          import generate as even_n_generate
from flu.container.communion  import CommunionEngine


class TensorNetworkSimulator:
    """
    FLU-structured tensor network state simulator.

    ┌─────────────────────────────────────────────────────┐
    │ SIMULATION ONLY — not a real quantum computing system│
    └─────────────────────────────────────────────────────┘

    Each "qubit" is modelled as a 1D FLU hyperprism of order n.
    Multi-qubit states are built via Kronecker fusion using
    CommunionEngine, producing a composite n^q state vector.

    Parameters
    ----------
    n     : int   FLU order per qubit (default 3, must be ≥ 2)
    signed: bool  Use balanced digit set (default True)

    Examples
    --------
    >>> sim = TensorNetworkSimulator(n=3)
    >>> state = sim.prepare_state(n_qubits=2)
    >>> state.shape
    (9,)
    >>> counts = sim.measure(state, n_shots=100)
    >>> sim.fidelity(state, state)   # self-overlap = 1.0
    1.0
    """

    SIMULATION_ONLY = True   # sentinel: this is never real quantum hardware

    def __init__(self, n: int = 3, signed: bool = True) -> None:
        if n < 2:
            raise ValueError(f"n must be ≥ 2, got {n}")
        self.n      = n
        self.signed = signed
        self._engine = CommunionEngine(
            mode = "kronecker",
            phi  = np.add,   # additive fusion for combining qubit slices
        )

    # ── State preparation ─────────────────────────────────────────────────

    def prepare_state(self, n_qubits: int) -> np.ndarray:
        """
        Prepare an n_qubits FLU tensor-network state vector.

        SIMULATION ONLY.

        Constructs a single-qubit FLU hyperprism of shape (n,) then fuses
        n_qubits copies via Kronecker product to form a composite state
        vector of length n^n_qubits.  The result is L2-normalised.

        Parameters
        ----------
        n_qubits : int   number of qubits (≥ 1)

        Returns
        -------
        np.ndarray  shape (n^n_qubits,)  float64, L2-normalised

        Raises
        ------
        ValueError   if n_qubits < 1
        """
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be ≥ 1, got {n_qubits}")

        # Single-qubit FLU vector (1D hyperprism)
        single = self._single_qubit_vector()

        # Fuse via Kronecker product for n_qubits, keeping 1D throughout
        state = single.copy()
        for _ in range(n_qubits - 1):
            fused = self._engine.commune(state, single)
            state = fused.flatten()   # Kronecker gives A.shape+B.shape; keep 1D

        # L2-normalise to unit probability amplitude
        norm = np.linalg.norm(state)
        if norm > 1e-12:
            state = state.astype(np.float64) / norm

        return state

    # ── Measurement ───────────────────────────────────────────────────────

    def measure(
        self,
        state  : np.ndarray,
        n_shots: int,
        rng    : Optional[np.random.Generator] = None,
    ) -> Dict[int, int]:
        """
        Sample from the FLU state distribution.

        SIMULATION ONLY.

        Computes a Born-rule probability distribution from the squared
        amplitudes of the state vector and draws n_shots samples.

        Parameters
        ----------
        state   : np.ndarray   1D state vector (need not be normalised)
        n_shots : int          number of measurement shots
        rng     : np.random.Generator | None  for reproducibility

        Returns
        -------
        dict  {outcome_index: count}  — only non-zero counts included

        Raises
        ------
        ValueError  if n_shots < 1 or state is not 1D
        """
        if state.ndim != 1:
            raise ValueError(f"state must be 1D, got shape {state.shape}")
        if n_shots < 1:
            raise ValueError(f"n_shots must be ≥ 1, got {n_shots}")

        rng    = rng or np.random.default_rng()
        probs  = state.astype(np.float64) ** 2
        total  = probs.sum()
        if total < 1e-12:
            raise ValueError("State vector has zero norm; cannot sample")
        probs /= total

        outcomes = rng.choice(len(probs), size=n_shots, p=probs)
        counts   = {}
        for o in outcomes:
            counts[int(o)] = counts.get(int(o), 0) + 1
        return counts

    # ── Fidelity ─────────────────────────────────────────────────────────

    def fidelity(
        self,
        state_a: np.ndarray,
        state_b: np.ndarray,
    ) -> float:
        """
        Compute the overlap (fidelity) between two FLU state vectors.

        SIMULATION ONLY.

        F(a, b) = |⟨a|b⟩|² = |dot(a*, b)|² / (||a||² · ||b||²)

        Parameters
        ----------
        state_a, state_b : np.ndarray   1D state vectors

        Returns
        -------
        float  fidelity ∈ [0, 1]

        Raises
        ------
        ValueError  if shapes mismatch or either norm is zero
        """
        if state_a.shape != state_b.shape:
            raise ValueError(
                f"Shape mismatch: {state_a.shape} vs {state_b.shape}"
            )
        if state_a.ndim != 1:
            raise ValueError("state vectors must be 1D")

        a = state_a.astype(np.float64)
        b = state_b.astype(np.float64)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            raise ValueError("Cannot compute fidelity for zero-norm state")

        overlap   = np.dot(a, b)
        fidelity  = float((overlap ** 2) / (norm_a ** 2 * norm_b ** 2))
        # Clamp to [0, 1] against floating-point overshoot
        return float(np.clip(fidelity, 0.0, 1.0))

    # ── Internal helpers ──────────────────────────────────────────────────

    def _single_qubit_vector(self) -> np.ndarray:
        """
        Build a 1D FLU hyperprism of shape (n,) as a single-qubit vector.

        Uses FM-Dance for odd n, sum-mod for even n.
        """
        if is_odd(self.n):
            raw = generate_fast(self.n, 1, signed=self.signed).flatten()
            # Centre step indices around zero
            raw = raw.astype(np.float64) - (self.n - 1) / 2.0
        else:
            raw = even_n_generate(self.n, 1, signed=self.signed).flatten().astype(np.float64)
        return raw

    def __repr__(self) -> str:
        return (
            f"TensorNetworkSimulator("
            f"n={self.n}, signed={self.signed}, "
            f"SIMULATION_ONLY=True)"
        )
