"""
flu/applications/lighthouse.py
===============================
LighthouseBeacon — Container-Based Post-Quantum Cryptography demo.

╔══════════════════════════════════════════════════════════════╗
║  SIMULATION ONLY — MATHEMATICAL DEMONSTRATION                ║
║  This is NOT a production cryptographic system.              ║
║  It provides NO real security guarantees.                    ║
║  Do not use to protect real data.                            ║
╚══════════════════════════════════════════════════════════════╝

LighthouseBeacon demonstrates how FLU hyperprisms can structure a
symmetric key exchange "handshake" using:
  • Factoradic arrow generation (Lehmer-code strand)
  • CommunionEngine fusion for key material derivation
  • Latin Hypercube padding for theoretical diffusion

The Schumann and OM frequencies are kept as named constants from
earlier versions; they play no cryptographic role here and are
preserved as contextual / cosmological markers only.

EPISTEMIC STATUS: CONJECTURE — no formal security proof exists.
  Real post-quantum cryptography requires NIST PQC standards
  (e.g. CRYSTALS-Kyber, FALCON, SPHINCS+).

CLI entry point: flu-lighthouse  (declared in pyproject.toml)
    $ flu-lighthouse --n 3 --rounds 1

Dependencies: flu.core.factoradic, flu.core.fm_dance,
              flu.container.communion. numpy only.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flu.core.factoradic     import factoradic_unrank, arrow_generator
from flu.core.fm_dance       import generate_fast, index_to_coords
from flu.container.communion import CommunionEngine
from flu.utils.math_helpers  import is_odd


# ── Physical / cosmological constants (context only, not cryptographic) ───────

SCHUMANN_HZ: float = 7.83      # Earth's Schumann resonance [Hz]
OM_HZ      : float = 136.1     # OM / AUM resonance [Hz]
PHI        : float = 1.61803398875   # Golden ratio φ


# ── BeaconKey ─────────────────────────────────────────────────────────────────

class BeaconKey:
    """
    Result of a LighthouseBeacon handshake.

    Attributes
    ----------
    material  : np.ndarray   raw integer key material (arrow fusion result)
    digest    : bytes        SHA-256 of the material (32 bytes)
    hex       : str          hex string of digest
    n_rounds  : int          number of factoradic rounds used
    n         : int          FLU order used
    """

    SIMULATION_ONLY = True

    def __init__(
        self,
        material: np.ndarray,
        n_rounds: int,
        n       : int,
    ) -> None:
        self.material  = material
        self.n_rounds  = n_rounds
        self.n         = n
        self.digest    = hashlib.sha256(material.tobytes()).digest()
        self.hex       = self.digest.hex()

    def __repr__(self) -> str:
        return (
            f"BeaconKey(n={self.n}, rounds={self.n_rounds}, "
            f"hex={self.hex[:16]}…, SIMULATION_ONLY=True)"
        )


# ── LighthouseBeacon ──────────────────────────────────────────────────────────

class LighthouseBeacon:
    """
    Container-Based Post-Quantum Cryptography demonstration.

    ┌─────────────────────────────────────────────────────────────┐
    │ SIMULATION ONLY — NOT a production cryptographic system.    │
    │ This is a mathematical toy demonstrating FLU structure.     │
    └─────────────────────────────────────────────────────────────┘

    The "handshake" combines:
      1. Factoradic arrow generation (Lehmer-code strand)
      2. FM-Dance hyperprism construction
      3. CommunionEngine Kronecker fusion of arrow + hyperprism slices

    The result is deterministic for a fixed (n, pivot, rounds) and can
    be used as a structured pseudo-random key material generator for
    demonstrations.

    Parameters
    ----------
    n      : int   FLU order (must be odd, ≥ 3; default 3)
    pivot  : int   Factoradic pivot value (default 0)
    rounds : int   Number of arrow fusion rounds (default 1)
    seed   : int | None  Optional numpy RNG seed for arrow selection

    Examples
    --------
    >>> beacon = LighthouseBeacon(n=3, rounds=1)
    >>> key = beacon.generate_key()
    >>> isinstance(key.hex, str)
    True
    >>> beacon.broadcast()      # prints SIMULATION ONLY, no network I/O
    """

    SIMULATION_ONLY = True   # sentinel — never perform real network I/O

    def __init__(
        self,
        n     : int           = 3,
        pivot : int           = 0,
        rounds: int           = 1,
        seed  : Optional[int] = None,
    ) -> None:
        if not is_odd(n):
            raise ValueError(f"n must be odd, got {n}")
        if n < 3:
            raise ValueError(f"n must be ≥ 3, got {n}")
        if rounds < 1:
            raise ValueError(f"rounds must be ≥ 1, got {rounds}")

        self.n       = n
        self.pivot   = pivot
        self.rounds  = rounds
        self._rng    = np.random.default_rng(seed)
        self._engine = CommunionEngine(mode="kronecker", phi=np.add)

    # ── Key generation ────────────────────────────────────────────────────

    def generate_key(self, k_start: int = 0) -> BeaconKey:
        """
        Generate structured key material via factoradic + FM-Dance fusion.

        SIMULATION ONLY.

        Algorithm (per round):
          1. Unrank the k_start-th factoradic arrow with fixed pivot.
          2. Take the FM-Dance 1D hyperprism slice at the pivot plane.
          3. Fuse arrow ⊗ hyperprism via Kronecker product.
          4. Accumulate across rounds by element-wise addition.

        The final material is hashed with SHA-256 to produce the key digest.

        Parameters
        ----------
        k_start : int   starting Lehmer-code rank (default 0)

        Returns
        -------
        BeaconKey  (SIMULATION_ONLY=True)
        """
        accumulated: Optional[np.ndarray] = None

        for r in range(self.rounds):
            k      = (k_start + r) % (self.n - 1)  # wrap within (n-1)! domain
            arrow  = factoradic_unrank(k, self.n, signed=True, pivot=self.pivot)

            # FM-Dance 1D hyperprism (step indices centred around zero)
            prism_1d = generate_fast(self.n, 1, signed=True).flatten()
            prism_1d = prism_1d.astype(np.int64) - (self.n - 1) // 2

            # Kronecker-fuse arrow and prism slice
            fused = self._engine.commune(arrow, prism_1d)

            if accumulated is None:
                accumulated = fused.copy()
            else:
                # Pad / crop to same length before accumulating
                l = min(len(accumulated), len(fused))
                accumulated = accumulated[:l] + fused[:l]

        material = accumulated if accumulated is not None else np.zeros(1, dtype=np.int64)
        return BeaconKey(material=material, n_rounds=self.rounds, n=self.n)

    # ── Broadcast ─────────────────────────────────────────────────────────

    def broadcast(self, key: Optional[BeaconKey] = None) -> None:
        """
        Print a SIMULATION ONLY broadcast to stdout.

        SIMULATION ONLY — no network I/O is ever performed.

        Parameters
        ----------
        key : BeaconKey | None   if None, a fresh key is generated
        """
        if key is None:
            key = self.generate_key()

        # ── Header ────────────────────────────────────────────────────────
        banner = "=" * 60
        print(banner)
        print("  LIGHTHOUSE BEACON — SIMULATION ONLY")
        print("  This is a mathematical demonstration.")
        print("  NOT a production cryptographic system.")
        print(banner)
        print(f"  FLU order n      : {self.n}")
        print(f"  Pivot            : {self.pivot}")
        print(f"  Rounds           : {self.rounds}")
        print(f"  Key digest (hex) : {key.hex[:32]}…")
        print(f"  Material length  : {len(key.material)} elements")
        print()
        print("  Context constants (cosmological, not cryptographic):")
        print(f"    SCHUMANN_HZ = {SCHUMANN_HZ} Hz")
        print(f"    OM_HZ       = {OM_HZ} Hz")
        print(f"    PHI         = {PHI}")
        print(banner)
        print("  NO network transmission has occurred.")
        print(banner)

    # ── Verification ──────────────────────────────────────────────────────

    def verify(self) -> Dict[str, Any]:
        """
        Run a self-consistency check on two independently generated keys.

        Two keys with different k_start values must produce distinct digests.
        The same k_start must reproduce the same digest (determinism).

        Returns
        -------
        dict with keys: deterministic (bool), distinct (bool), verified (bool)
        """
        key_a1 = self.generate_key(k_start=0)
        key_a2 = self.generate_key(k_start=0)   # same → must match
        key_b  = self.generate_key(k_start=1)   # different → must differ (n ≥ 3)

        deterministic = key_a1.digest == key_a2.digest
        distinct      = key_a1.digest != key_b.digest

        return {
            "deterministic": deterministic,
            "distinct"     : distinct,
            "verified"     : deterministic and distinct,
        }

    def __repr__(self) -> str:
        return (
            f"LighthouseBeacon("
            f"n={self.n}, pivot={self.pivot}, "
            f"rounds={self.rounds}, SIMULATION_ONLY=True)"
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

def cli_main(argv: Optional[List[str]] = None) -> None:
    """
    CLI entry point: flu-lighthouse

    Usage:  flu-lighthouse [--n N] [--pivot P] [--rounds R] [--seed S]

    Prints a SIMULATION ONLY broadcast.  No network I/O is performed.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog       = "flu-lighthouse",
        description=(
            "LighthouseBeacon CBPQC demo — SIMULATION ONLY.\n"
            "This is a mathematical demonstration, not a real cryptographic tool."
        ),
    )
    parser.add_argument("--n",      type=int, default=3, help="FLU order (odd, ≥3)")
    parser.add_argument("--pivot",  type=int, default=0, help="Factoradic pivot value")
    parser.add_argument("--rounds", type=int, default=1, help="Fusion rounds")
    parser.add_argument("--seed",   type=int, default=None, help="RNG seed")
    args = parser.parse_args(argv)

    try:
        beacon = LighthouseBeacon(
            n=args.n, pivot=args.pivot, rounds=args.rounds, seed=args.seed
        )
        beacon.broadcast()
        result = beacon.verify()
        print(f"\n  Self-check: deterministic={result['deterministic']}, "
              f"distinct={result['distinct']}, "
              f"verified={result['verified']}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
