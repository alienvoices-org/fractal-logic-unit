#!/usr/bin/env python3
"""
tools/generate_registry_json.py
================================
Auto-generates src/flu/theory/THEOREM_REGISTRY.json from the live Python
theorem registry (theorem_registry.py).

This is the ONLY way THEOREM_REGISTRY.json should ever be updated.
Never edit THEOREM_REGISTRY.json by hand — it will drift from the source.

USAGE
-----
    python tools/generate_registry_json.py
    python tools/generate_registry_json.py --output path/to/output.json
    python tools/generate_registry_json.py --verify    # check existing file matches

SCHEMA
------
The output JSON conforms to flu-srp-package-contract/v1.0#theorem-registry.
It contains the full dependency graph, cross-references, and proof sketches.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flu.theory.theorem_registry import REGISTRY
from flu._version import __version__


# ── Dependency graph ──────────────────────────────────────────────────────────

_ALL_IDS = set(REGISTRY.keys())
_ID_PATTERN = re.compile(
    r'\b(T[1-9]\d?|KIB|BPT|CGW|SRM|T7|PFNT-[1-5]|L[1-3]|S[12]|S2-Prime|C[2-4])\b'
)

def extract_depends_on(theorem_id: str, proof_text: str) -> list[str]:
    """Find theorem IDs referenced in proof text, excluding self-reference."""
    refs = _ID_PATTERN.findall(proof_text or "")
    return sorted(set(r for r in refs if r != theorem_id and r in _ALL_IDS))


def build_depended_on_by(depends_on_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """Invert the depends_on map to get depended_on_by."""
    result: dict[str, list[str]] = {tid: [] for tid in _ALL_IDS}
    for tid, deps in depends_on_map.items():
        for dep in deps:
            if dep in result:
                result[dep].append(tid)
    return {k: sorted(v) for k, v in result.items()}


# ── Perspective assignment ────────────────────────────────────────────────────

_PERSPECTIVE_MAP = {
    "T1": "matrix",    "T2": "kinetic",   "T3": "matrix",
    "T4": "kinetic",   "T5": "matrix",    "T6": "matrix",
    "T7": "algebraic", "SRM": "algebraic","KIB": "kinetic",
    "BPT": "kinetic",  "CGW": "algebraic","C4": "kinetic",
    "PFNT-1": "algebraic", "PFNT-2": "matrix", "PFNT-3": "matrix",
    "PFNT-4": "algebraic", "PFNT-5": "algebraic",
    "L1": "matrix",    "L2": "kinetic",   "L3": "kinetic",
    "S1": "matrix",    "S2": "spectral",  "S2-Prime": "spectral",
    "C2": "spectral",  "C3": "algebraic",
}

_CODE_REF_MAP = {
    "T1":     ["flu.core.fm_dance_path.path_coord", "flu.core.fm_dance_path.path_coord_to_rank"],
    "T2":     ["flu.core.fm_dance_path.traverse"],
    "T3":     ["flu.core.fm_dance.generate_fast", "flu.core.parity_switcher.generate"],
    "T4":     ["flu.core.fm_dance_path.step_bound_theorem"],
    "T5":     ["flu.core.fm_dance_path.verify_siamese_d2"],
    "T6":     ["flu.core.fm_dance_path.verify_fractal"],
    "T7":     ["flu.core.fm_dance_path.cayley_generators"],
    "SRM":    ["flu.core.fm_dance_path.identify_step", "flu.core.fm_dance_path.invert_fm_dance_step"],
    "KIB":    ["flu.core.fm_dance_path.identify_step", "flu.core.fm_dance_path.invert_fm_dance_step"],
    "BPT":    ["flu.core.fm_dance_path.boundary_partition_sizes", "flu.core.fm_dance_path.fractal_fault_lines"],
    "CGW":    ["flu.core.fm_dance_path.cayley_generators", "flu.core.fm_dance_path.cayley_inverse_generators", "flu.core.fm_dance_path.traverse_reverse"],
    "PFNT-4": ["flu.core.factoradic.factoradic_unrank", "flu.core.factoradic.unrank_optimal_seed"],
    "PFNT-5": ["flu.container.communion.CommunionEngine.commune"],
    "L2":     ["flu.theory.theory_latin.holographic_repair"],
    "L3":     ["flu.theory.theory_latin.holographic_repair"],
    "S2-Prime":["flu.theory.theory_spectral.spectral_dispersion_bound"],
}


# ── Main generator ────────────────────────────────────────────────────────────

def generate_registry_json(output_path: Path | None = None) -> dict:
    """
    Generate the theorem registry JSON from the live Python registry.

    Parameters
    ----------
    output_path : Path or None  if given, writes to file

    Returns
    -------
    dict  the full registry object
    """
    # Build dependency graph
    depends_on_map = {
        tid: extract_depends_on(tid, t.proof)
        for tid, t in REGISTRY.items()
    }
    depended_on_by_map = build_depended_on_by(depends_on_map)

    # Categorise theorems
    proven      = [tid for tid, t in REGISTRY.items() if t.status == "PROVEN"]
    conjectures = [tid for tid, t in REGISTRY.items() if t.status == "CONJECTURE"]
    partial     = [tid for tid, t in REGISTRY.items() if t.status == "PARTIAL"]
    disproven   = [tid for tid, t in REGISTRY.items() if "DISPROVEN" in t.status]
    retired     = [tid for tid, t in REGISTRY.items() if t.status == "RETIRED"]
    false_      = [tid for tid, t in REGISTRY.items() if t.status == "FALSE"]

    # Build theorem records
    theorems = {}
    for tid, t in REGISTRY.items():
        theorems[tid] = {
            "id":             tid,
            "name":           t.name,
            "status":         t.status,
            "proof_status":   getattr(t, "proof_status", "algebraic_sketch"),
            "statement":      t.statement,
            "proof_sketch":   t.proof,
            "conditions":     t.conditions,
            "perspective":    _PERSPECTIVE_MAP.get(tid, "general"),
            "depends_on":     depends_on_map[tid],
            "depended_on_by": depended_on_by_map[tid],
            "code_refs":      _CODE_REF_MAP.get(tid, []) + list(t.references),
            "implications":   depended_on_by_map[tid],  # alias for readability
        }

    registry = {
        "$schema":   "flu-srp-package-contract/v1.0#theorem-registry",
        "$comment":  "AUTO-GENERATED by tools/generate_registry_json.py — do not edit by hand.",
        "generated": str(date.today()),
        "source":    "src/flu/theory/theorem_registry.py",
        "package":   "FLU",
        "version":   __version__,

        "summary": {
            "total":            len(REGISTRY),
            "proven":           len(proven),
            "partial":          len(partial),
            "conjecture":       len(conjectures),
            "disproven_scoped": len(disproven),
            "retired":          len(retired),
            "false":            len(false_),
            "score":            (
                f"{len(proven)} PROVEN · "
                f"{len(partial)} PARTIAL · "
                f"{len(conjectures)} CONJECTURE · "
                f"{len(disproven)} DISPROVEN_SCOPED · "
                f"{len(retired)} RETIRED"
            ),
        },

        "dependency_chains": {
            "core_bijection_chain":  ["T1", "T2", "T3"],
            "kinetic_inverse_chain": ["T2", "KIB", "BPT", "SRM"],
            "algebraic_chain":       ["T2", "CGW", "T7", "SRM"],
            "healing_chain":         ["L1", "L2", "L3"],
            "spectral_chain":        ["S1", "S2-Prime", "S2"],
            "pfnt_chain":            ["PFNT-1", "PFNT-2", "PFNT-3", "PFNT-4", "PFNT-5"],
        },

        "perspective_groups": {
            "matrix":    [tid for tid in REGISTRY if _PERSPECTIVE_MAP.get(tid) == "matrix"],
            "kinetic":   [tid for tid in REGISTRY if _PERSPECTIVE_MAP.get(tid) == "kinetic"],
            "algebraic": [tid for tid in REGISTRY if _PERSPECTIVE_MAP.get(tid) == "algebraic"],
            "spectral":  [tid for tid in REGISTRY if _PERSPECTIVE_MAP.get(tid) == "spectral"],
        },

        "proven_ids":      proven,
        "conjecture_ids":  conjectures,
        "retired_ids":     retired,

        "theorems": theorems,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        print(f"Written: {output_path}  ({len(REGISTRY)} theorems)")

    return registry


def verify_registry_json(json_path: Path) -> bool:
    """
    Check that an existing THEOREM_REGISTRY.json matches the live Python registry.

    Returns True if they match, False (with details printed) otherwise.
    """
    if not json_path.exists():
        print(f"MISSING: {json_path}")
        return False

    with open(json_path, encoding="utf-8") as f:
        existing = json.load(f)

    live = generate_registry_json()

    live_ids   = set(live["theorems"].keys())
    saved_ids  = set(existing.get("theorems", {}).keys())

    if live_ids != saved_ids:
        print(f"DRIFT: Live IDs = {live_ids}, Saved IDs = {saved_ids}")
        return False

    drift = []
    for tid in live_ids:
        for field in ("status", "statement"):
            if live["theorems"][tid][field] != existing["theorems"][tid].get(field):
                drift.append(f"  {tid}.{field} changed")

    if drift:
        print(f"DRIFT in {len(drift)} fields:")
        for d in drift:
            print(d)
        print("Run: python tools/generate_registry_json.py")
        return False

    print(f"✓  THEOREM_REGISTRY.json is current ({len(live_ids)} theorems)")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate THEOREM_REGISTRY.json from Python registry")
    ap.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent.parent / "src" / "flu" / "theory" / "THEOREM_REGISTRY.json",
        help="Output path (default: src/flu/theory/THEOREM_REGISTRY.json)",
    )
    ap.add_argument(
        "--verify", action="store_true",
        help="Verify existing file matches live registry (no write)",
    )
    args = ap.parse_args()

    if args.verify:
        ok = verify_registry_json(args.output)
        sys.exit(0 if ok else 1)
    else:
        generate_registry_json(output_path=args.output)
        # Also verify after writing
        verify_registry_json(args.output)
