#!/usr/bin/env python3
"""
tools/theorem_dag.py
=====================
Renders the FLU theorem dependency DAG as ASCII art.

Addresses the external audit recommendation (March 2026):
    'Add a theorem graph visualizer. You already have dependency_chains.
     You could auto-generate a theorem DAG. That would instantly make
     the theory understandable.'

The full dependency structure is read from THEOREM_REGISTRY.json so this
output is always consistent with the live registry.

USAGE
-----
    python tools/theorem_dag.py              # full DAG, colour if terminal
    python tools/theorem_dag.py --chain kinetic_inverse_chain
    python tools/theorem_dag.py --from T2   # all theorems reachable from T2
    python tools/theorem_dag.py --no-colour # plain ASCII

OUTPUT FORMAT (excerpt)
-----------------------
    FLU V12 — Theorem Dependency DAG
    22 PROVEN · 3 CONJECTURE  (sketch+test=12, sketch=10, empirical=3)

    CORE BIJECTION CHAIN
    ─────────────────────────────────────
    T1 ──► T2 ──► T3
    (bijection)  (hamiltonian) (latin)

    T2 is also the root of:
      kinetic_inverse_chain : T2 ──► KIB ──► BPT ──► SRM
      algebraic_chain       : T2 ──► CGW ──► T7  ──► SRM

    (SRM is the convergence point of both kinetic and algebraic paths)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REGISTRY_JSON = (
    Path(__file__).parent.parent / "src" / "flu" / "theory" / "THEOREM_REGISTRY.json"
)

# ANSI colour codes (disabled when not a terminal or --no-colour)
_C = {
    "proven_sc": "\033[92m",   # green  — sketch+computational
    "proven_s":  "\033[96m",   # cyan   — sketch only
    "conjecture":"\033[93m",   # yellow — empirical conjecture
    "header":    "\033[1m",    # bold
    "dim":       "\033[2m",    # dim
    "reset":     "\033[0m",
}


def _colour(text: str, code: str, use_colour: bool) -> str:
    if not use_colour:
        return text
    return f"{code}{text}{_C['reset']}"


def _tier_colour(proof_status: str, use_colour: bool) -> str:
    if proof_status == "sketch_and_computational":
        return _C["proven_sc"]
    if proof_status == "algebraic_sketch":
        return _C["proven_s"]
    return _C["conjecture"]


def _node_label(tid: str, theorems: dict, short: bool = True) -> str:
    """Return display label for a node."""
    t = theorems.get(tid, {})
    name = t.get("name", tid)
    # Strip the ID prefix from the name if present (e.g. "T1 -- n-ary..." → "n-ary...")
    if " -- " in name:
        short_name = name.split(" -- ", 1)[1][:25]
    else:
        short_name = name[:25]
    if short:
        return tid
    return f"{tid} ({short_name})"


def render_chain(chain_name: str, nodes: list[str], theorems: dict, use_colour: bool) -> list[str]:
    """Render a single dependency chain as an ASCII row with annotations."""
    lines = []
    # Header
    header = chain_name.replace("_", " ").upper()
    lines.append(_colour(header, _C["header"], use_colour))
    lines.append("─" * 50)

    # Main arrow row
    arrow_parts = []
    for tid in nodes:
        t = theorems.get(tid, {})
        col = _tier_colour(t.get("proof_status", "algebraic_sketch"), use_colour)
        arrow_parts.append(_colour(tid, col, use_colour))
    lines.append(" ──► ".join(arrow_parts))

    # Annotation row (one-word descriptor per node)
    _DESCRIPTORS = {
        "T1": "bijection",     "T2": "hamiltonian", "T3": "latin",
        "T4": "step-bound",    "T5": "siamese",     "T6": "fractal",
        "T7": "group-sum",     "SRM": "self-ref",   "KIB": "kinetic-inv",
        "BPT": "partition",    "CGW": "cayley",     "C4": "cycle-close",
        "L1": "line-sum=0",    "L2": "repair",      "L3": "byz-fault",
        "S1": "dc-zero",       "S2-Prime": "disp-bound", "S2": "flatness?",
        "PFNT-1": "container", "PFNT-2": "mean-0",  "PFNT-3": "hyperprism",
        "PFNT-4": "lehmer",    "PFNT-5": "communion", "C2": "axial?", "C3": "tensor?",
    }
    desc_parts = []
    for tid in nodes:
        desc = _DESCRIPTORS.get(tid, "")
        padding = max(0, len(tid) + 4 - len(desc) - 1)
        desc_parts.append(desc + " " * padding)
    lines.append(_colour("(" + " ".join(desc_parts) + ")", _C["dim"], use_colour))
    lines.append("")
    return lines


def render_dag(reg: dict, filter_chain: str | None, from_node: str | None,
               use_colour: bool) -> str:
    """Render the full theorem dependency DAG."""
    theorems = reg["theorems"]
    chains   = reg["dependency_chains"]
    summary  = reg["summary"]
    lines    = []

    # ── Title ────────────────────────────────────────────────────────────────
    lines.append(_colour("FLU V12 — Theorem Dependency DAG", _C["header"], use_colour))
    from collections import Counter
    tier_counts = Counter(t.get("proof_status", "?") for t in theorems.values()
                          if theorems.get(list(theorems.keys())[0], {}).get("status") != "CONJECTURE")
    lines.append(
        f"{summary['score']}  "
        f"(sketch+test={sum(1 for t in theorems.values() if t.get('proof_status')=='sketch_and_computational')}, "
        f"sketch={sum(1 for t in theorems.values() if t.get('proof_status')=='algebraic_sketch')}, "
        f"empirical={sum(1 for t in theorems.values() if t.get('proof_status')=='empirical')})"
    )

    # Colour legend
    if use_colour:
        lines.append(
            f"  {_colour('■', _C['proven_sc'], use_colour)} sketch+test  "
            f"{_colour('■', _C['proven_s'],  use_colour)} sketch  "
            f"{_colour('■', _C['conjecture'],use_colour)} conjecture"
        )

    lines.append("")

    # ── Filter: single chain ──────────────────────────────────────────────────
    if filter_chain:
        if filter_chain not in chains:
            return f"Unknown chain '{filter_chain}'. Available: {list(chains.keys())}"
        chain_nodes_map = {filter_chain: chains[filter_chain]}
    elif from_node:
        # Build reachability from the given node using depended_on_by
        visited = set()
        queue = [from_node]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(theorems.get(node, {}).get("depended_on_by", []))
        # Find all chains that contain any visited node
        chain_nodes_map = {
            k: v for k, v in chains.items()
            if any(n in visited for n in v)
        }
        lines.append(f"Theorems reachable from {from_node}: {sorted(visited)}")
        lines.append("")
    else:
        chain_nodes_map = chains

    # ── Render each chain ────────────────────────────────────────────────────
    for chain_name, nodes in chain_nodes_map.items():
        lines.extend(render_chain(chain_name, nodes, theorems, use_colour))

    # ── Cross-chain joints ───────────────────────────────────────────────────
    if not filter_chain and not from_node:
        # Find nodes that appear in >1 chain
        node_to_chains: dict[str, list[str]] = defaultdict(list)
        for cname, cnodes in chains.items():
            for n in cnodes:
                node_to_chains[n].append(cname)
        joints = {n: cs for n, cs in node_to_chains.items() if len(cs) > 1}

        if joints:
            lines.append(_colour("CROSS-CHAIN JOINTS", _C["header"], use_colour))
            lines.append("─" * 50)
            lines.append("These nodes appear in more than one chain,")
            lines.append("making them critical load-bearing theorems.")
            lines.append("")
            for node, cs in sorted(joints.items()):
                t = theorems.get(node, {})
                col = _tier_colour(t.get("proof_status", "?"), use_colour)
                node_str = _colour(node, col, use_colour)
                stmt = (t.get("statement") or "")[:60]
                lines.append(f"  {node_str}  ←  bridges {' + '.join(cs)}")
                lines.append(_colour(f"     \"{stmt}...\"", _C["dim"], use_colour))
            lines.append("")

    # ── Full graph (compact) ─────────────────────────────────────────────────
    if not filter_chain and not from_node:
        lines.append(_colour("FULL DEPENDENCY GRAPH (compact)", _C["header"], use_colour))
        lines.append("─" * 50)
        lines.append("Each node: [ID / proof_tier / status]")
        lines.append("")

        tier_sym = {
            "sketch_and_computational": "●",  # filled
            "algebraic_sketch":         "○",  # open
            "empirical":                "◌",  # dashed — conjecture
        }

        for chain_name, nodes in chains.items():
            parts = []
            for n in nodes:
                t = theorems.get(n, {})
                sym = tier_sym.get(t.get("proof_status", "?"), "?")
                col = _tier_colour(t.get("proof_status", "?"), use_colour)
                parts.append(_colour(f"{sym}{n}", col, use_colour))
            lines.append("  " + " → ".join(parts))
        lines.append("")
        lines.append(_colour("  Legend: ● sketch+test  ○ sketch  ◌ conjecture", _C["dim"], use_colour))

    return "\n".join(lines)


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description="FLU theorem dependency DAG renderer")
    ap.add_argument("--chain",     help="Show only a specific chain (e.g. kinetic_inverse_chain)")
    ap.add_argument("--from",     dest="from_node", help="Show all theorems reachable from this node (e.g. T2)")
    ap.add_argument("--no-colour", action="store_true", help="Plain ASCII without ANSI colour")
    ap.add_argument("--registry",  type=Path, default=_REGISTRY_JSON,
                    help="Path to THEOREM_REGISTRY.json")
    args = ap.parse_args(argv)

    if not args.registry.exists():
        print(f"ERROR: registry not found at {args.registry}")
        print("Run: python tools/generate_registry_json.py")
        sys.exit(1)

    with open(args.registry) as f:
        reg = json.load(f)

    use_colour = not args.no_colour and sys.stdout.isatty()
    print(render_dag(reg, args.chain, args.from_node, use_colour))


if __name__ == "__main__":
    main()
