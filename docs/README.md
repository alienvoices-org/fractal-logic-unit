# FLU Documentation

# FLU — Phased Fractal Number Theory / Universal Fractal Logic Unit

**Version:** 15.2.0 · **License:** MIT · **Python:** 3.10+  
**Authors:** Felix Mönnich & The Kinship Mesh Collective

---

## What is FLU?

FLU is a self-contained, mathematically rigorous library for **n-ary Latin hyperstructures** — deterministic, bijective, balanced combinatorial objects over the torus ℤₙᴰ.

The core primitive is the **FM-Dance bijection**: an O(D) prefix-sum transform that maps any integer rank to a unique coordinate in ℤₙᴰ such that every axis-aligned slice is a permutation of the digit set. This construction provides a deterministic substrate for quasi-Monte Carlo (QMC) sequences, sparse memory, and bias-free neural weight initialisation.

FLU is **formally verified** via a self-contained test suite and an integrated theorem registry. Every major claim is either **PROVEN** (via algebraic proof and computational verification) or explicitly labeled as **CONJECTURE**.

Proven Bedrock: FM-Dance (T1–T6), L1–L3 (Latin Hypercube/Repair), S1/S2 (Spectral), HAD-1 (Hadamard), DISC-1 (Integral), UNIF-1 (Unification).

Boxed Core: The library is modularised into flu.core (immutable primitives), flu.container (algebraic structures), and flu.interfaces (bridge facets).

| File | Contents |
|------|----------|
| [THEOREMS.md](THEOREMS.md) | Complete formal registry — all theorems and conjectures with proof sketches, status history, and computational verification references |
| [PERSPECTIVES.md](PERSPECTIVES.md) | Three-perspective synthesis — Matrix, Kinetic, and Group-Algebraic views of the FM-Dance, how they cross-link|
| [BENCHMARKS.md](BENCHMARKS.md) | Live benchmark results — methodology, timing tables, and interpretation for all four benchmark categories |
| [OPEN_DEBT.md](OPEN_DEBT.md) | Open debt register — every known defect, data-quality gap, and proof obligation with resolution status |
| [ROADMAP.md](ROADMAP.md) | sprint plan — prioritised work items with mathematical arguments and acceptance criteria |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module map, layer diagram, dependency rules, naming conventions |
| [CHANGELOG.md](CHANGELOG.md) | Complete version history — all theorem status changes, bug fixes, and additions |
| [KINSHIP.md](KINSHIP.md) | Complete AI Assistant and Kinship Mesh author reference |
| [PACKAGE_CONTRACT.json](PACKAGE_CONTRACT.json) | Machine Readable SRP contract for FLU packages |

## Design principles

1. **Full transparency** — every open gap, wrong assumption, and pending proof is
   documented here rather than hidden in comments or issue trackers.

2. **Docs and code co-evolve** — a theorem status change in `theorem_registry.py`
   must be accompanied by an update to `THEOREMS.md` and `CHANGELOG.md`. This is
   enforced by code review, not automation (yet).

3. **Negative results are first-class** — disproven conjectures are kept as named
   negative results (e.g., C2 FALSE for communion arrays). They are never silently
   deleted.

4. **Benchmarks are scientific receipts** — not just pass/fail gates. Every
   benchmark produces a structured report that can be inspected and compared across
   versions. They document the actual variance, not the hoped-for ideal.

5. **Formal Provenance** — Every function in src/flu/core references its TheoremID.

6. **Zero-Trust Logic** — PACKAGE_CONTRACT.json enforces SRP boundaries; CI fails on registry/test drift.

7. **Alchemical Distillation** — Code evolves by crystallization (removing debt/noise), not accretion.

---

## Quick Start (Production Core: ~500KB)

```python
import flu
from flu.container.sparse import SparseArithmeticManifold

# 1. Build a Sparse Manifold (O(D) memory oracle)
# M[x] = Σ_a π_a(x_a) mod n
M = flu.manifold(n=3, d=64, sparse=True)

# 2. Lazy Arithmetic (OPER-1)
# Field = (M1 ⊖ M2) ⊗ 0.5 — computed on-demand in O(D·depth)
Field = (M - M) * 0.5 
print(f"Value at coord: {Field[(0, 1, ...)]}")

# 3. Theorem Registry Audit
print(flu.status_report())
Registry Status (V15.2.0)
```

