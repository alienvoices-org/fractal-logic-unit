# FLU — Architecture Reference

**Version:** 15.1.4

This document describes the internal structure of the FLU library: its layers,
modules, their responsibilities, and the dependency rules that keep the codebase
clean.

---

## Layer diagram
┌─────────────────────────────────────────────────────────────────┐
│  examples/          User-facing demos and application scripts   │
├─────────────────────────────────────────────────────────────────┤
│  src/flu/applications/   Domain application adapters            │
│    codes.py · design.py · lighthouse.py · neural.py · quantum.py│
├─────────────────────────────────────────────────────────────────┤
│  src/flu/interfaces/     Bridge facets to classical structures  │
│    base.py · lexicon.py · integrity.py · genetic.py · invariance.py│
│    hilbert.py · cohomology.py · gray_code.py · crypto.py        │
│    hadamard.py · digital_net.py                                 │
├─────────────────────────────────────────────────────────────────┤
│  src/flu/container/      Container algebra & sparse structures  │
│    communion.py · contract.py · manifold.py · sparse.py         │
│    export.py                                                    │
├────────────────────────────────┬────────────────────────────────┤
│  src/flu/theory/               │  src/flu/utils/                │
│    theorem_registry.py         │    benchmarks.py               │
│    theory_fm_dance.py          │    verification.py             │
│    theory_latin.py             │    math_helpers.py             │
│    theory_spectral.py          │    viz.py                      │
│    theory_container.py         │                                │
│    theory_communion_algebra.py │                                │
├────────────────────────────────┴────────────────────────────────┤
│  src/flu/core/               Core primitives (foundational)     │
│    factoradic.py · fm_dance.py · fm_dance_path.py               │
│    hypercell.py · lo_shu.py · fractal_3_6.py                    │
│    even_n.py · n_ary.py · parity_switcher.py                    │
│    vhdl_gen.py · fractal_net.py                                 │
├─────────────────────────────────────────────────────────────────┤
│  src/flu/constants.py        Global constants (no imports)      │
└─────────────────────────────────────────────────────────────────┘

**Dependency rule:** layers may only import from the same layer or from layers
*below* them. Upward imports are forbidden. The `interfaces` layer sits above
`core` and `theory`, but may also import from `applications` only if strictly
necessary (rare). `utils` is a utility layer that can be imported by any layer,
but must not import from higher layers (circular imports are avoided via lazy
imports in benchmarks).

---

## Module reference

### `src/flu/constants.py`
Global mathematical constants. No imports from flu. Imported by everything.

### `src/flu/core/`

| Module | Responsibility |
|--------|---------------|
| `factoradic.py` | Lehmer rank ↔ permutation bijection; APN Seed Hub (`GOLDEN_SEEDS`, `differential_uniformity`, `unrank_optimal_seed`); vectorised APN search |
| `fm_dance.py` | Addressing bijection `index_to_coords`, `coords_to_index`; `generate_fast` for full array |
| `fm_dance_path.py` | Kinetic traversal: `path_coord`, `path_coord_to_rank`, step vectors, Cayley generators, `FMDanceIterator` (OD-32), kinetic inverse |
| `hypercell.py` | `FLUHyperCell` – 3⁴ cell with UKMC contract and sparse manifold bridge |
| `lo_shu.py` | Lo Shu magic square, 72-perspective automorphisms, `LoShuHyperCell` |
| `fractal_3_6.py` | Recursive 3⁶ embedding (macro FLUHyperCell × micro Lo Shu) |
| `even_n.py` | Even‑n Latin hyperprism generation via n = 2ᵏ·m decomposition |
| `n_ary.py` | N‑ary generalisation of FM‑Dance; alignment principle; `nary_generate`, `nary_info` |
| `parity_switcher.py` | Unified factory `generate(n,d)` – dispatches to FM‑Dance (odd) or even‑n (even) |
| `vhdl_gen.py` | Synthesisable VHDL-93 RTL export of FM‑Dance odometer cascade |
| `fractal_net.py` | Quasi‑Monte Carlo digital nets: `FractalNet` (identity generator), `FractalNetKinetic` (T‑matrix generator), APN scrambling (DN2) |

### `src/flu/container/`

| Module | Responsibility |
|--------|---------------|
| `communion.py` | `CommunionEngine`: algebraic fusion (outer/direct/kronecker) of hyperprisms |
| `contract.py` | `UKMCContract`: immutable identity contract with freeze mechanism |
| `manifold.py` | Sparse manifold seam: `cell_to_sparse_coords`, `sparse_coords_to_norm0`, `verify_seam` |
| `sparse.py` | `SparseCommunionManifold` (O(D) memory oracle) and `ScarStore` (holographic sparse memory, HM‑1) |
| `export.py` | Zero‑copy buffer export to NumPy, PyTorch, JAX; `fill_weight_matrix` |

### `src/flu/theory/`

| Module | Responsibility |
|--------|---------------|
| `theorem_registry.py` | Central registry of all theorems (59 entries); `status_report()`, query helpers |
| `theory_fm_dance.py` | Formal theorem records for T1–T9, kinetic theorems, open conjectures (DN1, DN2, OD‑16, OD‑17) |
| `theory_latin.py` | L1–L3 theorems: constant line sum, holographic repair, Byzantine tolerance |
| `theory_spectral.py` | S1, S2, S2‑Prime theorems; `compute_spectral_profile`, `spectral_dispersion_bound` |
| `theory_container.py` | Permutation lattice algebra; generator roles, communion as tensor product |
| `theory_communion_algebra.py` | Algebraic investigation of different φ (add, max, lex) |
| `theory.py` | Phased Fractal Number Theory axioms (PFNT‑1…5) |

### `src/flu/interfaces/`

| Module | Responsibility |
|--------|---------------|
| `base.py` | `FluFacet` abstract base class |
| `lexicon.py` | `LexiconFacet` – bijective n‑ary alphanumeric encoding (LEX‑1) |
| `integrity.py` | `IntegrityFacet` – O(1) L1 conservative‑law sonde (INT‑1) |
| `genetic.py` | `GeneticFacet` – SHA‑256 verified APN seed reservoir (GEN‑1) |
| `invariance.py` | `InvarianceFacet` – cross‑branch structural isomorphism (INV‑1) |
| `hilbert.py` | `HilbertFacet` – FM‑Dance + RotationHub (HIL‑1, RETIRED) |
| `cohomology.py` | `CohomologyFacet` – discrete exterior calculus (DEC‑1 PROVEN) |
| `gray_code.py` | `GrayCodeFacet` – FM‑Dance as n‑ary Gray code (T8 PROVEN) |
| `crypto.py` | `CryptoFacet` – APN prime‑field structural immunity (CRYPTO‑1) |
| `hadamard.py` | `HadamardFacet` – Sylvester‑Hadamard via bit‑masked communion (HAD‑1) |
| `digital_net.py` | `FractalNetCorputFacet`, `FractalNetKineticFacet` – QMC facets (FMD‑NET, T9) |

### `src/flu/utils/`

| Module | Responsibility |
|--------|---------------|
| `benchmarks.py` | Benchmark suite: addressing, traversal, spectral variance, avalanche; `full_benchmark_report`, `spectral_probe_large_n` |
| `verification.py` | Cross‑cutting invariant checks: `check_latin`, `check_coverage`, `check_mean_centered`, `check_round_trip` |
| `math_helpers.py` | `factorial`, `is_odd`, `digits_signed`, `digits_unsigned`, `mean_of_digits`, `inv_mod` |
| `viz.py` | Optional matplotlib visualisations: `plot_hyperprism_2d`, `plot_lo_shu_grid`, `plot_bijection_path` |

### `src/flu/applications/`

| Module | Domain |
|--------|--------|
| `codes.py` | Latin square error‑correcting codes (DESIGN INTENT) |
| `design.py` | Latin hypercube experimental design (`ExperimentalDesign`) |
| `lighthouse.py` | Cryptographic beacon demo (SIMULATION ONLY) |
| `neural.py` | Bias‑free neural weight initialisation (`FLUInitializer`, `DynamicFLUNetwork`) |
| `quantum.py` | Tensor network state simulator (SIMULATION ONLY) |

---

## Key data types

| Type | Description |
|------|-------------|
| `D_set` | Balanced digit set `{-⌊n/2⌋, …, ⌊n/2⌋}` for odd n (unsigned `{0,…,n-1}` for even/unsigned) |
| `Arrow` | `np.ndarray` shape `(n,)` — a permutation of D_set |
| `HyperPrism` | `np.ndarray` shape `(n,n,…,n)` — D‑dimensional value array |
| `CommunionArray` | HyperPrism where `M[i] = Σ_j π_j(i_j)` (sum‑separable) |
| `ValueHyperPrism` | HyperPrism satisfying L1 (constant line sum) – e.g., shift‑sum construction |
| `SparseCommunionManifold` | Lazy oracle for n^D manifold, O(D) memory, O(D) evaluation |
| `ScarStore` | Holographic sparse memory: baseline + scars (HM‑1) |
| `FractalNet` | Digital net generator (identity matrix, control group) |
| `FractalNetKinetic` | Digital net generator with prefix‑sum matrix T (T9 PROVEN) |
| `FMDanceIterator` | O(1) amortised incremental traversal (OD‑32) |

**Important distinction:** `CommunionArray` and `ValueHyperPrism` are NOT the same.
A `ValueHyperPrism` satisfies L1 by construction (e.g., shift‑sum); a `CommunionArray`
generally does not. Theorems L2 and L3 apply only to `ValueHyperPrism`. See
`docs/THEOREMS.md` for details.

---

## Dependency graph (simplified)

applications → interfaces → container → theory ← core ← constants
                               ↑           ↑
                             utils        utils

`utils/benchmarks.py` imports lazily from `core` and `theory` to avoid circular
imports. `interfaces` may import from `core`, `theory`, and sometimes `applications`
(e.g., `hadamard.py` imports `HadamardGenerator` from `applications`). This is
acceptable as `applications` is above `interfaces` in the diagram, but we treat
`interfaces` as a bridge that can import from both `applications` and lower layers
without causing upward dependency because `applications` does not import `interfaces`
(internally). External users import from `interfaces` directly.

---

## Test structure

tests/
├── run_all.py                  Unified runner (no pytest required)
├── benchmarks/                 QMC / APN / discrepancy benchmarks
│   ├── bench_addressing.py
│   ├── bench_apn_hub.py
│   ├── bench_comparison.py
│   ├── bench_discrepancy.py
│   ├── bench_fusion.py
│   ├── bench_qmc_rigor.py
│   ├── bench_traversal.py
│   └── run_full_audit.py
├── test_applications/
│   ├── test_applications.py
│   └── test_hadamard.py
├── test_container/
│   ├── test_communion.py
│   ├── test_contract.py
│   ├── test_manifold.py
│   ├── test_scarstore.py
│   └── test_sparse_export.py
├── test_core/
│   ├── test_apn_seeds.py
│   ├── test_even_n.py
│   ├── test_factoradic.py
│   ├── test_fm_dance.py
│   ├── test_fm_dance_properties.py
│   ├── test_fmd_net.py
│   ├── test_fractal_3_6.py
│   ├── test_fractal_net.py
│   ├── test_hypothesis.py
│   ├── test_hypothesis_deterministic.py
│   ├── test_lo_shu_hypercell.py
│   ├── test_math_helpers.py
│   ├── test_nary_vhdl.py
│   ├── test_od32_iterator.py
│   ├── test_parity_neural.py
│   └── test_traversal.py
├── test_interfaces/
│   └── test_interfaces.py
├── test_theory/
│   ├── test_proofs.py
│   ├── test_registry.py
│   └── test_theory.py
└── test_utils/
    ├── test_verification.py
    └── test_viz.py

**Rule:** every new module must have a corresponding test file. Every new public
function must have at least one test. Theorem status changes must update both
`theorem_registry.py` and `docs/THEOREMS.md`, with a new test verifying the status.

---

## Naming conventions

| Convention | Example |
|------------|---------|
| Theorem IDs | `T1`, `T2`, `PFNT-1`, `L1`, `S2`, `C3`, `C3W`, `HAD-1`, `DEC-1` |
| Theorem status strings | `"PROVEN"`, `"CONJECTURE"`, `"DISPROVEN_SCOPED"`, `"PARTIAL"`, `"RETIRED"` |
| Test labels | `"V15 T9 three‑way comparison"` |
| Open debt IDs | `OD-1` through `OD-N` (sequential, never reuse) |
| Class names | `PascalCase` (e.g., `FLUHyperCell`, `SparseCommunionManifold`, `FractalNetKinetic`) |
| Functions | `snake_case` (e.g., `factoradic_unrank`, `path_coord`, `verify_seam`) |
| Constants | `UPPER_SNAKE` (e.g., `GOLDEN_SEEDS`, `D_SET`, `SCAR_TYPES`) |
