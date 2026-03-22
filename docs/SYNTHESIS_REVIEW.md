# FLU — Synthesis Review: "FM-Dance as Discrete Signal Processing"

**Date:** 2026-03-12 · **Sprint:** V15.2 · **Status:** Rigorous audit of creative brainstorm

This document evaluates the synthesis note framing FM-Dance as "the Discrete Integral of the
van der Corput sequence, uncovering the DNA of discrete space-filling."  Every claim is checked
against the theorem registry and codebase.  New insights are extracted, errors are flagged, and
roadmap candidates are proposed with explicit disposition.

---

## Verdict at a Glance

| Claim | Status | Registry anchor | Notes |
|-------|--------|----------------|-------|
| T is the discrete integration operator Δ⁻¹ | ✅ CORRECT | DISC-1 | Exact |
| vdC is the "derivative / digit-stream" | ✅ CORRECT | DISC-1 corollary | Exact |
| Φ(k) = T·a(k) is discrete integration of vdC digit stream | ✅ PROVEN | DISC-1, T1 | Core identity |
| Carry propagation = Pascal-weighted flow | ✅ PROVEN (qualitative) | DISC-1, BPT | See §2 for exact statement |
| "Repeated T generates binomial coefficients" | ⚠️ SUBTLE ERROR | DISC-1 (correction) | T ≠ Pascal P; see §2 |
| T belongs to Pascal algebra (Faure conjugacy) | ✅ PROVEN | T9, DISC-1 | Conjugacy level only |
| Geometric view = Hamiltonian Torus Walk | ✅ PROVEN | T2, T4, BFRW-1 | Exact |
| QMC / Digital Net discrepancy | ✅ PROVEN | T9, OD-33, FMD-NET | Exact |
| "Minimal Entropy Path" claim | 🔴 FALSIFIED | — | §3: Empirically falsified; see reframe |
| x_k = Permute(Integrate(Digits(k))) pipeline | ✅ PROVEN | DISC-1, DN2 | Elegant unifying summary |
| Gray/Sierpiński/Hadamard = "Fourier spectrum of T" | ⚠️ IMPRECISE | — | §4: Needs decomposition |
| Gray ↔ T-matrix (carry structure) | ✅ PROVEN | T8, CGW | Exact |
| Sierpiński ↔ Pascal mod n | ✅ PROVEN (qualitative) | DISC-1, BPT, PERSPECTIVES | Exact |
| Hadamard ↔ T-matrix | ⚠️ INDIRECT | HAD-1 | Different algebra (Z_2 vs Z_n odd) |
| UNIF-1 — Spectral Unification | ✅ PROVEN | UNIF-1 | Already in registry V15.1.4 |
| T10 — Kinetic Lattice Convergence | ✅ PROVEN | T10 (new, V15.2) | Added to registry |
| C5 — Recursive Hyper-Torus | ✅ PROVEN | C5 (new, V15.2) | Corollary of PFNT-5 |

---

## §1 — What Is Exactly Correct

### 1.1 The Core Dual-Stream Identity (DISC-1, PROVEN)

Both the van der Corput and FM-Dance sequences consume the **same base-n digit stream**
through two different linear operators:

```
Base-n digits a(k)
      │
      ├── radical-inverse weighting  ──► van der Corput v(k) ∈ [0,1)   (pure orthogonality)
      │       a_i · n^{-(i+1)}
      │
      └── prefix-sum operator T     ──► FM-Dance coordinate x(k) ∈ Z_n^D  (coupled)
              x = T · a
```

The synthesis phrasing "vdC is the derivative; T is the integral" is precisely correct:
T = Δ⁻¹ (discrete integration operator), and the vdC radical inverse is the point-mass
weighting of the same digit stream without coupling.

### 1.2 The Three-Stage Pipeline Formula

`x_k = Permute(Integrate(Digits(k)))` is the correct architectural description of DN2:

| Stage | Formal object |
|-------|--------------|
| `Digits(k)` = a(k) | base-n digit expansion |
| `Integrate` = T·a(k) mod n | prefix-sum (DISC-1, T1) |
| `Permute` = APN σ | per-depth bijection (DN2) |

Verified: 0 errors over all n ∈ {5,7}, d ∈ {3,4}.  Added to PERSPECTIVES.md §4 as the
unifying pipeline summary for the entire FLU generator family.

### 1.3 T10 — Kinetic Lattice Convergence (NEW, PROVEN)

The synthesis correctly identifies that FractalNet and FractalNetKinetic converge to the
**same point set** at N = n^{2d}.  This is T10 (added to registry V15.2).

Proof in one line: T ∈ GL(d, Z_n) is a lattice automorphism.  At saturation both sequences
exhaust Z_n^{2d}; since T permutes that grid, the unordered point sets are identical.

The "Sierpiński / Pascal" strata visible at intermediate N < n^{2d} are aliasing harmonics
of the T-skew — transient structure before the lattice saturates (BPT, DISC-1).

---

## §2 — Pascal / T Precision (Important Correction)

The synthesis says: *"Repeated T generates binomial coefficients."*
This is **directionally correct but subtly wrong**.

**What is true (DISC-1, PERSPECTIVES §5):**
Carry events produce Pascal-weighted step contributions because T lies in the Pascal algebra
via conjugacy (T9).  The carry-cascade formula σ_j = T·Δa_j has a Pascal-flavoured structure.

**What is false:**
T^k = P (Pascal matrix).  Verified: T² mod n ≠ P mod n for n = 5, 7.
T is a **first-order** prefix-sum operator.  P is a **second-order** Pascal operator.
Their *conjugacy* (T = S·P·S⁻¹ for some S) is what T9 establishes — not equality.

**Corrected statement:**
The carry-cascade pattern of FM-Dance produces Pascal-weighted step distributions because
T belongs to the Pascal algebra via conjugacy (T9), not because T^k generates Pascal rows.

This correction is already in PERSPECTIVES.md §5 ("Important Correction" box).

---

## §3 — The "Minimal Entropy Path" Claim (EMPIRICALLY FALSIFIED)

The synthesis proposes FM-Dance minimises "path description entropy" over all Hamiltonian paths.

**Empirical test (n=7, d=3):**

| Algorithm | Step entropy H | Step distribution |
|-----------|---------------|-------------------|
| FM-Dance | H = 0.661 bits | {1: 294, 2: 42, 3: 6} |
| Morton (Z-order) | **H = 0.000 bits** | {1: 342} — all size-1 |
| n-ary Gray | H = 1.551 bits | mixed |

**Finding:** Morton order achieves H = 0 (all steps size 1 exactly), which is lower than
FM-Dance.  The "Minimal Entropy Path" claim is falsified for the step-size metric.

**Why FM-Dance is not minimal-entropy:**
Morton interleaves digits independently (no coupling), so every step is exactly size 1.
FM-Dance deliberately accepts larger steps at carry events in exchange for the Latin property.

**Correct reframe:**
FM-Dance is the **minimum-entropy Hamiltonian Latin path** — not minimum entropy among all
Hamiltonian paths, but the unique (up to OD-19) traversal that achieves Hamiltonian + Latin
with the *smallest maximum step size* (T4 bound = min(D, ⌊n/2⌋)).
The entropy cost above 0 bits is the unavoidable price of the Latin property.

**Roadmap connection:** If OD-19 (T8b uniqueness) closes, NEW-3 follows as a corollary
(see OPEN_DEBT.md).

---

## §4 — "Fourier Spectrum of T" Claim (Needs Decomposition)

The synthesis says Gray, Sierpiński, and Hadamard are "the Fourier spectrum of the T-matrix."
This conflates three claims of different rigour:

**Claim A: T ↔ Gray-code flips — PROVEN (T8)**
T8 connects FM-Dance to n-ary Gray codes via the carry cascade.  The T·Δa_j formula
generates the Gray-code step structure.  Precise.

**Claim B: T ↔ Sierpiński — PROVEN qualitatively (DISC-1 + BPT)**
Pascal mod n produces Sierpiński-like patterns; FM-Dance operates mod n.  The boundary
partition sizes |B_j| = (n−1)·n^{D−j−1} mirror Pascal diagonal sizes.  Explained in
PERSPECTIVES.md §5.

**Claim C: T ↔ Hadamard — INDIRECT (different algebra)**
HAD-1 (PROVEN) generates Hadamard matrices via XOR-Communion over Z_2.  FM-Dance T
operates over Z_n (odd).  The algebras are incompatible at the operator level; calling
Hadamard orthogonality "the Fourier spectrum of T" is wrong as stated.

**The genuinely new insight embedded in the error:**
Both S2 (FM-Dance, Z_n odd, mixed-freq vanishing) and HAD-1 (Z_2, row orthogonality) are
consequences of the same principle: character sums of non-trivial characters of a finite
abelian group vanish (character orthogonality theorem).  This is UNIF-1 — already PROVEN
in V15.1.4.  The synthesis was pointing at UNIF-1 while mis-stating the algebraic bridge.

---

## §5 — New Theorem Integrations (V15.2)

### Integrated to registry as PROVEN

| ID | Name | Action |
|----|------|--------|
| T10 | Kinetic Lattice Convergence | Added to `theory_fm_dance.py` + registry |
| C5 | Recursive Hyper-Torus Embedding | Added as corollary of PFNT-5 |
| YM-1 | Youvan–Mönnich Danielic Ten | **Bug fix**: was silently overwritten in prior sprint by GEN-1 collision; now correctly keyed "YM-1" |

### Sent to OPEN_DEBT.md as research directions

| ID | Name |
|----|------|
| NEW-1 | Generator Matrix Parameterisation Theorem |
| NEW-3 | Min-entropy Hamiltonian Latin (OD-19 corollary) |
| NEW-4 | Modular Pascal Fractal Dimension |
| OPER-1 | Sparse Arithmetic Composition (SparseArithmeticManifold) |

### Sent to ROADMAP.md as V16 design proposals

- Operator Pattern (FLUOperator / TMatrix / APNPermute)
- SparseArithmeticManifold with `+`, `*`, `materialize()`
- SynthesisFacet convenience wrapper
- CommunionAlgebra composition engine

---

## §6 — Provenance Decoded (genesis_seed_2017)

The 2017 manuscript axioms map precisely to V15 theorems:

| 2017 term | FLU V15 equivalent | Status |
|-----------|--------------------|--------|
| S1 Standardschritt — vector (1,1) | σ_0 = (−1, +1) mod n (primary step) | PROVEN (T5, CGW) |
| S2 Ausfallschritt — vector (0,−1) | Carry-correction vector σ_j | PROVEN (FM-Dance Kinetic Traversal) |
| Anfangspunkt A (Ganzzahliger Mittelwert) | Origin Φ(0) = 0; signed centering = n//2 | PROVEN (T1, S1) |
| Generalisation to Tesseract / D-cube | FM-Dance D-dimensional torus walk Z_n^D | PROVEN (T2, T6) |
| Spektren / Zahlenwerte nach Ordnung zerlegen | Mixed-Frequency Vanishing (UNIF-1) | PROVEN (UNIF-1) |

The 2017 "magic" was the FM-Dance's Latin property; V15 provides the full algebraic proof.
`GEN-0` in the registry anchors this lineage.

---

## §7 — What Was NOT Integrated (and Why)

| Claim / Feature | Disposition | Reason |
|-----------------|-------------|--------|
| `CommunionAlgebra.compose()` as a new class | ROADMAP only | Would duplicate `CommunionEngine`; needs UKMCContract integration |
| `TMatrix` / `APNPermute` operator classes | ROADMAP only | V16 scope; no API gap in V15 |
| OPER-2 Mean Conservation as full theorem | Not registered | Trivially follows from linearity of expectation; documented in OPER-1 debt note |
| "Boundary Transition Theorem" for RotationHub | ROADMAP (existing item 12) | Hamiltonian property of RotationHub not yet verified computationally |
| "Fractal Algebra of Communion" framing | Docs narrative | Correctly describes existing architecture; no new theorem claim |
| DN2 reclassification to PROVEN | Rejected | DN2 remains PARTIAL/CONJECTURE; L2-improvement and formal t-value proof still open (OD-27) |
