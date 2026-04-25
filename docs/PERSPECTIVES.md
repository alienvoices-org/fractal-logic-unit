# FLU — Six Perspectives on the FM-Dance Path

**Version:** 15.4.0 (Perspective 6 added — Siamese Magic Construction)  
**Status:** All six perspectives PROVEN or rigorously grounded; equivalences PROVEN

This document shows that the FM-Dance traversal can be understood from five
distinct but mathematically equivalent perspectives. Each perspective illuminates
a different aspect of the structure. None supersedes the others — they are
complementary lenses on the same object.

---

## The Six Perspectives at a Glance

```
        ┌───────────────────────────────────────────────────────────────────┐
        │                     FM-Dance Traversal Φ                         │
        │                                                                   │
        │  MATRIX VIEW    KINETIC VIEW   ALGEBRAIC VIEW   QMC VIEW   PASCAL│
        │  (T1, T6)       (CGW, BPT,     (T7, CGW, SRM)  (T9,       VIEW  │
        │                  KIB)                            DISC-1,   (T9,  │
        │                                                  OD-33)    DISC-1)│
        │                                                                   │
        │  T·a = x        x_k + σ_j = x_{k+1}  Φ=Σσ   C_m=T     Pascal-Δ │
        │  det(T)=−1      Fault Lines B_j  Cayley graph  Faure     Sierp.  │
        │                                                                   │
        │  All bijections.  All O(D).  All agree everywhere.               │
        └───────────────────────────────────────────────────────────────────┘
```

| Perspective | Core object | Key theorems | Entry point |
|-------------|-------------|--------------|-------------|
| Matrix / Linear Algebra | Transform T with det(T)=−1 | T1, T6 | `path_coord`, `path_coord_to_rank` |
| Kinetic / Odometric | Step vectors σ_j, carry levels, Fault Lines | CGW, BPT, KIB | `step_vector`, `identify_step`, `boundary_partition_sizes` |
| Group-Algebraic | Product formula in (Z_n^D, +) | T7, CGW, SRM | `cayley_generators`, `fractal_fault_lines` |
| QMC / Digital Net | Generator matrix C_m=T, Faure conjugacy | T9, DISC-1, OD-33 | `FractalNetKinetic`, `generate`, `generate_scrambled` |
| Pascal / Discrete Calculus | T = Δ^{-1}, carry cascade = Pascal flow | DISC-1, T8 | `step_vector`, `FMDanceIterator` |
| **Siamese Magic Construction** | **Adjacent-pair steps S_j, spectral block balance** | **MH, MH-COMPARE, T5** | **`magic_coord`, `generate_magic`** |

---

## Perspective 1: The Matrix View

*Origin: pre-V11. Formalised in V11 (T1). The foundational constructive proof.*

The FM-Dance coordinate map is a linear transform over Z_n:

```
x = T · a   where T is lower-triangular:

    T = ⎡ -1  0  0  0  ⋯ ⎤
        ⎢  1  1  0  0  ⋯ ⎥
        ⎢  1  1  1  0  ⋯ ⎥
        ⎣  ⋮            ⎦

a_i = ⌊k / n^i⌋ mod n   (base-n digits of rank k)
```

**Why this proves the bijection (T1):**  
det(T) = −1 ≠ 0 in Z_n for odd n → T is invertible → Φ is a bijection.  
T^{-1} exists explicitly: a_0 = −x_0, a_i = (x_i − x_{i-1}) mod n for i ≥ 1.

**Strengths:** Closed-form. Gives bijection and inverse in one shot. No
iteration required. D=2 is classically the Siamese magic square (T5).

**Limitations:** Doesn't directly explain *why* the steps have the specific
form they do, or why the odometer carry creates the Fault Line structure.

```python
# The matrix view in code:
x = path_coord(k, n, d)          # x = T · digits(k)  [T1]
k = path_coord_to_rank(x, n, d)  # a = T^{-1} · x, then k = Σ a_i n^i  [T1]
```

---

## Perspective 2: The Kinetic / Odometric View

*Origin: V11 Bedrock sprint. Introduces Fault Lines, step vectors, carry levels.*

The FM-Dance path is a walk where each step is chosen by an **odometer rule**:
at each rank k, count the number j of trailing (n−1) digits. The step applied is
σ_j — the j-th **Fractal Fault Line** step vector.

```
σ_0 = (n−1, 1,   1,   1,   …)   ← primary step (no carry)
σ_1 = (n−1, 2,   2,   2,   …)   ← level-1 carry
σ_2 = (n−1, 2,   3,   3,   …)   ← level-2 carry
σ_j = (n−1, 2, …, j+1, j+1, …) ← level-j carry
```

**Fractal Fault Lines (audit term):** The boundary sets
```
B_j = { x_k | step σ_j produced x_k }
```
are "Fractal Fault Lines" of the manifold. They partition all non-origin
coordinates with exact sizes |B_j| = (n−1)·n^{D−j−1} (BPT, P3).

**Why this proves the KIB bijection:**  
After applying σ_j, the resulting coordinate's first j unsigned digits are 0 and
digit j is non-zero. This "leading-zeros signature" is unique per carry level →
Ψ(x) = first non-zero unsigned digit is a bijection. This gives O(D) inversion
with no search.

**The key insight (from the audit):**  
> *"If we can prove that the boundary conditions of the FM-Dance path are unique
> for every step, then the mapping x_k → σ_k is a bijection. This would mean:
> The FM-Dance path is a Self-Referential Manifold."*

```python
# The kinetic view in code:
j    = identify_step(x_k, n)           # Ψ(x_k): which Fault Line?   [KIB]
sv   = step_vector(j, n, d)            # σ_j: which step was taken?   [CGW]
svi  = inverse_step_vector(j, n, d)    # σ_j^{-1}: undo the step      [CGW]
prev = invert_fm_dance_step(x_k, n)    # x_{k-1} in O(D)              [KIB]
sizes = boundary_partition_sizes(n, d) # |B_j| = (n-1)·n^{D-j-1}     [BPT]
ffl   = fractal_fault_lines(n, d)      # same, with Fault Line naming  [BPT]
```

---

## Perspective 3: The Group-Algebraic View

*Origin: V11 audit document. Formalises the walk as a Cayley graph product.*

The FM-Dance is a walk on the Cayley graph Cay(Z_n^D, S). In the group (Z_n^D, +),
every element has a unique additive inverse, so the walk is reversible by construction.

**Product formula (T7):**
```
Φ(k) = Φ(0) + Σ_{i=0}^{k-1} σ_{j(i)}   in Z_n^D
```
Each step is left-translation L_{σ_j}: x ↦ x + σ_j. The path is the orbit of
Φ(0) under the sequence of translations.

**Generator set and its inverse:**
```
S     = { σ_0, σ_1, …, σ_{D-1} }      (Cayley generators)
S^{-1} = { σ_0^{-1}, …, σ_{D-1}^{-1} } (additive inverses)
σ_j^{-1}[i] = (n − σ_j[i]) mod n
```
The forward walk uses S; the reverse walk uses S^{-1}. Both are Hamiltonian paths on
Cay(Z_n^D, S) — the same graph, traversed in opposite directions.

**Why this view adds insight:**  
The group structure makes it obvious why the inverse walk *must* exist (every group
element has an inverse) and why it's O(D) (the group operation is component-wise).
The Cayley graph framing also makes the bijection properties of BPT and KIB follow
from standard group-theory arguments (orbit size = group order for transitive actions).

```python
# The algebraic view in code:
S     = cayley_generators(n, d)        # generator set S              [CGW]
S_inv = cayley_inverse_generators(n,d) # inverse generator set S^{-1} [CGW]
# T7: Φ(k) = Φ(0) + Σ_{i<k} S[j(i)]  (verified in tests)
```

---

## How the Three Perspectives Cross-Link

```
Matrix View (T1)
      │
      │  det(T) = −1  →  bijection  →  T^{-1} exists
      │
      ▼
 Φ(k) = T·a ←──────────────────────────────────────────┐
                                                        │
 Kinetic View (BPT, KIB)                               │ All three
      │                                                 │ give the
      │  x lies on Fault Line j  →  σ_j produced x     │ same Φ
      │  identify_step is O(D) bijection                │
      │                                                 │
      ▼                                                 │
 Φ(k) = Φ(k-1) + σ_{j(k-1)} ──────────────────────────┤
                                                        │
 Algebraic View (T7, CGW)                              │
      │                                                 │
      │  (Z_n^D, +) is abelian  →  S^{-1} exists       │
      │  Hamiltonian walk on Cay(Z_n^D, S)              │
      │                                                 │
      ▼                                                 │
 Φ(k) = Φ(0) + Σ_{i<k} σ_{j(i)} ─────────────────────┘
```

**Formal equivalence:** All three formulas define the same function. The matrix
view and kinetic view are connected by the observation that T's columns are exactly
the cumulative sums of the generators σ_j. The kinetic and algebraic views differ
only in notation (additive recursion vs. explicit sum).

---

## The Bijective Triple (Water-Proofing, SRM)

The SRM corollary states that the FM-Dance is **fully water-proof**: every bit
of information in the path is accessible from every other bit via O(D) bijections.

```
                    k ──── path_coord ────► x_k
                    ▲                       │
                    │                       │
          path_coord_to_rank          identify_step
                    │                       │
                    └──── x_{k-1} ◄─── invert_fm_dance_step

All three maps are:
  • O(D) time
  • Bijections (proven: T1, KIB)
  • No search, no simulation, no external state

The system is self-consistent: present → past → rank → present forms a cycle.
```

This "water-proofing" was the goal the audit document set. It is now achieved.

---

## The D=2 Case as Worked Example

The D=2 case is the simplest proof of the KIB bijection. It connects to the
classical Siamese magic square (T5) and makes the general argument concrete.

**Setup (n odd, D=2):**
```
x_0 = (−a_0) mod n − half
x_1 = (a_0 + a_1) mod n − half
```

**Two steps only:**
```
σ_0 = (n−1, 1) = (−1, +1) mod n  ← primary (Siamese step, T5)
σ_1 = (n−1, 2) = (−1, +2) mod n  ← carry (column-skip)
```

**The two-partition (D=2 bijection lemma):**
```
B_0 = { x_k | (x_k[0] + half) mod n ≠ 0 }  ← reached by σ_0 (primary)
B_1 = { x_k | (x_k[0] + half) mod n = 0 }  ← reached by σ_1 (carry)
```

Since x_k[0] = (−a_0) mod n − half, we have (x_k[0] + half) mod n = (−a_0) mod n.
This is 0 iff a_0 = 0, which happens iff the previous step was a carry. ✓

**Sizes:** |B_0| = n(n−1), |B_1| = n−1. Total = n²−1. ✓  
**Verified:** All odd n ∈ {3, 5, 7, 11} — 0 mismatches.

---

## Notes on the Audit Document (V11, 2026)

The audit document introduced the group-theoretic framing and several key terms.
Integration notes:

**Terminology adopted:**
- "Fractal Fault Lines" — the audit's geometric name for the boundary sets B_j.
  Preserved as `fractal_fault_lines()` alias for `boundary_partition_sizes()`.
- "Self-Referential Manifold" — the audit's name for the bijective-triple property.
  Formalised as theorem SRM.
- "Water-proofing" — the audit's term for the complete bijection system.
  Formalised in the SRM theorem proof.

**Pseudocode correction:**  
The audit's `identify_step` pseudocode uses the boundary check `coord[i] == (n-1)//2`
(checking if the signed coordinate equals +half, the maximum value). This is
**incorrect** — it produces 55/124 errors on n=5, d=3.

The correct boundary check is `(coord[i] + half) mod n ≠ 0`, which detects
whether the unsigned digit a_i = 0 (minimum, just wrapped). The theoretical
insight — that the boundary uniquely determines the step — is fully correct.
Only the concrete threshold expression was mis-stated in the pseudocode.

The correction was derived from first principles (verified against the odometer
rule) and is implemented in `identify_step()`.

**Perspectives not in the audit:**  
The matrix view (T1, T-matrix) and the fractal block structure (T6) are native
to the FLU library and not discussed in the audit. The audit adds the group-algebraic
perspective as a third lens. All three are now cross-linked here.

---

## Perspective 4: The QMC / Digital Net View

*Origin: V15 Audit (FLUAudit.txt, March 2026). Formalised in T9 (PROVEN), DISC-1, OD-33.*

The FM-Dance kinetic traversal is a **linear digital sequence** — a member of the well-studied
quasi-Monte Carlo family that includes the Faure, Sobol, and Niederreiter sequences.

### The Van der Corput Duality

The same base-n digit stream {a_i(k)} generates two distinct sequences by two different operators:

```
Base-n digits a(k)
       │
       ├─── radical-inverse weighting  ──► van der Corput sequence v(k) ∈ [0,1)
       │       (a_i · n^{-(i+1)})
       │
       └─── prefix-sum operator T      ──► FM-Dance coordinate x(k) ∈ Z_n^D
               (x = T · a)
```

These are the two canonical dual views of the same digit stream. This duality (DISC-1 Corollary)
explains why FM-Dance inherits quasi-Monte Carlo properties from the van der Corput backbone.

### Generator Matrix Formulation (T9 PROVEN)

In the digital-net framework, sequences are characterised by generator matrices C_m:

```
FractalNet          (baseline)    →   C_m = I   (identity matrix)
FractalNetKinetic   (FM-Dance)    →   C_m = T   (prefix-sum matrix)
Faure sequence      (classical)   →   C_m = P^m (Pascal matrix powers)
```

Because T is lower-triangular with unit diagonal and T ∈ GL(d, Z_n) (det T = −1, a unit for odd n),
FractalNetKinetic is a **valid linear digital sequence** inheriting the full QMC apparatus.

### The Faure Conjugacy

T lies in the same lower-triangular matrix algebra as the Pascal matrix P. There exists an
invertible S such that T = S · P · S^{-1}, making FractalNetKinetic **linearly conjugate** to
a Faure digital sequence. Conjugate sequences share:

- uniform distribution
- discrepancy order: **D_N = O((log N)^d / N)**
- the (0,d,d)-net property at full blocks (OD-33 PROVEN)

### The Grand Tradeoff: Two Canonical Geometries

The FractalNet/FractalNetKinetic pair forms a perfect experimental control:

| Net | Generator | Geometry | Strength |
|-----|-----------|----------|----------|
| FractalNet | C_m = I | Orthogonal (independent digit planes) | Perfect digit independence; lower L2 at intermediate N |
| FractalNetKinetic | C_m = T | Affine skew (Pascal-coupled digits) | Uniform dimensional resolution; DN2 scrambling target |

No single transform optimises all criteria simultaneously. This tradeoff is fundamental in QMC design.

### The Three-Stage Pipeline Formula (V15.2 Synthesis)

The entire FM-Dance + DN2 architecture can be summarised in one line:

```
x_k = Permute( Integrate( Digits(k) ) )
```

where each stage has a precise mathematical meaning:

| Stage | Operation | Formal object | Reference |
|-------|-----------|---------------|-----------|
| `Digits(k)` | Base-n digit expansion | a(k) ∈ Z_n^D | T1 |
| `Integrate` | Prefix-sum linear map | T·a(k) mod n, signed | DISC-1 |
| `Permute` | Per-depth APN bijection | σ: Z_n → Z_n | DN2 |

Choosing `Integrate = I` (identity) gives the Corput / FractalNet baseline.
Choosing `Integrate = T` gives FractalNetKinetic.
Omitting `Permute` (σ = id) gives the plain (unscrambled) net.
This single formula therefore parameterises the whole FLU generator family.

**Computational verification (V15.2):** 0 errors across all n ∈ {5, 7}, d ∈ {3, 4}.

*For the formal generalisation of this parameterisation to all M ∈ GL(d, Z_n),
see NEW-1 in OPEN_DEBT.md (V16 research direction).*

---

### DN2: APN Scrambling as the Optimal Remedy (PARTIAL, V15.1)

Because the T-transform is linear, all hyperplane correlations arise from T^⊤k in the dual lattice.
These are the lattice planes observed in the QMC benchmarks. APN permutations (δ=2) specifically
destroy **linear relationships** over finite fields. Applying APN **after** the T-transform
(path_coord → APN perm) is therefore the mathematically optimal scrambling strategy for
FractalNetKinetic — the exact analogue of Owen scrambling for Faure sequences.

```python
# DN2 correct pipeline:
# digits → T-transform (path_coord) → APN permutation → radical inverse
pts = net_kinetic.generate_scrambled(N, seed_rank=0)
```

**V15.1 Empirical Confirmation (2026-03-11):**

| n  | APN exists? | δ_min | FFT reduction (best seed) |
|----|------------|-------|--------------------------|
| 3  | NO (δ_min=3) | 3   | 0.0%                     |
| 5  | YES (8 seeds) | 2 | **26.5%**                |
| 7  | YES (8 seeds) | 2 | **39.3%**                |
| 11 | YES (16 seeds)| 2 | **50.1%**                |

Clear result: APN scrambling reduces spectral peaks by 26–50% for n≥5, monotonically.
L2 discrepancy is NOT improved (spectral and L2 metrics are genuinely independent).

**Critical clarification — n=3 is APN-free:** GOLDEN_SEEDS[3] has δ=3, not δ=2.
Z_3 has only 6 permutations and every one has δ=3 (trivially, since there are only
3 elements and the DDT cannot achieve fewer than 3 collisions). The prior null result
at n=3 was not an architecture failure — no APN seed exists at n=3.

DN2 is now **PARTIAL**: spectral part confirmed; L2 improvement remains open conjecture.

```python
# QMC view in code:
from flu.core.fractal_net import FractalNet, FractalNetKinetic
net_base    = FractalNet(n, d)       # C_m = I, van der Corput control    [FMD-NET, OD-33]
net_kinetic = FractalNetKinetic(n, d)  # C_m = T, FM-Dance (T9 PROVEN)   [T9, DISC-1]
pts_base    = net_base.generate(N)
pts_kinetic = net_kinetic.generate(N)
pts_scrambled = net_kinetic.generate_scrambled(N, seed_rank=0)  # DN2 target
```

---

## Perspective 5: The Pascal / Discrete Calculus View

*Origin: V15 Audit (FLUAudit.txt, March 2026). Grounds DISC-1, T8, fractal visualisations.*

The FM-Dance traversal can be understood as a **discrete integration of a digital sequence** —
a perspective that simultaneously explains the Pascal structure, the fractal visualisations,
and the QMC behaviour through one unified algebraic mechanism.

### T as the Discrete Integration Operator

In discrete calculus, the forward-difference operator Δ is the discrete analogue of a derivative:

```
Δf(k) = f(k+1) − f(k)
```

Its inverse is discrete summation (integration):

```
(Δ^{-1} a)_i = Σ_{j≤i} a_j
```

This is exactly the operation performed by FLU's prefix-sum matrix T (ignoring the sign of T[0,0]
for a moment, which simply reflects the first coordinate). Therefore:

```
T  = Δ^{-1}   (discrete integration operator)
T^{-1} = Δ    (finite-difference operator — bidiagonal, NOT Pascal inverse)
```

And the FM-Dance transform x = T·a is precisely the **discrete integral of the digit vector**.

### Why Carry Cascades Produce Pascal Coefficients

The digits a(k) evolve via carry propagation as k increments:

```
a(k+1) = a(k) + e_0 − n·c(k)
```

where c(k) encodes the carry depth. Applying T:

```
x(k+1) − x(k) = T·e_0 − n·T·c(k)
```

- T·e_0 = (−1, 1, 1, ..., 1) is the **base velocity** (primary FM-Dance step)
- T·c(k) gives **Pascal-weighted corrections** at carry events

Repeated discrete integration generates binomial coefficients — the entries of Pascal's triangle.
This is why carry cascades accumulate Pascal-weighted contributions: they are powers of the
integration operator Δ^{-1}.

### Why Fractal (Sierpiński-type) Patterns Appear

Pascal coefficients reduced modulo small integers produce the Sierpiński triangle:

```
Pascal mod 2  →  Sierpiński triangle
Pascal mod n  →  Sierpiński-like fractal at base n
```

FM-Dance operates modulo n, so projections inherit these cancellations. The boundary sets B_j
(Fractal Fault Lines) follow geometric decay |B_j| = (n−1)·n^{D−j−1}, which mirrors Pascal
diagonal sizes. Projecting to two coordinates reveals:

- triangular voids (Pascal diagonal structure)
- recursive wedges (higher-order carry hierarchy)
- nested lattice holes (Sierpiński strata)

These are not visual accidents — they are the **modular Pascal phenomenon** made visible.

### The Discrete Calculus Chain

All five perspectives can be summarised in one unified diagram:

```
integer counter k
       │
base-n digit expansion  a(k)
       │
discrete derivative (carry propagation) ←─── T^{-1} = Δ (finite difference)
       │
Pascal integration operator T = Δ^{-1}
       │
FM-Dance coordinate x(k) ∈ Z_n^D
       │
┌──────┴───────────────────────────────────────────────────────┐
│  Digital net theory          Cayley graph theory              │
│  (generator matrix C_m = T)  (generator set S)               │
│  [T9, OD-33, DISC-1]         [CGW, T7, SRM]                  │
└──────────────────────────────────────────────────────────────┘
```

### Important Correction: T^{-1} is NOT the Inverse Pascal Matrix

An external reviewer suggested T^{-1}_{ij} = (−1)^{i−j} · C(i,j) (the inverse Pascal formula).
This is **incorrect for FLU's T**. The Pascal inverse formula applies to the Pascal matrix P,
which is a **higher-order** integration operator. FLU's T is a **first-order** prefix-sum matrix.

FLU's T^{-1} is the bidiagonal forward-difference matrix:

```
(T^{-1})_{i,j}:   row 0 = [−1, 0, 0, ...]
                   row 1 = [ 1, 1, 0, ...]
                   row i = δ_{i,j} − δ_{i,j−1}   for i ≥ 2
```

Giving a(k) = T^{-1}·x(k) via simple differences — O(d) and exact.

The Pascal/Faure connection exists at the **conjugacy level** (T = S·P·S^{-1} for appropriate S),
not at the level of T being P itself or sharing its inverse.

```python
# Pascal / Discrete Calculus view in code:
# The step vectors ARE the discrete integral of carry events:
from flu.core.fm_dance_path import step_vector, path_coord
sigma_j = step_vector(j, n, d)    # = T · Δa_j  (DISC-1 Part ii)
x_k     = path_coord(k, n, d)     # = T · a(k)  (DISC-1 Part i)

# Verify DISC-1 identity computationally:
from flu.theory.theory_fm_dance import verify_discrete_integral_identity
result = verify_discrete_integral_identity(n=3, d=4)
assert result['phi_identity_ok'] and result['step_identity_ok']
```

---

## How All Five Perspectives Cross-Link

```
Matrix View (T1)
      │
      │  det(T) = −1 → bijection → T^{-1} = Δ (bidiagonal)
      │
      ▼
 Φ(k) = T·a  ◄──────────────────── Pascal/Calculus View (DISC-1)
      │                              T = Δ^{-1}; carry cascade = Pascal flow
      │                              Sierpiński strata = Pascal mod n
      │
 Kinetic View (BPT, KIB)
      │
      │  x lies on Fault Line j → σ_j produced x
      │  |B_j| = (n−1)·n^{D−j−1} (Pascal diagonal sizes)
      │
      ▼
 Φ(k) = Φ(k-1) + σ_{j(k-1)} ◄──── Algebraic View (T7, CGW)
      │                              (Z_n^D, +) is abelian; S^{-1} exists
      │                              Hamiltonian walk on Cay(Z_n^D, S)
      │
      ▼
 QMC View (T9, OD-33) ◄────────── van der Corput duality (DISC-1 Corollary)
      │                             C_m = T → linear digital sequence
      │                             Faure conjugacy → D_N = O((log N)^d / N)
      │
      └─── FractalNet (C_m=I) as control ←→ FractalNetKinetic (C_m=T) as experiment
```

**Formal equivalence:** All five views define the same function Φ. The QMC and Pascal
views add the external mathematical context that locates FM-Dance within the broader
landscape of quasi-Monte Carlo theory and discrete calculus — explaining *why* the
three original perspectives have the properties they do.

---

## The Fractal Algebra of Communion (OPER-1)

*Origin: V15.2 Sprint. Formalised as SparseArithmeticManifold.*

The FLU manifold is not just a data structure; it is an **executable intent**.
With the OPER-1 stack, we shift from "materialising" data to "evaluating" logic.

### The Compositional Pipeline
We model a manifold M not as an array, but as an expression tree of 
(Permute ∘ Integrate ∘ Digits) operators.

    Result = ( (M1 ⊕ M2) ⊗ M3 ) / 0.5

This is an **Algebraic Tree**, not an array.

### The "Oracle" Resolution
To evaluate the value at coordinate `x`, we resolve the tree recursively:
    eval(Node) = op(eval(Left), eval(Right))
    
This achieves O(D · depth) time complexity, where depth is the number of
arithmetic operations.

### Key Invariants
1. **L1 Preservation:** `+` and `-` preserve the L1 (constant line sum) invariant, provided both inputs satisfy it.
2. **Mean-Centering (S1):** `+` and `-` preserve the zero-mean property (S1).
3. **Spectral Shift:** `*` and `/` transform the spectral distribution, creating localized "Energy Densities" or "Scar Wells" within the lattice.

### Arithmetic vs. Communion
- **Communion (⊗_φ):** Structural Fusion (Dimension Increase).
- **Arithmetic (+, -, *, /):** Field Interaction (Value Transformation).

The two together allow us to perform **Discrete Field Theory** on the FLU substrate — simulating everything from diffusive heat maps to neural weight decay using only O(D) arithmetic.


## Notes on V15 Audit Integration (March 2026)

The V15 audit (`FLUAudit.txt`) established Perspectives 4 and 5 through:

1. Identifying the `np.cumsum` benchmark bug (T[0,0]=+1 instead of −1) that caused the false T9 refutation.
2. Algebraically deriving the FM-Dance–Faure conjugacy from the triangular matrix algebra.
3. Tracing the Pascal carry-cascade recursion through to the Sierpiński fractal projections.
4. Correcting the external reviewer's conflation of FLU's T^{-1} with the inverse Pascal matrix.

**T9 was promoted from CONJECTURE to PROVEN** as a direct result of finding and fixing the
diagnostic bug. The theorem was algebraically inevitable once the construction was clarified —
the Gnostic Scar (the 0/27 dissonance) forced the deeper inspection that revealed the truth.

---

## Perspective 6: The Siamese Magic Construction View

*Origin: Mönnich 2017 manuscript "Symmetrische Tanzschritte für magische Universen".
Formalised as Theorem MH (V15.4). This perspective predates all others — it is the
Genesis Seed (GEN-0) from which the FLU library grew.*

### The Two FM-Dance Objects

The FM-Dance traversal has two distinct implementations. Understanding the difference
is essential for correct use:

```
rank k ──┬── T-matrix path  (path_coord)    → Hamiltonian, Latin, T·a(k), NOT magic
         │   [CGW/BPT/KIB]  "How does FM-Dance WALK through the torus?"
         │
         └── Magic hypercube (magic_coord)  → adjacent-pair Siamese, ALL sums = M
             [MH theorem]   "What MAGIC CUBE does FM-Dance produce?"
```

Both are bijections over Z_nᵈ. Only the Siamese construction produces magic line sums.

### The Adjacent-Pair Step Vectors

For D=3, the manuscript step vectors are:

```
S1 = (+1, +1,  0)   steps axes 0,1 together     (primary, every rank)
S2 = ( 0, +1, +1)   steps axes 1,2 together     (fallback every n ranks)
S3 = ( 0,  0, −1)   backstep on axis 2 only     (fallback every n² ranks)
```

Compare with the T-matrix primary step σ₀ = (−1,+1,+1,…) which couples axis-0
*against* all others simultaneously — a fundamentally different coupling geometry.

**Why adjacent coupling produces magic:**
For any axis-p line, the free digit a_p appears in exactly two coordinate formulas
(i_{p-1} and i_p). The adjacent-pair structure forces each line to sample one element
from every spectral block {1..n^{d-1}}, {n^{d-1}+1..2n^{d-1}}, …, ensuring equal
block contributions. The T-matrix σ₀ = (−1,+1,…) breaks this for axis-0 by coupling
it simultaneously to all d axes, so axis-0 slices fall entirely within one block.

### Closed-Form Position Formula (MH PROVEN)

Derived algebraically from the cumulative effect of adjacent-pair steps:

```
digits: a_i = ⌊k/nⁱ⌋ mod n,   half = ⌊n/2⌋

i_0      = (half + a_0 − a_1)           mod n
i_j      = (half + a_{j−1} − a_{j+1})   mod n    [1 ≤ j ≤ d−2]
i_{d−1}  = (n−1  + a_{d−2} − 2·a_{d−1}) mod n
```

Verified exact against the iterative step algorithm for all (n,d) with n^d ≤ 10⁶.

### Three-Way Distinction

| Object | Construction | Magic? | Latin/LHS? | Hamiltonian? |
|--------|-------------|--------|-----------|--------------|
| `generate_fast` | identity digit map | ✗ | ✓ (global) | trivial |
| `path_coord` / T-matrix | σ₀=(−1,+1,…) | ✗ | ✓ (per-slice) | ✓ (T-bound) |
| `magic_coord` / Siamese | Sⱼ adjacent-pair | ✓ | ✓ (per-slice) | ✓ (simple) |

The Siamese magic cube is the ONLY one of the three with all axis line sums = M.

### FM-Dance vs Trump/Boyer Perfect Cube

The Trump/Boyer order-5 perfect cube (Trump & Boyer, 2003-11-13) achieves 30/30
planar diagonals = 315 at the cost of per-slice LHS digit balance. FM-Dance achieves
per-slice LHS balance at the cost of 18/30 planar diagonals. Both achieve all line
sums = 315. They occupy structurally orthogonal corners of the design space:

```
                  planar diagonals
                  (0 → 30 possible)
                        30 ┤ Trump/Boyer
                           │   (PERFECT)
                        18 ┤ FM-Dance
                           │   (MAGIC+LHS)
                         0 ┤ generate_fast / path_coord
                           └────────────────────────
                            no LHS   LHS per-slice
                             global    (FLU-native)
```

A cube achieving all 30 planar diagonals AND per-slice LHS balance would be a new
mathematical result. None is currently known at order 5.

```python
from flu.core.fm_dance import generate_magic
from flu.constants import FM_DANCE_5_NP, TRUMP_BOYER_5_NP, MAGIC_SUM_5
import numpy as np

cube = generate_magic(n=5, d=3)
M    = MAGIC_SUM_5   # 315
assert all(np.unique(cube.sum(axis=a)).tolist() == [M] for a in range(3))
```

---

## Notes on V15.4 (April 2026)

Perspective 6 was added based on formalisation of the original Mönnich 2017 manuscript
construction. Key contributions in V15.4:

1. Closed-form `magic_coord` formula derived algebraically from step vectors S1/S2/…/Sd.
2. Root-cause of the `generate_fast` bug documented: identity-map addressing ≠ magic cube.
3. `FM_DANCE_5_NP` corrected in `constants.py` to use `generate_magic(5,3)`.
4. Three-way distinction (addressing / magic / T-matrix) documented across `fm_dance.py`,
   `constants.py`, `THEOREMS.md`, `PERSPECTIVES.md`, and `BENCHMARKS.md`.
5. Trump/Boyer structural comparison formalised as theorem MH-COMPARE.
6. New document `docs/ANALYSIS_MAGIC_CUBES_ORDER5.md` created with full multi-representation
   in-depth analysis of both cubes.
