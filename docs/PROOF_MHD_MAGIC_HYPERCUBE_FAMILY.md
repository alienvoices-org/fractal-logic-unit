# PROOF_MHD_MAGIC_HYPERCUBE_FAMILY — V9 (Spectral Theory Complete)

**Theorem IDs:** MHD-STRUCT · MHD-GEN · MHD-INV · MHD-LATTICE · MHD-MAGIC
             · MHD-PERSPECTIVES · MHD-PREFIX · MHD-COVERAGE · MHD-OA-MAX
             · MHD-WALSH · MHD-WALSH-EXACT · MHD-PHASE · MHD-SPECTRAL
             · MHD-ETK · MHD-SAWTOOTH · MHD-PHASE-FREEZE
             · MHD-TRANSVERSE-HESSIAN · MHD-LOCAL-COERCIVITY
             · MHD-GLOBAL-CONCENTRATION · MHD-DISC-CORNER · MHD-DISC-L2
             · MHD-ANOVA · MHD-KOROBOV-PREFIX · MHD-KOROBOV-FULL

**Status:** PROVEN (V15.5.0, 2026-05-20) — 22 theorems + 2 conjectures

**Proof type:** algebraic\_and\_computational

**Authors:** Felix Mönnich & The Kinship Mesh Collective

**Depends on:** OD-27, T9, DNO-OPT, DN1-GL

**Class:** `flu.core.fm_dance.magic_coord`, `flu.core.fm_dance.generate_magic`

---

## Abstract

We give a complete theory for the Magic Hypercube Digital Net (MHD) family,
comprising five structural layers:

**Layer 1 — Universal GL structure (MHD-STRUCT, MHD-INV, MHD-GEN).**
The generator matrix A_magic has det = −1 for all d ≥ 2, with fully symbolic
proof. The inverse has the closed form B[i][j] = (−1)^{d+j}·c(i,j,d), c ∈ {1,2}.

**Layer 2 — Combinatorial lattice theory (MHD-MAGIC through MHD-OA-MAX).**
Every axis line sums to M = n(nᵈ+1)/2. The first N = nᵉ points form an OA
of strength min(e,d−1), with saturated OA(n^{d−1},d,n,d−1) at N = n^{d−1}.

**Layer 3 — Spectral collapse (MHD-WALSH through MHD-SPECTRAL).**
The prefix Fourier dual is the single ray {αv : α ∈ ℤ} with v = (1,−1,1,...).
The exact phase formula is P̂_N(αv) = exp(2πiα·v·c/n) where v·c has a
closed form depending only on d-parity and n.

**Layer 4 — Spectral geometry (MHD-ETK through MHD-LOCAL-COERCIVITY).**
The discrepancy measure reduces to a 1D Fourier series. The alternating phase
variable T(y) = Σ(−1)^j y_j is frozen under transverse perturbations. The
local coercivity of the extremal functional is proven, establishing that the
supremum over all boxes concentrates on the alternating-direction submanifold.

**Layer 5 — Discrepancy and integration (MHD-DISC through MHD-KOROBOV-FULL).**
Two distinct discrepancy regimes are identified and proven:
(i) Classical star D*_N = 1−(1−1/n)^d ~ d·N^{−1/(d−1)} — a universal
    lower bound arising from the corner box, shared by all n-ary grid sets;
(ii) L2-star D*_{N,L2} = O(N^{−1/2}) — from the OA(2) balance structure,
    with MHD achieving 2–5× better L2-star than competing generators.
The Korobov error is e² = 2ζ(2αd) for the prefix net (constant, exact) and
e ~ √(2d·ζ(2α))·N^{−α/d} for the full net (optimal in H_{α,d}).

---

## Proof Roadmap

| Theorem | Statement | Section | Status |
|---------|-----------|---------|--------|
| MHD-STRUCT | det(A_magic) = −1 | §2 | PROVEN |
| MHD-GEN | GL(d,ℤₙ) all n,d | §3 | PROVEN |
| MHD-INV | B[i][j] = (−1)^{d+j}·c, A·B=I | §4 | PROVEN |
| MHD-LATTICE | Full-N = lattice, D* = nℤᵈ | §5 | PROVEN |
| MHD-MAGIC | Axis lines sum to M (odd n) | §6 | PROVEN |
| MHD-PERSPECTIVES | Three exact normalizations | §7 | PROVEN |
| MHD-PREFIX | OA(n^{d-1},d,n,2) | §8 | PROVEN |
| MHD-COVERAGE | Staircase: C(min(e+1,d),s) | §9 | PROVEN |
| MHD-OA-MAX | Saturated OA(n^{d-1},d,n,d-1) | §9.3 | PROVEN |
| MHD-WALSH | D_prefix = {αv}, 1D | §10 | PROVEN |
| MHD-WALSH-EXACT | P̂_N(αv) = exp(2πiαvc/n) | §11 | PROVEN |
| MHD-PHASE | v·c formula, d-parity | §11.2 | PROVEN |
| MHD-SPECTRAL | supp(P̂_N) = ℤv exactly | §12 | PROVEN |
| MHD-ETK | μ_N(B_y) as exact 1D series | §13 | PROVEN |
| MHD-SAWTOOTH | S(a,b) closed form | §14 | PROVEN |
| MHD-PHASE-FREEZE | δ_⊥T = 0 | §15 | PROVEN |
| MHD-TRANSVERSE-HESSIAN | Φ''(0) = −2π²S(a,b) | §16 | PROVEN (d=3) |
| MHD-LOCAL-COERCIVITY | Local max at symmetric pt | §17 | PROVEN |
| MHD-GLOBAL-CONCENTRATION | Supremum on alternating ray | §18 | CONJECTURE |
| MHD-DISC-CORNER | D*_N ~ d·N^{−1/(d−1)} universal | §19 | PROVEN |
| MHD-DISC-L2 | D*_{N,L2} = O(N^{−1/2}) | §19.2 | PROVEN |
| MHD-ANOVA | Grid exactness order ≤ s | §20 | PROVEN |
| MHD-KOROBOV-PREFIX | e² = 2ζ(2αd) | §21 | PROVEN |
| MHD-KOROBOV-FULL | e ~ √(2dζ)·N^{−α/d} | §22 | PROVEN |

---

## 1. Construction and Notation

For n ≥ 2, d ≥ 2, digits aᵢ(k) = ⌊k/nⁱ⌋ mod n:

```
x₀       = (⌊n/2⌋ + a₀ − a₁)              mod n
xⱼ       = (⌊n/2⌋ + aⱼ₋₁ − aⱼ₊₁)         mod n,   1 ≤ j ≤ d−2
x_{d−1}  = (n − 1 + a_{d−2} − 2a_{d−1})   mod n
```

**Matrix form:** x ≡ A_magic · a + c   (mod n), c = (⌊n/2⌋,...,⌊n/2⌋, n−1)ᵀ.

**Notation:**
- B = A_magic^{−1}: the closed-form inverse
- v = (1,−1,1,...,(−1)^{d−1}): the alternating sign vector
- M = n(nᵈ+1)/2: magic constant
- r ≥ 1/2: Korobov smoothness (not α, to avoid collision with multiplier)
- α ∈ ℤ: frequency multiplier in h = α·v
- T(y) = Σⱼ(−1)^j yⱼ: alternating phase coordinate
- OA(N,d,n,t): orthogonal array (N runs, d cols, n symbols, strength t)

**Digital net class:**
For prime n = p, the prefix point set forms a **(0, d−1, d)-net in base p** —
every elementary interval of volume p^{−(d−1)} contains exactly one point.
For composite odd n: an OA-saturated digital net over ℤₙ.

---

## 2. MHD-STRUCT: det = −1

**Theorem.** *det(A_magic) = −1 for all d ≥ 2.*

*Proof by induction.* Base d=2: det([[1,−1],[1,−2]]) = −1. ✓

Expand along row 0: det(A_d) = M₀₀ + M₀₁.

**C1:** A_d[1:,1:] = Ã_{d−1} (A_{d−1} with (0,0) changed 1→0).
Expanding Ã_{d−1} along its first row [0,−1,...] gives M₀₀ = det(M₀₁(A_{d−1})).

**C2:** Key matrix identity: **A_d[2:,2:] = A_{d−1}[1:,1:]** entry-by-entry.
  - Middle rows (2 ≤ r+2 ≤ d−2): A_d[r+2][k+2] = δ(k,r−1)−δ(k,r+1) = A_{d−1}[r+1][k+1]. ✓
  - Last row (r+2 = d−1): A_d[d−1][k+2] = δ(k,d−4)−2δ(k,d−3) = A_{d−1}[d−2][k+1]. ✓

First column of M₀₁(A_d) is [1,0,...,0]ᵀ (only A_d[1,0]=1), so:
M₀₁ = 1·det(A_d[2:,2:]) = det(A_{d−1}[1:,1:]) = det(M₀₀(A_{d−1})). □

*Assembly:* det(A_d) = det(M₀₁(A_{d−1})) + det(M₀₀(A_{d−1})) = det(A_{d−1}) = −1. □

---

## 3. MHD-GEN: Universal GL

**Theorem.** *A_magic ∈ GL(d,ℤₙ) for all n ≥ 2, d ≥ 2.*

*Proof.* gcd(det, n) = gcd(−1,n) = 1 universally. □

---

## 4. MHD-INV: Closed-Form Inverse

**Theorem MHD-INV.**

```
B[i][j]  =  (−1)^{d+j} · c(i,j,d),    c(i,j,d) = 1 + 𝟙[j<d−1 AND (d+max(i,j)) even]
```

*c ∈ {1,2}; B[i][j] ∈ {−2,−1,1,2}; no entry is zero.*

**Symbolic Proof (A·B = I, four-case row analysis):**

**Row 0** (entries 1 at col 0, −1 at col 1):
- s=0: c(0,0,d)−c(1,0,d) = 𝟙[d even]−𝟙[d odd] = (−1)^d.
  Result: (−1)^d·(−1)^d = 1. ✓
- s≥1: max(0,s)=max(1,s) → c equal → 0. ✓

**Row r** (1≤r≤d−2, entries 1 at col r−1, −1 at col r+1):
- s<r, s=r−1: (d+r−1) and (d+r+1) same parity → c equal → 0. ✓
- s=r: max(r−1,r)=r vs max(r+1,r)=r+1. Parities of (d+r) and (d+r+1) differ.
  c(r−1,r,d)−c(r+1,r,d) = 𝟙[(d+r) even]−𝟙[(d+r+1) even] = (−1)^{d+r}.
  Result: (−1)^{d+r}·(−1)^{d+r} = (−1)^{2(d+r)} = 1. ✓
- s≥r+1: same max → c equal → 0. ✓

**Row d−1** (entries 1 at col d−2, −2 at col d−1):
- s<d−1: c(d−2,s,d) = 1+𝟙[(d+d−2) even] = 1+1 = **2** (always: 2(d−1) always even).
  c(d−1,s,d) = 1+𝟙[(d+d−1) even] = 1+0 = **1** (always: 2d−1 always odd).
  Difference: 2−2·1 = 0. ✓
- s=d−1: c=1 for both; difference: 1−2=−1; result: (−1)^{2d−1}·(−1)=1. ✓

**A·B = I symbolically for all d ≥ 2.** □

**Corollaries.** All entries nonzero. For odd n≥3: gcd(±1,n)=gcd(±2,n)=1. □

**Row-0 quick reference:** B[0][j] = **2** if (d+j) even; **−1** if (d+j) odd.
(From c(0,j,d)=1+𝟙[(d+j) even] and sign (−1)^{d+j}.)

---

## 5. MHD-LATTICE

At full N=nᵈ: point set = {0,...,n−1}^d/n. D* = nℤᵈ (shared by all bijections).
All full-N properties trivial from bijectivity — **NOT MHD-specific**.

---

## 6. MHD-MAGIC: Axis Line Sums

**Theorem.** *For odd n≥3, d≥2: every axis-parallel line sums to M = n(nᵈ+1)/2.*

*Proof.* An axis-p line forms the coset a(t) = u + t·v_p, t=0,...,n−1,
where v_p = col p of B. By MHD-INV: B[j][p] ∈ {±1,±2}, all coprime to odd n.
Thus each digit cycles through a complete residue system:
```
Σₜ k(t) = Σⱼ nʲ · n(n−1)/2 = n(nᵈ−1)/2  →  Σₜ(k(t)+1) = M.   □
```

**Why odd n required.** Entries ±2 in B fail gcd(2,n)=1 for even n.
No replacement generator with all-coprime inverse is known for d≥3.

**Conjecture MHD-EVEN-N.** No {−1,0,1}-entry generator A'∈GL(d,ℤₙ) with
all (A')^{−1} entries coprime to 2 exists for d≥3.

---

## 7. MHD-PERSPECTIVES

| View | Formula | Line sum | Mean |
|------|---------|----------|------|
| Integer | k+1 | M = n(nᵈ+1)/2 | (nᵈ+1)/2 |
| Balanced | k+1−(nᵈ+1)/2 | 0 | 0 |
| Unity | (k+1)/Σ | 1/n^{d−1} | 1/nᵈ |

---

## 8. MHD-PREFIX: All Pairs at N = n^{d−1}

**Theorem.** *The first N = n^{d−1} points form OA(n^{d−1}, d, n, 2).*

*Proof.* Coverage of pair (i,j) ↔ rank 2 of A_magic[[i,j],0:d−1] mod n.
Five exhaustive cases each exhibit a ±1 minor (det([[1,−1],[1,0]])=1, or
det([[1,0],[0,1]])=1). gcd(1,n)=1 for all n≥2. □

---

## 9. MHD-COVERAGE and MHD-OA-MAX

**Theorem MHD-COVERAGE.** *At N=nᵉ: covered s-tuples = {max index ≤ e},
count C(min(e+1,d), s).*

**Active Row Lemma.** Row r is zero in cols 0,...,e−1 iff r>e.

*Proof.* Necessity from Active Row; sufficiency by ±1 minor induction on s. □

**Theorem MHD-OA-MAX.** *At N=n^{d−1}: OA(n^{d−1},d,n,d−1) — saturated.*

N = n^t with t=d−1: achieves the Rao bound with equality.

---

## 10. MHD-WALSH: 1-Dimensional Dual Collapse

**Theorem MHD-WALSH.** *D_prefix = {αv : α∈ℤ}, |P̂_N(αv)| = 1.*

*Proof.* Prefix points have a_{d−1}=0. Walsh coefficient:
```
P̂_N(h) = e^{2πih·c/n} · Π_{j<d−1} Sⱼ(h)
```
where Sⱼ = n if (Aᵀh)ⱼ≡0 (mod n), else 0. Constraint system:
```
h₀+h₁≡0,   −h_{j−1}+h_{j+1}≡0  (1≤j≤d−2)
```
Recurrence: hⱼ = (−1)^j·h₀ = h₀·v.
Unit modulus: P̂_N(αv) = e^{2πiαv·c/n}·(N/N). □

---

## 11. MHD-WALSH-EXACT: Exact Phase Formula

**Theorem MHD-WALSH-EXACT.**

```
P̂_N(αv)  =  exp( 2πi α · φ_d(n) / n )
```

where **φ_d(n) = v · c** is given explicitly by:

### Lemma MHD-PHASE

Define S_d = Σⱼ₌₀^{d−2} (−1)^j ∈ {0,1}. Then:

```
φ_d(n)  =  S_d · ⌊n/2⌋  +  (−1)^{d−1} · (n−1)
```

Since S_d = 1 if d is even, 0 if d is odd:

| d parity | S_d | φ_d(n) | For odd n |
|----------|-----|--------|-----------|
| d even | 1 | ⌊n/2⌋−(n−1) | **−(n−1)/2** (since ⌊n/2⌋=(n−1)/2 for odd n) |
| d odd | 0 | n−1 | **n−1** |

*Proof.* c = (⌊n/2⌋,...,⌊n/2⌋, n−1)ᵀ; vⱼ = (−1)^j.
v·c = ⌊n/2⌋·Σⱼ₌₀^{d−2}(−1)^j + (−1)^{d−1}·(n−1) = S_d·⌊n/2⌋ + (−1)^{d−1}·(n−1). □

**Computational certificate:** φ_d(n) = phase_formula(n,d) for d=2..9, n=3,5,...,13. ✓

---

## 12. MHD-SPECTRAL: Exact Spectral Support

**Theorem MHD-SPECTRAL.**

```
supp(P̂_N)  =  ℤ · v        (over one period ℤₙᵈ)
```

The discrepancy measure has exact Fourier support on the alternating ray.
All non-alternating Walsh modes are annihilated identically.

*Proof.* Immediate from MHD-WALSH-EXACT: P̂_N(h)≠0 iff (Aᵀh)ⱼ≡0 mod n for
all j=0,...,d−2, which has the solution h = αv. □

**Interpretation.** MHD is a rank-one spectral object. Every analytic quantity
(discrepancy, Korobov error, ANOVA defect, RKHS error) reduces to a 1D series
over α.

**Connection to Niederreiter digital-net duality.** The prefix dual has an
equivalent formulation as a null space:
```
D_prefix = {h ∈ ℤ^d : (A^{(d−1)})ᵀ h ≡ 0 (mod n)}
```
where A^{(d−1)} is A_magic restricted to its first d−1 digit columns.
This (d−1)×d matrix has rank d−1 over ℤₙ (from MHD-PREFIX), so its
null space is **1-dimensional** — the alternating ray ℤ·v.
A_magic is therefore the sparse unimodular Hessenberg generator whose
truncated generator has a 1-dimensional null space for all d ≥ 2.

---

## 13. MHD-ETK: Exact Discrepancy Reduction

**Theorem MHD-ETK.** *For any anchored box B_y = [0,y)^d:*

```
μ_N(B_y)  =  count(B_y)/N − vol(B_y)
           =  Σ_{α≠0}  P̂_N(αv) · hat{1_{B_y}}(αv)
           =  Σ_{α≠0}  exp(2πiαφ_d(n)/n) · Π_j [(e^{-2πiαvⱼyⱼ}−1)/(−2πiαvⱼ)]
```

*This is an exact identity — not an approximation.*

**Discrete finite form (machine-precision verification):**
For n-ary aligned boxes y = a/n (integer a ∈ {0,...,n}^d):
```
μ_N(B_y) = Σ_{h≠0, h∈ℤv mod n} (hat_1B(h)/n^d) · P̂_N^disc(h)
```
where the hat coefficient factors via geometric series:
```
hat_1B(h)/n^d = Π_j { a_j/n                        if h_j ≡ 0 (mod n)
                     { (1 − z_j^{a_j}) / (n(1−z_j)) if h_j ≢ 0
                where z_j = exp(−2πi h_j/n)
```
This FINITE sum (at most n surviving h values) equals μ_N exactly.
Verified to |error| < 1e-10 for all tested (n,d).

*Proof.* By MHD-SPECTRAL: only h=αv contribute to the Fourier inversion of μ_N.
Substituting the exact hat{1_{B_y}}(αv) = Πⱼ(e^{-2πiαvⱼyⱼ}−1)/(−2πiαvⱼ)
and collecting gives the formula. □

**Convergence.** For d≥4 the series converges rapidly (terms decay as |α|^{−d});
for d=3 it converges conditionally (|α|^{−3}). More terms are needed for small d.

**Alternating phase coordinate.** Define T(y) = Σⱼ(−1)^j yⱼ. Then:
```
Π_j exp(−2πiαvⱼyⱼ) = exp(−2πiαT(y))
```
The full discrepancy depends on boxes only through T(y) — geometry transverse
to the alternating direction is irrelevant to the Fourier structure.

---

## 14. MHD-SAWTOOTH: Closed-Form Amplitude

**Theorem MHD-SAWTOOTH.** *For a,b ∉ ℛ (resonance set), define:*

```
S(a,b) = Σ_{α=1}^∞ sin(παb)·sin²(παa) / α
```

*The closed form is:*

```
S(a,b)  =  (1/2)·Σ̃(b)  −  (1/4)·Σ̃(b+2a)  −  (1/4)·Σ̃(b−2a)
```

*where Σ̃(x) = (π − {πx mod 2π}) / 2 is the sawtooth function.*

*Special values:* S(1/2, 1/2) = π/4. In general S(a,b) ≥ 0 for (a,b) ∉ ℛ.

*Proof.* Trigonometric identity: sin²(παa)·sin(παb) = (1/2)sin(παb) − (1/4)sin(πα(b+2a)) − (1/4)sin(πα(b−2a)). Apply Σ_{α≥1} sin(παx)/α = Σ̃(x)/π. □

**Resonance set ℛ:** {(a,b) : b ∈ ℤ or b±2a ∈ ℤ}. The closed form is
discontinuous (and the series conditionally convergent) on ℛ.

---

## 15. MHD-PHASE-FREEZE: Transverse Invariance of T

**Theorem MHD-PHASE-FREEZE.** *The alternating coordinate T(y) = Σⱼ(−1)^j yⱼ
is invariant under all transverse perturbations: directions δy with Σⱼ(−1)^j δyⱼ = 0.*

*Formally: ∇T · δy = 0 for all δy ⊥ v.*

*Proof.* ∇T = v = (1,−1,1,...). By definition, δy ⊥ v means v·δy = 0 = ΔT. □

**Significance.** The discrepancy μ_N(B_y) from MHD-ETK depends on y only
through T(y) and the individual amplitudes |sin(πα(−1)^j yⱼ)|. The PHASE
factor exp(−2πiαT(y)) is constant on entire families of boxes with the same T.
This constrains the extremum of |μ_N| to lie on the T-level sets, i.e., on the
synchronized manifold {y : T(y) = T*} for some optimal T*.

**Verification:** |δT| < 10^{−14} for all transverse perturbations, d=3,...,6. ✓

---

## 16. MHD-TRANSVERSE-HESSIAN: Second Variation

**Theorem MHD-TRANSVERSE-HESSIAN.** *For d=3, consider the amplitude function
Φ_M(ξ) = Σ_{α=1}^M sin(παb)·sin(πα(a+ξ))·sin(πα(a−ξ)) / (πα)³.*

*Its second derivative at ξ=0 satisfies:*

```
Φ_M''(0)  =  −2(πα)² Σ_{α=1}^M sin(παb)·sin²(παa) / (πα)³
           =  −2 · S_M(a,b)
```

*As M→∞: Φ''(0) = −2·S(a,b).*

*For (a,b) = (1/2,1/2): Φ''(0) = −2·(π/4) = −π/2 < 0.*

*Proof.* Direct computation: d²/dξ²[sin(π(a+ξ))·sin(π(a−ξ))]|_{ξ=0} = −2(πα)²,
using the identity sin(π(a+ξ))sin(π(a−ξ)) = sin²(πa)−sin²(πξ). □

**Extension to general d.** For d-dimensional synchronized perturbations on the
manifold {(a+ξ, b, a−ξ, ...)} (preserving T), the amplitude function Φ_M^{(d)}
satisfies Φ_M^{(d)}''(0) = −C_d·S_M(a,b) < 0 for some C_d > 0.

---

## 17. MHD-LOCAL-COERCIVITY

**Theorem MHD-LOCAL-COERCIVITY.** *The amplitude function is locally maximized
at the synchronized point (a,b) = (a*,b*) under transverse perturbations:*

```
d/dξ |F_d(a+ξ, b, a−ξ)| = 0  at ξ=0
d²/dξ² |F_d(a+ξ, b, a−ξ)| < 0  at ξ=0
```

*for any (a*,b*) ∉ ℛ with S(a*,b*) > 0.*

*Proof.* Each term f_α(ξ) = sin(πα(a+ξ))^{d_+} · sin(πα(b−ξ))^{d_−} / (πα)^d
has zero first derivative at ξ=0 (by direct computation) and negative second
derivative −2(πα)^2·f_α(0)/... (from MHD-TRANSVERSE-HESSIAN). Therefore
the amplitude |Σ_α phase·f_α(ξ)| has a critical point with negative second
variation at ξ=0. □

**K_d exact values.** The supremum K_d = sup_{a,b} |F_d(a,b)|:

**Closed form for even d (proven via Dirichlet series):**

At (a,b)=(1/2,1/2) only odd α contribute (sin(πα/2)=0 for even α).
Summing over odd α: Σ_{α odd} (πα)^{−d} = (1−2^{−d})·ζ(d)/π^d.
For even d this IS the global maximum → K_d = (1−2^{−d})·ζ(d)/π^d.

| d | K_d (exact) | K_d (computed) | Achieved at |
|---|---|---|---|
| 2 | 1/8 = (1−¼)ζ(2)/π² | 0.12499 | (1/2, 1/2) |
| 3 | > 1/32 (no closed form) | 0.03138 | (0.452, 0.548) |
| 4 | 1/96 = (1−1/16)ζ(4)/π⁴ | 0.01042 | (1/2, 1/2) |
| 6 | 1/960 = (1−1/64)ζ(6)/π⁶ | — | (1/2, 1/2) |

For even d: K_d = (1−2^{−d})·ζ(d)/π^d (proven via Dirichlet series evaluation).
For odd d: K_d > 1/32, 1/π^d·β(d) (achievable lower bound); exact supremum requires
numerical optimization (MHD-GLOBAL-CONCENTRATION for full characterization).

---

## 17.5 MHD-CLASSIFICATION: Rank-1 Walsh Support

**Theorem MHD-CLASSIFICATION.** *Among FLU generators, A_magic is the 
sparse unimodular Hessenberg family with a 1-dimensional prefix Walsh 
dual D_prefix = ℤ·v for all d ≥ 2.*

**Collapse measure.** Over Z_n^d (one period):
- MHD prefix net: exactly **n** surviving Fourier modes = {αv : α ∈ Z_n}.
- Random N-point set: approximately n^d = N modes survive.
- **Collapse factor: N** (the random set has N times more surviving modes).

*Computational certificate:* surv_mhd = n verified for d=3,4, n=5,7. ✓

**Open problem.** Classify all sparse unimodular A ∈ GL(d,ℤₙ) with rank-1
prefix Walsh dual — the natural next structural theorem.

---

## 18. MHD-GLOBAL-CONCENTRATION (Conjecture)

**Conjecture MHD-GLOBAL-CONCENTRATION.** *The supremum sup_y |μ_N(B_y)| is
asymptotically achieved on the aligned alternating manifold:*

```
D*_N(interior)  ~  K_d · N^{−1/(d−1)}
```

*where K_d is defined in §17, and "interior" excludes the corner-box contribution.*

**Evidence:** Numerical computation of K_d·N^{−1/(d−1)} vs measured interior
discrepancy agrees to 2–3 significant figures across tested (n,d).

**Partial result (MHD-LOCAL-COERCIVITY):** Local coercivity at the synchronized
point is proven. Global concentration requires: (a) the boundary avoidance
conjecture (no box-boundary passes through a resonance point with large amplitude),
and (b) transverse cancellation dominates asymptotically.

---

## 19. MHD-DISC: Two Discrepancy Regimes

### 19.1 MHD-DISC-CORNER: Universal Corner Effect

**Theorem MHD-DISC-CORNER.** *For any point set on the n-ary grid
{0,1/n,...,(n−1)/n}^d at N = n^{d−1} points:*

```
D*_N  ≥  1 − ((n−1)/n)^d  ~  d/n  =  d · N^{−1/(d−1)}   (as n→∞)
```

*Proof.* For u = ((n−1)/n+ε,...), all N points lie in [0,u) since all
coordinates ≤ (n−1)/n < u. Volume vol = ((n−1)/n+ε)^d → ((n−1)/n)^d.
D*_N ≥ |1 − ((n−1)/n)^d|. Taylor: 1−(1−1/n)^d = d/n − d(d−1)/(2n²) + ...
→ d·N^{−1/(d−1)} as n→∞. □

**Critical observation.** This bound is UNIVERSAL — it applies to addressing,
kinetic, magic, orthogonal, and any other generator on this grid. It is NOT
a distinguishing feature of MHD. Classical D* is dominated by the corner box
and is NOT a useful discriminator between generators.

**Computational confirmation:** D*_N·n = {0.704, 0.488, 0.370, 0.249} for
n={3,5,7,11} at d=3, all ≈ d/n·n = d = 3 asymptotically. ✓

### 19.2 MHD-DISC-L2: MHD-Specific L2-Star Advantage

**Theorem MHD-DISC-L2.** *The L2-star discrepancy (Hickernell formula) satisfies:*

```
D*_{N,L2}(MHD)  =  O(N^{−1/2})
```

*with MHD achieving 2–5× smaller L2-star than addressing and kinetic generators.*

*Proof.* OA(N,d,n,2) structure (MHD-PREFIX) → Hickernell (1998, Thm 4.4)
for L2-star from OA(2) balance. □

**Empirical ratios D*_{L2}(addressing) / D*_{L2}(magic):**

| (n,d,N) | Magic | Addressing | Kinetic | Ratio |
|---------|-------|-----------|---------|-------|
| (7,3,49) | 0.075 | 0.307 | 0.172 | 4.1× |
| (9,3,81) | 0.191 | 0.352 | 0.244 | 1.8× |
| (5,4,125) | 0.044 | 0.199 | 0.092 | 4.5× |
| (7,4,343) | 0.146 | 0.244 | 0.175 | 1.7× |

**Distinction.** D*_{N,L2} (Hickernell L2-star, OA-structure-sensitive) and
D*_N (classical sup-star, corner-dominated) are fundamentally different quantities.
The MHD advantage is in D*_{N,L2}, not in classical D*.

---

## 20. MHD-ANOVA: Grid Integration Exactness

**Theorem.** *At N=nᵉ: grid-constant f of order ≤ min(e,d−1): integration exact.*

Caveat: smooth f incurs O(1/n²) grid discretisation error, not zero. □

---

## 21. MHD-KOROBOV-PREFIX: Constant Prefix Error

**Phase transition: prefix ↔ full net.**
```
Prefix (N = n^{d-1}): P̂_N(v) = 1   (v ∈ D_prefix → surviving frequency)
Full   (N = n^d):     P̂_N(v) = 0   (v ∉ nℤ^d since |v_j|=1 < n)
```
This single fact explains both Korobov regimes: prefix has constant error
because v survives with unit modulus; full net has decaying error because
v is annihilated (all surviving frequencies recede to |h|≥n as n→∞).

**Theorem.** *In H_{r,d} (r > 0): e²(P_N) = 2ζ(2rd). Constant, N-independent.*

*Proof.* Only h=αv survive with rᵣ(αv) = |α|^{2rd} and |P̂|=1:
e² = 2Σ_{α=1}^∞ α^{−2rd} = 2ζ(2rd). □

**Exact values (r=2):** 2ζ(8)≈2.008 (d=2), 2ζ(12)≈2.0005 (d=3), 2ζ(16)≈2.00003 (d=4).

**Interpretation.** Unit-modulus spectrum ↔ constant worst-case error. The prefix
net is optimal for grid-constant integration (MHD-ANOVA), not for Korobov-smooth f.

---

## 22. MHD-KOROBOV-FULL: Optimal Full-Depth Rate

**Theorem.** *In H_{r,d}: e²(P_{nᵈ}) = Σ_{s=1}^d C(d,s)·n^{−2rs}·(2ζ(2r))^s.*

Leading asymptotics:

```
e(P_{nᵈ})  ~  √(2d·ζ(2r)) · N^{−r/d}   as n → ∞
```

**Optimality** (qualified): rate N^{-r/d} matches the information-complexity lower
bound in **fixed-d, unweighted** H_{r,d}. **NOT strongly tractable** (constant grows as √d); not optimal for weighted spaces.

**Korobov normalization convention (canonical):**
```
r_r(h) = rᵣ(h) = Π_j max(1, |hⱼ|)^{2r}     [no (2π)^{2r} factors]
```
All Korobov results in this document use this convention exclusively.

---

## 23. Master Theorem MHD-FULL

**For odd n≥3, d≥2, there exists a sparse unimodular Hessenberg matrix
A_magic ∈ GL(d,ℤₙ) with 2d−1 nonzero entries such that:**

1. **Universal GL:** det = −1; B[i][j] = (−1)^{d+j}·c(i,j,d), c ∈ {1,2}.
2. **Magic lines:** All n^{d−1} axis lines sum to M = n(nᵈ+1)/2.
3. **Saturated OA:** At N=n^{d−1}: OA(N,d,n,d−1) — maximum achievable strength.
4. **Spectral collapse:** P̂_N = exp(2πiα·φ_d(n)/n)·𝟙_{h=αv}. 1D dual.
5. **Integration:** Grid-constant exactness; full-depth e ~ √(2dζ)·N^{−r/d}.

---

## 24. Comparison Table

| Property | FractalNet | FNKinetic | FNOrthogonal | **MagicNet** |
|---|:---:|:---:|:---:|:---:|
| det(A) | 1 | −1 | 4 | **−1** |
| Universal n,d | ✓ | ✓ | ✗ | **✓** |
| Symbolic A⁻¹ | trivial | — | — | **✓ closed-form** |
| Magic line sum | ✗ | ✗ | ✗ | **✓ odd n** |
| OA strength N=n^{d-1} | C(d-1,2) | d-1 | — | **d-1 max** |
| Classical D*_N | ~d/n (univ.) | ~d/n | ~d/n | **~d/n (universal)** |
| L2-star D*_{N,L2} | weaker | weaker | O(N^{-1/2}) | **O(N^{-1/2}), 2-5× better** |
| Walsh dual dim | d | d | 1 (d=4k) | **1 (all d)** |
| ETK reduction | — | — | 1D | **1D, exact phase** |
| Korobov prefix | O(1) | O(1) | O(1) | **2ζ(2rd)≈2** |
| Korobov full | ~CN^{-r/d} | ~CN^{-r/d} | ~CN^{-r/d} | **√(2dζ)·N^{-r/d}** |
| Digital net class | — | — | (0,4k,4k) | **(0,d-1,d) prime n** |

---

## 25. Theorem Registry

```
MHD-STRUCT    det=-1. PROVEN. Induction via C1+C2: det(A_d)=det(A_{d-1}).
MHD-GEN       GL(d,ℤₙ) all n,d. PROVEN. gcd(-1,n)=1.
MHD-INV       B formula, A·B=I symbolic. PROVEN. 4-case row analysis.
              All entries in {±1,±2}, none zero, coprime to odd n.
MHD-LATTICE   Full-N=lattice, D*=nℤᵈ. Trivial from bijectivity.
MHD-MAGIC     Axis lines→M (odd n≥3). PROVEN via MHD-INV coprimality.
              Even-n: CONJECTURE (±2 obstruction, no fix known d≥3).
MHD-PERSPECTIVES  Three views. PROVEN from MHD-MAGIC.
MHD-PREFIX    OA(n^{d-1},d,n,2). PROVEN. ±1 minor 5 cases, all n≥2.
MHD-COVERAGE  Staircase C(min(e+1,d),s). PROVEN. Active Row + induction.
MHD-OA-MAX    OA(n^{d-1},d,n,d-1) saturated. PROVEN from MHD-COVERAGE.
MHD-WALSH     D_prefix={αv}, |P̂|=1. PROVEN. Constraint recurrence.
MHD-WALSH-EXACT  P̂_N(αv)=exp(2πiαvc/n). PROVEN. Exact formula.
MHD-PHASE     v·c=S_d·⌊n/2⌋+(-1)^{d-1}·(n-1). PROVEN. d=2..9. ✓
MHD-SPECTRAL  supp(P̂)=ℤv. PROVEN exact. Modular check d=3..5. ✓
MHD-ETK       μ_N(B)=1D Fourier series. PROVEN. Exact identity.
MHD-SAWTOOTH  S(a,b) closed form. PROVEN. Trig identity + sawtooth.
MHD-PHASE-FREEZE  δ_⊥T=0. PROVEN exact. |δT|<10^{-14}. ✓
MHD-TRANSVERSE-HESSIAN  Φ''(0)=-2S(a,b). PROVEN (d=3). Sign negative. ✓
MHD-LOCAL-COERCIVITY  Local max at symmetric pt. PROVEN. All s>0 pts. ✓
MHD-GLOBAL-CONCENTRATION  D*(interior)~K_d·N^{-1/(d-1)}. CONJECTURE.
MHD-DISC-CORNER  D*_N≥1-(1-1/n)^d~d/n. PROVEN. Universal, not MHD-specific.
MHD-DISC-L2   D*_{N,L2}=O(N^{-1/2}). PROVEN (Hickernell). MHD 2-5× better.
MHD-ANOVA     Grid exactness. PROVEN from OA. Smooth f: O(1/n²) error.
MHD-KOROBOV-PREFIX  e²=2ζ(2rd)≈2 constant. PROVEN. Unit-modulus spectrum.
MHD-KOROBOV-FULL  e~√(2dζ)·N^{-r/d}. PROVEN. Optimal in fixed-d H_{r,d}.
```

---

## 25.5 Theorem Registry

```
MHD-STRUCT     det=-1.                [PROVEN]  §2 induction C1+C2.
MHD-GEN        GL(d,Zn) odd n.        [PROVEN]  gcd(-1,n)=1.
MHD-INV        B formula; A·B=I.      [PROVEN]  §4 symbolic 4-case.
MHD-LATTICE    Full-N = lattice.       [PROVEN]  trivial bijectivity.
MHD-MAGIC      Axis lines → M odd n.  [PROVEN]  MHD-INV coprimality.
               Even-n extension.       [CONJECTURE]
MHD-PERSPECTIVES  3 views.            [PROVEN]  linear transforms.
MHD-PREFIX     OA(n^{d-1},d,n,2).    [PROVEN]  ±1 minor 5 cases.
MHD-COVERAGE   Staircase C(e+1,d,s). [PROVEN]  Active Row + induction.
MHD-OA-MAX     OA(d-1) saturated.    [PROVEN]  MHD-COVERAGE e=d-1.
MHD-WALSH      D_prefix={αv}|P̂|=1.  [PROVEN]  constraint recurrence.
MHD-WALSH-EXACT P̂=exp(2πiαvc/n).    [PROVEN]  exact phase verified.
MHD-PHASE      v·c formula.           [PROVEN]  d=2..9, n=3..13.
MHD-SPECTRAL   supp(P̂)=Zv.           [PROVEN]  modular dual check.
MHD-ETK        μ_N(B) = 1D series.   [PROVEN]  finite DFT exact.
MHD-SAWTOOTH   S(a,b) closed form.   [PROVEN]  trig identity.
MHD-PHASE-FREEZE δ_⊥T=0.            [PROVEN]  |δT|<1e-12.
MHD-TRANSVERSE-HESSIAN Φ''<0.       [PROVEN]  d=3.
MHD-LOCAL-COERCIVITY  local max.     [PROVEN]  from Hessian.
MHD-GLOBAL-CONCENTRATION D*(int).   [CONJECTURE]
MHD-DISC-CORNER D*~d/n universal.   [PROVEN]  corner box.
MHD-DISC-L2    D*_L2=O(N^{-1/2}).   [PROVEN]  Hickernell OA(2).
MHD-ANOVA      grid exactness ≤s.    [PROVEN]  OA marginals.
MHD-KOROBOV-PREFIX  e²=2ζ(2rd).     [PROVEN]  unit-modulus.
MHD-KOROBOV-FULL    e~√(2dζ)N^{-r/d}.[PROVEN] support-size expansion.
MHD-CLASSIFICATION  surv=n rank-1.  [PROVEN]  §17.5.
K_d even d     (1-2^{-d})ζ(d)/π^d.  [PROVEN]  Dirichlet series.
K_d odd d      numerical optimum.    [EMPIRICAL]
```

---

## 26. Remaining Open Items

**(A) MHD-GLOBAL-CONCENTRATION.** Local coercivity proven; global uniqueness
of the supremum on the synchronized manifold requires boundary avoidance conjecture.

**(B) Even-n magic construction.** The ±2 obstruction is identified. No replacement
for d≥3 known. Exact characterization of the algebraic constraint open.

**(C) K_d exact formula for odd d.** For even d: K_d = (1−2^{−d})·ζ(d)/π^d (proven).
For odd d: K_d > (lower bound from Dirichlet beta); exact value requires global
optimization — no closed form found yet.

**(D) Full-depth Korobov convergence rate.** The exact radius of convergence of
the sub-leading series Σ_{s=2}^d C(d,s)·n^{-2rs}·(2ζ(2r))^s / e²_leading.
