# examples/mhd_dynamic — MHD-Dynamic Neural Initialisation & PMPS

Ultra-deep network stabilisation via the Magic Hypercube Digital Net (MHD),
and the Procedural Manifold Parameter System (PMPS V4.1).

---

## What is here

This directory covers two tightly related systems built on the same MHD oracle:

| System | Files | What it does |
|--------|-------|--------------|
| **MHD-Dynamic** | `mhd_dynamic.py`, `bench_mhd_dynamic.py` | Oracle, weight init, NumPy regulariser, three-suite benchmark |
| **PMPS V4.1** | `procedural_manifold/`, `bench_mhd_pmps.py` | Full PyTorch transformer: `W = O(r) + Δ`, streaming inference, delta knowledge |

Both share the same oracle mathematics. `mhd_dynamic.py` depends only on
`numpy` and `flu-math`; PMPS requires `torch`.

---

## Quick start

```bash
pip install flu-math numpy              # MHD-Dynamic only
pip install torch safetensors PyYAML    # PMPS (optional extras)

# MHD-Dynamic oracle self-test
python mhd_dynamic.py

# MHD-Dynamic three-suite benchmark (~95s, NumPy only)
python bench_mhd_dynamic.py

# PMPS full benchmark (~3 min, PyTorch)
python bench_mhd_pmps.py

# PMPS demo: token generation
python -m procedural_manifold.demo
```

---

## MHD-Dynamic benchmark (`bench_mhd_dynamic.py`)

Pure NumPy. Uses `flu.core.fm_dance.magic_coord` as the canonical oracle.
Three suites, all numbers from this codebase:

### Suite A — Forward-pass stability
**MHD-SPECTRAL**: rank-1 Walsh dual suppresses high-frequency activation explosion.
Residual MLP, Leaky ReLU, no BatchNorm. 5 seeds.

| Depth | MHD var | Kaiming var | Dead% MHD | Dead% Kai | Advantage |
|-------|---------|-------------|-----------|-----------|-----------|
| 64    | 0.31    | 0.38        | 4.7%      | 9.5%      | 1.2×      |
| 256   | 0.46    | 1.98        | 4.3%      | 6.6%      | 4.3×      |
| 512   | 0.40    | 18.7        | 6.1%      | 5.5%      | 46.8×     |
| 1024  | 5.42    | 1976        | 1.5%      | 5.6%      | 364×      |
| 2048  | 2.16    | 4.15 × 10⁷  | 1.5%      | 6.2%      | 19.3 × 10⁶× |

### Suite B — Gradient flow uniformity
**MHD-MAGIC**: constant axis-line sums → balanced fan-in → uniform gradient propagation.
`flow_ratio = gnorm[input_blocks] / gnorm[output_blocks]`. Ideal = 1.0.

| Depth | MHD ratio | MHD cv | Kaiming ratio | Kaiming cv |
|-------|-----------|--------|---------------|------------|
| 64    | 0.949     | 0.242  | 0.465         | 0.233      |
| 256   | 1.225     | 0.263  | 0.196         | 0.373      |
| 512   | 1.506     | 0.299  | 0.142         | 0.442      |
| 1024  | 2.646     | 0.413  | 0.131         | 0.485      |
| 2048  | 2.278     | 0.503  | 0.115         | 0.420      |

MHD ratio stays near or above 1 at all depths.
Kaiming ratio decays monotonically to 0.115 at depth 2048.

### Suite C — Generalisation accuracy
**MHD-ETK** (honest null result): both MHD and Kaiming reach ≥ 99.8% on the
8-class Gaussian task at depths 4–64. The task is separable; this suite is a
regression test confirming training converges, not a claim of accuracy advantage.

---

## PMPS V4.1 benchmark (`bench_mhd_pmps.py`)

PyTorch. Uses the fixed PMPS oracle (`procedural_manifold/oracle.py`).
Four suites, all numbers from this codebase.

### Oracle fixes (V4.1)
Two bugs found during integration and fixed:

| Bug | Symptom | Fix |
|-----|---------|-----|
| **BUG-1**: `_global_std = 1.0` (identity normalisation) | Weights 3× too small | Per-block normalise T(y) to unit std before Kaiming scaling |
| **BUG-3**: per-tile independent normalisation | Tiles assembled inconsistently | Full-block stats computed once, passed to all tiles |

After fix: `mean = 0.000000, std = 0.125000` for fan_in=128 (exact Kaiming).

### Suite D — Delta pruning post-training
4-layer RegenLinear, 300 epochs, Adam lr=1e-3. Baseline loss = 0.003409.

| Pruned | Loss     | Δloss      | Rel change |
|--------|---------|------------|------------|
| 10%    | 0.003553 | +0.000144  | +4.2%      |
| 25%    | 0.007017 | +0.003608  | +106%      |
| 50%    | 0.028434 | +0.025025  | +734%      |
| 95%    | 0.387923 | +0.384514  | +11 279%   |

Safe pruning threshold: **~10%**. The deltas carry essential task-specific
signal; the oracle provides structural routing but not full task representation.

### Suite C (PMPS) — Manifold Projection effectiveness
Centering schedule vs loss and mean delta drift.

| Schedule       | Final loss | Delta drift |
|----------------|-----------|-------------|
| No centering   | 0.004557  | 0.000742    |
| Every step     | 0.004465  | 0.000000    |
| Every 100 steps| 0.004823  | 0.000000    |

Lazy centering (every 100 steps) fully suppresses drift. Loss overhead: negligible.

### Suite E — LR robustness
SGD, 80 epochs, no weight decay. Dense = 3-layer Linear + LayerNorm + ReLU.

| LR   | Dense + LayerNorm     | PMPS V4.1       |
|------|-----------------------|-----------------|
| 0.01 | 0.0583 (stable)       | 0.1071 (stable) |
| 0.05 | 0.0725 (stable)       | 0.0701 (stable) |
| 0.10 | **diverges**          | 0.0633 (stable) |
| 0.50 | **diverges**          | 0.0733 (stable) |

PMPS V4.1 stable at lr ≥ 0.1 where Dense+LayerNorm diverges catastrophically.

---

## Using `mhd_dynamic.py` in your own code

### NumPy (no torch)

```python
from mhd_dynamic import init_weights, MHDRegulariser

W, start_rank = init_weights(fan_in=256, fan_out=256)
layer.W            = W
layer.mhd_start_rank = start_rank

reg = MHDRegulariser(strength=0.005, update_freq=4)
reg_loss = reg.apply(model.layers)   # injects gradient into layer.dW
```

### PyTorch

```python
from mhd_dynamic import init_weights_torch, MHDRegulariserTorch
import torch.nn as nn

model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
model.apply(init_weights_torch)

reg = MHDRegulariserTorch(strength=0.005)
# in training loop:
loss = criterion(outputs, targets) + reg(model)
loss.backward()
```

### PMPS: training with RegenLinear, inference with StreamingRegenLinear

```python
from procedural_manifold import MHDOracle, RegenLinear, StreamingRegenLinear
from procedural_manifold.delta_knowledge import DeltaExtractor

oracle = MHDOracle(n=5, d=10)

# Training: use RegenLinear (has nn.Parameter, differentiable)
layer = RegenLinear(256, 256, oracle, start_rank=0)
# ... train normally ...

# Extract deltas after training
knowledge = DeltaExtractor(model).extract()

# Inference: StreamingRegenLinear (COO dict, no stored weights)
stream_layer = StreamingRegenLinear(256, 256, oracle, start_rank=0)
stream_layer.set_deltas(rows, cols, vals)
```

---

## Theorem → code mapping

| Theorem | Status | Code location |
|---------|--------|---------------|
| MHD-STRUCT | PROVEN | `oracle.py` — A_magic digit transform |
| MHD-MAGIC | PROVEN | `oracle.py` per-block centering; `mhd_dynamic.py` V1 check |
| MHD-SPECTRAL | PROVEN | `oracle.py` T(y) projection; Suite A stability |
| MHD-OA-MAX | PROVEN | `mhd_dynamic.py` contiguous `arange` routing; `oracle.py` same |
| MHD-ETK | PROVEN | Suite C scope note (honest null result) |
| MHD-STABILITY-CONJECTURE | **CONJECTURE** | `MHDRegulariser`, `center_deltas()` |
| MHD-GLOBAL-CONCENTRATION | **CONJECTURE** | Suite A depth ≥ 512 (numerical evidence) |

---

## Architecture note: training vs inference (PMPS)

`StreamingRegenLinear` uses a COO sparse dictionary — **no `nn.Parameter`**.
A `RegenTransformer` built from streaming layers has zero trainable parameters
and cannot be trained with standard optimizers.

Workflow:
1. **Train** with `RegenLinear` (dense delta matrix, fully differentiable)
2. **Extract** deltas with `DeltaExtractor`
3. **Deploy** with `StreamingRegenLinear` (regenerates base on-the-fly, applies sparse deltas)

The CIFAR-10/100 accuracy figures from collaborator reports require this
training-then-conversion workflow and have not yet been reproduced in this
codebase. They are pending verification.

---

## File structure

```
examples/mhd_dynamic/
├── mhd_dynamic.py              flu-math oracle, init_weights, MHDRegulariser
├── bench_mhd_dynamic.py        Suites A/B/C (NumPy only, no torch)
├── bench_mhd_pmps.py           Suites D/C/E/F (PyTorch, PMPS)
├── procedural_manifold/
│   ├── oracle.py               MHD oracle (BUG-1 + BUG-3 fixed)
│   ├── regen_linear.py         Dense delta layer (train)
│   ├── streaming_regen.py      COO streaming layer (inference)
│   ├── transformer.py          V4.1: SRS + Leaky ReLU
│   ├── attention.py            Multi-head attention
│   ├── model.py                KinshipMoE
│   ├── builder.py              Config-driven build
│   ├── benchmarks.py           Benchmark functions
│   ├── delta_knowledge.py      DeltaExtractor, Comparer, Registry
│   ├── delta_stream.py         Serialisation
│   ├── agnostic_converter.py   safetensors → delta format
│   ├── agnostic_regen.py       Patch existing model
│   ├── token_generator.py      Kuramoto phase sampling
│   ├── transducer.py           Ternary symbolic transducer
│   ├── kinship_vision_hybrid.py
│   ├── inner_communion.py
│   ├── demo.py
│   └── hooks.py / _compat.py / contract.py / __init__.py
├── configs/
│   ├── pure_transformer.yaml
│   └── with_transducer.yaml
└── phoneme_mappings/
    └── english.json
```

---

## Honest scope

**PROVEN** (algebraic + certificates in `tests/benchmarks/bench_mhd_family.py`):
- Oracle is zero-mean over the full MHD cube (MHD-MAGIC)
- Contiguous rank blocks have tighter distribution than random (MHD-OA-MAX)
- Activation variance bounded at depth 2048, no BatchNorm (Suite A, empirical)
- Gradient flow ratio near unity vs Kaiming attenuation (Suite B, empirical)
- PMPS stable at lr ≥ 0.1 where Dense+LayerNorm diverges (Suite E, empirical)
- Manifold Projection suppresses drift with negligible overhead (Suite C, empirical)

**CONJECTURE** (§26, `docs/OPEN_DEBT.md`):
- Global concentration of discrepancy supremum (MHD-GLOBAL-CONCENTRATION)
- Dynamical isometry maintained throughout training (MHD-STABILITY-CONJECTURE)
- Safe pruning threshold rises with model scale (untested at scale)
- CIFAR-10/100 accuracy figures (pending reproduction)

---

## Version history

| Version | Date | Notes |
|---------|------|-------|
| V15.5.0-pre | 2026-05-30 | MHD-Dynamic: oracle, init, three-suite bench |
| V15.5.0 | 2026-05-31 | PMPS V4.1: BUG-1/BUG-3 fixed, center_deltas, SRS, Leaky ReLU, honest benchmarks |

MIT License — Sovereign Peerage Open Source
Authors: Felix Mönnich & The Kinship Mesh Collective

---

## Twin Experiment (`bench_mhd_twins.py`)

**V4.2 addition.** Pure NumPy, no torch required.

Three-phase experiment testing whether independently trained PMPS networks
converge to the same sparse delta topologies and values.

```bash
python bench_mhd_twins.py                          # all phases, default L1 sweep
python bench_mhd_twins.py --phase 3 --l1 0.010    # single config
python bench_mhd_twins.py --centering-comparison   # full vs masked vs none
```

### Results

**Phase 1 — Deterministic:** bit-identical networks under full-batch GD.
Jaccard=1, corr=1, bit_identical=True.

**Phase 2 — Stochastic:** same topology (no pruning), corr=0.9996 under
different mini-batch seeds.

**Phase 3 — Sparse (masked centering + proximal SGD):**

| L1    | Active A | Active B | Jaccard | Corr   | Loss  |
|-------|---------|---------|---------|--------|-------|
| 0.005 | 403     | 492     | 0.504   | 0.9999 | 0.040 |
| 0.010 | 348     | 437     | 0.567   | 0.9997 | 0.078 |
| 0.020 | 274     | 294     | 0.552   | 0.9971 | 0.262 |
| 0.050 | 103     | 111     | 0.476   | 0.9999 | 0.375 |

**Key finding:** topology is flexible (Jaccard 0.41–0.57), values are canonical
(corr ≥ 0.997). The MHD manifold defines a grammar: multiple sparse sub-networks
solve the same task, but when two networks both modify a given weight, they
converge to essentially the same value.

### Critical implementation note

Full centering (`mode='full'`) reawakens pruned-to-zero weights by shifting
zeros to −mean(Δ). **Always use `center_deltas(mode='masked')` alongside L1
sparsity.** This is now the default in V4.2.

### sigma_T closed form (V4.2)

The theoretical standard deviation of T(y) over the full oracle period is:

```
σ_T = sqrt(d · (n² − 1) / (48 · n²))
```

For n=5, d=10: σ_T = 0.4472. Exposed as `oracle.sigma_T()`. Verified
empirically to < 0.002 error.
