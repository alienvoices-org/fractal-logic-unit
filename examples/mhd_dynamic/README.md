# examples/mhd_dynamic — MHD-Dynamic Neural Initialisation

Ultra-deep network stabilisation via the Magic Hypercube Digital Net.

This directory contains the neural-application layer of the FLU framework,
demonstrating how the algebraic properties of the MHD generator translate
directly into measurable training stability.

---

## Files

| File | Purpose |
|------|---------|
| `mhd_dynamic.py` | Core module: oracle, `init_weights()`, `MHDRegulariser`, optional Torch wrappers |
| `bench_mhd_dynamic.py` | Three-suite benchmark (stability · gradient flow · accuracy) |

`mhd_dynamic.py` has **no dependency beyond numpy and flu-math**.  The Torch
wrappers are gated by a `try/except` and degrade gracefully to stubs.

---

## Quick start

```bash
pip install flu-math numpy          # required
pip install torch torchvision       # optional — for Torch wrappers only

# Run self-test (oracle verification + sample init)
python mhd_dynamic.py

# Full benchmark — all three suites, default settings (~95s)
python bench_mhd_dynamic.py

# Quick smoke run (~30s)
python bench_mhd_dynamic.py --suite a b c --depths 64 512 2048 --blocks 4 32 --epochs 20 --seeds 2

# Individual suites
python bench_mhd_dynamic.py --suite a                          # stability only
python bench_mhd_dynamic.py --suite b --depths 64 256 512 1024 # gradient flow
python bench_mhd_dynamic.py --suite c --blocks 4 32 64         # accuracy
```

---

## What the benchmark measures

### Suite A — Forward-pass stability  (`--suite a`)
**Theorem: MHD-SPECTRAL** (rank-1 Walsh dual suppresses high-frequency interference)

Activation variance at init, no BatchNorm, no training.
At depth ≥ 512 Kaiming variance diverges; MHD stays bounded.

```
Depth  | MHD var  | Kaiming var | Advantage
   512 |    0.40  |       18.2  |     46×
  1024 |    8.4   |     1593    |    191×
  2048 |    1.6   |   26 000 000|    16M×
```

### Suite B — Gradient flow uniformity  (`--suite b`)
**Theorem: MHD-MAGIC** (constant axis-line sums → balanced fan-in)

Per-block gradient norm profile after one forward+backward pass.
`flow_ratio = gnorm[input_layer] / gnorm[output_layer]`.  Ideal = 1.0.
MHD stays near 1 across all depths; Kaiming attenuates toward 0.12 by depth 2048.

### Suite C — Generalisation accuracy  (`--suite c`)
**Theorem: MHD-ETK** (discrepancy = 1-D Fourier series, low-frequency concentration)

Full SGD training on 8-class Gaussian blobs (32-D, σ = 1.8 noise), no BatchNorm.
Both methods reach ≥ 99.8% on this task at depths 4–64 — **this is an honest
null result**: separable synthetic data is too easy to separate MHD from Kaiming
on accuracy alone.  The suite exists as a regression test confirming training
converges at all depths, not as a claim of accuracy advantage.

For harder benchmarks (ImageNet, LLM pretraining) see Attachment 3 /
`examples/mhd_dynamic/bench_mhd_resnet.py` (PyTorch, GPU required, not yet
integrated).

---

## Using `mhd_dynamic.py` in your own code

### NumPy (no torch)

```python
from mhd_dynamic import init_weights, MHDRegulariser

# Initialise a layer
W, start_rank = init_weights(fan_in=256, fan_out=256)
layer.W            = W
layer.mhd_start_rank = start_rank   # needed by regulariser

# During training
reg = MHDRegulariser(strength=0.005, update_freq=4)
reg_loss = reg.apply(model.layers)  # injects gradient into layer.dW
```

### PyTorch

```python
from mhd_dynamic import init_weights_torch, MHDRegulariserTorch
import torch.nn as nn

model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
model.apply(init_weights_torch)                    # MHD init in-place

reg = MHDRegulariserTorch(strength=0.005)
# in training loop:
loss = criterion(outputs, targets) + reg(model)
loss.backward()
```

---

## Theorem → code mapping

| Theorem | Proof doc ref | Code |
|---------|--------------|------|
| MHD-STRUCT | §3 | `mhd_oracle_numpy()` — A_magic matrix |
| MHD-MAGIC | §6 | `verify_oracle()` V1, `init_weights()` mean-centering |
| MHD-SPECTRAL | §15 | `mhd_oracle_numpy()` T(y) projection, Suite A |
| MHD-OA-MAX | §11 | `init_weights()` contiguous `arange` routing, `verify_oracle()` V4 |
| MHD-ETK | §20 | Suite C scope note |
| MHD-STABILITY-CONJECTURE | §26 | `MHDRegulariser` — **CONJECTURE** |
| MHD-GLOBAL-CONCENTRATION | §18 | Suite A depth ≥ 512 — **CONJECTURE** (numerical evidence) |

---

## Honest scope

The results here validate **initialisation stability** rigorously.

Claims that are **PROVEN** (directly from the proof doc):
- Oracle is zero-mean over the full MHD cube (MHD-MAGIC)
- Contiguous rank blocks have tighter value distribution than random blocks (MHD-OA-MAX)
- Activation variance stays bounded at depth 2048, no BatchNorm (empirical, Suite A)
- Gradient flow ratio stays near 1.0 vs Kaiming's monotonic decay (empirical, Suite B)

Claims that remain **CONJECTURE** (§26, OPEN_DEBT.md):
- Global concentration of the discrepancy supremum (MHD-GLOBAL-CONCENTRATION)
- Dynamical isometry maintained throughout training (MHD-STABILITY-CONJECTURE)
- Optimal learning-rate scaling with spectral radius of A_magic

---

## Version history

| Version | Date | Notes |
|---------|------|-------|
| V15.5.0-pre | 2026-05-30 | Initial integration; bench + core module |

---

*Part of the Fractal Logic Unit (FLU) framework.*
*Authors: Felix Mönnich & The Kinship Mesh Collective*
