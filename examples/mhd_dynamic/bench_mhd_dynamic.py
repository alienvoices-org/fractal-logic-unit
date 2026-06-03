#!/usr/bin/env python3
"""
MHD-Dynamic Initialization & Training Benchmark
================================================
Standalone NumPy benchmark for the Magic Hypercube Digital Net (MHD).
Tests three distinct phenomena, each tied to a specific theorem:

  Suite A — Forward-pass stability    (MHD-SPECTRAL / rank-1 Walsh dual)
  Suite B — Gradient flow uniformity  (MHD-MAGIC / constant line sums)
  Suite C — Generalisation accuracy   (MHD-ETK / low-frequency concentration)

Requires only: numpy, flu-math >= 15.4.0
No PyTorch, no GPU.

Author  : Felix Mönnich & The Kinship Mesh Collective
Date    : 2026-05-30
Version : V15.5.0-pre
"""

import argparse
import time
import sys
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ORACLE + INIT  — imported from mhd_dynamic.py
# ═══════════════════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhd_dynamic import (
    mhd_oracle_numpy as mhd_oracle,
    init_weights     as _init_weights_base,
    kaiming_init,
    verify_oracle,
)

def init_mhd(fan_in: int, fan_out: int, n: int = 5, d: int = 10):
    """Thin wrapper — returns (W, start_rank) matching benchmark internal API."""
    return _init_weights_base(fan_in, fan_out, n=n, d=d)

init_kaiming = kaiming_init   # alias matching benchmark internal usage


def _check_oracle(n: int = 5, d: int = 3) -> None:
    vals = mhd_oracle(np.arange(n**d, dtype=np.int64), n=n, d=d)
    assert abs(float(vals.mean())) < 1e-6, "Oracle balance FAIL (MHD-MAGIC violated)"


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  NETWORK PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

_relu   = lambda x: np.maximum(0.0, x)
_drelu  = lambda z: (z > 0).astype(np.float64)
_smax   = lambda x: (lambda e: e / e.sum(1, keepdims=True))(
               np.exp(x - x.max(1, keepdims=True)))


class ResBlock:
    """
    Pre-activation residual block with full autodiff.
    out = ReLU( x  +  scale · (ReLU(x @ W1 + b1) @ W2 + b2) )
    """
    def __init__(self, dim: int, mode: str, scale: float = 0.1,
                 n: int = 5, d: int = 10):
        self.scale = scale
        if mode == 'mhd':
            self.W1, _ = init_mhd(dim, dim, n, d)
            self.W2, _ = init_mhd(dim, dim, n, d)
        else:
            self.W1 = init_kaiming(dim, dim)
            self.W2 = init_kaiming(dim, dim)
        self.b1 = np.zeros(dim);  self.b2 = np.zeros(dim)
        self.dW1 = self.dW2 = self.db1 = self.db2 = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x   = x
        self.z1  = x @ self.W1 + self.b1
        self.a1  = _relu(self.z1)
        self.z2  = self.a1 @ self.W2 + self.b2
        self.out = _relu(x + self.scale * self.z2)
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        d_sum       = dout * _drelu(self.x + self.scale * self.z2)
        dz2         = self.scale * d_sum
        self.dW2    = self.a1.T @ dz2
        self.db2    = dz2.sum(0)
        dz1         = (dz2 @ self.W2.T) * _drelu(self.z1)
        self.dW1    = self.x.T @ dz1
        self.db1    = dz1.sum(0)
        return (dz1 @ self.W1.T) + d_sum

    def step(self, lr: float) -> None:
        self.W1 -= lr * self.dW1;  self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2;  self.b2 -= lr * self.db2

    @property
    def layer_gnorm(self) -> float:
        return float(((self.dW1 ** 2).sum() + (self.dW2 ** 2).sum()) ** 0.5)

    @property
    def dead_frac(self) -> float:
        return float(np.mean(self.out == 0.0))

    @property
    def act_var(self) -> float:
        return float(np.var(self.out))


class ResMLP:
    """Stem → n_blocks × ResBlock → Head, with full autodiff."""

    def __init__(self, in_dim: int, hidden: int, n_blocks: int,
                 n_classes: int, mode: str, scale: float = 0.1,
                 n: int = 5, d: int = 10):
        self.blocks = [ResBlock(hidden, mode, scale, n, d)
                       for _ in range(n_blocks)]
        if mode == 'mhd':
            self.W_in,  _ = init_mhd(in_dim, hidden, n, d)
            self.W_out, _ = init_mhd(hidden, n_classes, n, d)
        else:
            self.W_in  = init_kaiming(in_dim, hidden)
            self.W_out = init_kaiming(hidden, n_classes)
        self.b_in  = np.zeros(hidden)
        self.b_out = np.zeros(n_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_raw = x
        self.z_in  = x @ self.W_in + self.b_in
        h = _relu(self.z_in)
        for blk in self.blocks:
            h = blk.forward(h)
        self.pre_out = h
        return h @ self.W_out + self.b_out

    def backward(self, dl: np.ndarray) -> None:
        """dl should already be divided by batch size."""
        self.dW_out = self.pre_out.T @ dl
        self.db_out = dl.sum(0)
        da = dl @ self.W_out.T
        for blk in reversed(self.blocks):
            da = blk.backward(da)
        dh = da * _drelu(self.z_in)
        self.dW_in = self.x_raw.T @ dh
        self.db_in = dh.sum(0)

    def step(self, lr: float) -> None:
        self.W_in  -= lr * self.dW_in;   self.b_in  -= lr * self.db_in
        self.W_out -= lr * self.dW_out;  self.b_out -= lr * self.db_out
        for blk in self.blocks:
            blk.step(lr)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=1)

    def total_gnorm(self) -> float:
        gn = (self.dW_in**2).sum() + (self.dW_out**2).sum()
        gn += sum(b.layer_gnorm ** 2 for b in self.blocks)
        return float(gn ** 0.5)

    def per_block_gnorms(self) -> list:
        """Block 0 = nearest input, last = nearest output."""
        return [b.layer_gnorm for b in self.blocks]

    def per_block_act_vars(self) -> list:
        return [b.act_var for b in self.blocks]

    def mean_dead_frac(self) -> float:
        return float(np.mean([b.dead_frac for b in self.blocks]))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DATA
# ═══════════════════════════════════════════════════════════════════════════════

def make_gaussian_blobs(N: int = 1600, D: int = 32, C: int = 8,
                        noise: float = 1.8, seed: int = 42):
    """
    C-class Gaussian clusters in R^D.  Shuffled before split so every
    class appears in both train and test sets.
    """
    np.random.seed(seed)
    centers = np.random.randn(C, D) * 3.5
    X = np.vstack([centers[c] + np.random.randn(N // C, D) * noise
                   for c in range(C)])
    y = np.repeat(np.arange(C), N // C)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)         # standardise
    idx = np.random.permutation(N);  X, y = X[idx], y[idx]
    s = int(0.75 * N)
    return X[:s], y[:s], X[s:], y[s:]


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_run(mode: str, n_blocks: int, n_epochs: int, lr: float,
                  batch_size: int, data_seed: int, init_seed: int,
                  D: int = 32, H: int = 64, C: int = 8,
                  n: int = 5, d_oracle: int = 10, scale: float = 0.1):
    """
    One full training run.  Returns a dict with per-epoch curves
    and per-block gradient norm profile from the final epoch.
    """
    X_tr, y_tr, X_te, y_te = make_gaussian_blobs(seed=data_seed, D=D, C=C)
    np.random.seed(init_seed)
    net = ResMLP(D, H, n_blocks, C, mode, scale=scale, n=n, d=d_oracle)

    train_losses, test_accs, total_gnorms = [], [], []
    final_block_gnorms = None
    final_act_vars     = None

    for ep in range(n_epochs):
        idx = np.random.permutation(len(X_tr))
        ep_loss = ep_gn = n_batches = 0

        for i in range(0, len(X_tr), batch_size):
            xb = X_tr[idx[i: i + batch_size]]
            yb = y_tr[idx[i: i + batch_size]]
            N  = len(yb)

            logits = net.forward(xb)
            probs  = _smax(logits)
            loss   = -np.log(probs[np.arange(N), yb] + 1e-12).mean()

            dl = probs.copy();  dl[np.arange(N), yb] -= 1;  dl /= N
            net.backward(dl)
            net.step(lr)

            ep_loss   += loss
            ep_gn     += net.total_gnorm()
            n_batches += 1

        acc = np.mean(net.predict(X_te) == y_te)
        train_losses.append(ep_loss / n_batches)
        test_accs.append(float(acc))
        total_gnorms.append(ep_gn / n_batches)

    # Run one more forward+backward to get per-block diagnostics
    xb = X_tr[:batch_size];  yb = y_tr[:batch_size];  N = len(yb)
    logits = net.forward(xb);  probs = _smax(logits)
    dl = probs.copy();  dl[np.arange(N), yb] -= 1;  dl /= N
    net.backward(dl)

    final_block_gnorms = net.per_block_gnorms()
    final_act_vars     = net.per_block_act_vars()
    final_dead         = net.mean_dead_frac()

    return {
        'train_losses'     : train_losses,
        'test_accs'        : test_accs,
        'total_gnorms'     : total_gnorms,
        'block_gnorms'     : final_block_gnorms,   # list len = n_blocks
        'block_act_vars'   : final_act_vars,
        'final_dead'       : final_dead,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SUITE A — FORWARD-PASS STABILITY   (MHD-SPECTRAL)
# ═══════════════════════════════════════════════════════════════════════════════

def suite_a_stability(depths, dim, n_seeds, n_oracle, d_oracle, scale, verbose):
    """
    Activation variance and dead-ReLU fraction at init (no training).
    Replicates the original bench_mhd_dynamic.py result in a unified script.
    """
    if verbose:
        hdr = (f"{'Depth':>8} | {'MHD var':>12} | {'Kai var':>12} | "
               f"{'MHD dead%':>10} | {'Kai dead%':>10} | {'Advantage':>10}")
        print(hdr);  print("─" * len(hdr))

    rows = {}
    for depth in depths:
        mhd_vars, kai_vars, mhd_d, kai_d = [], [], [], []
        for seed in range(n_seeds):
            np.random.seed(seed * 137 + depth)
            x = np.random.randn(32, dim)

            # MHD
            xm, ok = x.copy(), True
            for _ in range(depth // 2):
                W1, _ = init_mhd(dim, dim, n_oracle, d_oracle)
                W2, _ = init_mhd(dim, dim, n_oracle, d_oracle)
                h = _relu(xm @ W1)
                xm = _relu(xm + scale * (h @ W2))
                if not np.isfinite(xm).all(): ok = False; break
            mhd_vars.append(float(np.var(xm)) if ok else float('nan'))
            mhd_d.append(float(np.mean(xm == 0)) if ok else 1.0)

            # Kaiming
            xk, ok = x.copy(), True
            for _ in range(depth // 2):
                W1 = init_kaiming(dim, dim);  W2 = init_kaiming(dim, dim)
                h = _relu(xk @ W1)
                xk = _relu(xk + scale * (h @ W2))
                if not np.isfinite(xk).all(): ok = False; break
            kai_vars.append(float(np.var(xk)) if ok else float('nan'))
            kai_d.append(float(np.mean(xk == 0)) if ok else 1.0)

        mv = np.nanmean(mhd_vars);  kv = np.nanmean(kai_vars)
        adv = kv / mv if (np.isfinite(mv) and np.isfinite(kv) and mv > 1e-12) else float('nan')

        rows[depth] = dict(mhd_var=mv, kai_var=kv, advantage=adv,
                           mhd_dead=np.nanmean(mhd_d), kai_dead=np.nanmean(kai_d))
        if verbose:
            def _fv(v):
                if not np.isfinite(v): return "         NaN"
                return f"{v:>12.4f}" if v < 1e5 else f"{v:>12.2e}"
            print(f"{depth:>8} | {_fv(mv)} | {_fv(kv)} | "
                  f"{np.nanmean(mhd_d):>9.1%} | {np.nanmean(kai_d):>9.1%} | "
                  f"{adv:>9.1f}x")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SUITE B — GRADIENT FLOW UNIFORMITY   (MHD-MAGIC)
# ═══════════════════════════════════════════════════════════════════════════════

def suite_b_gradient_flow(depths, dim, n_seeds, n_oracle, d_oracle, scale, verbose):
    """
    Measures per-block gradient norm profile after one forward+backward pass.

    Healthy gradient flow: norms roughly constant from output to input.
    Vanishing:             norms decay exponentially toward input.

    Reports:
      flow_ratio = gnorm[input_end] / gnorm[output_end]
        ~1.0 → uniform (ideal)
        <<1  → vanishing gradient toward input
        >>1  → exploding gradient toward input
      flow_cv = std(block_gnorms) / mean(block_gnorms)
        ~0   → perfectly uniform
    """
    if verbose:
        hdr = (f"{'Depth':>8} | {'MHD ratio':>10} | {'MHD cv':>8} | "
               f"{'Kai ratio':>10} | {'Kai cv':>8} | {'Notes'}")
        print(hdr);  print("─" * len(hdr))

    rows = {}
    for depth in depths:
        mhd_ratios, kai_ratios, mhd_cvs, kai_cvs = [], [], [], []
        for seed in range(n_seeds):
            np.random.seed(seed * 137 + depth + 1000)
            x_init = np.random.randn(64, dim)
            y_init = np.random.randint(0, 8, 64)

            for mode, r_list, cv_list in [
                ('mhd',     mhd_ratios, mhd_cvs),
                ('kaiming', kai_ratios, kai_cvs),
            ]:
                blocks = [ResBlock(dim, mode, scale, n_oracle, d_oracle)
                          for _ in range(depth // 2)]
                if mode == 'mhd':
                    W_in,  _ = init_mhd(dim, dim, n_oracle, d_oracle)
                    W_out, _ = init_mhd(dim, 8,   n_oracle, d_oracle)
                else:
                    W_in  = init_kaiming(dim, dim)
                    W_out = init_kaiming(dim, 8)

                h = _relu(x_init @ W_in)
                for blk in blocks: h = blk.forward(h)
                logits = h @ W_out
                probs  = _smax(logits)
                dl = probs.copy(); dl[np.arange(64), y_init] -= 1; dl /= 64
                da = dl @ W_out.T
                for blk in reversed(blocks): da = blk.backward(da)

                gnorms = [blk.layer_gnorm for blk in blocks]
                if len(gnorms) < 2 or gnorms[0] < 1e-15:
                    r_list.append(float('nan'));  cv_list.append(float('nan'))
                    continue
                # ratio: norm near input / norm near output (blocks[0]=input, [-1]=output)
                ratio = gnorms[0] / gnorms[-1]
                cv    = np.std(gnorms) / (np.mean(gnorms) + 1e-12)
                r_list.append(ratio);  cv_list.append(cv)

        mr = np.nanmean(mhd_ratios);   kr = np.nanmean(kai_ratios)
        mc = np.nanmean(mhd_cvs);      kc = np.nanmean(kai_cvs)

        # A ratio of 1 is ideal; lower = Kaiming vanishes toward input
        note = ""
        if np.isfinite(mr) and np.isfinite(kr):
            if mr > 0.7 and kr < 0.5: note = "MHD more uniform"
            elif abs(mr - 1.0) < abs(kr - 1.0): note = "MHD closer to ideal"

        rows[depth] = dict(mhd_ratio=mr, kai_ratio=kr, mhd_cv=mc, kai_cv=kc)
        if verbose:
            def _ff(v): return f"{v:>9.3f}" if np.isfinite(v) else "      NaN"
            print(f"{depth:>8} | {_ff(mr)} | {_ff(mc)} | {_ff(kr)} | {_ff(kc)} | {note}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  SUITE C — GENERALISATION ACCURACY   (MHD-ETK)
# ═══════════════════════════════════════════════════════════════════════════════

def suite_c_generalisation(configs, n_epochs, lr, batch_size,
                            n_seeds, n_oracle, d_oracle, scale, verbose):
    """
    Full training runs: loss curves, test accuracy curves, gradient norm curves.

    configs = list of (n_blocks, label) tuples.
    Reports final test accuracy and convergence speed (epoch to reach 80% acc).
    """
    D, H, C = 32, 64, 8

    if verbose:
        hdr = (f"{'Config':>20} | {'MHD acc':>9} | {'Kai acc':>9} | "
               f"{'MHD ep→80%':>11} | {'Kai ep→80%':>11} | "
               f"{'MHD gnorm':>10} | {'Kai gnorm':>10}")
        print(hdr);  print("─" * len(hdr))

    rows = {}
    for n_blocks, label in configs:
        mhd_accs, kai_accs = [], []
        mhd_ep80, kai_ep80 = [], []
        mhd_gn,   kai_gn   = [], []
        mhd_curves, kai_curves = [], []

        for seed in range(n_seeds):
            for mode, acc_list, ep80_list, gn_list, curves in [
                ('mhd',     mhd_accs, mhd_ep80, mhd_gn, mhd_curves),
                ('kaiming', kai_accs, kai_ep80, kai_gn, kai_curves),
            ]:
                r = train_one_run(
                    mode=mode, n_blocks=n_blocks, n_epochs=n_epochs,
                    lr=lr, batch_size=batch_size,
                    data_seed=seed * 13 + 1, init_seed=seed * 7,
                    D=D, H=H, C=C, n=n_oracle, d_oracle=d_oracle, scale=scale,
                )
                acc_list.append(r['test_accs'][-1])
                gn_list.append(r['total_gnorms'][-1])
                curves.append(r['test_accs'])
                # Epoch at which 80% first exceeded (-1 if never)
                over80 = [i for i, a in enumerate(r['test_accs']) if a >= 0.80]
                ep80_list.append(over80[0] + 1 if over80 else -1)

        mhd_m = np.mean(mhd_accs);  mhd_s = np.std(mhd_accs)
        kai_m = np.mean(kai_accs);  kai_s = np.std(kai_accs)
        mhd_e80 = np.mean([e for e in mhd_ep80 if e > 0]) if any(e > 0 for e in mhd_ep80) else float('nan')
        kai_e80 = np.mean([e for e in kai_ep80 if e > 0]) if any(e > 0 for e in kai_ep80) else float('nan')

        rows[(n_blocks, label)] = dict(
            mhd_acc=mhd_m, mhd_acc_std=mhd_s,
            kai_acc=kai_m, kai_acc_std=kai_s,
            mhd_ep80=mhd_e80, kai_ep80=kai_e80,
            mhd_gn=np.mean(mhd_gn), kai_gn=np.mean(kai_gn),
            mhd_curves=mhd_curves, kai_curves=kai_curves,
        )

        if verbose:
            def _fa(m, s): return f"{m:.1%}±{s:.1%}"
            def _fe(e): return f"{e:>6.1f}" if np.isfinite(e) else "  never"
            cfg_str = f"{label}({n_blocks}blk)"
            print(f"{cfg_str:>20} | {_fa(mhd_m,mhd_s):>9} | {_fa(kai_m,kai_s):>9} | "
                  f"{_fe(mhd_e80):>11} | {_fe(kai_e80):>11} | "
                  f"{np.mean(mhd_gn):>10.4f} | {np.mean(kai_gn):>10.4f}")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  ORACLE SELF-VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_oracle_verification(n: int = 5, d: int = 3) -> int:
    """5 checks; returns number of failures."""
    print("\n── Oracle self-verification ────────────────────────────────────")
    errors = 0

    # V1  MHD-MAGIC: full-cube T(y) mean = 0
    all_r = np.arange(n**d, dtype=np.int64)
    vals  = mhd_oracle(all_r, n=n, d=d)
    m     = abs(float(vals.mean()))
    ok    = m < 1e-6
    print(f"  V1 MHD-MAGIC    full-cube T(y) mean = {m:.2e}  {'✓' if ok else '✗ FAIL'}")
    errors += 0 if ok else 1

    # V2  MHD-STRUCT: oracle(rank=0) matches closed-form expectation
    r0          = float(mhd_oracle(np.array([0], dtype=np.int64), n=n, d=d)[0])
    expected_r0 = float(((n - 1) - n // 2) * ((-1) ** (d - 1))) / (n * 2.0)
    ok          = abs(r0 - expected_r0) < 1e-5
    print(f"  V2 MHD-STRUCT   oracle(0)={r0:.4f} expected={expected_r0:.4f}  "
          f"{'✓' if ok else '✗ FAIL'}")
    errors += 0 if ok else 1

    # V3  MHD-SPECTRAL: values bounded within d/n
    bound = float(d) / float(n)
    lo, hi = float(vals.min()), float(vals.max())
    ok = lo >= -bound - 1e-4 and hi <= bound + 1e-4
    print(f"  V3 MHD-SPECTRAL range=[{lo:.3f},{hi:.3f}] ⊆ [{-bound:.3f},{bound:.3f}]  "
          f"{'✓' if ok else '✗ FAIL'}")
    errors += 0 if ok else 1

    # V4  arange tighter than randint (contiguous OA block property)
    # Must use d_full >= 8 so the cube is large enough (n^d >> block_size).
    # d=3 gives n^3=125 < block_size=512, so blocks wrap and look random.
    d_full = max(d, 8)
    stds_a, stds_r = [], []
    for _ in range(50):
        start = int(np.random.randint(0, n ** (d_full - 1)))
        N_blk = 512
        ra = (start + np.arange(N_blk, dtype=np.int64)) % (n ** d_full)
        rr = np.random.randint(0, n ** d_full, N_blk, dtype=np.int64)
        stds_a.append(float(mhd_oracle(ra, n=n, d=d_full).std()))
        stds_r.append(float(mhd_oracle(rr, n=n, d=d_full).std()))
    ok = np.mean(stds_a) < np.mean(stds_r)
    print(f"  V4 OA-balance   arange_std={np.mean(stds_a):.4f} < randint_std={np.mean(stds_r):.4f}  "
          f"(d={d_full})  {'✓' if ok else '✗ FAIL'}")
    errors += 0 if ok else 1

    # V5  Non-degenerate (multiple distinct values)
    unique_vals = len(np.unique(np.round(vals * 100).astype(int)))
    ok = unique_vals > 2
    print(f"  V5 MHD-GEN      unique value classes = {unique_vals}  "
          f"{'✓' if ok else '✗ FAIL (degenerate)'}")
    errors += 0 if ok else 1

    print(f"\n  {'ALL PASS ✓' if errors == 0 else str(errors) + ' FAILURE(S) ✗'}")
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MHD-Dynamic full benchmark: stability + gradient flow + accuracy"
    )
    parser.add_argument("--suite",    nargs="+",
                        choices=["a", "b", "c", "all"], default=["all"],
                        help="Which suites to run: a=stability, b=grad-flow, c=accuracy (default: all)")
    parser.add_argument("--depths",   nargs="+", type=int,
                        default=[64, 128, 256, 512, 1024, 2048],
                        help="Depths for suites A and B (default: 64 128 256 512 1024 2048)")
    parser.add_argument("--blocks",   nargs="+", type=int,
                        default=[4, 32, 64],
                        help="ResBlock counts for suite C (default: 4 32 64; use 128+ for longer runs)")
    parser.add_argument("--epochs",   type=int, default=25,
                        help="Training epochs for suite C (default: 25)")
    parser.add_argument("--seeds",    type=int, default=3,
                        help="Random seeds per config (default: 3)")
    parser.add_argument("--dim",      type=int, default=64,
                        help="Hidden dimension (default: 64)")
    parser.add_argument("--lr",       type=float, default=0.025,
                        help="Learning rate for suite C (default: 0.025)")
    parser.add_argument("--n",        type=int, default=5,
                        help="MHD radix, odd ≥ 3 (default: 5)")
    parser.add_argument("--d-oracle", type=int, default=10,
                        help="Oracle dimension (default: 10)")
    parser.add_argument("--scale",    type=float, default=0.1,
                        help="Residual scale factor (default: 0.1)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip oracle verification")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress per-row output (summary only)")
    args = parser.parse_args()

    run_all  = "all" in args.suite
    run_a    = run_all or "a" in args.suite
    run_b    = run_all or "b" in args.suite
    run_c    = run_all or "c" in args.suite

    if args.n % 2 == 0:
        print("ERROR: --n must be odd (even-n obstruction, MHD §6)."); sys.exit(1)

    print("=" * 72)
    print("  MHD-Dynamic Initialization & Training Benchmark  (V15.5.0-pre)")
    print("=" * 72)
    print(f"  Oracle  : n={args.n}, d={args.d_oracle}")
    print(f"  Network : dim={args.dim}, residual_scale={args.scale}")
    print(f"  Seeds   : {args.seeds} per configuration")
    if run_a: print(f"  Suite A : stability depths = {args.depths}")
    if run_b: print(f"  Suite B : grad-flow depths = {args.depths}")
    if run_c: print(f"  Suite C : accuracy blocks  = {args.blocks}, epochs={args.epochs}, lr={args.lr}")
    print("=" * 72)

    verbose = not args.quiet
    t_total = time.time()

    # ── Oracle verification ──────────────────────────────────────────────────
    if not args.no_verify:
        n_fail = run_oracle_verification(n=args.n, d=3)
        if n_fail > 0:
            print("\nOracle verification failed — aborting."); sys.exit(1)
        print()

    results = {}

    # ── Suite A — Forward-pass stability ────────────────────────────────────
    if run_a:
        print("\n┌─ Suite A: Forward-pass Stability  (MHD-SPECTRAL / rank-1 Walsh dual)")
        print("│  Activation variance at init, NO BatchNorm, NO training.")
        print("│  MHD advantage = Kaiming_var / MHD_var  (higher = MHD more stable)\n│")
        t0 = time.time()
        results['a'] = suite_a_stability(
            args.depths, args.dim, args.seeds,
            args.n, args.d_oracle, args.scale, verbose,
        )
        print(f"│\n└─ Suite A complete in {time.time()-t0:.1f}s")

    # ── Suite B — Gradient flow ──────────────────────────────────────────────
    if run_b:
        print("\n┌─ Suite B: Gradient Flow Uniformity  (MHD-MAGIC / constant line sums)")
        print("│  Per-block grad norm profile after 1 fwd+bwd pass.")
        print("│  flow_ratio = gnorm[input_layer] / gnorm[output_layer]")
        print("│  Ideal = 1.0 (uniform).  <1 = vanishing.  >1 = exploding.")
        print("│  flow_cv = coeff. of variation across blocks (lower = more uniform)\n│")
        t0 = time.time()
        results['b'] = suite_b_gradient_flow(
            args.depths, args.dim, args.seeds,
            args.n, args.d_oracle, args.scale, verbose,
        )
        print(f"│\n└─ Suite B complete in {time.time()-t0:.1f}s")

    # ── Suite C — Generalisation accuracy ───────────────────────────────────
    if run_c:
        configs = [(nb, f"{nb}blk") for nb in args.blocks]
        print("\n┌─ Suite C: Generalisation Accuracy  (MHD-ETK / low-frequency concentration)")
        print("│  8-class Gaussian blob classification, 32-D features, σ=1.8 noise.")
        print("│  Full train/test loop, NO BatchNorm.")
        print("│  ep→80% = first epoch test accuracy exceeds 80% (-1 if never)\n│")
        t0 = time.time()
        results['c'] = suite_c_generalisation(
            configs, args.epochs, args.lr, 128,
            args.seeds, args.n, args.d_oracle, args.scale, verbose,
        )
        print(f"│\n└─ Suite C complete in {time.time()-t0:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  BENCHMARK SUMMARY")
    print(f"{'═'*72}")

    if 'a' in results:
        adv_vals = [r['advantage'] for r in results['a'].values() if np.isfinite(r['advantage'])]
        if adv_vals:
            print(f"\n  Suite A  max MHD advantage: {max(adv_vals):.1f}x  "
                  f"(at depth {max(results['a'], key=lambda d: results['a'][d]['advantage'])})")

    if 'b' in results:
        print(f"\n  Suite B  gradient flow ratios (closer to 1.0 = more uniform):")
        for depth, r in results['b'].items():
            print(f"    depth={depth:5d}  MHD={r['mhd_ratio']:.3f}  Kaiming={r['kai_ratio']:.3f}")

    if 'c' in results:
        print(f"\n  Suite C  final test accuracy (avg over {args.seeds} seeds):")
        for (nb, lbl), r in results['c'].items():
            diff = r['mhd_acc'] - r['kai_acc']
            sign = "+" if diff >= 0 else ""
            print(f"    {lbl:>8}  MHD={r['mhd_acc']:.1%}  Kaiming={r['kai_acc']:.1%}  "
                  f"Δ={sign}{diff:.1%}")

    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")
    print()
    print("  Theorem mapping:")
    print("    Suite A → MHD-SPECTRAL : rank-1 Walsh dual suppresses activation explosion")
    print("    Suite B → MHD-MAGIC    : constant line sums → uniform gradient propagation")
    print("    Suite C → MHD-ETK      : low-freq concentration → faster, stable convergence")
    print()
    print("  Scope note: this benchmark uses synthetic clustered data (NumPy only).")
    print("  For real-image results, see examples/mhd_dynamic/bench_mhd_resnet.py (PyTorch).")
    print("=" * 72)


if __name__ == "__main__":
    main()
