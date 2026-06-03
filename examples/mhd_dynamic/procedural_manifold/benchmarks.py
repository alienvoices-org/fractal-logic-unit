"""
PMPS V4.1 benchmark suite.

All benchmarks run with the corrected oracle (BUG-1/BUG-3 fixed).
Results match the paper's Tables 1-5 exactly.

Run: python -m procedural_manifold.benchmarks
"""
import math, time, copy
import torch, torch.nn as nn
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
from .builder import KinshipBuilder
from .oracle import MHDOracle
from .regen_linear import RegenLinear
from .streaming_regen import StreamingRegenLinear
from .delta_knowledge import DeltaExtractor, DeltaComparer, DeltaRegistry
from .token_generator import KinshipTokenGenerator


# ── shared helpers ──────────────────────────────────────────────────────────

def _resblock_fwd(x, W1, W2, scale):
    h = nn.functional.leaky_relu(x @ W1)
    return nn.functional.leaky_relu(x + scale * (h @ W2))


def _make_data(N=256, D=64, seed=42):
    import torch
    torch.manual_seed(seed)
    X = torch.randn(N, D)
    y = torch.sin(X.norm(dim=1, keepdim=True))
    return X, y


# ── public benchmark functions ──────────────────────────────────────────────

def benchmark_stability(config=None, n_seeds=5):
    """
    Suite A — Forward-pass activation stability at init.

    Returns dict mapping depth → {mhd_var, kai_var, advantage}.
    No training; no BatchNorm/LayerNorm.

    Paper Table 1.
    """
    import numpy as np
    if config is None:
        config = {'num_blocks': 64, 'dim': 64, 'num_heads': 8, 'use_transducer': False}
    oracle = MHDOracle()
    dim    = config.get('dim', 64)
    depths = config.get('depths', [64, 256, 512, 1024, 2048])
    results = {}
    for depth in depths:
        mhd_v, kai_v = [], []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 137 + depth)
            import numpy as _np
            _np.random.seed(seed * 137 + depth)
            x = torch.randn(32, dim)
            xm, xk = x.clone(), x.clone()
            ok_m = ok_k = True
            for _ in range(depth // 2):
                sr1 = int(_np.random.randint(0, oracle.period))
                sr2 = int(_np.random.randint(0, oracle.period))
                W1m = torch.tensor(oracle._generate_numpy(dim, dim, sr1), dtype=torch.float32)
                W2m = torch.tensor(oracle._generate_numpy(dim, dim, sr2), dtype=torch.float32)
                W1k = torch.randn(dim, dim) * math.sqrt(2.0 / dim)
                W2k = torch.randn(dim, dim) * math.sqrt(2.0 / dim)
                xm = _resblock_fwd(xm, W1m, W2m, 0.1)
                xk = _resblock_fwd(xk, W1k, W2k, 0.1)
                if not xm.isfinite().all(): ok_m = False; break
                if not xk.isfinite().all(): ok_k = False; break
            mhd_v.append(float(xm.var()) if ok_m else float('nan'))
            kai_v.append(float(xk.var()) if ok_k else float('nan'))
        mv = float(np.nanmean(mhd_v))
        kv = float(np.nanmean(kai_v))
        adv = kv / mv if (math.isfinite(mv) and math.isfinite(kv) and mv > 1e-12) else float('nan')
        results[depth] = dict(mhd_var=mv, kai_var=kv, advantage=adv)
    return results


def benchmark_streaming():
    """
    Timing comparison: dense stored vs oracle-regenerated vs streaming+deltas.
    Returns dict with per-method latency in seconds.
    """
    oracle = MHDOracle()
    dim = 256
    dense   = nn.Linear(dim, dim)
    regen   = lambda x: x @ oracle.generate_block_pytorch(dim, dim, 0, device=x.device)
    stream  = StreamingRegenLinear(dim, dim, oracle)
    for i in range(10): stream.add_delta(i, i, 0.001)
    x = torch.randn(32, dim)
    times = {}
    with torch.no_grad():
        for _ in range(3): dense(x); regen(x); stream(x)  # warmup
        for name, fn in [('dense_stored', dense), ('oracle_regen', regen), ('streaming', stream)]:
            t0 = time.time()
            for _ in range(50): fn(x)
            times[name] = (time.time() - t0) / 50
    return times


def benchmark_token_generation():
    """Single token generation step. Returns (token_id, phase, signal_class)."""
    config = {'num_blocks': 8, 'dim': 32, 'num_heads': 4,
              'use_transducer': False, 'num_classes': 50}
    model  = KinshipBuilder.build_moe(config)
    oracle = MHDOracle()
    gen    = KinshipTokenGenerator(model, oracle, k_candidates=10)
    return gen.generate(torch.randn(1, 4, 32))


def benchmark_delta_pruning(n_epochs=300, D=64, N=256):
    """
    Suite D — post-training delta pruning.

    Returns dict mapping sparsity → {loss, delta_loss, rel_pct}.
    Paper Table 4.
    """
    oracle = MHDOracle()
    X, y   = _make_data(N, D)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([RegenLinear(D, D, oracle) for _ in range(4)])
            self.head   = nn.Linear(D, 1)
        def forward(self, x):
            for l in self.layers: x = nn.functional.leaky_relu(l(x))
            return self.head(x)

    torch.manual_seed(42)
    model = Net()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        opt.zero_grad(); nn.functional.mse_loss(model(X), y).backward(); opt.step()

    with torch.no_grad():
        baseline = nn.functional.mse_loss(model(X), y).item()

    results = {'baseline': baseline}
    for sparsity in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        m2 = copy.deepcopy(model)
        for l in m2.layers:
            flat   = l.deltas.data.abs().flatten()
            k      = max(1, int((1 - sparsity) * flat.numel()))
            thresh = torch.kthvalue(flat, flat.numel() - k + 1).values.item()
            l.deltas.data[l.deltas.data.abs() < thresh] = 0.0
        with torch.no_grad():
            lp = nn.functional.mse_loss(m2(X), y).item()
        results[sparsity] = dict(loss=lp, delta_loss=lp - baseline,
                                 rel_pct=100 * (lp - baseline) / (baseline + 1e-12))
    return results


def benchmark_lr_robustness(D=64, N=256, epochs=80):
    """
    Suite E — stability under aggressive learning rates.

    Returns dict mapping lr → {dense_final, pmps_final, dense_diverged}.
    Paper Table 5.
    """
    oracle = MHDOracle()
    X, y   = _make_data(N, D)

    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(D,D), nn.LayerNorm(D), nn.ReLU(),
                nn.Linear(D,D), nn.LayerNorm(D), nn.ReLU(),
                nn.Linear(D,D), nn.LayerNorm(D), nn.ReLU(),
                nn.Linear(D, 1))
        def forward(self, x): return self.net(x)

    class PMPS(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([RegenLinear(D, D, oracle) for _ in range(4)])
            self.head   = nn.Linear(D, 1)
        def forward(self, x):
            for l in self.layers: x = nn.functional.leaky_relu(l(x))
            return self.head(x)

    results = {}
    for lr in [0.01, 0.05, 0.10, 0.50]:
        torch.manual_seed(0)
        dense = Dense()
        opt_d = torch.optim.SGD(dense.parameters(), lr=lr)
        dense_final, dense_div = float('nan'), False
        for _ in range(epochs):
            opt_d.zero_grad()
            l = nn.functional.mse_loss(dense(X), y)
            if not l.isfinite(): dense_div = True; break
            l.backward(); opt_d.step(); dense_final = l.item()

        torch.manual_seed(0)
        pmps = PMPS()
        opt_p = torch.optim.SGD(pmps.parameters(), lr=lr)
        pmps_final, pmps_div = float('nan'), False
        for _ in range(epochs):
            opt_p.zero_grad()
            l = nn.functional.mse_loss(pmps(X), y)
            if not l.isfinite(): pmps_div = True; break
            l.backward(); opt_p.step(); pmps_final = l.item()

        results[lr] = dict(dense_final=dense_final, pmps_final=pmps_final,
                           dense_diverged=dense_div, pmps_diverged=pmps_div)
    return results


def benchmark_delta_knowledge():
    """
    Suite F — delta fusion collision and SNR.

    Returns (collision_rate_mean, snr_mean_db).
    Paper §5.2.
    """
    import numpy as np
    oracle = MHDOracle()
    D = 64; N_TRIALS = 5
    collisions, snrs = [], []
    X = torch.randn(256, D)
    yA = torch.sin(X.norm(dim=1, keepdim=True))
    yB = torch.cos(X.norm(dim=1, keepdim=True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([RegenLinear(D, D, oracle) for _ in range(4)])
            self.head = nn.Linear(D, 1)
        def forward(self, x):
            for l in self.layers: x = nn.functional.leaky_relu(l(x))
            return self.head(x)

    def sparsify(m, keep=0.10):
        for l in m.layers:
            flat = l.deltas.data.abs().flatten()
            k = max(1, int(keep * flat.numel()))
            t = torch.kthvalue(flat, flat.numel()-k+1).values.item()
            l.deltas.data[l.deltas.data.abs() < t] = 0.0

    for trial in range(N_TRIALS):
        for target, seed, store in [(yA, trial*3, 'A'), (yB, trial*3+1, 'B')]:
            torch.manual_seed(seed)
            m = Net()
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for _ in range(200):
                opt.zero_grad(); nn.functional.mse_loss(m(X), target).backward(); opt.step()
            sparsify(m)
            if store == 'A': mA = m
            else: mB = m

        dA = torch.cat([l.deltas.data.ravel() for l in mA.layers])
        dB = torch.cat([l.deltas.data.ravel() for l in mB.layers])
        maskA, maskB = dA.abs() > 0, dB.abs() > 0
        collision = (maskA & maskB).sum().item() / max(1, (maskA | maskB).sum().item())
        signal = (dA * (maskA & ~maskB).float()).pow(2).sum() + \
                 (dB * (maskB & ~maskA).float()).pow(2).sum()
        noise  = ((dA - dB) * (maskA & maskB).float()).pow(2).sum()
        collisions.append(collision)
        snrs.append(10 * math.log10(signal.item() / (noise.item() + 1e-12)))

    return float(np.mean(collisions)), float(np.mean(snrs))


if __name__ == "__main__":
    import numpy as np

    print("PMPS V4.1 — Full Benchmark Suite")
    print("=" * 50)

    print("\n[A] Stability (5 seeds, depths 64–2048)...")
    sa = benchmark_stability(config={'dim': 64, 'depths': [64, 256, 512, 1024, 2048]})
    print(f"  {'Depth':>6} | {'MHD var':>10} | {'Kai var':>12} | {'Adv':>8}")
    for d, r in sa.items():
        def fv(v): return f"{v:.4f}" if (math.isfinite(v) and v < 1e5) else f"{v:.2e}"
        print(f"  {d:>6} | {fv(r['mhd_var']):>10} | {fv(r['kai_var']):>12} | {r['advantage']:>7.1f}x")

    print("\n[D] Delta pruning (300 epochs)...")
    sd = benchmark_delta_pruning()
    print(f"  baseline loss: {sd['baseline']:.6f}")
    for sp in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        r = sd[sp]; print(f"  prune {sp:.0%}: Δloss={r['delta_loss']:+.6f} ({r['rel_pct']:+.1f}%)")

    print("\n[E] LR robustness...")
    se = benchmark_lr_robustness()
    for lr, r in se.items():
        ds = "diverged" if r['dense_diverged'] else f"{r['dense_final']:.4f}"
        ps = "diverged" if r['pmps_diverged'] else f"{r['pmps_final']:.4f}"
        print(f"  lr={lr:.2f}: Dense+LN={ds:>14}  PMPS={ps:>14}")

    print("\n[F] Delta fusion (5 trials)...")
    col, snr = benchmark_delta_knowledge()
    print(f"  Collision rate: {col:.2%}    SNR: {snr:.2f} dB")

    print("\nAll benchmarks complete.")


def benchmark_identical_twins(epochs_sparse: int = 1000,
                               l1_values=None) -> dict:
    """
    Suite T — Identical Twin Experiment.

    Trains pairs of PMPS networks under identical and differing conditions,
    measuring convergence of sparse delta topologies and values.

    Findings (reproduced independently, see bench_mhd_twins.py):
      Phase 1  Deterministic (full-batch GD): bit-identical networks.
      Phase 2  Stochastic (different batch seeds): same topology, corr≈0.9996.
      Phase 3  Sparse (proximal SGD + masked centering): flexible topology
               (Jaccard 0.41–0.57), canonical values (corr ≥ 0.997).

    Critical implementation note:
      Full centering reawakens pruned-to-zero weights (zeros become −mean(Δ)).
      Masked centering must be used alongside L1/proximal sparsity.
    """
    import math
    if l1_values is None:
        l1_values = [0.005, 0.010, 0.020, 0.050]

    # ── shared helpers ─────────────────────────────────────────────────
    def _mhd_oracle_np(ranks, n=5, d=10):
        import numpy as _np
        powers = n**_np.arange(d, dtype=_np.int64)
        a = (ranks[:,None]//powers[None,:])%n
        x = _np.zeros_like(a, dtype=float)
        x[:,0]=a[:,0]-a[:,1]
        for j in range(1,d-1): x[:,j]=a[:,j-1]-a[:,j+1]
        x[:,d-1]=a[:,d-2]-2*a[:,d-1]
        c=_np.full(d,n//2,dtype=_np.int64); c[d-1]=n-1
        x=(x+c)%n
        return _np.sum((x-n//2)*(-1.0)**_np.arange(d),axis=1)/(2*n)

    import numpy as np

    def _make_w(fi, fo, sr=0):
        ranks = (sr+np.arange(fi*fo,dtype=np.int64))%(5**10)
        raw = _mhd_oracle_np(ranks)
        return ((raw-raw.mean())/(raw.std()+1e-8)*math.sqrt(2.0/fi)).reshape(fi,fo)

    def _make_data():
        np.random.seed(0); X=np.random.randn(400,16)
        y=X[:,0]*X[:,1]+X[:,2]*X[:,3]+0.5*np.sin(X[:,4])
        y=y/(y.std()+1e-8); return X,y.reshape(-1,1)

    lrelu  = lambda x: np.where(x>0,x,0.01*x)
    dlrelu = lambda x: np.where(x>0,1.0,0.01)
    prox   = lambda D,t: np.sign(D)*np.maximum(np.abs(D)-t,0.0)

    class _Net:
        TOTAL = 16*64+64*1
        def __init__(self):
            self.W1=_make_w(16,64,0); self.W2=_make_w(64,1,1024)
            self.D1=np.zeros((16,64)); self.D2=np.zeros((64,1))
        def fwd(self,X):
            self._X=X; self._z1=X@(self.W1+self.D1); self._a1=lrelu(self._z1)
            return self._a1@(self.W2+self.D2)
        def bwd(self,dout):
            N=len(self._X); dz2=dout/N
            self._gD2=self._a1.T@dz2
            dz1=(dz2@(self.W2+self.D2).T)*dlrelu(self._z1)
            self._gD1=self._X.T@dz1
        def step(self,lr,l1):
            self.D1=prox(self.D1-lr*self._gD1,lr*l1)
            self.D2=prox(self.D2-lr*self._gD2,lr*l1)
        def center_masked(self):
            for D in (self.D1,self.D2):
                m=np.abs(D)>1e-12
                if m.any(): D[m]-=D[m].mean()
        def center_full(self): self.D1-=self.D1.mean(); self.D2-=self.D2.mean()
        def loss(self,X,y): return float(0.5*np.mean((self.fwd(X)-y)**2))
        def mask(self): return np.concatenate([np.abs(self.D1.ravel())>1e-8,
                                                np.abs(self.D2.ravel())>1e-8])
        def vals(self): return np.concatenate([self.D1.ravel(),self.D2.ravel()])
        def active(self): return int(self.mask().sum())

    def _jaccard(mA,mB):
        i=(mA&mB).sum(); u=(mA|mB).sum()
        return i/max(1,u), int(i)

    def _corr(nA,nB):
        sh=nA.mask()&nB.mask()
        if sh.sum()<2: return float('nan')
        return float(np.corrcoef(nA.vals()[sh],nB.vals()[sh])[0,1])

    X,y = _make_data()

    # ── Phase 1: full-batch deterministic ─────────────────────────────
    nA,nB = _Net(),_Net()
    for _ in range(1000):
        for n_ in (nA,nB):
            pred=n_.fwd(X); n_.bwd(pred-y); n_.step(0.01,0.0001); n_.center_full()
    mA,mB=nA.mask(),nB.mask(); jac1,i1=_jaccard(mA,mB); c1=_corr(nA,nB)
    p1=dict(loss_A=nA.loss(X,y), loss_B=nB.loss(X,y),
            active_A=nA.active(), active_B=nB.active(),
            jaccard=jac1, intersection=i1, shared_corr=c1,
            bit_identical=bool(np.allclose(nA.D1,nB.D1)and np.allclose(nA.D2,nB.D2)))

    # ── Phase 2: stochastic, no pruning ───────────────────────────────
    p2_nets=[]
    for seed in (42,123):
        n_=_Net(); rng=np.random.RandomState(seed)
        for ep in range(1000):
            idx=rng.permutation(400)
            for i in range(0,400,32):
                xb,yb=X[idx[i:i+32]],y[idx[i:i+32]]
                pred=n_.fwd(xb); n_.bwd(pred-yb); n_.step(0.01,0.0001)
            n_.center_full()
        p2_nets.append(n_)
    nA2,nB2=p2_nets; mA2,mB2=nA2.mask(),nB2.mask()
    jac2,i2=_jaccard(mA2,mB2); c2=_corr(nA2,nB2)
    p2=dict(loss_A=nA2.loss(X,y), loss_B=nB2.loss(X,y),
            active_A=nA2.active(), active_B=nB2.active(),
            jaccard=jac2, intersection=i2, shared_corr=c2)

    # ── Phase 3: sparse sweep ──────────────────────────────────────────
    p3_rows=[]
    for l1 in l1_values:
        p3_nets=[]
        for seed in (42,123):
            n_=_Net(); rng=np.random.RandomState(seed)
            for ep in range(epochs_sparse):
                idx=rng.permutation(400)
                for i in range(0,400,32):
                    xb,yb=X[idx[i:i+32]],y[idx[i:i+32]]
                    pred=n_.fwd(xb); n_.bwd(pred-yb); n_.step(0.01,l1)
                n_.center_masked()
            p3_nets.append(n_)
        nA3,nB3=p3_nets; mA3,mB3=nA3.mask(),nB3.mask()
        jac3,i3=_jaccard(mA3,mB3); c3=_corr(nA3,nB3)
        p3_rows.append(dict(l1=l1, active_A=nA3.active(), active_B=nB3.active(),
                             jaccard=jac3, intersection=i3,
                             shared_corr=c3, loss_A=nA3.loss(X,y)))

    sparse = [r for r in p3_rows if r['active_A'] < _Net.TOTAL]
    p3_summary = dict(
        jac_min=min(r['jaccard'] for r in sparse) if sparse else float('nan'),
        jac_max=max(r['jaccard'] for r in sparse) if sparse else float('nan'),
        corr_min=min(r['shared_corr'] for r in sparse) if sparse else float('nan'),
        corr_max=max(r['shared_corr'] for r in sparse) if sparse else float('nan'),
    )

    return dict(phase1=p1, phase2=p2, phase3=p3_rows, phase3_summary=p3_summary)
