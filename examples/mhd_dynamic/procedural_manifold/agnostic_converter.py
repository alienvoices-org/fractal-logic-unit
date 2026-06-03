#!/usr/bin/env python3
"""Convert safetensors → sparse delta dictionary."""
import os, json, argparse, hashlib
from ._compat import HAS_TORCH, HAS_SAFETENSORS, HAS_MSGPACK, _md5_hash
if not HAS_TORCH or not HAS_SAFETENSORS: raise ImportError("torch and safetensors required")
import torch
from safetensors.torch import load_file
from collections import defaultdict
from .oracle import MHDOracle

def convert_checkpoint(ckpt_dir, output_path, sparsity=0.95, n=5, d=10,
                       skip_patterns=("norm","bias","scale","embed","head","lm_head"),
                       binary=False):
    oracle = MHDOracle(n=n, d=d)
    all_deltas = {}
    stats = defaultdict(lambda: {"total":0,"deltas":0})
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.safetensors'))
    for fname in files:
        path = os.path.join(ckpt_dir, fname)
        tensors = load_file(path)
        for name, weight in tensors.items():
            if weight.ndim!=2 or any(p in name.lower() for p in skip_patterns): continue
            out_f_pt, in_f = weight.shape
            start_rank = _md5_hash(name) % oracle.period
            baseline_internal = oracle.generate_block_pytorch(in_f, out_f_pt, start_rank)
            baseline_pt = baseline_internal.T
            deviations = weight.float() - baseline_pt
            k = max(1, int(in_f*out_f_pt*(1-sparsity)))
            flat = deviations.abs().flatten()
            threshold = torch.kthvalue(flat, max(1, flat.numel()-k)).values
            mask = deviations.abs() >= threshold
            rows_pt, cols_pt = torch.where(mask)
            deltas = {(int(c), int(r)): float(deviations[r,c]) for r,c in zip(rows_pt, cols_pt)}
            all_deltas[name] = deltas
            stats[name]["total"] = in_f*out_f_pt; stats[name]["deltas"] = len(deltas)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    data = {"contract": {}, "deltas": {n: [[r,c,v] for (r,c),v in d.items()] for n,d in all_deltas.items()}}
    if binary and HAS_MSGPACK:
        import msgpack
        with open(output_path,'wb') as f: msgpack.dump(data, f)
    else:
        with open(output_path,'w') as f: json.dump(data, f)
    tp = sum(s["total"] for s in stats.values()); tr = sum(s["deltas"] for s in stats.values())
    print(f"Saved {tr} deltas ({100*(1-tr/max(1,tp)):.1f}% compression) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--sparsity", type=float, default=0.95)
    parser.add_argument("--binary", action='store_true')
    parser.add_argument("--n", type=int, default=5); parser.add_argument("--d", type=int, default=10)
    args = parser.parse_args()
    convert_checkpoint(args.ckpt_dir, args.output, args.sparsity, args.n, args.d, binary=args.binary)
