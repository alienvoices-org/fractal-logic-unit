"""Patch PyTorch model → StreamingRegenLinear with delta dictionary."""
import json, torch, torch.nn as nn
from ._compat import HAS_TORCH, HAS_MSGPACK, _md5_hash
if not HAS_TORCH: raise ImportError("torch required")
from .oracle import MHDOracle
from .streaming_regen import StreamingRegenLinear

def patch_model(model, delta_path, n=5, d=10):
    oracle = MHDOracle(n=n, d=d)
    if delta_path.endswith('.mpk') and HAS_MSGPACK:
        import msgpack
        with open(delta_path,'rb') as f: data = msgpack.load(f)
    else:
        with open(delta_path) as f: data = json.load(f)
    deltas_dict = data["deltas"]
    def _replace(module, prefix=""):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                rlist = deltas_dict.get(full, [])
                start_rank = _md5_hash(full) % oracle.period
                new_layer = StreamingRegenLinear(child.in_features, child.out_features, oracle, start_rank)
                if rlist:
                    rows, cols, vals = zip(*[(int(r),int(c),float(v)) for r,c,v in rlist])
                    new_layer.set_deltas(rows, cols, vals)
                setattr(module, name, new_layer)
            else:
                _replace(child, full)
    _replace(model); return model
