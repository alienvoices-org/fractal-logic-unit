"""Compressed delta knowledge - pure Python."""
import json, os, math
from typing import Dict, List, Tuple, Set
from ._compat import HAS_NUMPY, HAS_MSGPACK, HAS_TORCH
if HAS_TORCH: import torch
if HAS_NUMPY: import numpy as np

class DeltaExtractor:
    def __init__(self, model): self.model = model

    def extract(self) -> 'DeltaKnowledge':
        deltas = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'delta_rows') and module.delta_rows:
                rows, cols, vals = module.delta_rows, module.delta_cols, module.delta_vals
                deltas[name] = {(r,c): v for r,c,v in zip(rows, cols, vals)}
            elif hasattr(module, 'deltas'):
                if isinstance(module.deltas, dict) and module.deltas:
                    deltas[name] = dict(module.deltas)
                elif HAS_TORCH and isinstance(module.deltas, torch.nn.Parameter):
                    dense = module.deltas.data
                    indices = torch.nonzero(dense.abs() > 1e-7, as_tuple=False)
                    deltas[name] = {(r.item(),c.item()): dense[r,c].item() for r,c in indices}
        return DeltaKnowledge(deltas=deltas)

class DeltaKnowledge:
    def __init__(self, deltas: dict, contract=None, metadata=None):
        self.deltas = deltas
        self.contract = contract or {}
        self.metadata = metadata or {}

    @property
    def total_deltas(self): return sum(len(v) for v in self.deltas.values())

    def to_dict(self):
        return {"contract": self.contract, "metadata": self.metadata,
                "deltas": {n: [[r,c,v] for (r,c),v in d.items()] for n,d in self.deltas.items()}}

    @classmethod
    def from_dict(cls, data):
        deltas = {n: {(int(r),int(c)): float(v) for r,c,v in lst} for n,lst in data["deltas"].items()}
        return cls(deltas, data.get("contract"), data.get("metadata"))

    def export_json(self, path): json.dump(self.to_dict(), open(path,'w'))
    def export_msgpack(self, path):
        if not HAS_MSGPACK: raise ImportError("msgpack required")
        import msgpack
        with open(path,'wb') as f: msgpack.dump(self.to_dict(), f)

    @classmethod
    def import_json(cls, path): return cls.from_dict(json.load(open(path)))
    @classmethod
    def import_msgpack(cls, path):
        if not HAS_MSGPACK: raise ImportError("msgpack required")
        import msgpack
        return cls.from_dict(msgpack.load(open(path,'rb')))

class DeltaComparer:
    @staticmethod
    def jaccard(k1: DeltaKnowledge, k2: DeltaKnowledge) -> float:
        s1 = {n: set(d.keys()) for n,d in k1.deltas.items()}
        s2 = {n: set(d.keys()) for n,d in k2.deltas.items()}
        inter = sum(len(s1.get(n,set())&s2.get(n,set())) for n in set(s1)|set(s2))
        union = sum(len(s1.get(n,set())|s2.get(n,set())) for n in set(s1)|set(s2))
        return inter/max(1, union)

    @staticmethod
    def structural_similarity(k1: DeltaKnowledge, k2: DeltaKnowledge) -> float:
        energy1 = energy2 = overlap = 0.0
        for n in set(k1.deltas)&set(k2.deltas):
            d1, d2 = k1.deltas[n], k2.deltas[n]
            for c in set(d1)&set(d2): overlap += abs(d1[c]*d2[c])
            for v in d1.values(): energy1 += v*v
            for v in d2.values(): energy2 += v*v
        return overlap/max(1e-12, math.sqrt(energy1*energy2))

class DeltaRegistry:
    def __init__(self): self.entries: Dict[str,dict] = {}; self._cache: Dict[str,DeltaKnowledge] = {}
    def register(self, name, path=None, metadata=None, load_now=True, knowledge=None):
        if knowledge is not None:
            self.entries[name] = {"path": None, "metadata": metadata or {}}
            self._cache[name] = knowledge; return
        if path is None: raise ValueError("path or knowledge required")
        if not os.path.exists(path): raise FileNotFoundError(path)
        self.entries[name] = {"path": path, "metadata": metadata or {}}
        if load_now: self._load(name)

    def _load(self, name):
        if name not in self._cache:
            path = self.entries[name]["path"]
            self._cache[name] = DeltaKnowledge.import_msgpack(path) if path.endswith('.mpk') else DeltaKnowledge.import_json(path)

    def query(self, query_deltas, top_k=5, min_similarity=0.0):
        qk = DeltaKnowledge(deltas=query_deltas)
        comp = DeltaComparer()
        results = []
        for name in list(self.entries.keys()):
            if name not in self._cache: self._load(name)
            sim = comp.structural_similarity(qk, self._cache[name])
            if sim >= min_similarity: results.append((name, sim, self.entries[name].get("metadata",{})))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
