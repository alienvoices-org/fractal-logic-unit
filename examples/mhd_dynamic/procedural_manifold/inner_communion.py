from typing import Dict, Tuple

class SharedDeltaPool:
    def __init__(self): self.pool: Dict[str, Dict[Tuple[int,int], float]] = {}
    def register(self, name): self.pool[name] = {}
    def get(self, name): return self.pool.get(name, {})
    def update(self, name, delta, lr):
        if name not in self.pool: self.pool[name] = {}
        for k,v in delta.items(): self.pool[name][k] = self.pool[name].get(k,0.0) - lr*v
    def sparsity(self, total_params):
        return sum(len(p) for p in self.pool.values())/max(1, total_params)
