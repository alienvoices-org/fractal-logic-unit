"""DeltaStream - export/import of sparse delta dictionaries."""
import json, torch, torch.nn as nn
from ._compat import HAS_TORCH, HAS_MSGPACK
if not HAS_TORCH: raise ImportError("torch required")
from .contract import Contract
from .regen_linear import RegenLinear
from .streaming_regen import StreamingRegenLinear

class DeltaStream:
    def __init__(self, contract, deltas: dict):
        self.contract = contract
        self.deltas = deltas

    def export_json(self) -> str:
        return json.dumps({"contract": {"logos": self.contract.logos, "omega": self.contract.omega, "phi": self.contract.phi},
                           "deltas": {n: [[r,c,v] for (r,c),v in d.items()] for n,d in self.deltas.items()}})

    @classmethod
    def import_json(cls, json_str: str):
        data = json.loads(json_str)
        contract = Contract(logos=data["contract"]["logos"], omega=data["contract"]["omega"], phi=data["contract"]["phi"])
        deltas = {n: {(int(r),int(c)): float(v) for r,c,v in lst} for n,lst in data["deltas"].items()}
        return cls(contract, deltas)

    def apply_to_model(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, StreamingRegenLinear):
                if name in self.deltas:
                    d = self.deltas[name]
                    rows, cols, vals = zip(*[(r,c,v) for (r,c),v in d.items()]) if d else ([],[],[])
                    module.set_deltas(rows, cols, vals)
            elif isinstance(module, RegenLinear):
                if name in self.deltas:
                    module.deltas.data.zero_()
                    for (r,c), v in self.deltas[name].items():
                        if r < module.in_f and c < module.out_f:
                            module.deltas.data[r,c] = v
