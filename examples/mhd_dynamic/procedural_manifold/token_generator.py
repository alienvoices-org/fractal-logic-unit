import torch, torch.nn as nn
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
from .oracle import MHDOracle

class KinshipTokenGenerator:
    def __init__(self, model, oracle, k_candidates=20, coupling=1.0, sync_steps=5, mode_gate=None):
        self.model = model; self.oracle = oracle
        self.k = k_candidates; self.K = coupling; self.sync_steps = sync_steps
        self.mode_gate = mode_gate

    def _get_mode(self, context):
        if self.mode_gate is None: return "interference"
        score = torch.sigmoid(self.mode_gate(context)).mean().item()
        return "interference" if score>0.5 else "inference"

    @torch.no_grad()
    def generate(self, context, symbols=None, temperature=1.0):
        self.model.eval()
        output = self.model(context, symbols)
        logits = output[:,-1,:]
        phases = torch.tanh(logits/temperature)
        abs_phases = torch.abs(phases)
        mode = self._get_mode(context)
        k = min(self.k, abs_phases.shape[-1])
        if mode=="inference":
            idx = torch.argmax(abs_phases, dim=-1)
            return idx[0].item(), phases[0,idx[0]].item(), self._classify(phases[0,idx[0]].item())
        _, top = torch.topk(abs_phases, k, dim=-1)
        cand = phases[0, top[0]].clone().float()
        for _ in range(self.sync_steps):
            diff = cand.unsqueeze(1)-cand.unsqueeze(0)
            cand += 0.1*(self.K/k)*torch.sum(torch.sin(diff), dim=1)
        chosen = torch.argmax(torch.abs(cand))
        return top[0,chosen].item(), cand[chosen].item(), self._classify(cand[chosen].item())

    def _classify(self, p):
        if p>0.8: return "CONSTRUCTIVE"
        if p<-0.8: return "DESTRUCTIVE"
        return "STILL_POINT"
