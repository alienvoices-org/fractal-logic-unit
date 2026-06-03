import torch
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
from .builder import KinshipBuilder
from .oracle import MHDOracle
from .token_generator import KinshipTokenGenerator

def main():
    print("Procedural Manifold Parameter System V4.0 Demo")
    config = {'num_blocks':8, 'dim':32, 'num_heads':4, 'num_classes':100}
    model = KinshipBuilder.build_moe(config)
    oracle = MHDOracle()
    gen = KinshipTokenGenerator(model, oracle, k_candidates=15)
    ctx = torch.randn(1,4,32)
    for i in range(5):
        tok, phase, sig = gen.generate(ctx)
        print(f"Step {i}: Token {tok:3d}  Phase {phase:+.3f}  [{sig}]")

if __name__ == "__main__":
    main()
