"""
flu.applications — FLU application layer.

All applications are thin wrappers over the proven core layer.
See individual modules for epistemic status of each class.

SIMULATION ONLY markers:
    TensorNetworkSimulator.SIMULATION_ONLY = True
    LighthouseBeacon.SIMULATION_ONLY       = True
"""

from flu.applications.codes      import LatinSquareCode, build_code_matrix
from flu.applications.design     import ExperimentalDesign, DesignResult
from flu.applications.neural     import FLUInitializer
from flu.applications.quantum    import TensorNetworkSimulator
from flu.applications.lighthouse import LighthouseBeacon, BeaconKey, cli_main

__all__ = [
    "ExperimentalDesign",
    "DesignResult",
    "FLUInitializer",
    "TensorNetworkSimulator",
    "LighthouseBeacon",
    "BeaconKey",
    "cli_main",
]
