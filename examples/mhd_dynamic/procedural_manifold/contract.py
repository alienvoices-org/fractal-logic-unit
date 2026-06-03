"""Minimal container identity."""
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Contract:
    logos: Dict[str, Any] = field(default_factory=dict)
    omega: float = 1.0
    phi: Dict[str, Any] = field(default_factory=dict)
