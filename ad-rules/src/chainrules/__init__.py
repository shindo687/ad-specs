"""A minimal, AD-system-agnostic rule interface for scientific Python."""

from ._zero import ZERO
from .api import grad, jvp, value_and_grad, vjp
from .errors import NonDifferentiablePoint, RuleNotFound, UnsupportedWrt
from .registry import RuleRegistry, rules

__all__ = [
    "ZERO",
    "NonDifferentiablePoint",
    "RuleNotFound",
    "RuleRegistry",
    "UnsupportedWrt",
    "grad",
    "jvp",
    "rules",
    "value_and_grad",
    "vjp",
]

__version__ = "0.1.0"
