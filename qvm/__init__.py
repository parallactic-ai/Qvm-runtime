"""Public package interface for QVM."""

from .decorators import grad, hybrid
from .exceptions import (
    BackendError,
    CircuitError,
    ExecutionError,
    QVMError,
)
from .optimize import minimize
from .optimizers import SGD, Adam, Momentum, Optimizer
from .qapp import OptimizationResult, Qapp
from .runtime import QuantumRuntime

# Note: ``ParameterError`` exists in ``qvm.exceptions`` but is not part of
# the top-level surface — the library does not raise it today, and exporting
# it would invite users to write ``except qvm.ParameterError`` clauses that
# never trigger. Import it from ``qvm.exceptions`` directly if you need it.

__all__ = [
    "QuantumRuntime",
    "Qapp",
    "OptimizationResult",
    "Optimizer",
    "SGD",
    "Momentum",
    "Adam",
    "minimize",
    "QVMError",
    "BackendError",
    "CircuitError",
    "ExecutionError",
    "hybrid",
    "grad",
]
