"""Custom exceptions for the Quantum Runtime."""

from __future__ import annotations


class QVMError(Exception):
    """Base exception for all Quantum VM errors."""
    pass


class BackendError(QVMError):
    """Raised when there is an issue with the quantum backend."""
    pass


class CircuitError(QVMError):
    """Raised for invalid circuit definitions or usage."""
    pass


class ParameterError(QVMError):
    """Raised when invalid parameters are passed to a circuit."""
    pass


class ExecutionError(QVMError):
    """Raised when circuit execution fails."""
    pass
