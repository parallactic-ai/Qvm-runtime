"""Module-level convenience decorators and helpers.

These exist so users can write the natural form ``from qvm import hybrid, grad``
without having to thread a ``QuantumRuntime`` instance through their code.

- :func:`hybrid` works without any runtime - it just wraps a function so
  exceptions surface as :class:`ExecutionError`.
- :func:`grad` recognizes ``@qvm.circuit`` functions and routes them through
  their bound runtime for noise-free analytic gradients. Anything else is
  passed straight to :func:`pennylane.grad`.
"""

from __future__ import annotations

from typing import Any, Callable

import pennylane as qml

from .exceptions import ExecutionError


def hybrid(func: Callable) -> Callable:
    """Mark a function as hybrid (classical + quantum).

    Exceptions raised inside ``func`` are re-raised as :class:`ExecutionError`
    so callers can rely on a single error hierarchy when composing hybrid
    pipelines. Equivalent to :meth:`QuantumRuntime.hybrid`, but standalone.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise ExecutionError(
                f"Hybrid function '{func.__name__}' failed: {e}"
            ) from e

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def grad(fn: Callable) -> Callable:
    """Return the gradient function for ``fn``.

    If ``fn`` is a ``@qvm.circuit``, the runtime it was registered with
    builds an analytic QNode internally so gradients are free of shot noise.
    Otherwise ``fn`` is passed straight to :func:`pennylane.grad`. The
    differentiable input must be a ``pennylane.numpy`` array with
    ``requires_grad=True``.
    """
    if getattr(fn, "_is_qvm_circuit", False):
        runtime = getattr(fn, "_qvm_runtime", None)
        if runtime is not None:
            return runtime.grad(fn)
    try:
        return qml.grad(fn)
    except Exception as e:
        raise ExecutionError(f"Failed to build gradient function: {e}") from e
