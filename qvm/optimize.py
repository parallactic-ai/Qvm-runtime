"""Bridge to SciPy classical optimizers.

The Qapp's built-in optimizers (SGD, Momentum, Adam) cover the common
gradient-descent regime. For everything else — L-BFGS-B, COBYLA,
Nelder-Mead, trust-region, and the dozens of methods SciPy ships —
:func:`minimize` hands the problem off to ``scipy.optimize.minimize``
while keeping the same :class:`OptimizationResult` contract.

Gradient-based SciPy methods receive a Jacobian automatically; gradient-free
methods (Nelder-Mead, COBYLA, Powell) ignore it harmlessly.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from pennylane import numpy as pnp
from scipy.optimize import minimize as _scipy_minimize

from .exceptions import CircuitError, ExecutionError, QVMError
from .qapp import OptimizationResult
from .runtime import QuantumRuntime

ParamLike = Union[Sequence[float], np.ndarray]


def minimize(
    cost_fn: Callable,
    x0: ParamLike,
    runtime: QuantumRuntime,
    method: str = "L-BFGS-B",
    options: Optional[dict[str, Any]] = None,
) -> OptimizationResult:
    """Optimize a quantum cost function with any SciPy method.

    Args:
        cost_fn: A ``@qvm.circuit`` returning a scalar measurement.
        x0: Initial parameter vector.
        runtime: The :class:`QuantumRuntime` that owns the device.
        method: Any SciPy ``minimize`` method. ``"L-BFGS-B"`` (gradient-based,
            handles bound constraints, fast) is a sensible default for most
            variational problems. ``"Nelder-Mead"`` and ``"COBYLA"`` are
            gradient-free alternatives.
        options: Extra options forwarded to ``scipy.optimize.minimize``
            (e.g. ``{"maxiter": 200}``).

    Returns:
        An :class:`OptimizationResult` with the converged parameters and a
        history populated from SciPy's per-iteration callback.

    Raises:
        CircuitError: ``cost_fn`` is not a ``@qvm.circuit``.
        ExecutionError: SciPy or the underlying circuit raised during the run.
    """
    if not getattr(cost_fn, "_is_qvm_circuit", False):
        raise CircuitError(
            "minimize() expects cost_fn to be decorated with @qvm.circuit"
        )

    qnode = runtime.make_qnode(cost_fn, analytic=True)
    grad_fn = runtime.grad(cost_fn)

    def fun(x: np.ndarray) -> float:
        return float(qnode(pnp.array(x, requires_grad=True)))

    def jac(x: np.ndarray) -> np.ndarray:
        return np.asarray(grad_fn(pnp.array(x, requires_grad=True)))

    x0_arr = np.asarray(x0, dtype=float)
    history: list[tuple[int, np.ndarray, float]] = [
        (0, x0_arr.copy(), fun(x0_arr))
    ]

    def _callback(xk: np.ndarray) -> None:
        # SciPy calls this once per accepted iterate (line-search converged).
        history.append((len(history), np.array(xk, dtype=float), fun(xk)))

    try:
        with warnings.catch_warnings():
            # Gradient-free methods (Nelder-Mead, COBYLA, Powell) warn that
            # ``jac`` is ignored. We always pass jac so callers don't have to
            # think about which methods need it — the warning is noise here.
            warnings.filterwarnings(
                "ignore",
                message=".*does not use gradient information.*",
                category=RuntimeWarning,
            )
            scipy_result = _scipy_minimize(
                fun,
                x0_arr,
                jac=jac,
                method=method,
                callback=_callback,
                options=options or {},
            )
    except QVMError:
        raise
    except Exception as e:
        raise ExecutionError(f"SciPy minimize failed: {e}") from e

    result = OptimizationResult()
    result.history = history
    result.best_params = np.asarray(scipy_result.x, dtype=float)
    result.best_cost = float(scipy_result.fun)
    result.converged = bool(scipy_result.success)
    result.steps_taken = max(0, len(history) - 1)
    return result
