"""High-level Qapp abstraction: a variational quantum algorithm in one object.

A :class:`Qapp` bundles a parameterized ``@qvm.circuit`` (the cost function),
the runtime it executes on, and an optimizer. Calling :meth:`Qapp.fit` runs
the standard variational loop — gradient → step → repeat — and returns a
:class:`OptimizationResult` with the full trace.

This compresses the boilerplate of writing a VQE / QAOA / QML loop by hand:

    >>> app = Qapp(cost_circuit, runtime=qvm, learning_rate=0.4)
    >>> result = app.fit(initial_params=[0.5, 0.3], steps=20)
    >>> result.best_cost
    -0.9998...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from pennylane import numpy as pnp

from .exceptions import CircuitError, ExecutionError, QVMError
from .optimizers import SGD, Optimizer
from .runtime import QuantumRuntime

ParamLike = Union[Sequence[float], np.ndarray]


@dataclass
class OptimizationResult:
    """The outcome of a :meth:`Qapp.fit` run.

    Attributes:
        history: One ``(step, params, cost)`` tuple per optimization step,
            including step 0 (the initial evaluation).
        best_params: Parameters at the step with the lowest observed cost.
        best_cost: The lowest cost observed across all steps.
        converged: ``True`` if the loop stopped because the cost delta fell
            below ``tol``; ``False`` if it ran out of steps.
        steps_taken: Number of optimization steps actually executed
            (not counting the initial evaluation).
    """

    history: List[Tuple[int, np.ndarray, float]] = field(default_factory=list)
    best_params: Optional[np.ndarray] = None
    best_cost: float = float("inf")
    converged: bool = False
    steps_taken: int = 0

    def costs(self) -> List[float]:
        """Return just the cost values, one per step."""
        return [c for _, _, c in self.history]


class Qapp:
    """A variational quantum algorithm as a single composable object.

    A Qapp is the minimal bundle needed to run a hybrid optimization loop:
    a parameterized quantum cost function, a runtime to execute it, and a
    learning rate. Call :meth:`fit` to optimize; inspect the returned
    :class:`OptimizationResult` for parameters, history, and convergence.

    Phase 1 ships with a single optimizer — plain gradient descent — chosen
    deliberately to keep the abstraction small. More optimizers will come
    in Phase 2; the loop structure is the same.
    """

    def __init__(
        self,
        cost_fn: Callable,
        runtime: QuantumRuntime,
        optimizer: Optional[Optimizer] = None,
        learning_rate: float = 0.1,
    ) -> None:
        """Build a Qapp.

        Args:
            cost_fn: A ``@qvm.circuit``-decorated function returning a scalar
                measurement (typically ``qml.expval``). Its only argument is
                the parameter vector being optimized.
            runtime: The :class:`QuantumRuntime` that owns the device.
            optimizer: Gradient-based optimizer (see :mod:`qvm.optimizers`).
                Defaults to :class:`SGD` with the given ``learning_rate``.
            learning_rate: Shorthand for ``optimizer=SGD(learning_rate)``.
                Ignored if ``optimizer`` is provided explicitly.

        Raises:
            CircuitError: ``cost_fn`` is not a ``@qvm.circuit``.
        """
        if not getattr(cost_fn, "_is_qvm_circuit", False):
            raise CircuitError(
                "Qapp expects cost_fn to be decorated with @qvm.circuit"
            )
        self._cost_fn = cost_fn
        self._runtime = runtime
        self._optimizer: Optimizer = optimizer if optimizer is not None else SGD(learning_rate)
        self._grad_fn = runtime.grad(cost_fn)
        self._eval_fn = runtime.make_qnode(cost_fn, analytic=True)

    @property
    def optimizer(self) -> Optimizer:
        """The :class:`Optimizer` driving updates inside :meth:`fit`."""
        return self._optimizer

    @property
    def runtime(self) -> QuantumRuntime:
        """The :class:`QuantumRuntime` this Qapp executes against."""
        return self._runtime

    def evaluate(self, params: ParamLike) -> float:
        """Compute the cost at a single point (analytic, no shot noise).

        Useful for inspecting the cost surface, validating an initial guess,
        or sanity-checking a converged result without re-running ``fit``.
        """
        try:
            arr = pnp.array(np.asarray(params, dtype=float), requires_grad=True)
            return float(self._eval_fn(arr))
        except QVMError:
            raise
        except Exception as e:
            raise ExecutionError(f"Cost evaluation failed: {e}") from e

    def grad_at(self, params: ParamLike) -> np.ndarray:
        """Compute the cost gradient at a single point."""
        try:
            arr = pnp.array(np.asarray(params, dtype=float), requires_grad=True)
            return np.asarray(self._grad_fn(arr))
        except QVMError:
            raise
        except Exception as e:
            raise ExecutionError(f"Gradient evaluation failed: {e}") from e

    def fit(
        self,
        initial_params: ParamLike,
        steps: int = 50,
        tol: float = 1e-6,
        verbose: bool = False,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> OptimizationResult:
        """Run gradient descent on the cost function.

        Args:
            initial_params: Starting parameter vector.
            steps: Maximum number of gradient steps.
            tol: Stop early if ``|cost(t) - cost(t-1)| < tol``.
            verbose: If True, print one line per step.
            callback: Optional ``fn(step, params, cost)`` invoked after each
                step (including step 0). Receives a copy of the parameters,
                safe to store.

        Returns:
            An :class:`OptimizationResult` with the full trace.
        """
        # Reset any per-run state on the optimizer so the same instance can
        # be reused across multiple ``fit`` calls without leaking momentum
        # or Adam moments from a previous run.
        self._optimizer.reset()

        params = pnp.array(np.asarray(initial_params, dtype=float), requires_grad=True)
        try:
            initial_cost = float(self._eval_fn(params))
        except QVMError:
            raise
        except Exception as e:
            raise ExecutionError(f"Initial cost evaluation failed: {e}") from e

        result = OptimizationResult()
        result.history.append((0, np.array(params), initial_cost))
        result.best_params = np.array(params)
        result.best_cost = initial_cost

        if verbose:
            print(f"step  0: cost={initial_cost:+.6f}  params={params}")
        if callback is not None:
            callback(0, np.array(params), initial_cost)

        prev_cost = initial_cost
        for step in range(1, steps + 1):
            try:
                grads = self._grad_fn(params)
                params = pnp.array(
                    np.asarray(self._optimizer.step(np.asarray(params), np.asarray(grads))),
                    requires_grad=True,
                )
                cost = float(self._eval_fn(params))
            except QVMError:
                raise
            except Exception as e:
                raise ExecutionError(
                    f"Optimization failed at step {step}: {e}"
                ) from e

            result.history.append((step, np.array(params), cost))
            result.steps_taken = step

            if cost < result.best_cost:
                result.best_cost = cost
                result.best_params = np.array(params)

            if verbose:
                print(f"step {step:>2}: cost={cost:+.6f}  params={params}")
            if callback is not None:
                callback(step, np.array(params), cost)

            if abs(cost - prev_cost) < tol:
                result.converged = True
                break
            prev_cost = cost

        return result
