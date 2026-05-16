"""Core Quantum Runtime (QVM) implementation - Phase 1."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pennylane as qml

from .exceptions import (
    BackendError,
    CircuitError,
    ExecutionError,
    QVMError,
)

ParamLike = Union[Sequence[float], np.ndarray]
RunResult = Union[float, np.ndarray, tuple]


# Short labels for common gates — keeps trace output readable.
_GATE_SHORT_NAMES = {
    "Hadamard": "H",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
}


def _format_op(op: Any) -> str:
    """Render a PennyLane operation as ``Gate(params, wires)`` for trace labels."""
    name = _GATE_SHORT_NAMES.get(op.name, op.name)
    wires = ",".join(str(w) for w in op.wires)
    if op.parameters:
        param_str = ",".join(f"{float(p):.3f}" for p in op.parameters)
        return f"{name}({param_str}, {wires})"
    return f"{name}({wires})"


class QuantumRuntime:
    """Minimal Viable Quantum Runtime for Phase 1.

    Provides a clean, developer-friendly interface over PennyLane for
    running quantum circuits and hybrid quantum-classical workflows.
    """

    def __init__(
        self,
        backend: str = "default.qubit",
        shots: int = 1024,
        seed: Optional[int] = None,
        diff_method: str = "parameter-shift",
    ) -> None:
        """Initialize the runtime and its underlying device.

        Args:
            backend: PennyLane device name (e.g. "default.qubit", "lightning.qubit").
            shots: Default number of measurement shots per execution.
            seed: Optional RNG seed for reproducibility.
            diff_method: Differentiation method passed to the QNode.
        """
        self._backend_name = backend
        self._shots = shots
        self._seed = seed
        self._diff_method = diff_method
        self._device: Optional[qml.Device] = None
        self._initialize_device()

    def _initialize_device(self) -> None:
        try:
            self._device = qml.device(
                self._backend_name,
                wires=None,
            )
        except Exception as e:
            raise BackendError(
                f"Failed to initialize backend '{self._backend_name}': {e}"
            ) from e

    @property
    def backend(self) -> str:
        """Name of the currently active backend."""
        return self._backend_name

    def set_backend(self, backend: str) -> None:
        """Switch to a different backend, re-initializing the device."""
        if backend == self._backend_name:
            return
        self._backend_name = backend
        self._initialize_device()

    def available_backends(self) -> list[str]:
        """Return a list of backend names recognized by this runtime."""
        return [
            "default.qubit",
            "lightning.qubit",
            "qsim",
            "default.mixed",
        ]

    # =====================
    # Decorators
    # =====================

    def circuit(self, func: Callable) -> Callable:
        """Decorator to mark a function as a quantum circuit.

        The decorated function should apply PennyLane gates and return a
        measurement (e.g. ``qml.expval``, ``qml.sample``, ``qml.state``).
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise CircuitError(f"Error in circuit '{func.__name__}': {e}") from e

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._is_qvm_circuit = True  # type: ignore[attr-defined]
        wrapper._original_fn = func  # type: ignore[attr-defined]
        wrapper._qvm_runtime = self  # type: ignore[attr-defined]
        return wrapper

    def hybrid(self, func: Callable) -> Callable:
        """Decorator to mark a function as hybrid (classical + quantum).

        Exceptions raised by the underlying function are converted to
        :class:`ExecutionError` so callers can rely on a single error
        hierarchy.
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

    # =====================
    # Execution
    # =====================

    def run(
        self,
        circuit_fn: Callable,
        params: Optional[ParamLike] = None,
        shots: Optional[int] = None,
    ) -> RunResult:
        """Execute a ``@qvm.circuit``-decorated function on the active device.

        Args:
            circuit_fn: A function decorated with ``@qvm.circuit``.
            params: Optional sequence of parameters passed to the circuit.
            shots: Optional per-call shots override.

        Returns:
            The circuit's measurement result, normalized for predictability:
              - A scalar measurement (e.g. ``qml.expval``) is returned as a
                Python ``float``.
              - A vector measurement (e.g. ``qml.probs``, ``qml.state``,
                ``qml.sample``) is returned as a NumPy ``ndarray``.
              - Multiple measurements come back as a ``tuple`` of the above.
        """
        if not getattr(circuit_fn, "_is_qvm_circuit", False):
            raise CircuitError("Function must be decorated with @qvm.circuit")

        effective_shots = shots if shots is not None else self._shots
        try:
            qnode = self._create_qnode(circuit_fn, shots=effective_shots)
            raw = qnode() if params is None else qnode(np.asarray(params, dtype=float))
            return self._normalize_result(raw)
        except QVMError:
            raise
        except Exception as e:
            raise ExecutionError(
                f"Execution failed for '{circuit_fn.__name__}': {e}"
            ) from e

    def run_batch(
        self,
        circuit_fn: Callable,
        params_batch: Union[Sequence[Sequence[float]], np.ndarray],
        shots: Optional[int] = None,
    ) -> np.ndarray:
        """Run a circuit over a batch of parameter vectors.

        Useful for parameter sweeps, hyperparameter searches, and vectorized
        loss surfaces — anywhere you'd otherwise write a Python loop calling
        :meth:`run` repeatedly.

        Args:
            circuit_fn: A function decorated with ``@qvm.circuit``.
            params_batch: Array-like of shape ``(N, k)`` where ``N`` is the
                batch size and ``k`` is the per-call parameter count.
            shots: Optional per-call shots override applied uniformly.

        Returns:
            A NumPy ``ndarray`` of stacked results — shape ``(N,)`` for scalar
            measurements, ``(N, m)`` for ``m``-dimensional measurements.
        """
        if not getattr(circuit_fn, "_is_qvm_circuit", False):
            raise CircuitError("Function must be decorated with @qvm.circuit")

        arr = np.asarray(params_batch, dtype=float)
        if arr.ndim != 2:
            raise CircuitError(
                f"params_batch must be a 2D array of shape (N, num_params); "
                f"got shape {arr.shape}"
            )
        if arr.shape[0] == 0:
            raise CircuitError("params_batch must contain at least one parameter set")

        # Phase 1.5: explicit Python loop. Robust on every backend, predictable
        # output shape, no broadcasting edge cases. Native PennyLane broadcasting
        # is a Phase 2 optimization to add when profiling shows it matters.
        results = [self.run(circuit_fn, params=arr[i], shots=shots) for i in range(arr.shape[0])]
        return np.asarray(results)

    def sample(
        self,
        circuit_fn: Callable,
        params: Optional[ParamLike] = None,
        shots: Optional[int] = None,
    ) -> np.ndarray:
        """Run a sampling circuit and return raw samples as an ``ndarray``.

        The decorated circuit must return ``qml.sample(...)``. ``shots`` must
        be positive (uses the runtime's default if not overridden).
        """
        if not getattr(circuit_fn, "_is_qvm_circuit", False):
            raise CircuitError("Function must be decorated with @qvm.circuit")

        effective_shots = shots if shots is not None else self._shots
        if not effective_shots or effective_shots <= 0:
            raise ExecutionError("sample() requires a positive shots value")

        try:
            qnode = self._create_qnode(circuit_fn, shots=effective_shots)
            raw = qnode() if params is None else qnode(np.asarray(params, dtype=float))
            return np.asarray(raw)
        except QVMError:
            raise
        except Exception as e:
            raise ExecutionError(
                f"Sampling failed for '{circuit_fn.__name__}': {e}"
            ) from e

    def state(
        self,
        circuit_fn: Callable,
        params: Optional[ParamLike] = None,
    ) -> np.ndarray:
        """Compute the statevector for a circuit (analytic, no shots).

        The decorated circuit must return ``qml.state()``. This is intended
        for small circuits on state-supporting devices (e.g. ``default.qubit``,
        ``lightning.qubit``). If the active device cannot produce a state,
        :class:`BackendError` is raised.
        """
        if not getattr(circuit_fn, "_is_qvm_circuit", False):
            raise CircuitError("Function must be decorated with @qvm.circuit")

        try:
            qnode = self._create_qnode(circuit_fn, shots=None)
            raw = qnode() if params is None else qnode(np.asarray(params, dtype=float))
            return np.asarray(raw)
        except QVMError:
            raise
        except Exception as e:
            msg = str(e).lower()
            if "state" in msg and ("support" in msg or "not " in msg):
                raise BackendError(
                    f"Backend '{self._backend_name}' does not support state output: {e}"
                ) from e
            raise ExecutionError(
                f"State computation failed for '{circuit_fn.__name__}': {e}"
            ) from e

    # =====================
    # Inspection
    # =====================

    def trace(
        self,
        circuit_fn: Callable,
        params: Optional[ParamLike] = None,
    ) -> list[tuple[str, np.ndarray]]:
        """Walk a circuit gate by gate, returning the state after each step.

        This is the educational view of a quantum program: every entry in the
        returned list is ``(gate_label, state)`` showing the wavefunction
        immediately after applying that gate, with the initial ``|0…0⟩`` state
        as the first entry. Useful for teaching, debugging, and notebook
        explanations.

        Args:
            circuit_fn: A function decorated with ``@qvm.circuit``. The
                circuit's returned measurement is ignored — ``trace`` always
                captures the full statevector.
            params: Optional parameter vector forwarded to the circuit.

        Returns:
            A list of ``(label, statevector)`` pairs. Labels are formatted as
            ``"i: GateName(args, wires)"``; the first entry is ``"0: init"``.
            Statevectors are NumPy arrays of length ``2**n_wires``.
        """
        if not getattr(circuit_fn, "_is_qvm_circuit", False):
            raise CircuitError("Function must be decorated with @qvm.circuit")

        try:
            # 1. Materialize the user's circuit as a PennyLane tape so we can
            #    enumerate its operations.
            qnode = self._create_qnode(circuit_fn, shots=None)
            tape_builder = qml.workflow.construct_tape(qnode)
            tape = (
                tape_builder()
                if params is None
                else tape_builder(np.asarray(params, dtype=float))
            )

            # 2. Size a fresh device so every snapshot returns the full-width
            #    state vector (avoids the auto-resize that hides untouched wires).
            wires_used: set[int] = set()
            for op in tape.operations:
                for w in op.wires.tolist():
                    if isinstance(w, int):
                        wires_used.add(w)
            n_wires = max(wires_used) + 1 if wires_used else 1
            trace_device = qml.device("default.qubit", wires=n_wires)

            # 3. Build a new circuit that interleaves snapshots between gates.
            ops = list(tape.operations)
            labels = ["0: init"] + [
                f"{i + 1}: {_format_op(op)}" for i, op in enumerate(ops)
            ]

            def _snapshotted() -> qml.measurements.StateMP:
                qml.Snapshot(labels[0])
                for op, label in zip(ops, labels[1:]):
                    qml.apply(op)
                    qml.Snapshot(label)
                return qml.state()

            snap_qnode = qml.QNode(_snapshotted, trace_device)
            snaps = qml.snapshots(snap_qnode)()

            # 4. Walk through snapshots in insertion order. ``qml.snapshots``
            #    returns a dict with one entry per Snapshot plus a final
            #    ``"execution_results"`` key we discard.
            result: list[tuple[str, np.ndarray]] = []
            for label in labels:
                if label in snaps:
                    result.append((label, np.asarray(snaps[label])))
            return result
        except QVMError:
            raise
        except Exception as e:
            raise ExecutionError(
                f"Failed to trace circuit '{circuit_fn.__name__}': {e}"
            ) from e

    def draw(
        self,
        circuit_fn: Callable,
        params: Optional[ParamLike] = None,
    ) -> str:
        """Return an ASCII diagram of a ``@qvm.circuit``.

        A thin wrapper over :func:`pennylane.draw`. Pass ``params`` if the
        circuit takes them; omit otherwise. The diagram is structural —
        shot count and device are irrelevant.
        """
        if not getattr(circuit_fn, "_is_qvm_circuit", False):
            raise CircuitError("Function must be decorated with @qvm.circuit")
        try:
            qnode = self._create_qnode(circuit_fn, shots=None)
            drawer = qml.draw(qnode)
            if params is None:
                return drawer()
            return drawer(np.asarray(params, dtype=float))
        except QVMError:
            raise
        except Exception as e:
            raise CircuitError(
                f"Failed to draw circuit '{circuit_fn.__name__}': {e}"
            ) from e

    # =====================
    # Differentiation
    # =====================

    def make_qnode(
        self,
        circuit_fn: Callable,
        shots: Optional[int] = None,
        analytic: bool = False,
    ) -> qml.QNode:
        """Return the underlying PennyLane QNode for a ``@qvm.circuit``.

        Args:
            circuit_fn: Function decorated with ``@qvm.circuit``.
            shots: Optional shot count. Falls back to the runtime default
                if not provided.
            analytic: If ``True``, build an analytic (no-shots) QNode. Takes
                precedence over ``shots``.
        """
        if not getattr(circuit_fn, "_is_qvm_circuit", False):
            raise CircuitError("Function must be decorated with @qvm.circuit")
        effective_shots = None if analytic else (
            shots if shots is not None else self._shots
        )
        return self._create_qnode(circuit_fn, shots=effective_shots)

    def grad(self, fn: Callable) -> Callable:
        """Return the gradient of a scalar-valued function via ``qml.grad``.

        If ``fn`` is a ``@qvm.circuit``, an analytic QNode is built
        internally so gradients are free of shot noise. For raw QNodes or
        ``@qvm.hybrid`` functions returning a Python float, ``fn`` is
        passed straight to ``qml.grad``. The differentiable input must be
        a ``pennylane.numpy`` array with ``requires_grad=True``.
        """
        target = fn
        if getattr(fn, "_is_qvm_circuit", False):
            target = self._create_qnode(fn, shots=None)
        try:
            return qml.grad(target)
        except Exception as e:
            raise ExecutionError(f"Failed to build gradient function: {e}") from e

    # =====================
    # Internals
    # =====================

    def _create_qnode(
        self, circuit_fn: Callable, shots: Optional[int]
    ) -> qml.QNode:
        """Build a PennyLane QNode for the active device.

        ``shots`` is the effective shot count: a positive ``int`` for
        finite-shot execution, or ``None`` for analytic (statevector)
        execution. Callers are responsible for resolving defaults.
        """
        if self._device is None:
            raise BackendError("Device not initialized")

        try:
            qnode = qml.QNode(
                circuit_fn,
                self._device,
                diff_method=self._diff_method,
            )
            return qml.set_shots(qnode, shots=shots)
        except Exception as e:
            raise CircuitError(f"Failed to create QNode: {e}") from e

    @staticmethod
    def _normalize_result(result: Any) -> RunResult:
        """Coerce PennyLane outputs to predictable Python/NumPy types.

        0-d arrays/tensors become Python floats; n-d outputs become
        ``np.ndarray``; tuples (multi-measurement circuits) are normalized
        element-wise.
        """
        if isinstance(result, tuple):
            return tuple(QuantumRuntime._normalize_result(r) for r in result)
        arr = np.asarray(result)
        if arr.ndim == 0:
            return float(arr)
        return arr

    def __repr__(self) -> str:
        return f"QuantumRuntime(backend={self._backend_name}, shots={self._shots})"
