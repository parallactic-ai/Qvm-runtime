"""Basic tests for QuantumRuntime (Phase 1)."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from qvm import QuantumRuntime, grad, hybrid
from qvm.exceptions import BackendError, CircuitError, ExecutionError


@pytest.fixture
def qvm() -> QuantumRuntime:
    return QuantumRuntime(backend="default.qubit", shots=2048)


# ---------- run() ----------

def test_run_without_params_returns_float(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def circuit() -> qml.measurements.ExpectationMP:
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    result = qvm.run(circuit)
    assert isinstance(result, float)
    assert abs(result) < 0.2  # Hadamard gives <Z> = 0 within shot noise


def test_run_with_params_returns_float(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def circuit(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    result = qvm.run(circuit, params=[0.0])
    assert isinstance(result, float)
    assert result == pytest.approx(1.0, abs=0.1)


def test_run_rejects_undecorated_function(qvm: QuantumRuntime) -> None:
    def bare(params: np.ndarray) -> qml.measurements.ExpectationMP:
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(CircuitError):
        qvm.run(bare, params=[0.0])


def test_run_propagates_circuit_error_for_invalid_gate(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def circuit(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires="not-a-wire-index-but-fine")
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ExecutionError):
        qvm.run(circuit, params=["this is not a float"])  # type: ignore[list-item]


# ---------- sample() ----------

def test_sample_returns_ndarray_of_eigenvalues(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def sampler() -> qml.measurements.SampleMP:
        qml.Hadamard(wires=0)
        return qml.sample(qml.PauliZ(0))

    samples = qvm.sample(sampler, shots=128)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (128,)
    assert set(np.unique(samples).tolist()).issubset({-1, 1})


def test_sample_requires_positive_shots(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def sampler() -> qml.measurements.SampleMP:
        qml.Hadamard(wires=0)
        return qml.sample(qml.PauliZ(0))

    with pytest.raises(ExecutionError):
        qvm.sample(sampler, shots=0)


# ---------- state() ----------

def test_state_returns_normalized_statevector(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def prep() -> qml.measurements.StateMP:
        qml.Hadamard(wires=0)
        return qml.state()

    state = qvm.state(prep)
    assert isinstance(state, np.ndarray)
    inv_sqrt2 = 1.0 / np.sqrt(2)
    np.testing.assert_allclose(state, np.array([inv_sqrt2, inv_sqrt2]), atol=1e-6)


# ---------- hybrid decorator ----------

def test_hybrid_passes_value_through(qvm: QuantumRuntime) -> None:
    @qvm.hybrid
    def add(a: float, b: float) -> float:
        return a + b

    assert add(2.0, 3.0) == 5.0


def test_hybrid_wraps_exceptions_in_execution_error(qvm: QuantumRuntime) -> None:
    @qvm.hybrid
    def bad() -> float:
        raise ValueError("boom")

    with pytest.raises(ExecutionError):
        bad()


# ---------- backend ----------

def test_invalid_backend_raises_backend_error() -> None:
    with pytest.raises(BackendError):
        QuantumRuntime(backend="not.a.real.backend")


# ---------- gradient ----------

def test_grad_of_quantum_circuit(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def cost(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    grad_fn = qvm.grad(cost)  # analytic QNode built internally
    params = pnp.array([np.pi / 2], requires_grad=True)
    g = grad_fn(params)
    # d/dtheta <Z> = -sin(theta); at pi/2 -> -1
    assert g[0] == pytest.approx(-1.0, abs=1e-5)


# ---------- module-level shortcuts ----------

def test_module_level_hybrid_wraps_exception() -> None:
    @hybrid
    def explode() -> float:
        raise RuntimeError("kaboom")

    with pytest.raises(ExecutionError):
        explode()


def test_module_level_grad_routes_through_bound_runtime(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def cost(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    grad_fn = grad(cost)  # module-level: finds qvm via _qvm_runtime
    params = pnp.array([np.pi / 2], requires_grad=True)
    g = grad_fn(params)
    assert g[0] == pytest.approx(-1.0, abs=1e-5)
