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


# ---------- run_batch() ----------

def test_run_batch_scalar_returns_1d_ndarray(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def circuit(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    batch = np.linspace(0.0, np.pi, 5).reshape(-1, 1)
    result = qvm.run_batch(circuit, batch)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    # cos(0)=1, cos(pi)=-1 — endpoints should be close within shot noise
    assert result[0] == pytest.approx(1.0, abs=0.1)
    assert result[-1] == pytest.approx(-1.0, abs=0.1)


def test_run_batch_vector_measurement_stacks_results(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def circuit(params: np.ndarray) -> qml.measurements.ProbabilityMP:
        qml.RX(params[0], wires=0)
        return qml.probs(wires=0)

    batch = np.linspace(0.0, np.pi, 4).reshape(-1, 1)
    result = qvm.run_batch(circuit, batch)
    assert result.shape == (4, 2)
    # probabilities sum to 1 for each row
    np.testing.assert_allclose(result.sum(axis=1), np.ones(4), atol=0.05)


def test_run_batch_rejects_1d_params(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def circuit(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(CircuitError):
        qvm.run_batch(circuit, np.array([0.1, 0.2, 0.3]))


def test_run_batch_rejects_empty_batch(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def circuit(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(CircuitError):
        qvm.run_batch(circuit, np.zeros((0, 1)))


def test_run_batch_rejects_undecorated_function(qvm: QuantumRuntime) -> None:
    def bare(params: np.ndarray) -> qml.measurements.ExpectationMP:
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(CircuitError):
        qvm.run_batch(bare, np.zeros((3, 1)))


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


# ---------- trace() ----------

def test_trace_returns_state_after_each_gate(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def bell() -> qml.measurements.StateMP:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    trace = qvm.trace(bell)
    # init + Hadamard + CNOT = 3 entries
    assert len(trace) == 3

    # Initial state is |00⟩
    _, init_state = trace[0]
    np.testing.assert_allclose(init_state, [1.0, 0.0, 0.0, 0.0], atol=1e-9)

    # After Hadamard: (|00⟩ + |10⟩) / √2 — amplitudes at indices 0 and 2
    _, h_state = trace[1]
    inv_sqrt2 = 1.0 / np.sqrt(2)
    np.testing.assert_allclose(h_state, [inv_sqrt2, 0.0, inv_sqrt2, 0.0], atol=1e-9)

    # After CNOT: Bell state (|00⟩ + |11⟩) / √2 — amplitudes at indices 0 and 3
    _, cnot_state = trace[2]
    np.testing.assert_allclose(cnot_state, [inv_sqrt2, 0.0, 0.0, inv_sqrt2], atol=1e-9)


def test_trace_labels_include_step_index_and_gate(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def bell() -> qml.measurements.StateMP:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    labels = [label for label, _ in qvm.trace(bell)]
    assert labels[0].startswith("0:") and "init" in labels[0]
    assert labels[1].startswith("1:") and "H" in labels[1] and "0" in labels[1]
    assert labels[2].startswith("2:") and "CNOT" in labels[2]


def test_trace_formats_parameterized_gates(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def parameterized(p: np.ndarray) -> qml.measurements.StateMP:
        qml.RX(p[0], wires=0)
        return qml.state()

    trace = qvm.trace(parameterized, params=[0.5])
    labels = [label for label, _ in trace]
    # Second label should contain the gate name and the rendered parameter
    assert "RX" in labels[1]
    assert "0.500" in labels[1]


def test_trace_rejects_undecorated_function(qvm: QuantumRuntime) -> None:
    def bare() -> qml.measurements.StateMP:
        return qml.state()

    with pytest.raises(CircuitError):
        qvm.trace(bare)


# ---------- draw ----------

def test_draw_renders_circuit_structure(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def bell() -> qml.measurements.StateMP:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    diagram = qvm.draw(bell)
    assert isinstance(diagram, str)
    # The Hadamard and the CNOT control/target should appear in the diagram.
    assert "H" in diagram
    assert "0:" in diagram and "1:" in diagram


def test_draw_renders_param_values(qvm: QuantumRuntime) -> None:
    @qvm.circuit
    def parameterized(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    diagram = qvm.draw(parameterized, params=[0.5])
    assert "RX" in diagram
    assert "0.50" in diagram  # PennyLane formats floats to 2 decimal places


def test_draw_rejects_undecorated_function(qvm: QuantumRuntime) -> None:
    def bare() -> qml.measurements.ExpectationMP:
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(CircuitError):
        qvm.draw(bare)


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
