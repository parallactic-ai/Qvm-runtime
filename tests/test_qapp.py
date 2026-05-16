"""Tests for the Qapp abstraction."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from qvm import OptimizationResult, Qapp, QuantumRuntime
from qvm.exceptions import CircuitError


@pytest.fixture
def qvm() -> QuantumRuntime:
    return QuantumRuntime(backend="default.qubit", shots=1024)


@pytest.fixture
def single_qubit_cost(qvm: QuantumRuntime):
    @qvm.circuit
    def cost(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    return cost


def test_qapp_requires_qvm_circuit(qvm: QuantumRuntime) -> None:
    def bare(params: np.ndarray) -> qml.measurements.ExpectationMP:
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(CircuitError):
        Qapp(bare, runtime=qvm)


def test_qapp_evaluate_returns_python_float(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    app = Qapp(single_qubit_cost, runtime=qvm)
    value = app.evaluate([0.0, 0.0])
    assert isinstance(value, float)
    # RX(0)·RY(0)|0⟩ = |0⟩ → ⟨Z⟩ = 1
    assert value == pytest.approx(1.0, abs=1e-6)


def test_qapp_grad_at_returns_ndarray(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    app = Qapp(single_qubit_cost, runtime=qvm)
    g = app.grad_at([np.pi / 2, 0.0])
    assert isinstance(g, np.ndarray)
    assert g.shape == (2,)
    # ∂/∂θ₀ ⟨Z⟩ for RX(θ₀)·RY(0) at θ₀=π/2 is -sin(π/2) = -1
    assert g[0] == pytest.approx(-1.0, abs=1e-5)


def test_qapp_fit_minimizes_cost(qvm: QuantumRuntime, single_qubit_cost) -> None:
    app = Qapp(single_qubit_cost, runtime=qvm, learning_rate=0.4)
    result = app.fit(initial_params=[0.5, 0.3], steps=50)

    assert isinstance(result, OptimizationResult)
    assert result.best_cost < 0.0  # got past the equator
    assert result.best_cost == pytest.approx(-1.0, abs=0.01)  # near minimum
    assert result.steps_taken > 0


def test_qapp_fit_records_full_history(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    app = Qapp(single_qubit_cost, runtime=qvm, learning_rate=0.4)
    result = app.fit(initial_params=[0.5, 0.3], steps=10)

    # history includes step 0 plus each step that ran
    assert len(result.history) == result.steps_taken + 1
    # each entry is (step, params, cost)
    step_0, params_0, cost_0 = result.history[0]
    assert step_0 == 0
    assert isinstance(params_0, np.ndarray)
    assert isinstance(cost_0, float)
    assert result.costs()[0] == cost_0


def test_qapp_fit_converges_flag(qvm: QuantumRuntime, single_qubit_cost) -> None:
    # Aggressive tol: should converge well before 200 steps
    app = Qapp(single_qubit_cost, runtime=qvm, learning_rate=0.4)
    result = app.fit(initial_params=[0.1, 0.1], steps=200, tol=1e-8)
    assert result.converged is True
    assert result.steps_taken < 200


def test_qapp_fit_callback_invoked_per_step(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    calls: list[tuple[int, float]] = []

    def cb(step: int, params: np.ndarray, cost: float) -> None:
        calls.append((step, cost))

    app = Qapp(single_qubit_cost, runtime=qvm, learning_rate=0.4)
    result = app.fit(initial_params=[0.5, 0.3], steps=5, tol=0.0, callback=cb)

    # one callback per history entry (includes step 0)
    assert len(calls) == len(result.history)
    assert calls[0][0] == 0
