"""Tests for the SciPy minimize() bridge."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from qvm import OptimizationResult, QuantumRuntime, minimize
from qvm.exceptions import CircuitError


@pytest.fixture
def qvm() -> QuantumRuntime:
    return QuantumRuntime(backend="default.qubit")


@pytest.fixture
def single_qubit_cost(qvm: QuantumRuntime):
    @qvm.circuit
    def cost(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    return cost


def test_minimize_rejects_undecorated_function(qvm: QuantumRuntime) -> None:
    def bare(params: np.ndarray) -> qml.measurements.ExpectationMP:
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(CircuitError):
        minimize(bare, x0=[0.0], runtime=qvm)


def test_minimize_lbfgsb_converges_to_minimum(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    result = minimize(single_qubit_cost, x0=[0.5, 0.3], runtime=qvm, method="L-BFGS-B")
    assert isinstance(result, OptimizationResult)
    assert result.best_cost == pytest.approx(-1.0, abs=1e-4)
    assert result.converged is True


def test_minimize_nelder_mead_converges_to_minimum(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    # Gradient-free method — verifies that passing jac= doesn't break it.
    result = minimize(
        single_qubit_cost,
        x0=[0.5, 0.3],
        runtime=qvm,
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6},
    )
    assert result.best_cost == pytest.approx(-1.0, abs=1e-3)


def test_minimize_records_history_with_initial_point(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    result = minimize(single_qubit_cost, x0=[0.5, 0.3], runtime=qvm, method="L-BFGS-B")
    # Initial point + at least one iteration must be recorded
    assert len(result.history) >= 2
    step_0, params_0, cost_0 = result.history[0]
    assert step_0 == 0
    np.testing.assert_allclose(params_0, [0.5, 0.3])
    assert isinstance(cost_0, float)
    # Final history cost should match best_cost (monotonic for L-BFGS-B here)
    assert result.history[-1][2] == pytest.approx(result.best_cost, abs=1e-6)


def test_minimize_options_maxiter_is_honored(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    result = minimize(
        single_qubit_cost,
        x0=[0.5, 0.3],
        runtime=qvm,
        method="L-BFGS-B",
        options={"maxiter": 1},
    )
    # maxiter=1 caps the number of accepted iterates; steps_taken should be small
    assert result.steps_taken <= 2
