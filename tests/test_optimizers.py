"""Tests for the gradient-based optimizers."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from qvm import SGD, Adam, Momentum, Qapp, QuantumRuntime


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


# ---------- Standalone optimizer math ----------

def test_sgd_step_is_plain_subtraction() -> None:
    opt = SGD(learning_rate=0.5)
    params = np.array([1.0, 2.0])
    grads = np.array([0.2, -0.4])
    new = opt.step(params, grads)
    np.testing.assert_allclose(new, [0.9, 2.2])


def test_momentum_accumulates_velocity() -> None:
    opt = Momentum(learning_rate=0.1, momentum=0.5)
    params = np.array([1.0])
    # First step: velocity starts at 0, becomes -0.1 * 1.0 = -0.1
    p1 = opt.step(params, np.array([1.0]))
    np.testing.assert_allclose(p1, [0.9])
    # Second step: velocity = 0.5 * -0.1 - 0.1 * 1.0 = -0.15
    p2 = opt.step(p1, np.array([1.0]))
    np.testing.assert_allclose(p2, [0.75])


def test_momentum_reset_clears_velocity() -> None:
    opt = Momentum(learning_rate=0.1, momentum=0.9)
    opt.step(np.array([1.0]), np.array([1.0]))
    opt.reset()
    # After reset, first step should match a fresh Momentum at the same params
    p_after_reset = opt.step(np.array([1.0]), np.array([1.0]))
    fresh = Momentum(learning_rate=0.1, momentum=0.9)
    p_fresh = fresh.step(np.array([1.0]), np.array([1.0]))
    np.testing.assert_allclose(p_after_reset, p_fresh)


def test_adam_bias_correction_on_first_step() -> None:
    # On step 1, bias-corrected m_hat = grads, v_hat = grads**2.
    # Update = lr * grads / (|grads| + eps) ≈ lr * sign(grads).
    opt = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999, eps=1e-12)
    params = np.array([1.0, 1.0])
    grads = np.array([2.0, -3.0])
    new = opt.step(params, grads)
    # update[0] ≈ params[0] - lr * sign(grads[0]) = 1 - 0.1 = 0.9
    # update[1] ≈ params[1] - lr * sign(grads[1]) = 1 + 0.1 = 1.1
    np.testing.assert_allclose(new, [0.9, 1.1], atol=1e-6)


def test_adam_reset_clears_moments_and_step_counter() -> None:
    opt = Adam(learning_rate=0.1)
    opt.step(np.array([1.0]), np.array([1.0]))
    opt.step(np.array([0.9]), np.array([0.5]))
    opt.reset()
    # After reset, internal _t goes back to 0 and moments back to zero
    assert opt._t == 0
    assert opt._m is None
    assert opt._v is None


# ---------- Optimizer integration with Qapp ----------

def test_qapp_with_sgd_explicit(qvm: QuantumRuntime, single_qubit_cost) -> None:
    app = Qapp(single_qubit_cost, runtime=qvm, optimizer=SGD(0.4))
    result = app.fit(initial_params=[0.5, 0.3], steps=50)
    assert result.best_cost == pytest.approx(-1.0, abs=0.01)


def test_qapp_with_momentum_converges(qvm: QuantumRuntime, single_qubit_cost) -> None:
    app = Qapp(
        single_qubit_cost, runtime=qvm, optimizer=Momentum(learning_rate=0.1, momentum=0.9)
    )
    result = app.fit(initial_params=[0.5, 0.3], steps=100)
    assert result.best_cost == pytest.approx(-1.0, abs=0.01)


def test_qapp_with_adam_converges(qvm: QuantumRuntime, single_qubit_cost) -> None:
    app = Qapp(single_qubit_cost, runtime=qvm, optimizer=Adam(learning_rate=0.1))
    result = app.fit(initial_params=[0.5, 0.3], steps=100)
    assert result.best_cost == pytest.approx(-1.0, abs=0.01)


def test_qapp_optimizer_property_exposes_instance(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    adam = Adam(learning_rate=0.05)
    app = Qapp(single_qubit_cost, runtime=qvm, optimizer=adam)
    assert app.optimizer is adam


def test_qapp_reuses_optimizer_across_fits(
    qvm: QuantumRuntime, single_qubit_cost
) -> None:
    # Reusing an Adam instance must not leak state — reset() should be called.
    adam = Adam(learning_rate=0.1)
    app = Qapp(single_qubit_cost, runtime=qvm, optimizer=adam)

    r1 = app.fit(initial_params=[0.5, 0.3], steps=5)
    r2 = app.fit(initial_params=[0.5, 0.3], steps=5)

    # Same start, same steps, same optimizer => same trajectory cost-wise
    np.testing.assert_allclose(r1.costs(), r2.costs(), atol=1e-8)
