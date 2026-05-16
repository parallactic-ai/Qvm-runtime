"""Gradient example for QVM hybrid workflows.

Demonstrates how to:
  1. Get a raw PennyLane QNode out of a ``@qvm.circuit`` via ``make_qnode``.
  2. Differentiate it with ``qvm.grad`` (a thin wrapper around ``qml.grad``).
  3. Combine the gradient with a classical step to take a hybrid update.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

from qvm import QuantumRuntime

qvm = QuantumRuntime(backend="default.qubit")


@qvm.circuit
def cost(params: pnp.ndarray) -> qml.measurements.ExpectationMP:
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


grad_fn = qvm.grad(cost)
analytic_qnode = qvm.make_qnode(cost, analytic=True)


@qvm.hybrid
def gradient_descent_step(params: pnp.ndarray, lr: float = 0.1) -> pnp.ndarray:
    """One classical gradient step over the quantum cost function."""
    grads = grad_fn(params)
    return params - lr * grads


if __name__ == "__main__":
    params = pnp.array([0.5, 0.3], requires_grad=True)

    print(f"initial params: {params}")
    print(f"initial cost:   {float(analytic_qnode(params)):.6f}")

    for step in range(5):
        params = gradient_descent_step(params, lr=0.4)
        value = float(analytic_qnode(params))
        print(f"step {step + 1}: cost={value:.6f}, params={params}")
