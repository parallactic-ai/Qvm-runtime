"""Variational quantum algorithm in ten lines — the Qapp story.

A Qapp bundles a parameterized circuit + runtime + optimizer into one
object with a ``fit`` method. This is the same gradient-descent loop you
see in ``examples/gradient_example.py`` — but written by Qapp, not by hand.

What you'll see:
  - 5 lines of setup, 1 line to fit
  - The cost surface ⟨Z⟩ minimized toward -1
  - A clean OptimizationResult with history, best params, convergence flag
"""

from __future__ import annotations

import pennylane as qml

from qvm import Qapp, QuantumRuntime


def main() -> None:
    qvm = QuantumRuntime()

    @qvm.circuit
    def cost(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    app = Qapp(cost, runtime=qvm, learning_rate=0.4)
    result = app.fit(initial_params=[0.5, 0.3], steps=30, tol=1e-6)

    print("─" * 60)
    print("  Optimization summary")
    print("─" * 60)
    print(f"  steps taken : {result.steps_taken}")
    print(f"  converged   : {result.converged}")
    print(f"  best cost   : {result.best_cost:+.6f}   (minimum of ⟨Z⟩ is -1)")
    print(f"  best params : {result.best_params}")
    print()

    print("─" * 60)
    print("  Cost trajectory")
    print("─" * 60)
    for step, params, c in result.history[::5]:  # every 5th step
        bar = "█" * int((1 + c) * 25)  # ⟨Z⟩ ∈ [-1, +1] -> bar length 0..50
        print(f"  step {step:>2} | cost={c:+.4f} | {bar}")
    print()

    print("─" * 60)
    print("  Spot-check at the converged params")
    print("─" * 60)
    spot = app.evaluate(result.best_params)
    grad = app.grad_at(result.best_params)
    print(f"  cost(best_params)  = {spot:+.6f}")
    print(f"  grad(best_params)  = {grad}     (should be near zero)")


if __name__ == "__main__":
    main()
