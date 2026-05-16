"""Parameter sweep — visualize a quantum cost surface in one call.

The cost ``⟨Z⟩`` on a single rotated qubit is the textbook ``cos(θ)`` curve.
This example sweeps θ from 0 to 2π in 32 steps using :meth:`run_batch`,
then renders the resulting wave as an ASCII bar chart.

Pattern this demonstrates:

  - One call instead of a hand-rolled for-loop.
  - Stacked NumPy output you can immediately plot, fit, or analyze.
  - Works the same for vector measurements — try swapping ``qml.expval``
    for ``qml.probs(wires=0)`` and inspect ``result.shape`` afterwards.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from qvm import QuantumRuntime


def _bar(value: float, width: int = 30) -> str:
    """Render a value in [-1, +1] as a bar centered around the midpoint."""
    half = width // 2
    if value >= 0:
        cells = int(round(value * half))
        return " " * half + "│" + "█" * cells + " " * (half - cells)
    cells = int(round(-value * half))
    return " " * (half - cells) + "█" * cells + "│" + " " * half


def main() -> None:
    qvm = QuantumRuntime()

    @qvm.circuit
    def cost(params):
        qml.RX(params[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    n_points = 32
    thetas = np.linspace(0.0, 2 * np.pi, n_points)
    batch = thetas.reshape(-1, 1)

    values = qvm.run_batch(cost, batch)

    print(f"  Swept θ from 0 to 2π in {n_points} points.")
    print(f"  qvm.run_batch returned shape={values.shape}, dtype={values.dtype}.")
    print()
    print("  θ          ⟨Z⟩       -1                  0                  +1")
    print("  " + "─" * 70)
    for theta, v in zip(thetas, values):
        print(f"  {theta:5.3f}   {v:+7.4f}    {_bar(v)}")
    print()
    print("  That cosine you see is ⟨Z⟩ = cos(θ) — the analytic expectation")
    print("  value after rotating |0⟩ by θ around the X-axis.")


if __name__ == "__main__":
    main()
