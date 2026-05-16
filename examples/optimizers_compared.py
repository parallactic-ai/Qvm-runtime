"""Three optimizers, one cost — see how convergence differs.

Same starting point, same cost surface (⟨Z⟩ of a 2-parameter circuit),
same step budget. The only thing that changes between runs is which
:class:`Optimizer` is plugged into the :class:`Qapp`.

You'll see:

  - SGD plodding steadily downward.
  - Momentum accelerating through the curved region.
  - Adam reaching the floor first by adapting its per-coordinate step.
"""

from __future__ import annotations

from typing import List

import pennylane as qml

from qvm import SGD, Adam, Momentum, Optimizer, Qapp, QuantumRuntime


def _run(opt: Optimizer, name: str, qvm: QuantumRuntime, cost_fn) -> List[float]:
    app = Qapp(cost_fn, runtime=qvm, optimizer=opt)
    result = app.fit(initial_params=[0.5, 0.3], steps=40, tol=0.0)
    final = result.history[-1][2]
    print(f"  {name:<10} final cost = {final:+.6f}   steps = {result.steps_taken}")
    return result.costs()


def _bar(c: float, width: int = 30) -> str:
    """Render a cost in [-1, +1] as a bar of the given width."""
    cells = max(0, min(width, int(round((1 + c) / 2 * width))))
    return "█" * cells + "·" * (width - cells)


def main() -> None:
    qvm = QuantumRuntime()

    @qvm.circuit
    def cost(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    print("─" * 64)
    print("  Final results")
    print("─" * 64)
    sgd_curve = _run(SGD(learning_rate=0.2), "SGD", qvm, cost)
    mom_curve = _run(Momentum(learning_rate=0.1, momentum=0.9), "Momentum", qvm, cost)
    adam_curve = _run(Adam(learning_rate=0.2), "Adam", qvm, cost)

    print()
    print("─" * 64)
    print("  Cost over time (bar = how close to +1; empty = how close to -1)")
    print("─" * 64)
    print(f"  {'step':<5} {'SGD':<32} {'Momentum':<32} {'Adam':<32}")
    for step in [0, 5, 10, 15, 20, 30, 40]:
        if step >= len(sgd_curve):
            continue
        s = _bar(sgd_curve[step])
        m = _bar(mom_curve[step])
        a = _bar(adam_curve[step])
        print(f"  {step:<5} {s} {m} {a}")
    print()
    print("  All three reach the minimum, but watch how fast each gets there.")


if __name__ == "__main__":
    main()
