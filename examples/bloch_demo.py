"""See entanglement collapse the Bloch sphere — a step-by-step Bloch tour.

Combines :func:`qvm.trace` with :func:`qvm.bloch`. At every gate in a Bell
circuit we render the Bloch sphere of qubit 0. Watch the vector slide from
the north pole down to the equator after Hadamard, then collapse to the
*origin* after CNOT — that collapse is entanglement in its rawest visual
form: a perfectly mixed local state.
"""

from __future__ import annotations

import pennylane as qml

from qvm import QuantumRuntime, bloch


def main() -> None:
    qvm = QuantumRuntime()

    @qvm.circuit
    def bell():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    trace = qvm.trace(bell)

    for label, state in trace:
        print("─" * 60)
        print(f"  {label}")
        print("─" * 60)
        print(bloch(state, wire=0))
        print()

    print("─" * 60)
    print("  Reading the trace")
    print("─" * 60)
    print("  ▸ At step 0, qubit 0 starts as |0⟩ — Bloch vector at the north pole.")
    print("  ▸ After H(0), it lands on the +X equator — a pure superposition state.")
    print("  ▸ After CNOT, the local Bloch vector collapses to the origin: qubit 0")
    print("    has become a maximally mixed state. The information is no longer")
    print("    in either qubit individually — it lives in the *correlations*")
    print("    between them. That's entanglement, drawn at the geometric level.")


if __name__ == "__main__":
    main()
