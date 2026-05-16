"""Step-by-step trace of a Bell-state circuit — see entanglement build up.

``qvm.trace`` returns the full statevector after every gate in a circuit.
This example renders that trace as an annotated walkthrough: at each step
you see the gate applied, the wavefunction's non-zero amplitudes, and a
plain-English description of what just changed.

This is the kind of view that makes "quantum is weird" click — the H gate
creates superposition on one qubit, then CNOT *correlates* the qubits so
that measuring one instantly determines the other.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from qvm import QuantumRuntime


def _format_state(state: np.ndarray, n_qubits: int) -> str:
    """Pretty-print non-zero amplitudes as ``α|basis⟩``."""
    terms = []
    for i, amp in enumerate(state):
        if abs(amp) > 1e-9:
            basis = format(i, f"0{n_qubits}b")
            mag = abs(amp)
            if abs(amp.imag) < 1e-9:
                terms.append(f"{amp.real:+.4f}|{basis}⟩")
            else:
                terms.append(f"({amp:+.4f})|{basis}⟩")
            _ = mag  # silence unused
    return "  ".join(terms) if terms else "0"


EXPLANATIONS = {
    "init": "The system starts in the computational |00⟩ ground state.",
    "H": "Hadamard rotates qubit 0 into an equal superposition of |0⟩ and |1⟩.",
    "CNOT": "CNOT flips qubit 1 iff qubit 0 is |1⟩ — entangling the two.",
}


def _explain(label: str) -> str:
    for key, sentence in EXPLANATIONS.items():
        if key in label:
            return sentence
    return ""


def main() -> None:
    qvm = QuantumRuntime()

    @qvm.circuit
    def bell():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    trace = qvm.trace(bell)
    n_qubits = int(np.log2(len(trace[0][1])))

    print("─" * 70)
    print("  Building a Bell state, one gate at a time")
    print("─" * 70)
    for label, state in trace:
        print()
        print(f"  {label}")
        print(f"    state    = {_format_state(state, n_qubits)}")
        explanation = _explain(label)
        if explanation:
            print(f"    meaning  = {explanation}")
    print()
    print("─" * 70)
    print("  The final state (|00⟩ + |11⟩) / √2 is the canonical Bell pair —")
    print("  the qubits are now perfectly correlated: measuring one tells you")
    print("  the other, no matter how far apart you put them.")
    print("─" * 70)


if __name__ == "__main__":
    main()
