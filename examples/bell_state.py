"""Bell state — the simplest interesting quantum program.

Two qubits prepared so that measuring qubit 0 instantly tells you what
qubit 1 will be, even though *individually* each qubit looks like a
fair coin. That's entanglement.

This example walks through four views of the same circuit:

  1. The diagram          (qvm.draw)
  2. The exact state      (qvm.state)
  3. Joint sample outcomes (qvm.sample)        - notice both qubits always agree
  4. Marginal expectations (qvm.run)            - each qubit alone is unbiased
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from qvm import QuantumRuntime

qvm = QuantumRuntime(shots=2048)


@qvm.circuit
def bell_state():
    """Apply H to q0, then CNOT to entangle q0 and q1."""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()


@qvm.circuit
def bell_samples():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.sample(wires=[0, 1])


@qvm.circuit
def bell_marginals():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    # Two simultaneous single-qubit expectations.
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))


def _format_state(state: np.ndarray) -> str:
    """Pretty-print non-zero amplitudes as |basis⟩ : amplitude."""
    n_qubits = int(np.log2(len(state)))
    lines = []
    for i, amp in enumerate(state):
        if abs(amp) > 1e-9:
            basis = format(i, f"0{n_qubits}b")
            lines.append(f"  |{basis}⟩  :  {amp.real:+.4f}{amp.imag:+.4f}j")
    return "\n".join(lines)


def main() -> None:
    print("─" * 60)
    print("  1. Circuit")
    print("─" * 60)
    print(qvm.draw(bell_state))

    print()
    print("─" * 60)
    print("  2. State vector (analytic)")
    print("─" * 60)
    state = qvm.state(bell_state)
    print(_format_state(state))
    print()
    print("  Only |00⟩ and |11⟩ have amplitude — never |01⟩ or |10⟩.")
    print("  That's the Bell state: (|00⟩ + |11⟩) / √2.")

    print()
    print("─" * 60)
    print("  3. Twelve measurement shots")
    print("─" * 60)
    samples = qvm.sample(bell_samples, shots=12)
    print("    q0  q1")
    for shot in samples:
        match = "✓ agree" if shot[0] == shot[1] else "✗ disagree"
        print(f"     {shot[0]}   {shot[1]}     {match}")
    print()
    n_agreeing = int(np.sum(samples[:, 0] == samples[:, 1]))
    print(f"  {n_agreeing}/12 shots: qubits always agree. That's entanglement.")

    print()
    print("─" * 60)
    print("  4. Marginal expectations (2048 shots each)")
    print("─" * 60)
    z0, z1 = qvm.run(bell_marginals)
    print(f"  ⟨Z⟩ on qubit 0:  {z0:+.3f}   (expected ≈ 0)")
    print(f"  ⟨Z⟩ on qubit 1:  {z1:+.3f}   (expected ≈ 0)")
    print()
    print("  Each qubit on its own is a fair coin — but they are not")
    print("  independent. The correlation lives between them, not in either one.")


if __name__ == "__main__":
    main()
