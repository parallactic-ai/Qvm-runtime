"""Basic usage example for Phase 1 Quantum Runtime."""

import numpy as np
import pennylane as qml

from qvm import QuantumRuntime
from qvm.exceptions import ExecutionError, QVMError

qvm = QuantumRuntime(backend="default.qubit", shots=1024)


@qvm.circuit
def variational_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


@qvm.hybrid
def hybrid_step(x: float) -> float:
    angle = float(np.sin(x) * np.pi)
    try:
        result = qvm.run(variational_circuit, params=[angle, 0.3])
        return float(result)
    except ExecutionError as e:
        print(f"Quantum execution failed: {e}")
        return 0.0


if __name__ == "__main__":
    try:
        output = hybrid_step(0.7)
        print(f"Hybrid result: {output}")
    except QVMError as e:
        print(f"QVM Error: {e}")
