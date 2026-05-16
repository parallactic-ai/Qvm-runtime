# qvm · Quantum Runtime

```text
┌──────────────────────────────────────────────────────────────┐
│                     qvm  ·  Quantum Runtime                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│           ┌─────┐                                            │
│      |0⟩ ─┤  H  ├────●────┐                                  │
│           └─────┘    │    │      ┌────────────┐              │
│                      │    │─────▶│            │───  ⟨Z⟩      │
│           ┌─────┐    │    │      │    run     │              │
│      |0⟩ ─┤     ├────X────┘      │            │───  |ψ⟩      │
│           └─────┘                └────────────┘              │
│                                                              │
│      Methods:   run   ·   sample   ·   state   ·   grad      │
└──────────────────────────────────────────────────────────────┘
```

> A clean, beginner-friendly **Quantum Runtime** built on top of PennyLane — for people who want to run quantum circuits without drowning in boilerplate.

`qvm-runtime` wraps PennyLane in a small, well-typed surface that makes it natural to write hybrid quantum-classical programs and take gradients through them.

```python
from qvm import QuantumRuntime, grad
import pennylane as qml
from pennylane import numpy as pnp

qvm = QuantumRuntime()

@qvm.circuit
def cost(params):
    qml.RX(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))

g = grad(cost)(pnp.array([pnp.pi / 2], requires_grad=True))
print(g)   # [-1.0]
```

---

## Why this exists

PennyLane is powerful but exposes a lot of moving pieces (devices, QNodes, shots, diff methods, interfaces). For learning and quick experimentation, that's friction. `qvm-runtime` gives you:

- One object — `QuantumRuntime` — that owns the device, shots, and diff method.
- Three measurement paths — `run`, `sample`, `state` — instead of one overloaded entry point.
- Decorators that wrap exceptions into a clean error hierarchy.
- A `grad()` helper that picks the right execution mode automatically.

It's a **Phase 1** project: small, opinionated, and easy to read end-to-end (`qvm/runtime.py` is ~250 lines).

---

## Features

- Clean API surface — `circuit`, `hybrid`, `run`, `sample`, `state`, `grad`
- Predictable return types — scalars come back as Python `float`, arrays as `np.ndarray`
- First-class hybrid workflows — mix Python and quantum code freely
- Analytic-by-default gradients — no shot-noise gotchas
- Custom exception hierarchy with informative messages
- Strong type hints throughout
- Compact test suite

---

## Installation

Requires **Python 3.10+** and PennyLane.

```bash
pip install -e .
```

With dev tooling (`pytest`, `ruff`):

```bash
pip install -e ".[dev]"
```

Verify:

```bash
python -c "from qvm import QuantumRuntime; print(QuantumRuntime())"
```

---

## Hello, Quantum

```python
from qvm import QuantumRuntime
import pennylane as qml

qvm = QuantumRuntime()

@qvm.circuit
def hello():
    qml.Hadamard(wires=0)
    return qml.probs(wires=0)

print(qvm.run(hello))   # ~ [0.5 0.5]
```

That's the whole loop: build a runtime, decorate a circuit, call `run`.

---

## Concepts

### The three measurement paths

`qvm-runtime` exposes three execution methods — one for each common shape of output. Pick the one matching your circuit's measurement:

| Method | Use when your circuit returns | Returns | Shots |
| --- | --- | --- | --- |
| `qvm.run(circuit, params, shots=None)` | `qml.expval(...)`, `qml.probs(...)`, etc. | `float` for scalar measurements, `np.ndarray` for vectors | finite (default 1024) |
| `qvm.sample(circuit, params, shots=None)` | `qml.sample(...)` | `np.ndarray` of shape `(shots,)` (or wider) | finite, required |
| `qvm.state(circuit, params)` | `qml.state()` | `np.ndarray` (complex statevector) | analytic — no shots |

### Shots semantics

- The runtime's `shots` (default `1024`) applies to every call unless overridden.
- `run()` and `sample()` accept a per-call `shots=N` override.
- `state()` always runs **analytically** (no shots) — it's for inspecting the wavefunction, not sampling from it.

```python
qvm = QuantumRuntime(shots=2048)
qvm.run(my_circuit)                       # 2048 shots
qvm.run(my_circuit, shots=128)            # 128 shots for this call only
qvm.state(my_state_circuit)               # analytic, ignores shots entirely
```

### Return-type contract for `run()`

`run()` normalizes PennyLane's output so callers can rely on Python-native types:

- A scalar measurement (e.g. `qml.expval`) returns a Python `float`.
- A vector measurement (e.g. `qml.probs`, `qml.state`) returns an `np.ndarray`.
- Multiple measurements come back as a `tuple` of the above.

This means `result + 1` works without any casting for the common scalar case.

---

## Hybrid programs

Use the `@hybrid` decorator to mark a function that mixes Python and quantum work. Any exception raised inside gets re-raised as `ExecutionError`, so error handling stays uniform.

```python
from qvm import QuantumRuntime, hybrid
import numpy as np
import pennylane as qml

qvm = QuantumRuntime()

@qvm.circuit
def expval(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

@hybrid
def loss(x):
    angle = float(np.sin(x) * np.pi)
    return qvm.run(expval, params=[angle, 0.3]) * 2 + 1

print(loss(0.7))
```

`@hybrid` is both a runtime method (`@qvm.hybrid`) and a free function (`from qvm import hybrid`) — use whichever reads better.

---

## Gradients

`grad()` returns a function that computes the gradient of its input. When given a `@qvm.circuit`, it builds an **analytic** QNode internally so gradients are free of shot noise — a common footgun.

```python
from qvm import QuantumRuntime, grad
import pennylane as qml
from pennylane import numpy as pnp

qvm = QuantumRuntime()

@qvm.circuit
def cost(params):
    qml.RX(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))

gradient = grad(cost)
params = pnp.array([pnp.pi / 2], requires_grad=True)
print(gradient(params))   # [-1.0]
```

> **Note** — PennyLane's autograd backend differentiates only `pennylane.numpy` arrays with `requires_grad=True`. Passing a plain `np.array(...)` will silently return zeros.

### Gradient descent loop

A complete example lives in `examples/gradient_example.py`:

```bash
python examples/gradient_example.py
```

It runs five SGD steps over a parameterized circuit; the cost drops from `~0.84` to `~-0.15`.

---

## Advanced

### Raw QNode access

For PennyLane interop (custom optimizers, transforms, batched execution), grab the underlying QNode:

```python
qnode = qvm.make_qnode(my_circuit, analytic=True)
# Now `qnode` is a vanilla pennylane.QNode you can pass anywhere.
```

### Switching backends

```python
qvm.set_backend("lightning.qubit")   # fast C++ simulator
qvm.available_backends()             # ['default.qubit', 'lightning.qubit', ...]
```

### Custom diff method

```python
qvm = QuantumRuntime(diff_method="adjoint")   # for lightning devices
```

---

## Error reference

All exceptions inherit from `QVMError`. The most common ones you'll see:

| Exception | When it fires | Typical fix |
| --- | --- | --- |
| `CircuitError: Function must be decorated with @qvm.circuit` | You passed a bare function to `run`/`sample`/`state` | Add the `@qvm.circuit` decorator |
| `BackendError: Failed to initialize backend '...'` | The backend name is unknown or its plugin isn't installed | Check `qvm.available_backends()`; install the plugin |
| `BackendError: Backend '...' does not support state output` | You called `state()` on a device that can't produce a wavefunction (e.g. some noisy simulators) | Switch to `default.qubit` / `lightning.qubit` |
| `ExecutionError: sample() requires a positive shots value` | `state()`-style analytic mode used with `sample()` | Pass `shots=N` or set a default on the runtime |
| `ExecutionError: Execution failed for '...'` | PennyLane raised during circuit execution — wrong wire index, mismatched param shape, etc. | Read the chained cause |

All of these chain the original exception via `__cause__`, so the underlying PennyLane error is one `.__cause__` away if you need it.

---

## Project status

**Phase 1 — Minimal Viable Runtime.** What works today:

- `run`, `sample`, `state` for the common measurement types
- `@circuit` and `@hybrid` decorators
- Analytic gradients via `grad()` and `make_qnode(analytic=True)`
- 13-test pytest suite (`pytest tests/ -v`)

### Honest limitations

- **No automatic batching.** One execution per `run()` call. If you want to run a parameter sweep, write a Python loop.
- **No GPU or distributed dispatch.** Whatever PennyLane device you pick is what you get.
- **No shared wire registers.** Each circuit infers wires independently — there's no `Qubit` / `Register` abstraction yet.
- **No mid-circuit measurement helpers.** Use raw PennyLane primitives inside the circuit for that.
- **No noise modeling sugar.** Use `default.mixed` or a noise plugin and configure it yourself.
- **Not production-ready.** Error messages favor clarity over machine-readability; APIs may shift in Phase 2.

### Phase 2 (rough sketch)

- Higher-level **Qapp** abstraction — quantum programs as composable objects.
- Built-in batching and parameter sweeps.
- Pluggable schedulers for parallel hybrid workflows.
- Richer educational tooling (circuit visualizers, step traces).

Nothing here is committed yet — feedback welcome.

---

## Testing

```bash
pytest tests/ -v
```

The whole suite runs in under a second on `default.qubit`.

---

## Project layout

```
qvm/
  __init__.py        # public API
  runtime.py         # QuantumRuntime class
  decorators.py      # module-level hybrid, grad
  exceptions.py      # QVMError hierarchy
examples/
  basic_usage.py
  gradient_example.py
tests/
  test_runtime.py
```

---

## Contributing

This is an early, opinionated project — issues and PRs are welcome. The code is intentionally small so you can read all of it in one sitting; start with `qvm/runtime.py`.

---

## License

MIT.
