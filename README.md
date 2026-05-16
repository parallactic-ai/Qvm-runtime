# qvm · Quantum Runtime

[![tests](https://github.com/parallactic-ai/Qvm-runtime/actions/workflows/tests.yml/badge.svg)](https://github.com/parallactic-ai/Qvm-runtime/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

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
pip install qvm-runtime
```

From source (for development):

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

## Qapps — variational algorithms in ten lines

The pattern *parameterized circuit → cost → gradient descent → repeat* is the kernel of every variational quantum algorithm (VQE, QAOA, quantum classifiers). `Qapp` bundles it into a single object so you stop writing the loop by hand.

```python
from qvm import QuantumRuntime, Qapp
import pennylane as qml

qvm = QuantumRuntime()

@qvm.circuit
def cost(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

app = Qapp(cost, runtime=qvm, learning_rate=0.4)
result = app.fit(initial_params=[0.5, 0.3], steps=30, tol=1e-6)

print(result.best_cost)     # ~ -1.0
print(result.steps_taken)   # ~ 22
print(result.converged)     # True
```

`fit()` returns an `OptimizationResult` with:

| Attribute | What it is |
| --- | --- |
| `history` | `[(step, params, cost), ...]` — full trace including step 0 |
| `best_params` / `best_cost` | Snapshot at the lowest-cost step seen |
| `converged` | `True` if cost delta dropped below `tol`, `False` if `steps` ran out |
| `steps_taken` | How many gradient steps actually ran |

For one-off inspection without re-running `fit`:

```python
app.evaluate(params)   # cost at a point (analytic, no shot noise)
app.grad_at(params)    # gradient at a point
```

A live demo with a cost-trajectory bar chart lives in `examples/vqa_demo.py`.

### Parameter sweeps

`run_batch` runs a circuit on many parameter vectors at once and stacks the results into one NumPy array. No Python loop, no shape juggling:

```python
import numpy as np

@qvm.circuit
def cost(params):
    qml.RX(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))

thetas = np.linspace(0, 2 * np.pi, 32).reshape(-1, 1)
values = qvm.run_batch(cost, thetas)
print(values.shape)   # (32,)
```

Vector measurements stack along a new leading axis — `qml.probs(...)` of width `m` over `N` parameter sets gives shape `(N, m)`. Useful for hyperparameter searches, training data over a cost surface, or just plotting `f(θ)`. A full ASCII visualisation of the resulting cosine wave lives in `examples/parameter_sweep.py`.

### Choosing an optimizer

By default `Qapp` uses plain SGD. Two stronger options ship out of the box — instantiate one and pass it to `Qapp(..., optimizer=...)`:

```python
from qvm import Adam, Momentum, Qapp, SGD

Qapp(cost, runtime=qvm, optimizer=SGD(learning_rate=0.2))
Qapp(cost, runtime=qvm, optimizer=Momentum(learning_rate=0.1, momentum=0.9))
Qapp(cost, runtime=qvm, optimizer=Adam(learning_rate=0.1))
```

| Optimizer | When to reach for it |
| --- | --- |
| `SGD` | Predictable, no per-parameter state. Good baseline. |
| `Momentum` | Smooths gradient noise; helps in long curved valleys. |
| `Adam` | Adapts per-parameter step size. Fastest convergence on most QML cost surfaces. |

All three are gradient-based and share the same interface (`Optimizer.step(params, grads)` returning new params). Implement your own by subclassing `Optimizer` if you need a custom rule. `examples/optimizers_compared.py` runs the same cost surface through all three side by side.

### When you want something SciPy has

For methods that aren't gradient descent — L-BFGS-B, COBYLA, Nelder-Mead, trust-region, etc. — use `qvm.minimize`. It hands the problem to `scipy.optimize.minimize` and hands back the same `OptimizationResult`:

```python
from qvm import minimize

result = minimize(cost, x0=[0.5, 0.3], runtime=qvm, method="L-BFGS-B")
print(result.best_cost, result.converged)
```

Gradient-based methods (`L-BFGS-B`, `CG`, `BFGS`, `Newton-CG`) receive the analytic Jacobian automatically. Gradient-free methods (`Nelder-Mead`, `COBYLA`, `Powell`) ignore it harmlessly. Pass any other SciPy option via `options={"maxiter": 100, ...}`.

---

## Command-line interface

Installing the package also installs a `qvm` command. Five subcommands, each useful inside its first five seconds:

```bash
qvm                # banner + hint at what to try next
qvm version        # qvm-runtime 0.1.0
qvm info           # PennyLane version + available backends + Python
qvm demo bell      # Bell-state walkthrough — entanglement in your terminal
qvm demo vqa       # optimization loop converging to ⟨Z⟩ = -1
```

`qvm` with no args shows a quick banner:

```text
╭──────────────────────────────────────────────────╮
│  qvm · quantum runtime · v0.1.0                  │
│                                                  │
│   |0⟩ ─┤H├──●──── ⟨Z⟩                            │
│             │                                    │
│   |0⟩ ──────X──── |ψ⟩ = (|00⟩+|11⟩)/√2           │
╰──────────────────────────────────────────────────╯

  Try:  qvm info   ·   qvm demo bell   ·   qvm --help
```

The CLI is built on [Typer](https://typer.tiangolo.com/) — extending it is just `@app.command()` on a new function.

---

## Inspecting circuits

`qvm.draw(circuit, params=...)` returns an ASCII diagram of any `@qvm.circuit`. Great for tutorials, debug prints, and notebook output.

```python
@qvm.circuit
def bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

print(qvm.draw(bell))
# 0: ──H─╭●─┤  State
# 1: ────╰X─┤  State
```

A complete walkthrough — diagram, state vector, samples, marginals — lives in `examples/bell_state.py`. It's the smallest example that shows real entanglement.

### Step-by-step traces

`qvm.trace(circuit, params)` returns the wavefunction after every gate in the circuit — useful for tutorials, debugging, and notebook explanations:

```python
@qvm.circuit
def bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

for label, state in qvm.trace(bell):
    print(label, abs(state))
# 0: init      [1. 0. 0. 0.]
# 1: H(0)      [0.707 0.    0.707 0.   ]
# 2: CNOT(0,1) [0.707 0.    0.    0.707]
```

The first entry is always the initial `|0…0⟩` state. Labels are formatted as `i: GateName(params, wires)` so they stay readable in deep circuits. `examples/trace_demo.py` walks through a Bell state with a plain-English explanation at each step.

### Bloch-sphere visualization

`qvm.bloch(state, wire=0)` renders a single-qubit state on a Bloch sphere — directly for 2-amplitude states, or via partial trace on the requested wire for multi-qubit states:

```python
from qvm import bloch
import numpy as np

print(bloch(np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)))
```

```text
          │
      ·········
   ····   │    ···
  ··      │      ··
 ··       │       ··
─·────────┼────────●─
 ··       │       ··
  ··      │      ··
   ···    │    ···
      ·········
          │

  Bloch vector  (x, y, z) = (+1.0000, -0.0000, +0.0000)
  Spherical     (θ, φ)    = ( 90.00°,   -0.00°)
  Magnitude     |r|       = 1.0000   (pure-state qubit)
```

Combined with `trace()` this becomes the most useful learning tool in the library — `examples/bloch_demo.py` walks through a Bell circuit and you can literally **watch the Bloch vector collapse to the origin** as entanglement forms. The magnitude `|r|` drops from `1.0` (pure) to `0.0` (maximally mixed) at the moment CNOT entangles the qubits, which is entanglement as raw geometry.

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

### Supported hardware (via PennyLane plugins)

Anything PennyLane recognizes as a device string works out of the box — `qvm-runtime` is hardware-agnostic. Install the relevant plugin and pass its device string to `QuantumRuntime(backend=...)`:

| Plugin | Devices | Install |
| --- | --- | --- |
| Built-in (always available) | `default.qubit`, `default.mixed`, `lightning.qubit` | — |
| [pennylane-qiskit](https://github.com/PennyLaneAI/pennylane-qiskit) | IBM Quantum hardware, Qiskit Aer simulators | `pip install pennylane-qiskit` |
| [pennylane-cirq](https://github.com/PennyLaneAI/pennylane-cirq) | Google Cirq simulators | `pip install pennylane-cirq` |
| [pennylane-rigetti](https://github.com/PennyLaneAI/pennylane-rigetti) | Rigetti Forest QPUs, QVM, wavefunction simulator | `pip install pennylane-rigetti` |
| [amazon-braket-pennylane](https://github.com/amazon-braket/amazon-braket-pennylane-plugin-python) | IonQ, Rigetti, IQM, AWS simulators | `pip install amazon-braket-pennylane-plugin` |

Example with Qiskit Aer:

```python
qvm = QuantumRuntime(backend="qiskit.aer")   # after pip install pennylane-qiskit
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

**Phase 1 — complete.** Minimal viable runtime:

- `run`, `run_batch`, `sample`, `state` for the common measurement types
- `@circuit` and `@hybrid` decorators (instance and module-level)
- Analytic gradients via `grad()` and `make_qnode(analytic=True)`
- `Qapp` abstraction with `fit` / `evaluate` / `grad_at` and a full `OptimizationResult`
- Pluggable optimizers: `SGD`, `Momentum`, `Adam` (write your own by subclassing `Optimizer`)
- `draw()` for ASCII circuit diagrams; `trace()` for gate-by-gate statevector snapshots; `bloch()` for single-qubit Bloch-sphere rendering (with partial-trace support for multi-qubit states)
- `qvm` CLI with `info`, `version`, `demo bell`, `demo vqa`
- Eight runnable examples: basic, gradient, Bell state, VQA demo, optimizers compared, parameter sweep, trace demo, bloch demo
- 71-test pytest suite covering runtime + Qapp + optimizers + CLI + bloch

### Honest limitations

- **Batching is sequential under the hood.** `run_batch` works on any backend but loops in Python; native PennyLane broadcasting is a Phase 2 perf optimization.
- **No GPU or distributed dispatch.** Whatever PennyLane device you pick is what you get.
- **No shared wire registers.** Each circuit infers wires independently — there's no `Qubit` / `Register` abstraction yet.
- **No mid-circuit measurement helpers.** Use raw PennyLane primitives inside the circuit for that.
- **No noise modeling sugar.** Use `default.mixed` or a noise plugin and configure it yourself.
- **Not production-ready.** Error messages favor clarity over machine-readability; APIs may shift in Phase 2.

### Phase 2 (sketched, not committed)

- Native broadcasting in `run_batch` for backends that support it (perf).
- More CLI subcommands — saved-experiment replay, history, sweep launchers.
- Pluggable schedulers for parallel hybrid workflows.
- Animated traces (terminal recordings) and Bloch trajectories.

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
  qapp.py            # Qapp + OptimizationResult
  optimizers.py      # Optimizer base + SGD / Momentum / Adam
  optimize.py        # scipy.minimize bridge
  bloch.py           # Bloch-sphere math + ASCII renderer (partial trace)
  cli.py             # `qvm` command-line interface (Typer)
  decorators.py      # module-level hybrid, grad
  exceptions.py      # QVMError hierarchy
examples/
  basic_usage.py             # smallest hybrid example
  gradient_example.py        # manual gradient-descent loop
  bell_state.py              # entanglement walkthrough
  vqa_demo.py                # the Qapp story in 10 lines
  optimizers_compared.py     # SGD vs Momentum vs Adam on the same cost
  parameter_sweep.py         # run_batch over θ ∈ [0, 2π], cosine wave in ASCII
  trace_demo.py              # trace() walking through a Bell state with commentary
  bloch_demo.py              # trace() + bloch() — entanglement collapsing the sphere
tests/
  test_runtime.py
  test_qapp.py
  test_optimizers.py
  test_optimize.py
  test_bloch.py
  test_cli.py
```

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md). Current version is **0.2.0** — "Phase 2 done, Phase 3 begun" — adding `Qapp`, three optimizers, the SciPy bridge, the CLI, batched execution, `trace()`, and Bloch-sphere rendering on top of the Phase 1 runtime.

## Contributing

This is an early, opinionated project — issues and PRs are welcome. The code is intentionally small so you can read all of it in one sitting; start with `qvm/runtime.py`. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup and the release process.

---

## License

MIT.
