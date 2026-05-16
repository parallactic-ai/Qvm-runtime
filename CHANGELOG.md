# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-05-16

The "Phase 2 done, Phase 3 begun" release. Adds variational-algorithm tooling, optimizer plugins, a SciPy bridge, a CLI, batched execution, gate-by-gate tracing, and ASCII Bloch-sphere rendering on top of the Phase 1 runtime.

### Added

#### Variational quantum algorithms
- `Qapp` class — bundles a `@qvm.circuit` cost function, runtime, and optimizer into one object with a `.fit()` method.
- `OptimizationResult` dataclass — full per-step history plus `best_params`, `best_cost`, `converged`, `steps_taken`.
- `Qapp.evaluate(params)` / `Qapp.grad_at(params)` for inspecting the cost surface without re-running `fit`.
- `fit()` accepts a `callback(step, params, cost)` for per-step monitoring.

#### Optimizers
- `Optimizer` abstract base class with `step()` / `reset()`.
- `SGD`, `Momentum`, `Adam` — first-class pluggable optimizers; instantiate one and pass `optimizer=` to `Qapp`.
- `qvm.minimize(cost, x0, runtime, method=...)` — bridge to `scipy.optimize.minimize` for L-BFGS-B, COBYLA, Nelder-Mead, and friends. Returns the same `OptimizationResult` shape.

#### Execution
- `QuantumRuntime.run_batch(circuit, params_batch)` — stacks N circuit evaluations into one ndarray. `(N,)` for scalar measurements, `(N, m)` for vector.

#### Inspection / education
- `QuantumRuntime.trace(circuit, params=None)` — returns `(label, statevector)` pairs after every gate, with the initial `|0…0⟩` state as the first entry. Built on `qml.snapshots` so all states share the full Hilbert-space dimension.
- `QuantumRuntime.draw(circuit, params=None)` — ASCII circuit diagram via `qml.draw`.
- `qvm.bloch(state, wire=0)` — ASCII Bloch-sphere rendering. Single-qubit states directly; multi-qubit states via partial trace on the requested wire (entangled marginals collapse to the origin as expected).

#### CLI
- `qvm` command installed via `[project.scripts]`. Subcommands:
  - `qvm version` — installed version.
  - `qvm info` — PennyLane / NumPy / Python versions and available backends.
  - `qvm demo bell` — Bell-state walkthrough.
  - `qvm demo vqa` — variational optimization demo.
- Quantum-themed Bell-circuit banner when invoked with no subcommand.

#### Module-level convenience
- `from qvm import grad, hybrid` — equivalent to the instance methods but require no runtime instance to import.

#### Examples (six new)
- `gradient_example.py` — hand-rolled gradient descent.
- `bell_state.py` — entanglement walkthrough (state + samples + marginals).
- `vqa_demo.py` — `Qapp` story in ten lines plus a live cost-bar chart.
- `optimizers_compared.py` — SGD vs Momentum vs Adam side by side.
- `parameter_sweep.py` — `run_batch` rendering a cosine wave in ASCII.
- `trace_demo.py` — Bell-state evolution with plain-English commentary.
- `bloch_demo.py` — same Bell circuit, but the Bloch sphere is drawn at every gate. Watch entanglement collapse the local Bloch vector to the origin.

#### Documentation
- Full README rewrite with concepts section, measurement-paths table, error reference, hardware-plugin matrix, and Phase 1 / Phase 2 status.
- `CONTRIBUTING.md` with dev setup, code standards, and contribution flow.
- `.github/CODEOWNERS` for review routing.
- `.github/dependabot.yml` for weekly GitHub Actions version bumps.

#### CI
- `.github/workflows/tests.yml` split into three parallel jobs: `lint` (ruff), `test` (matrix on Python 3.10/3.11/3.12), and `package` (build wheel + import smoke test in a clean venv).
- `.github/workflows/release.yml` — publishes to PyPI via Trusted Publishing on `v*` tag push.

### Changed
- Project name and description in `pyproject.toml` reflect the broader scope.
- `QuantumRuntime` constructor docstring expanded; arguments documented.
- `_create_qnode` shots contract clarified: `None` means analytic, positive int means finite shots; callers resolve defaults.

### Fixed
- `state()` no longer silently used the runtime's default shots — analytic execution is now correctly forced.
- Typed errors (`CircuitError`, `BackendError`) no longer get re-wrapped as `ExecutionError` by outer try/except blocks: every relevant method has an `except QVMError: raise` passthrough.
- `optimizers.SGD/Momentum/Adam` methods now carry docstrings (previously class-level only).
- Banner row widths align consistently across Unicode glyphs.

### Removed
- `ParameterError` is no longer in the top-level `__all__` — it was exported but never raised. Still importable from `qvm.exceptions` if needed.

### Stats
- 71 tests (up from 13 in v0.1.0).
- 8 runnable examples (up from 1).
- 5 source modules in `qvm/` (up from 3).
- CI matrix across Python 3.10 / 3.11 / 3.12.

## [0.1.0] — 2026-05-16

Initial release.

### Added
- `QuantumRuntime` class with `run`, `sample`, `state`.
- `@qvm.circuit` and `@qvm.hybrid` decorators.
- `QuantumRuntime.grad` for analytic gradient functions.
- `QVMError` hierarchy (`BackendError`, `CircuitError`, `ParameterError`, `ExecutionError`).
- Single basic-usage example.
- Initial pytest suite.

[0.2.0]: https://github.com/parallactic-ai/Qvm-runtime/releases/tag/v0.2.0
[0.1.0]: https://github.com/parallactic-ai/Qvm-runtime/releases/tag/v0.1.0
