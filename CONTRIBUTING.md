# Contributing to qvm-runtime

Thanks for your interest. `qvm-runtime` is an early, opinionated project — feedback, issues, and PRs are all welcome.

## Project philosophy

We're in **Phase 1**: a minimal viable quantum runtime focused on clarity over completeness. Before contributing larger changes, please read `CLAUDE.md` — it captures the design priorities (clarity → type safety → DX → extensibility → performance, in that order).

## Development setup

```bash
git clone https://github.com/parallactic-ai/Qvm-runtime.git
cd Qvm-runtime
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the suite

```bash
pytest tests/ -v          # 13 tests, runs in under a second
ruff check qvm tests      # lint
python examples/basic_usage.py
python examples/gradient_example.py
```

## Code standards

Non-negotiable for any PR touching `qvm/`:

- **Type hints** on every function and method signature.
- **Docstrings** on every public class and method.
- **Custom exceptions only** — use `QVMError`, `BackendError`, `CircuitError`, `ParameterError`, `ExecutionError` from `qvm/exceptions.py`. Don't introduce new top-level exception types without discussion.
- **No bare PennyLane leaks in the public API** — wrap or re-export rather than pushing PennyLane idioms onto users.

Style preferences:

- Keep functions small and focused.
- Prefer explicit, simple solutions over clever ones.
- Avoid premature abstraction. Three similar lines beat an early helper.
- Don't add comments that just restate what the code says.

## Submitting changes

1. **Open an issue first** for anything non-trivial. A 30-second sanity check saves a wasted PR.
2. **One concern per PR.** Refactors, features, and bug fixes shouldn't share a branch.
3. **Tests required** for new features and bug fixes. Add to `tests/test_runtime.py` (or a new module under `tests/`) using the existing pytest style.
4. **CI must be green.** Lint + tests run on Python 3.10, 3.11, and 3.12.
5. **Describe the why**, not just the what, in the PR body.

## Filing issues

Useful issues include:

- A minimal reproducer (5–10 lines if possible).
- The PennyLane version (`pip show pennylane`).
- What you expected vs. what happened.
- Stack trace if there is one — `__cause__` chains the underlying PennyLane error, so include the full chain.

## Phase 2 ideas

We have a rough sketch in the README (`Qapp` abstraction, batching, schedulers, educational tooling). None of it is committed. If you want to drive one of these, open an issue to discuss the design before writing code.

## Code of conduct

Be kind. Assume good intent. Disagree on substance, not on people.
