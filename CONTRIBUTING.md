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

## Releasing

This repo publishes to PyPI via Trusted Publishing — no API tokens, no secrets in CI. The release workflow at `.github/workflows/release.yml` runs on any `v*` tag push, builds an sdist + wheel, verifies the wheel imports in a clean venv, and uploads.

### One-time setup (already done — keep for reference)

1. Reserve the project name on PyPI:
   - Log in to https://pypi.org
   - Go to **Your projects → Publishing → Add a new pending publisher**
   - Project name: `qvm-runtime`
   - Owner: `parallactic-ai`
   - Repository: `Qvm-runtime`
   - Workflow filename: `release.yml`
   - Environment: `pypi`
2. In the GitHub repo settings:
   - Create a `pypi` Environment (Settings → Environments → New environment).
   - Optional: add a required reviewer or wait timer for extra safety.

### Cutting a release

```bash
# 1. Bump version in pyproject.toml and CHANGELOG.md, commit, push.
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.X.Y"
git push

# 2. Tag the commit on main.
git tag v0.X.Y
git push origin v0.X.Y
```

The `release` workflow runs on the tag push, builds, and publishes. Watch with:

```bash
gh run watch -R parallactic-ai/Qvm-runtime
```

Within a minute the new version appears on https://pypi.org/p/qvm-runtime/.

### Versioning policy

[Semantic Versioning](https://semver.org/):

- **Patch** (`0.2.0 → 0.2.1`) — bug fixes, doc-only changes.
- **Minor** (`0.2.0 → 0.3.0`) — new public API, backward compatible.
- **Major** (`0.2.0 → 1.0.0`) — breaking changes. Pre-`1.0.0`, minor releases are allowed to break things; flag prominently in `CHANGELOG.md`.

## Code of conduct

Be kind. Assume good intent. Disagree on substance, not on people.
