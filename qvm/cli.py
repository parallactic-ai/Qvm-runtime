"""Command-line interface for qvm-runtime.

Four commands, each useful within its first five seconds:

  - ``qvm version``       Print the installed version.
  - ``qvm info``          Show PennyLane version, available backends, Python.
  - ``qvm demo bell``     Walk through Bell-state entanglement.
  - ``qvm demo vqa``      Run a small variational optimization to ⟨Z⟩ = -1.

Running ``qvm`` with no arguments prints a banner and exits ``0``.
Exit codes are conventional: ``0`` for success, non-zero for any error.
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pennylane as qml
import typer

from .qapp import Qapp
from .runtime import QuantumRuntime

# Resolve version from package metadata; fall back to a sentinel if running
# from source without an install.
try:
    from importlib.metadata import version as _pkg_version

    QVM_VERSION = _pkg_version("qvm-runtime")
except Exception:  # pragma: no cover - only hits in unusual layouts
    QVM_VERSION = "0.0.0+source"


app = typer.Typer(
    name="qvm",
    help="Quantum Runtime CLI — inspect your install and run quick demos.",
    add_completion=False,
)

demo_app = typer.Typer(
    help="Self-contained demonstrations of qvm-runtime in action.",
    no_args_is_help=True,
)
app.add_typer(demo_app, name="demo")


_BANNER_WIDTH = 50


def _row(content: str = "") -> str:
    """One row of the banner box, padded to uniform inner width."""
    return f"│{content:<{_BANNER_WIDTH}}│"


def _banner() -> str:
    """Build the qvm banner displayed when ``qvm`` runs with no subcommand.

    Uses Unicode box-drawing with rounded corners and bra-ket notation —
    box characters render in every modern terminal; the quantum glyphs
    give immediate subject-matter signal.
    """
    border = "─" * _BANNER_WIDTH
    return "\n".join(
        [
            f"╭{border}╮",
            _row(f"  qvm · quantum runtime · v{QVM_VERSION}"),
            _row(),
            _row("   |0⟩ ─┤H├──●──── ⟨Z⟩"),
            _row("             │"),
            _row("   |0⟩ ──────X──── |ψ⟩ = (|00⟩+|11⟩)/√2"),
            f"╰{border}╯",
            "",
            "  Try:  qvm info   ·   qvm demo bell   ·   qvm --help",
        ]
    )


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Print the banner when called without any subcommand."""
    if ctx.invoked_subcommand is None:
        typer.echo(_banner())


# =====================
# Top-level commands
# =====================

@app.command()
def version() -> None:
    """Print the installed qvm-runtime version."""
    typer.echo(f"qvm-runtime {QVM_VERSION}")


@app.command()
def info() -> None:
    """Show environment info: versions, available backends, Python build."""
    qvm = QuantumRuntime()
    typer.echo(f"qvm-runtime    {QVM_VERSION}")
    typer.echo(f"PennyLane      {qml.__version__}")
    typer.echo(f"NumPy          {np.__version__}")
    typer.echo(f"Python         {platform.python_version()}  ({platform.system()})")
    typer.echo("")
    typer.echo("Recognized backends:")
    for backend in qvm.available_backends():
        typer.echo(f"  · {backend}")


# =====================
# Demos
# =====================

@demo_app.command("bell")
def demo_bell() -> None:
    """Build a Bell state, show the circuit, sample it, prove entanglement."""
    qvm = QuantumRuntime(shots=2048)

    @qvm.circuit
    def bell():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    @qvm.circuit
    def bell_samples():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.sample(wires=[0, 1])

    typer.echo("Circuit:")
    typer.echo(qvm.draw(bell))
    typer.echo("")

    typer.echo("State vector:")
    state = qvm.state(bell)
    for i, amp in enumerate(state):
        if abs(amp) > 1e-9:
            basis = format(i, "02b")
            typer.echo(f"  |{basis}⟩ : {amp.real:+.4f}{amp.imag:+.4f}j")
    typer.echo("")

    typer.echo("Eight measurement shots — both qubits always agree:")
    samples = qvm.sample(bell_samples, shots=8)
    for shot in samples:
        typer.echo(f"  q0={shot[0]}  q1={shot[1]}")


@demo_app.command("vqa")
def demo_vqa(
    steps: int = typer.Option(30, help="Maximum optimization steps."),
    learning_rate: float = typer.Option(0.4, help="Gradient-descent step size."),
) -> None:
    """Optimize a parameterized circuit to minimize ⟨Z⟩ on qubit 0."""
    qvm = QuantumRuntime()

    @qvm.circuit
    def cost(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    app_obj = Qapp(cost, runtime=qvm, learning_rate=learning_rate)
    result = app_obj.fit(initial_params=[0.5, 0.3], steps=steps, tol=1e-6)

    typer.echo(f"steps taken : {result.steps_taken}")
    typer.echo(f"converged   : {result.converged}")
    typer.echo(f"best cost   : {result.best_cost:+.6f}   (minimum of ⟨Z⟩ is -1)")
    typer.echo(f"best params : {result.best_params}")


def main() -> None:
    """Entry point referenced by ``[project.scripts]`` in pyproject.toml."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.", err=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
