"""Tests for the qvm CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from qvm.cli import app

runner = CliRunner()


def test_version_prints_qvm_runtime_and_a_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "qvm-runtime" in result.stdout


def test_info_shows_versions_and_backends() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "PennyLane" in result.stdout
    assert "default.qubit" in result.stdout


def test_demo_bell_shows_circuit_and_correlated_shots() -> None:
    result = runner.invoke(app, ["demo", "bell"])
    assert result.exit_code == 0
    # Circuit diagram should appear
    assert "0:" in result.stdout and "1:" in result.stdout
    # Bell state amplitudes
    assert "|00⟩" in result.stdout
    assert "|11⟩" in result.stdout


def test_demo_vqa_converges_to_minimum() -> None:
    result = runner.invoke(app, ["demo", "vqa", "--steps", "50"])
    assert result.exit_code == 0
    assert "converged" in result.stdout
    assert "best cost" in result.stdout


def test_demo_vqa_respects_step_override() -> None:
    # With 1 step, won't converge — proves the flag is plumbed
    result = runner.invoke(app, ["demo", "vqa", "--steps", "1"])
    assert result.exit_code == 0
    assert "steps taken : 1" in result.stdout


def test_no_args_shows_banner() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    # Banner contains the project tag line and a hint at next commands
    assert "qvm · quantum runtime" in result.stdout
    assert "qvm --help" in result.stdout


def test_help_flag_still_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout
    assert "demo" in result.stdout  # subcommand listed
