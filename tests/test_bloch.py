"""Tests for the Bloch-sphere visualization."""

from __future__ import annotations

import numpy as np
import pytest

from qvm import bloch
from qvm.bloch import (
    density_to_bloch_vector,
    reduced_density_matrix,
    state_to_bloch_vector,
    state_to_density,
)
from qvm.exceptions import CircuitError

INV_SQRT2 = 1.0 / np.sqrt(2)


# ---------- Bloch vector math ----------

def test_zero_state_points_north() -> None:
    x, y, z = state_to_bloch_vector(np.array([1.0, 0.0], dtype=complex))
    assert (x, y, z) == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)


def test_one_state_points_south() -> None:
    x, y, z = state_to_bloch_vector(np.array([0.0, 1.0], dtype=complex))
    assert (x, y, z) == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)


def test_plus_state_points_to_positive_x() -> None:
    x, y, z = state_to_bloch_vector(np.array([INV_SQRT2, INV_SQRT2], dtype=complex))
    assert (x, y, z) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_minus_state_points_to_negative_x() -> None:
    x, y, z = state_to_bloch_vector(np.array([INV_SQRT2, -INV_SQRT2], dtype=complex))
    assert (x, y, z) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-9)


def test_plus_i_state_points_to_positive_y() -> None:
    x, y, z = state_to_bloch_vector(
        np.array([INV_SQRT2, 1j * INV_SQRT2], dtype=complex)
    )
    assert (x, y, z) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_pure_state_lies_on_unit_sphere() -> None:
    # Random single-qubit state of the form cos(θ/2)|0⟩ + e^iφ sin(θ/2)|1⟩
    theta = 1.234
    phi = 0.567
    state = np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex
    )
    x, y, z = state_to_bloch_vector(state)
    assert x * x + y * y + z * z == pytest.approx(1.0, abs=1e-9)


# ---------- Partial trace (multi-qubit reduction) ----------

def test_bell_pair_reduced_state_is_maximally_mixed() -> None:
    bell = np.array([INV_SQRT2, 0.0, 0.0, INV_SQRT2], dtype=complex)
    rho = reduced_density_matrix(bell, wire=0, n_qubits=2)
    # ρ should be I/2
    np.testing.assert_allclose(rho, 0.5 * np.eye(2), atol=1e-9)


def test_bell_pair_bloch_vector_is_origin() -> None:
    bell = np.array([INV_SQRT2, 0.0, 0.0, INV_SQRT2], dtype=complex)
    x, y, z = state_to_bloch_vector(bell, wire=0)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    # Same for the other qubit
    x2, y2, z2 = state_to_bloch_vector(bell, wire=1)
    assert (x2, y2, z2) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_product_state_marginals_are_pure() -> None:
    # |+⟩ ⊗ |0⟩ — both reduced states should be pure
    plus = np.array([INV_SQRT2, INV_SQRT2], dtype=complex)
    zero = np.array([1.0, 0.0], dtype=complex)
    joint = np.kron(plus, zero)
    x0, y0, z0 = state_to_bloch_vector(joint, wire=0)
    x1, y1, z1 = state_to_bloch_vector(joint, wire=1)
    assert (x0, y0, z0) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert (x1, y1, z1) == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)


def test_density_matrix_from_pure_state_has_unit_trace() -> None:
    state = np.array([0.6, 0.8], dtype=complex)
    rho = state_to_density(state)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)


def test_density_to_bloch_vector_rejects_wrong_shape() -> None:
    with pytest.raises(CircuitError):
        density_to_bloch_vector(np.eye(3, dtype=complex))


# ---------- Renderer ----------

def test_bloch_returns_multiline_string_with_axes() -> None:
    output = bloch(np.array([1.0, 0.0], dtype=complex))
    assert isinstance(output, str)
    # Sphere uses these characters
    assert "·" in output
    assert "─" in output and "│" in output
    # Numeric readout is present
    assert "Bloch vector" in output
    assert "Spherical" in output


def test_bloch_marks_state_position_for_zero_state() -> None:
    """The state marker for |0⟩ should appear above the equator line."""
    output = bloch(np.array([1.0, 0.0], dtype=complex))
    lines = output.split("\n")
    sphere_lines = [line for line in lines if "─" in line or "·" in line]
    equator_idx = next(i for i, line in enumerate(sphere_lines) if "─" in line)
    # State marker ● should be on a row above the equator
    marker_rows = [i for i, line in enumerate(sphere_lines) if "●" in line]
    assert marker_rows, "expected a ● marker for |0⟩"
    assert marker_rows[0] < equator_idx, "|0⟩ should be marked above the equator"


def test_bloch_marks_state_position_for_one_state() -> None:
    """The state marker for |1⟩ should appear below the equator line."""
    output = bloch(np.array([0.0, 1.0], dtype=complex))
    lines = output.split("\n")
    sphere_lines = [line for line in lines if "─" in line or "·" in line]
    equator_idx = next(i for i, line in enumerate(sphere_lines) if "─" in line)
    marker_rows = [i for i, line in enumerate(sphere_lines) if "●" in line]
    assert marker_rows, "expected a ● marker for |1⟩"
    assert marker_rows[0] > equator_idx, "|1⟩ should be marked below the equator"


def test_bloch_reports_mixed_state_for_bell_marginal() -> None:
    bell = np.array([INV_SQRT2, 0.0, 0.0, INV_SQRT2], dtype=complex)
    output = bloch(bell, wire=0)
    assert "mixed-state" in output


def test_bloch_reports_pure_state_for_unit_vector() -> None:
    output = bloch(np.array([INV_SQRT2, INV_SQRT2], dtype=complex))
    assert "pure-state" in output


def test_bloch_rejects_out_of_range_wire() -> None:
    state = np.array([INV_SQRT2, 0.0, 0.0, INV_SQRT2], dtype=complex)
    with pytest.raises(CircuitError):
        bloch(state, wire=5)
