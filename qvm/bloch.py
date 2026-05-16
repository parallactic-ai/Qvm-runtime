"""Bloch-sphere visualization for single-qubit states.

A :func:`bloch` call takes a state vector — or any wire of a multi-qubit state
via partial trace — and returns a printable ASCII rendering of the Bloch
sphere with the state marked, plus the numeric Bloch vector and spherical
angles.

The math:

  - Pure single-qubit state α|0⟩ + β|1⟩
    → ``x = 2 Re(α* β)``, ``y = 2 Im(α* β)``, ``z = |α|² − |β|²``.
  - Multi-qubit state: trace out every wire except the target, leaving a
    2×2 density matrix ρ. The Bloch vector is then
    ``(x, y, z) = (2 Re ρ_{01}, −2 Im ρ_{01}, ρ_{00} − ρ_{11})``.

Entangled qubits have a reduced state with ``|r| < 1`` — the Bloch vector
lives *inside* the sphere. That's not a rendering bug: a maximally mixed
reduced state (e.g. one qubit of a Bell pair) sits at the origin, which is
exactly the geometric statement of entanglement-induced decoherence.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .exceptions import CircuitError


def state_to_density(state: np.ndarray) -> np.ndarray:
    """Return ``|ψ⟩⟨ψ|`` for a pure state vector."""
    psi = np.asarray(state, dtype=complex).flatten()
    return np.outer(psi, np.conj(psi))


def reduced_density_matrix(
    state: np.ndarray, wire: int, n_qubits: int
) -> np.ndarray:
    """Trace out every wire except ``wire``, returning a 2×2 density matrix.

    Wires are indexed left-to-right: in a 3-qubit state, ``wire=0`` is the
    leftmost qubit, matching PennyLane's convention.
    """
    if not (0 <= wire < n_qubits):
        raise CircuitError(
            f"wire {wire} is out of range for an {n_qubits}-qubit state"
        )
    if state.size != 2 ** n_qubits:
        raise CircuitError(
            f"state has {state.size} amplitudes but {n_qubits} qubits expects "
            f"{2 ** n_qubits}"
        )

    # Reshape into the n_qubits-dimensional tensor, contract every axis
    # except ``wire`` against itself via einsum.
    tensor = np.asarray(state, dtype=complex).reshape([2] * n_qubits)
    rho = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            # Build index lists: tensor[..., i_wire, ...] for the i side,
            # conj(tensor)[..., j_wire, ...] for the j side, contracted on
            # every other axis.
            sl_i: list = [slice(None)] * n_qubits
            sl_j: list = [slice(None)] * n_qubits
            sl_i[wire] = i
            sl_j[wire] = j
            a = tensor[tuple(sl_i)]
            b = np.conj(tensor[tuple(sl_j)])
            rho[i, j] = np.sum(a * b)
    return rho


def density_to_bloch_vector(rho: np.ndarray) -> Tuple[float, float, float]:
    """Convert a 2×2 density matrix to its Bloch vector ``(x, y, z)``."""
    if rho.shape != (2, 2):
        raise CircuitError(f"density matrix must be 2x2, got {rho.shape}")
    x = float(2.0 * rho[0, 1].real)
    y = float(-2.0 * rho[0, 1].imag)
    z = float((rho[0, 0] - rho[1, 1]).real)
    return x, y, z


def state_to_bloch_vector(
    state: np.ndarray, wire: int = 0
) -> Tuple[float, float, float]:
    """Convert a state vector to a Bloch vector.

    For a single-qubit state, ``wire`` is ignored. For multi-qubit states,
    the reduced density matrix on ``wire`` is computed first.
    """
    flat = np.asarray(state, dtype=complex).flatten()
    if flat.size == 2:
        return density_to_bloch_vector(state_to_density(flat))
    # Multi-qubit: figure out n_qubits and reduce.
    n_qubits = int(round(math.log2(flat.size)))
    if 2 ** n_qubits != flat.size:
        raise CircuitError(
            f"state size {flat.size} is not a power of 2"
        )
    rho = reduced_density_matrix(flat, wire=wire, n_qubits=n_qubits)
    return density_to_bloch_vector(rho)


def _spherical_angles(x: float, y: float, z: float) -> Tuple[float, float]:
    """Return ``(θ, φ)`` in degrees for a Bloch vector.

    For a Bloch vector with magnitude ``r`` (which may be less than 1 for
    mixed states), ``θ`` is the polar angle from +z, ``φ`` is the azimuth in
    the x-y plane. Both undefined when ``r = 0``; we report ``(0, 0)``.
    """
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-12:
        return 0.0, 0.0
    theta = math.degrees(math.acos(max(-1.0, min(1.0, z / r))))
    phi = math.degrees(math.atan2(y, x))
    return theta, phi


def _render_sphere(
    x: float, y: float, z: float, width: int = 21, height: int = 11
) -> str:
    """Render the X–Z projection of the Bloch sphere with the vector marked.

    ``y`` is the out-of-page component — it doesn't appear in the diagram,
    but is included in the numeric readout produced by :func:`bloch`.
    """
    grid = [[" " for _ in range(width)] for _ in range(height)]
    cx = width // 2
    cy = height // 2
    radius_x = cx - 1
    radius_y = cy - 1

    # Sphere outline as a dense ring of dots.
    for deg in range(0, 360, 4):
        rad = math.radians(deg)
        gx = cx + int(round(radius_x * math.cos(rad)))
        gy = cy - int(round(radius_y * math.sin(rad)))
        if 0 <= gx < width and 0 <= gy < height:
            grid[gy][gx] = "·"

    # Vertical and horizontal axes inside the sphere.
    for j in range(height):
        if grid[j][cx] == " ":
            grid[j][cx] = "│"
    for i in range(width):
        if grid[cy][i] == " ":
            grid[cy][i] = "─"
    grid[cy][cx] = "┼"

    # State marker. Clamp to the grid so mixed states with |r| > 1 (which
    # shouldn't happen but might due to floating-point) still render safely.
    px = cx + int(round(max(-1.0, min(1.0, x)) * radius_x))
    py = cy - int(round(max(-1.0, min(1.0, z)) * radius_y))
    if 0 <= px < width and 0 <= py < height:
        grid[py][px] = "●"

    return "\n".join("".join(row) for row in grid)


def bloch(state: np.ndarray, wire: int = 0) -> str:
    """Return a printable Bloch-sphere rendering of ``state``.

    Args:
        state: A NumPy array of amplitudes. Length 2 for a single qubit,
            length ``2**n`` for an ``n``-qubit state.
        wire: For multi-qubit input, which wire to visualise (the others
            are traced out).

    Returns:
        A multi-line string containing the ASCII sphere, the Bloch vector
        ``(x, y, z)``, the spherical angles ``(θ°, φ°)``, and a label.
    """
    x, y, z = state_to_bloch_vector(state, wire=wire)
    theta, phi = _spherical_angles(x, y, z)
    r = math.sqrt(x * x + y * y + z * z)
    sphere = _render_sphere(x, y, z)

    n_amps = int(np.asarray(state).size)
    n_qubits = int(round(math.log2(n_amps))) if n_amps > 1 else 1
    target = f"qubit {wire} of {n_qubits}" if n_qubits > 1 else "qubit"

    purity = "pure" if r > 0.999 else "mixed" if r < 1.0 - 1e-6 else "boundary"

    lines = [
        sphere,
        "",
        f"  Bloch vector  (x, y, z) = ({x:+.4f}, {y:+.4f}, {z:+.4f})",
        f"  Spherical     (θ, φ)    = ({theta:6.2f}°, {phi:7.2f}°)",
        f"  Magnitude     |r|       = {r:.4f}   ({purity}-state {target})",
    ]
    return "\n".join(lines)
