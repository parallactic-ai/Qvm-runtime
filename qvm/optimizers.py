"""Optimizers for variational quantum algorithms.

A small, deliberate set of gradient-based optimizers that plug into
:class:`Qapp.fit`. The :class:`Optimizer` base class is stateful — it
keeps any per-run state (momentum buffers, Adam moments) on the instance,
and exposes :meth:`reset` so the same instance can be reused across
``fit`` calls without leaking state.

Phase 1 ships three:

  - :class:`SGD`       — plain gradient descent.
  - :class:`Momentum`  — heavy-ball momentum.
  - :class:`Adam`      — adaptive moments; the workhorse for QML.

All three accept and return ``pennylane.numpy`` arrays so the
``requires_grad`` flag survives the update, which the autograd-based
gradient function on the next step relies on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Optimizer(ABC):
    """Abstract base class for gradient-based optimizers.

    Subclasses implement :meth:`step` to apply a single parameter update
    given the current parameters and their gradient. They may also
    override :meth:`reset` to clear accumulated state when an instance
    is reused across multiple ``fit`` calls.
    """

    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Return the parameters after one update step."""

    def reset(self) -> None:
        """Clear any accumulated state. Default is a no-op."""


class SGD(Optimizer):
    """Plain stochastic gradient descent: ``θ ← θ − lr · ∇θ``."""

    def __init__(self, learning_rate: float = 0.1) -> None:
        self._lr = learning_rate

    @property
    def learning_rate(self) -> float:
        return self._lr

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        return params - self._lr * grads


class Momentum(Optimizer):
    """Heavy-ball momentum: maintains a velocity that smooths gradient noise.

    ``v ← momentum · v − lr · ∇θ``
    ``θ ← θ + v``
    """

    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.9) -> None:
        self._lr = learning_rate
        self._momentum = momentum
        self._velocity: Optional[np.ndarray] = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self._velocity is None:
            self._velocity = np.zeros_like(np.asarray(params))
        self._velocity = self._momentum * self._velocity - self._lr * grads
        return params + self._velocity

    def reset(self) -> None:
        self._velocity = None


class Adam(Optimizer):
    """Adaptive Moment Estimation (Kingma & Ba, 2014).

    Tracks both a first-moment (gradient mean) and second-moment
    (squared-gradient mean) estimate, with bias correction. Generally
    converges faster than SGD on variational quantum cost surfaces.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._m: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._t: int = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(np.asarray(params))
            self._v = np.zeros_like(np.asarray(params))
        self._t += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * grads
        self._v = self._beta2 * self._v + (1 - self._beta2) * (grads ** 2)
        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)
        return params - self._lr * m_hat / (np.sqrt(v_hat) + self._eps)

    def reset(self) -> None:
        self._m = None
        self._v = None
        self._t = 0
