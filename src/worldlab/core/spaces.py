"""Small dependency-free spaces for the built-in demo."""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence, cast

import numpy as np


class DiscreteSpace:
    """A minimal integer space compatible with WorldLab's Space protocol."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be greater than zero")
        self.size = size
        self._random = random.Random()

    def sample(self) -> int:
        return self._random.randrange(self.size)

    def contains(self, value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < self.size

    def seed(self, seed: Optional[int] = None) -> None:
        self._random.seed(seed)


class ArraySpace:
    """Small NumPy space for fixed-shape actions and observations."""

    def __init__(
        self,
        shapes: Sequence[int] | Sequence[tuple[int, ...]],
        *,
        dtype: Any = np.float32,
    ) -> None:
        if not shapes:
            raise ValueError("shapes must not be empty")
        if shapes and isinstance(shapes[0], int):
            dimensions = cast(Sequence[int], shapes)
            normalized: tuple[tuple[int, ...], ...] = (
                tuple(int(value) for value in dimensions),
            )
        else:
            shape_values = cast(Sequence[tuple[int, ...]], shapes)
            normalized = tuple(
                tuple(int(value) for value in shape) for shape in shape_values
            )
        if any(not shape or any(value <= 0 for value in shape) for shape in normalized):
            raise ValueError("array shapes must contain positive dimensions")
        self.shapes = normalized
        self.dtype = np.dtype(dtype)
        self._random = np.random.default_rng()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the primary shape for callers that expect a single shape."""

        return self.shapes[0]

    def sample(self) -> np.ndarray[Any, Any]:
        return self._random.normal(size=self.shape).astype(self.dtype)

    def contains(self, value: object) -> bool:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return False
        return array.shape in self.shapes and np.issubdtype(array.dtype, np.number)

    def seed(self, seed: Optional[int] = None) -> None:
        self._random = np.random.default_rng(seed)
