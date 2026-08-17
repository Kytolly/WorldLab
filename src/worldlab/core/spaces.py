"""Small dependency-free spaces for the built-in demo."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Optional, cast

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
        low: Any = -np.inf,
        high: Any = np.inf,
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
        self.low = _normalize_bound(low, "low", self.shapes, self.dtype)
        self.high = _normalize_bound(high, "high", self.shapes, self.dtype)
        if np.any(self.low > self.high):
            raise ValueError("low must not be greater than high")
        self._random = np.random.default_rng()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the primary shape for callers that expect a single shape."""

        return self.shapes[0]

    def sample(self) -> np.ndarray[Any, Any]:
        low = np.broadcast_to(self.low, self.shape)
        high = np.broadcast_to(self.high, self.shape)
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            value = self._random.uniform(low, high, size=self.shape)
        else:
            value = self._random.normal(size=self.shape)
            value = np.maximum(value, low)
            value = np.minimum(value, high)
        return value.astype(self.dtype)

    def contains(self, value: object) -> bool:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return False
        if array.shape not in self.shapes or not np.issubdtype(array.dtype, np.number):
            return False
        try:
            low = np.broadcast_to(self.low, array.shape)
            high = np.broadcast_to(self.high, array.shape)
        except ValueError:
            return False
        return bool(np.all(array >= low) and np.all(array <= high))

    def seed(self, seed: Optional[int] = None) -> None:
        self._random = np.random.default_rng(seed)


class DictSpace:
    """A named recursive product of child spaces."""

    def __init__(self, spaces: Mapping[str, Any]) -> None:
        self.spaces = dict(spaces)
        for name, space in self.spaces.items():
            if not isinstance(name, str) or not name:
                raise ValueError("DictSpace keys must be non-empty strings")
            _validate_child_space(space, name)

    def sample(self) -> dict[str, Any]:
        return {name: space.sample() for name, space in self.spaces.items()}

    def contains(self, value: object) -> bool:
        if not isinstance(value, Mapping) or set(value) != set(self.spaces):
            return False
        return all(self.spaces[name].contains(value[name]) for name in self.spaces)

    def seed(self, seed: Optional[int] = None) -> None:
        _seed_children(self.spaces.values(), seed)


class TupleSpace:
    """A positional recursive product of child spaces."""

    def __init__(self, spaces: Sequence[Any]) -> None:
        self.spaces = tuple(spaces)
        for index, space in enumerate(self.spaces):
            _validate_child_space(space, str(index))

    def sample(self) -> tuple[Any, ...]:
        return tuple(space.sample() for space in self.spaces)

    def contains(self, value: object) -> bool:
        if not isinstance(value, tuple) or len(value) != len(self.spaces):
            return False
        return all(space.contains(item) for space, item in zip(self.spaces, value))

    def seed(self, seed: Optional[int] = None) -> None:
        _seed_children(self.spaces, seed)


def _normalize_bound(
    value: Any,
    name: str,
    shapes: tuple[tuple[int, ...], ...],
    dtype: np.dtype[Any],
) -> np.ndarray[Any, Any]:
    bound_dtype: Any = dtype if np.issubdtype(dtype, np.inexact) else np.float64
    try:
        bound = np.asarray(value, dtype=bound_dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric or array-like") from error
    for shape in shapes:
        try:
            np.broadcast_to(bound, shape)
        except ValueError as error:
            raise ValueError(f"{name} cannot broadcast to shape {shape}") from error
    return bound


def _validate_child_space(space: Any, name: str) -> None:
    if not all(callable(getattr(space, method, None)) for method in ("sample", "contains", "seed")):
        raise TypeError(f"space {name!r} must implement sample(), contains(), and seed()")


def _seed_children(spaces: Sequence[Any], seed: Optional[int]) -> None:
    random_source = random.Random(seed)
    for space in spaces:
        space.seed(random_source.randrange(2**32))
