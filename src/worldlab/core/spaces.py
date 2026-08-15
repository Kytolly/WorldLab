"""Small dependency-free spaces for the built-in demo."""

from __future__ import annotations

import random
from typing import Optional, Sequence


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


class FrameSpace:
    """A fixed-length integer frame space for the random-frame demo."""

    def __init__(self, frame_size: int, *, value_max: int = 255) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be greater than zero")
        if value_max <= 0:
            raise ValueError("value_max must be greater than zero")
        self.frame_size = frame_size
        self.value_max = value_max
        self._random = random.Random()

    def sample(self) -> tuple[int, ...]:
        return tuple(
            self._random.randrange(self.value_max + 1)
            for _ in range(self.frame_size)
        )

    def contains(self, value: object) -> bool:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return False
        return len(value) == self.frame_size and all(
            isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= self.value_max
            for item in value
        )

    def seed(self, seed: Optional[int] = None) -> None:
        self._random.seed(seed)
