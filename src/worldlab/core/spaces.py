"""Small dependency-free spaces for the built-in demo."""

from __future__ import annotations

import random
from typing import Optional


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

