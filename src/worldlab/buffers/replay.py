"""In-memory uniform replay buffer."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar

from .base import Buffer


ItemT = TypeVar("ItemT")


class ReplayBuffer(Buffer[ItemT], Generic[ItemT]):
    def __init__(self, capacity: int, *, seed: Optional[int] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        self.capacity = capacity
        self._items: Deque[ItemT] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def add(self, item: ItemT) -> None:
        self._items.append(item)

    def sample(self, batch_size: int) -> List[ItemT]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if batch_size > len(self._items):
            raise ValueError(
                f"cannot sample {batch_size} items from a buffer of size {len(self._items)}"
            )
        return self._random.sample(list(self._items), batch_size)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
