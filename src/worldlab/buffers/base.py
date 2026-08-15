"""Generic experience-buffer contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar


ItemT = TypeVar("ItemT")


class Buffer(ABC, Generic[ItemT]):
    @abstractmethod
    def add(self, item: ItemT) -> None:
        raise NotImplementedError

    def extend(self, items: Sequence[ItemT]) -> None:
        for item in items:
            self.add(item)

    @abstractmethod
    def sample(self, batch_size: int) -> Sequence[ItemT]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
