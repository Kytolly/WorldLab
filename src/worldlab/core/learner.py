"""Optional learning contract, independent of environment interaction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, TypeVar


BatchT = TypeVar("BatchT")


class Learner(ABC, Generic[BatchT]):
    @abstractmethod
    def update(self, batch: BatchT) -> Mapping[str, float]:
        raise NotImplementedError

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        return None
