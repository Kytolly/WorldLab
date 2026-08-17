"""Typed term contracts and shared manager utilities."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Generic, Mapping, TypeVar

from worldlab.data import TerminationResult


ResetContextT = TypeVar("ResetContextT")
ContextT = TypeVar("ContextT")
ValueT = TypeVar("ValueT")


class ObservationTerm(ABC, Generic[ResetContextT, ContextT, ValueT]):
    """One observation component managed by ``ObservationManager``."""

    @abstractmethod
    def reset(self, context: ResetContextT) -> ValueT:
        """Reset term state and return its initial observation value."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, context: ContextT) -> ValueT:
        raise NotImplementedError


class RewardTerm(ABC, Generic[ResetContextT, ContextT]):
    """One scalar reward component before manager weighting."""

    def reset(self, context: ResetContextT) -> None:
        return None

    @abstractmethod
    def compute(self, context: ContextT) -> float:
        raise NotImplementedError


class TerminationTerm(ABC, Generic[ResetContextT, ContextT]):
    """One termination component preserving terminated/truncated semantics."""

    def reset(self, context: ResetContextT) -> None:
        return None

    @abstractmethod
    def compute(self, context: ContextT) -> TerminationResult:
        raise NotImplementedError


@dataclass(frozen=True)
class ObservationTermSpec:
    term: ObservationTerm[Any, Any, Any]
    space: Any
    group: str = "policy"
    enabled: bool = True


@dataclass(frozen=True)
class RewardTermSpec:
    term: RewardTerm[Any, Any]
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not _is_finite_scalar(self.weight):
            raise ValueError("reward term weight must be a finite scalar")


@dataclass(frozen=True)
class TerminationTermSpec:
    term: TerminationTerm[Any, Any]
    enabled: bool = True


class ManagerTermError(RuntimeError):
    """Error raised when a manager term fails or returns an invalid value."""

    def __init__(
        self,
        *,
        manager: str,
        term: str,
        cause: BaseException,
        group: str | None = None,
    ) -> None:
        location = manager
        if group is not None:
            location += f"/group={group}"
        location += f"/term={term}"
        super().__init__(f"{location} failed: {cause}")
        self.manager = manager
        self.group = group
        self.term = term
        self.__cause__ = cause


def namespace_info(prefix: str, info: Mapping[str, Any]) -> dict[str, Any]:
    """Prefix diagnostic keys without allowing a term to escape its namespace."""

    return {f"{prefix}.{key}": value for key, value in info.items()}


def immutable_info(info: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(info))


def _is_finite_scalar(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
