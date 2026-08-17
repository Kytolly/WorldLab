"""Composable reward term manager."""

from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar

from worldlab.core import RewardProvider
from worldlab.data import RewardResult

from .terms import ManagerTermError, RewardTermSpec, immutable_info, namespace_info


ResetContextT = TypeVar("ResetContextT")
StepContextT = TypeVar("StepContextT")


class RewardManager(
    RewardProvider[ResetContextT, StepContextT],
    Generic[ResetContextT, StepContextT],
):
    """Evaluate ordered reward terms and return their weighted scalar sum."""

    def __init__(self, terms: Mapping[str, RewardTermSpec]) -> None:
        self.terms = dict(terms)
        for name in self.terms:
            if not name:
                raise ValueError("reward term names must be non-empty")

    def reset(self, context: ResetContextT) -> None:
        for name, spec in self.terms.items():
            if not self._active(spec):
                continue
            self._call_reset(name, spec, context)

    def compute(self, context: StepContextT) -> RewardResult:
        total = 0.0
        info: dict[str, Any] = {}
        for name, spec in self.terms.items():
            if not self._active(spec):
                continue
            try:
                raw = spec.term.compute(context)
                if not _is_finite_scalar(raw):
                    raise ValueError("term output must be a finite scalar")
                weighted = float(raw) * spec.weight
                if not _is_finite_scalar(weighted):
                    raise ValueError("weighted term output must be a finite scalar")
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ManagerTermError(
                    manager="reward",
                    term=name,
                    cause=error,
                ) from error
            total += weighted
            info.update(
                namespace_info(
                    f"worldlab.reward.terms.{name}",
                    {"raw": float(raw), "weighted": weighted},
                )
            )
        if not _is_finite_scalar(total):
            raise ValueError("reward manager produced a non-finite total")
        info["worldlab.reward.total"] = total
        return RewardResult(total, immutable_info(info))

    @staticmethod
    def _active(spec: RewardTermSpec) -> bool:
        return spec.enabled and spec.weight != 0.0

    def _call_reset(
        self,
        name: str,
        spec: RewardTermSpec,
        context: ResetContextT,
    ) -> None:
        try:
            spec.term.reset(context)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ManagerTermError(manager="reward", term=name, cause=error) from error


def _is_finite_scalar(value: Any) -> bool:
    import math
    from numbers import Real

    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
