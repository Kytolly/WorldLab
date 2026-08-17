"""Composable termination term manager."""

from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar

from worldlab.core import TerminationProvider
from worldlab.data import TerminationResult

from .terms import ManagerTermError, TerminationTermSpec, immutable_info, namespace_info


ResetContextT = TypeVar("ResetContextT")
StepContextT = TypeVar("StepContextT")


class TerminationManager(
    TerminationProvider[ResetContextT, StepContextT],
    Generic[ResetContextT, StepContextT],
):
    """Evaluate ordered terms and OR their terminated/truncated flags."""

    def __init__(self, terms: Mapping[str, TerminationTermSpec]) -> None:
        self.terms = dict(terms)
        for name in self.terms:
            if not name:
                raise ValueError("termination term names must be non-empty")

    def reset(self, context: ResetContextT) -> None:
        for name, spec in self.terms.items():
            if not spec.enabled:
                continue
            self._call_reset(name, spec, context)

    def compute(self, context: StepContextT) -> TerminationResult:
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        for name, spec in self.terms.items():
            if not spec.enabled:
                continue
            try:
                result = spec.term.compute(context)
                if not isinstance(result, TerminationResult):
                    raise TypeError("term output must be a TerminationResult")
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ManagerTermError(
                    manager="termination",
                    term=name,
                    cause=error,
                ) from error
            terminated = terminated or result.terminated
            truncated = truncated or result.truncated
            info.update(
                namespace_info(
                    f"worldlab.termination.terms.{name}",
                    {
                        "terminated": result.terminated,
                        "truncated": result.truncated,
                        **dict(result.info),
                    },
                )
            )
        return TerminationResult(terminated, truncated, immutable_info(info))

    def _call_reset(
        self,
        name: str,
        spec: TerminationTermSpec,
        context: ResetContextT,
    ) -> None:
        try:
            spec.term.reset(context)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ManagerTermError(manager="termination", term=name, cause=error) from error
