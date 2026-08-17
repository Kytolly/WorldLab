"""Composable observation term manager."""

from __future__ import annotations

from typing import Any, Generic, Mapping, TypeAlias, TypeVar

from worldlab.core import DictSpace, ObservationProvider, Space
from worldlab.data import ObservationResult

from .terms import (
    ManagerTermError,
    ObservationTermSpec,
    immutable_info,
)


ResetContextT = TypeVar("ResetContextT")
StepContextT = TypeVar("StepContextT")
ObservationGroups: TypeAlias = dict[str, dict[str, Any]]


class ObservationManager(
    ObservationProvider[ResetContextT, StepContextT, ObservationGroups],
    Generic[ResetContextT, StepContextT],
):
    """Evaluate ordered observation terms and assemble named groups."""

    def __init__(self, terms: Mapping[str, ObservationTermSpec]) -> None:
        self.terms = dict(terms)
        self._validate_names()
        self.group_spaces: dict[str, DictSpace] = self._build_group_spaces()
        self.space = DictSpace(self.group_spaces)

    def reset(self, context: ResetContextT) -> ObservationResult[ObservationGroups]:
        groups: dict[str, dict[str, Any]] = {
            group: {} for group in self.group_spaces
        }
        info: dict[str, Any] = {}
        for name, spec in self.terms.items():
            if not spec.enabled:
                continue
            try:
                value = spec.term.reset(context)
                if not spec.space.contains(value):
                    raise ValueError("term output is outside its declared space")
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ManagerTermError(
                    manager="observation",
                    group=spec.group,
                    term=name,
                    cause=error,
                ) from error
            groups[spec.group][name] = value
            info[f"worldlab.observation.terms.{spec.group}.{name}.computed"] = True
        observation = {group: values for group, values in groups.items() if values}
        if not self.space.contains(observation):
            raise ValueError("observation manager produced output outside its space")
        return ObservationResult(observation, immutable_info(info))

    def compute(self, context: StepContextT) -> ObservationResult[ObservationGroups]:
        groups: dict[str, dict[str, Any]] = {
            group: {} for group in self.group_spaces
        }
        info: dict[str, Any] = {}
        for name, spec in self.terms.items():
            if not spec.enabled:
                continue
            try:
                value = spec.term.compute(context)
                if not spec.space.contains(value):
                    raise ValueError("term output is outside its declared space")
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ManagerTermError(
                    manager="observation",
                    group=spec.group,
                    term=name,
                    cause=error,
                ) from error
            groups[spec.group][name] = value
            info[f"worldlab.observation.terms.{spec.group}.{name}.computed"] = True

        observation = {group: values for group, values in groups.items() if values}
        if not self.space.contains(observation):
            raise ValueError("observation manager produced output outside its space")
        return ObservationResult(observation, immutable_info(info))

    def _validate_names(self) -> None:
        for name, spec in self.terms.items():
            if not name:
                raise ValueError("observation term names must be non-empty")
            if not spec.group:
                raise ValueError(f"observation term {name!r} group must be non-empty")
            if not _is_space(spec.space):
                raise TypeError(f"observation term {name!r} must declare a Space")

    def _build_group_spaces(self) -> dict[str, DictSpace]:
        groups: dict[str, dict[str, Space[Any]]] = {}
        for name, spec in self.terms.items():
            if spec.enabled:
                groups.setdefault(spec.group, {})[name] = spec.space
        return {group: DictSpace(spaces) for group, spaces in groups.items()}

def _is_space(value: Any) -> bool:
    return all(callable(getattr(value, method, None)) for method in ("sample", "contains", "seed"))
