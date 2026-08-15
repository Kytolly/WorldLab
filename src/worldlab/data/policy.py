"""Policy inference outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar


ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class PolicyOutput(Generic[ActionT]):
    action: ActionT
    state: Any = None
    info: Mapping[str, Any] = field(default_factory=dict)
