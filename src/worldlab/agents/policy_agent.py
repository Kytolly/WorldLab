"""Adapt an inference policy to the Agent lifecycle."""

from __future__ import annotations

from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.core import Agent, Policy
from worldlab.data import PolicyOutput


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class PolicyAgent(Agent[ObservationT, ActionT], Generic[ObservationT, ActionT]):
    def __init__(self, policy: Policy[ObservationT, ActionT]) -> None:
        self.policy = policy

    def reset(self, *, seed: Optional[int] = None) -> None:
        self.policy.reset(seed=seed)

    def act(
        self,
        observation: ObservationT,
        *,
        info: Mapping[str, Any],
        training: bool,
        deterministic: bool,
    ) -> PolicyOutput[ActionT]:
        del training
        infer = getattr(type(self.policy), "infer", None)
        if infer is not None and infer is not Policy.infer:
            return self.policy.infer(
                observation,
                info=info,
                deterministic=deterministic,
            )
        return self.policy.act(observation, info=info, deterministic=deterministic)
