"""High-level generative world-model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.data import WorldModelContext, WorldModelPrediction


ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class WorldModel(ABC, Generic[ContextT, StateT, ActionT]):
    """A model that samples the next world state from context and action.

    The context may contain a history window, latent state, diffusion
    conditioning features, cached encodings, or any other model-specific data.
    Core code deliberately does not depend on a tensor or diffusion library.
    """

    @abstractmethod
    def initialize(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> WorldModelContext[ContextT, StateT]:
        raise NotImplementedError

    @abstractmethod
    def sample_step(
        self,
        context: ContextT,
        action: ActionT,
    ) -> WorldModelPrediction[ContextT, StateT]:
        """Sample one conditional transition from the model."""

        raise NotImplementedError

    def rollout(
        self,
        context: ContextT,
        actions: list[ActionT],
    ) -> list[WorldModelPrediction[ContextT, StateT]]:
        """Optional multi-step rollout; concrete models may override it."""

        raise NotImplementedError("multi-step rollout is not implemented by this model")

    def render(self, context: ContextT, state: StateT) -> Any:
        return None

    def close(self) -> None:
        return None
