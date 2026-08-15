"""Dependency-free stochastic frame generator for the WorldModel boundary."""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional, Tuple

from worldlab.core import WorldModel
from worldlab.data import WorldModelContext, WorldModelPrediction


Frame = Tuple[int, ...]


class RandomFrameWorldModel(WorldModel[int, Frame, int]):
    """Generate a random discrete frame conditioned on an action.

    This is intentionally not a diffusion implementation. It exercises the
    same context/sample boundary that a diffusion model can implement later.
    """

    def __init__(self, frame_size: int = 8, *, value_max: int = 255) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be greater than zero")
        if value_max <= 0:
            raise ValueError("value_max must be greater than zero")
        self.frame_size = frame_size
        self.value_max = value_max
        self._random = random.Random()

    def initialize(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> WorldModelContext[int, Frame]:
        del options
        self._random.seed(seed)
        frame = self._sample_frame()
        return WorldModelContext(context=0, state=frame, info={"step": 0})

    def sample_step(
        self,
        context: int,
        action: int,
    ) -> WorldModelPrediction[int, Frame]:
        del action
        next_context = context + 1
        frame = self._sample_frame()
        return WorldModelPrediction(
            context=next_context,
            state=frame,
            info={"step": next_context},
        )

    def _sample_frame(self) -> Frame:
        return tuple(
            self._random.randrange(self.value_max + 1)
            for _ in range(self.frame_size)
        )
