"""Stateful simulator runtime backed by a high-level WorldModel."""

from __future__ import annotations

import time
from typing import Any, Generic, Mapping, Optional, TypeVar, cast

from worldlab.core import Simulator
from worldlab.data import (
    SIMULATION_CHUNK_INDEX,
    SIMULATION_MODEL_LATENCY_S,
    SIMULATION_OUTPUT,
    SIMULATION_STATE,
    SimulationReset,
    SimulationStep,
)
from worldlab.world_models.base import WorldModel


ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
_UNSET = object()


class WorldModelSimulator(Simulator[StateT, ActionT], Generic[ContextT, StateT, ActionT]):
    """Own episode state while delegating generation to a WorldModel."""

    def __init__(
        self,
        model: WorldModel[ContextT, StateT, ActionT],
        *,
        chunk_size: Optional[int] = None,
    ) -> None:
        self.model = model
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        self.chunk_size = chunk_size
        self._context: object = _UNSET
        self._state: object = _UNSET
        self._active = False
        self._closed = False
        self._chunk_index = 0

    @property
    def context(self) -> ContextT:
        if not self._active or self._context is _UNSET:
            raise RuntimeError("simulator has not been reset")
        return cast(ContextT, self._context)

    @property
    def state(self) -> StateT:
        if not self._active or self._state is _UNSET:
            raise RuntimeError("simulator has not been reset")
        return cast(StateT, self._state)

    @property
    def chunk_index(self) -> int:
        """Index of the next chunk that will be sent to the model."""

        return self._chunk_index

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> SimulationReset[StateT]:
        self._ensure_open()
        result = self.model.initialize(seed=seed, options=options)
        self._context = result.context
        self._state = result.state
        self._active = True
        self._chunk_index = 0
        return SimulationReset(result.state, result.info)

    def step(self, action: ActionT) -> SimulationStep[StateT]:
        self._ensure_open()
        chunk_index = self._chunk_index
        started_at = time.perf_counter()
        prediction = self.model.sample_step(self.context, action)
        model_latency_s = time.perf_counter() - started_at
        prediction.validate()
        self._context = prediction.context
        self._state = prediction.state
        info = dict(prediction.info)
        info[SIMULATION_CHUNK_INDEX] = chunk_index
        info[SIMULATION_MODEL_LATENCY_S] = model_latency_s
        info[SIMULATION_STATE] = prediction.state
        if prediction.output is not None:
            info[SIMULATION_OUTPUT] = prediction.output
        self._chunk_index += 1
        return SimulationStep(
            state=prediction.state,
            info=info,
        )

    def render(self) -> Any:
        self._ensure_open()
        return self.model.render(self.context, self.state)

    def close(self) -> None:
        if self._closed:
            return
        self.model.close()
        self._closed = True
        self._active = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("simulator is closed")
