from __future__ import annotations

import numpy as np
import pytest

from worldlab import (
    CounterWorldModel,
    ExampleEnvironment,
    WorldModelStepResult,
    WorldModelSimulator,
    build_configured_demo,
    load_config,
)


def test_world_model_step_result_validates_chunk_outputs() -> None:
    result = WorldModelStepResult(
        context=1,
        state=np.zeros((2, 4), dtype=np.float32),
        frames=np.zeros((2, 3, 2, 2, 2), dtype=np.float32),
    )

    result.validate(chunk_size=2)
    assert not hasattr(result, "action")
    assert not hasattr(result, "reward")
    assert result.frames.shape == (2, 3, 2, 2, 2)
    assert result.state.shape == (2, 4)


def test_world_model_step_result_rejects_invalid_outputs() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        WorldModelStepResult(
            context=1,
            state=np.zeros((3, 4), dtype=np.float32),
            frames=np.zeros((3, 3, 2, 2, 2), dtype=np.float32),
        ).validate(chunk_size=2)

    with pytest.raises(ValueError, match="same leading sequence"):
        WorldModelStepResult(
            context=1,
            state=np.zeros((1, 4), dtype=np.float32),
            frames=np.zeros((2, 3, 2, 2, 2), dtype=np.float32),
        )


def test_world_model_simulator_propagates_normalized_chunk_fields() -> None:
    simulator = WorldModelSimulator(CounterWorldModel())
    reset = simulator.reset(options={"start": 0})
    step = simulator.step(1)

    assert reset.state == 0
    assert step.state == 1
    assert step.action == 1
    assert step.frames is None


def test_example_environment_keeps_model_and_task_boundaries_explicit() -> None:
    environment, _, options = build_configured_demo(load_config())
    try:
        reset = environment.reset(seed=0, options=options)
        step = environment.step(np.zeros((4, 16), dtype=np.float32))
    finally:
        environment.close()

    assert reset.observation.shape == (3, 3, 32, 32)
    assert isinstance(environment.unwrapped, ExampleEnvironment)
    assert step.observation.shape == (4, 16)
    assert step.reward == 1.0
    assert step.terminated is False
    assert step.info["action"].shape == (4, 16)
    assert step.info["frames"].shape == (4, 3, 3, 32, 32)
