from __future__ import annotations

import numpy as np
import pytest

from worldlab import (
    CounterWorldModel,
    ExampleEnvironment,
    SIMULATION_CHUNK_INDEX,
    SIMULATION_FRAMES,
    SIMULATION_MODEL_LATENCY_S,
    SIMULATION_OUTPUT,
    SIMULATION_STATE,
    WorldModelStepResult,
    WorldModelSimulator,
    build_configured_demo,
    load_config,
)


def test_world_model_step_result_validates_chunk_outputs() -> None:
    result = WorldModelStepResult(
        context=1,
        state=np.zeros((2, 4), dtype=np.float32),
        output={"opaque": "model-specific"},
    )

    result.validate()
    assert not hasattr(result, "action")
    assert not hasattr(result, "reward")
    assert result.output == {"opaque": "model-specific"}
    assert result.state.shape == (2, 4)


def test_world_model_step_result_does_not_interpret_output_shape() -> None:
    WorldModelStepResult(
        context=1,
        state=np.zeros((3, 4), dtype=np.float32),
        output=np.zeros((3, 3, 2, 2, 2), dtype=np.float32),
    ).validate()


def test_world_model_simulator_propagates_normalized_chunk_fields() -> None:
    simulator = WorldModelSimulator(CounterWorldModel())
    reset = simulator.reset(options={"start": 0})
    step = simulator.step(1)

    assert reset.state == 0
    assert step.state == 1
    assert not hasattr(step, "action")
    assert not hasattr(step, "frames")
    assert step.info[SIMULATION_CHUNK_INDEX] == 0
    assert step.info[SIMULATION_MODEL_LATENCY_S] >= 0.0
    assert step.info[SIMULATION_STATE] == 1
    assert SIMULATION_OUTPUT not in step.info
    assert simulator.chunk_index == 1

    simulator.reset(options={"start": 10})
    assert simulator.chunk_index == 0


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
    assert step.info[SIMULATION_CHUNK_INDEX] == 0
    assert step.info[SIMULATION_MODEL_LATENCY_S] >= 0.0
    assert step.info[SIMULATION_FRAMES].shape == (4, 3, 3, 32, 32)
    assert step.info[SIMULATION_OUTPUT].shape == (4, 3, 3, 32, 32)
    assert step.info[SIMULATION_STATE].shape == (4, 16)
